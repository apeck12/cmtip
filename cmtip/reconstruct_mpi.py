import argparse, time, os
import numpy as np
import skopi as sk
import h5py, sys
from mpi4py import MPI

import cmtip.alignment as alignment
from cmtip.prep_data import *
from cmtip.phasing import phase_mpi as phaser
from cmtip.phasing import maps
from cmtip.autocorrelation import autocorrelation_mpi as autocorrelation


def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser(description="Reconstruct an SPI dataset using the MTIP algorithm with MPI parallelization.")
    parser.add_argument('-i', '--input', help='Input h5 file containing intensities and exp information.')
    parser.add_argument('-m', '--M', help='Cubic length of reconstruction volume at each resolution', required=True, nargs='+', type=int)
    parser.add_argument('-o', '--output', help='Path to output directory', required=True, type=str)
    parser.add_argument('-t', '--n_images', help='Total number of images to process', required=True, type=int)
    parser.add_argument('-n', '--niter', help='Number of MTIP iterations at each resolution/size', required=False, type=int, default=10)
    parser.add_argument('-r', '--res_limit_ac', help='Resolution limit for solving AC at each iteration', required=False, nargs='+', type=float)
    parser.add_argument('-b', '--bin_factor', help='Factor by which to bin data', required=False, type=int, default=1)
    parser.add_argument('-ar', '--a_nref', help='Number of model slices for alignment at each resolution', required=True, nargs='+',type=int)
    parser.add_argument('-ab', '--a_nbatches', help='Number of batches for alignment at each resolution', required=True, nargs='+',type=int)
    parser.add_argument('-aw', '--a_weighting', help='Alignment parameters: (res, order)', required=False, nargs=2, type=float, default=[8.0,-2])
    parser.add_argument('-g', '--use_gpu', help='Use cufinufft for GPU-accelerated NUFFT calculations', action='store_true')
    parser.add_argument('-a', '--aligned', help='Alignment from reference quaternions', action='store_true')

    return vars(parser.parse_args())


def save_output(generation, output, ac, rho_):
    """
    Save output from each MTIP iteration.

    :param generation: mtip iteration
    :param output: output directory
    :param ac: 3d array of autocorrelation volumes
    :param rho_: 3d array of fft shifted density volumes
    """
    rho_unshifted = np.fft.ifftshift(rho_)
    maps.save_mrc(os.path.join(output, f"density{generation}.mrc"), rho_unshifted)
    
    return


def run_mtip_mpi(comm, data, M, output, a_params, use_gpu=False, aligned=True, n_iterations=10, res_limit_ac=None):
    """
    Run MTIP algorithm.
    
    :param comm: MPI intracommunicator instance
    :param data: dictionary containing images, pixel positions, orientations, etc.
    :param M: array of length n_iterations of cubic autocorrelation volume
    :param output: path to output directory
    :param use_gpu: boolean; if True, use GPU-based NUFFT calculations
    :param a_params: dictionary of alignment parameters
    :param aligned: if False use ground truth quaternions
    :param n_iterations: number of MTIP iterations to run, default=10
    :param res_limit_ac: resolution limit to which to solve AC at each iteration. if None use full res.
    """  
    print("Running MTIP")
    start_time = time.time()
    rank = comm.rank
    
    # track resolution limits at each iteration
    if res_limit_ac is None:
        res_limit_ac = np.zeros(n_iterations)
    reciprocal_extents = np.zeros(n_iterations)

    # iteration 0: ac_estimate is unknown
    generation = 0
    orientations = None
    if aligned:
        print("Using ground truth quaternions")
        orientations = data['orientations']

    # clip resolution
    pixel_position_reciprocal, intensities = trim_dataset(data['pixel_index_map'], 
                                                          data['pixel_position_reciprocal'], 
                                                          data['intensities'], 
                                                          data['det_shape'], 
                                                          res_limit_ac[generation])
    reciprocal_extent = np.linalg.norm(pixel_position_reciprocal, axis=0).max()
    reciprocal_extents[generation] = reciprocal_extent
    print(f"Iteration {generation}: trimmed data to {1e10/reciprocal_extent} A resolution")

    ac = autocorrelation.solve_ac_mpi(comm, 
                                      generation,
                                      pixel_position_reciprocal,
                                      reciprocal_extent,
                                      intensities,
                                      M[generation],
                                      orientations=orientations,
                                      use_gpu=use_gpu)
    ac_phased, support_, rho_ = phaser.phase_mpi(comm, generation, ac)
    if rank == 0:
        save_output(generation, output, ac, rho_)

    # iterations 1-n_iterations: ac_estimate from phasing
    for generation in range(1, n_iterations):
        # align slices using clipped data
        if not aligned:
            print(f"Aligning using {a_params['n_ref'][generation]} model slices to {a_params['res_limit']} A resolution")
            pixel_position_reciprocal = clip_data(data['pixel_position_reciprocal'], 
                                                  data['pixel_position_reciprocal'],
                                                  a_params['res_limit'])
            intensities = clip_data(data['intensities'], 
                                    data['pixel_position_reciprocal'], a_params['res_limit'])

            orientations = alignment.match_orientations(generation,
                                                        pixel_position_reciprocal,
                                                        reciprocal_extent, # needs to match resolution of ac
                                                        intensities,
                                                        ac_phased.astype(np.float32),
                                                        a_params['n_ref'][generation],
                                                        nbatches=a_params['nbatches'][generation],
                                                        use_gpu=use_gpu,
                                                        order=a_params['order'])
            
            # save orientations from each rank (probably faster than broadcasting?)
            np.save(os.path.join(output, f"quats{generation}_r{rank}.npy"), orientations)

        else:
            print("Using ground truth quaternions")
            orientations = data['orientations']

        # trim data for solving autocorrelation
        pixel_position_reciprocal, intensities = trim_dataset(data['pixel_index_map'], 
                                                              data['pixel_position_reciprocal'], 
                                                              data['intensities'], 
                                                              data['det_shape'], 
                                                              res_limit_ac[generation])
        reciprocal_extent = np.linalg.norm(pixel_position_reciprocal, axis=0).max()
        reciprocal_extents[generation] = reciprocal_extent
        print(f"Iteration {generation}: trimmed data to {1e10/reciprocal_extent} A resolution")

        # if M or resolution has changed, do not use support_ and rho_ from previous iteration
        if (M[generation] != M[generation-1]) or (res_limit_ac[generation] != res_limit_ac[generation-1]):
            support_, rho_ = None, None

        # solve for autocorrelation
        ac = autocorrelation.solve_ac_mpi(comm,
                                          generation,
                                          pixel_position_reciprocal,
                                          reciprocal_extent,
                                          intensities,
                                          M[generation],
                                          orientations=orientations,
                                          use_gpu=use_gpu)
        # phase
        ac_phased, support_, rho_ = phaser.phase_mpi(comm, generation, ac, support_, rho_)
        if rank == 0:
            save_output(generation, output, ac, rho_)
    
    print("elapsed time is %.2f" %((time.time() - start_time)/60.0))
    return


def main():

    # set up MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    # gather command line input and set up output directory
    args = parse_input()
    if rank == 0:
        if not os.path.isdir(args['output']):
            os.mkdir(args['output'])
    
    # set resolution and volume size at each iteration 
    if len(args['M']) != len(args['res_limit_ac']) != len(args['a_nref']) != len(args['a_nbatches']):
        print("Error: lengths of res_limit_ac, M, a_nref, and a_batches inputs must match.")
        sys.exit()
    args['M'] = np.repeat(np.array(args['M']), args['niter'])
    args['res_limit_ac'] = np.repeat(np.array(args['res_limit_ac']), args['niter'])

    # assemble alignment parameters
    args['a_params'] = dict()
    args['a_params']['nbatches'] = np.repeat(np.array(args['a_nbatches']), args['niter']).astype(int)
    args['a_params']['n_ref'] = np.repeat(np.array(args['a_nref']), args['niter']).astype(int)
    args['a_params']['res_limit'], args['a_params']['order'] = args['a_weighting']

    # load subset of data onto each rank and bin if requested
    n_images_batch = int(args['n_images'] / size)
    data = load_h5(args['input'], start=rank*n_images_batch, end=(rank+1)*n_images_batch)
    if args['bin_factor']!=1:
        for key in ['intensities', 'pixel_position_reciprocal']:
            data[key] = bin_data(data[key], args['bin_factor'], data['det_shape'])
        data['reciprocal_extent'] = np.linalg.norm(data['pixel_position_reciprocal'], axis=0).max()
        data['pixel_index_map'] = bin_pixel_index_map(data['pixel_index_map'], args['bin_factor'])
        data['det_shape'] = data['pixel_index_map'].shape[:3]

    # run MTIP to reconstruct density from simulated diffraction images
    run_mtip_mpi(comm, data, args['M'], args['output'], aligned=args['aligned'], use_gpu=args['use_gpu'],
                 n_iterations=len(args['M']), res_limit_ac=args['res_limit_ac'], a_params=args['a_params'])

    # compile orientations arrays and tidy up
    if rank == 0:
        for label in ['quats']:
            for generation in range(1,len(args['M'])):
                temp = np.zeros((args['n_images'], 4))
                for nr in range(size):
                    fname = os.path.join(args['output'], f"{label}{generation}_r{nr}.npy")
                    temp[nr*n_images_batch:(nr+1)*n_images_batch,:] = np.load(fname)
                    os.system(f'rm {fname}')
                np.save(os.path.join(args['output'], f"{label}{generation}.npy"), temp)


if __name__ == '__main__':
    main()
