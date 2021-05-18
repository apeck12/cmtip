import argparse, time, os
import numpy as np
import skopi as sk
import h5py, sys
from mpi4py import MPI

import cmtip.alignment as alignment
from cmtip.prep_data import *
from cmtip.reconstruct import save_output
from cmtip.phasing import phase_mpi as phaser
from cmtip.autocorrelation import autocorrelation_mpi as autocorrelation


def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser(description="Reconstruct an SPI dataset using the MTIP algorithm with MPI parallelization.")
    parser.add_argument('-i', '--input', help='Input h5 file containing intensities and exp information.')
    parser.add_argument('-m', '--M', help='Cubic length of reconstruction volume', required=True, nargs='+', type=int)
    parser.add_argument('-o', '--output', help='Path to output directory', required=True, type=str)
    parser.add_argument('-t', '--n_images', help='Total number of images to process', required=True, type=int)
    parser.add_argument('-n', '--niter', help='Number of MTIP iterations', required=False, type=int, default=10)
    parser.add_argument('-r', '--res_limit_ac', help='Resolution limit for solving AC at each iteration, overrides niter', 
                        required=False, nargs='+', type=float)
    parser.add_argument('-b', '--bin_factor', help='Factor by which to bin data', required=False, type=int, default=1)
    parser.add_argument('-ap', '--a_params', help='Alignment parameters: (n_reference, res_limit, weighting_order)',
                        required=False, nargs=3, type=float, default=[2500, 8.0, -2])
    parser.add_argument('-a', '--aligned', help='Alignment from reference quaternions', action='store_true')

    return vars(parser.parse_args())


def run_mtip_mpi(comm, data, M, output, aligned=True, n_iterations=10, res_limit_ac=None, a_params=[2500,8,-2]):
    """
    Run MTIP algorithm.
    
    :param comm: MPI intracommunicator instance
    :param data: dictionary containing images, pixel positions, orientations, etc.
    :param M: array of length n_iterations of cubic autocorrelation volume
    :param output: path to output directory
    :param aligned: if False use ground truth quaternions
    :param n_iterations: number of MTIP iterations to run, default=10
    :param res_limit_ac: resolution limit to which to solve AC at each iteration. if None use full res.
    :param a_params: list of alignment parameters, (n_reference, res_limit, weight_order)
    """  
    print("Running MTIP")
    start_time = time.time()
    rank = comm.rank
    
    # track resolution limits at each iteration, retrieve alignment parameters
    if res_limit_ac is None:
        res_limit_ac = np.zeros(n_iterations)
    reciprocal_extents = np.zeros(n_iterations)
    n_ref, res_limit_align, weight_order = int(a_params[0]), a_params[1], a_params[2]

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
                                      orientations=orientations)
    ac_phased, support_, rho_ = phaser.phase_mpi(comm, generation, ac)
    if rank == 0:
        save_output(generation, output, ac, rho_, orientations=None)

    # iterations 1-n_iterations: ac_estimate from phasing
    for generation in range(1, n_iterations):
        # align slices using clipped data
        if not aligned:
            print(f"Aligning using {n_ref} ref slices to {res_limit_align} A res, order weight of {weight_order}")
            pixel_position_reciprocal = clip_data(data['pixel_position_reciprocal'], 
                                                  data['pixel_position_reciprocal'],
                                                  res_limit_align)
            intensities = clip_data(data['intensities'], 
                                    data['pixel_position_reciprocal'], res_limit_align)

            orientations = alignment.match_orientations(generation,
                                                        pixel_position_reciprocal,
                                                        reciprocal_extent, # needs to match resolution of ac
                                                        intensities,
                                                        ac_phased.astype(np.float32),
                                                        n_ref,
                                                        order=weight_order)
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

        # if M or resolution has changed, resize the support_ and rho_ from previous iteration
        if (M[generation] != M[generation-1]) or (res_limit_ac[generation] != res_limit_ac[generation-1]):
            support_, rho_ = None, None
#            support_, rho_ = phaser.resize_mpi(comm, support_, rho_, M[generation], 
#                                               reciprocal_extents[generation-1], reciprocal_extents[generation])

        # solve for autocorrelation
        ac = autocorrelation.solve_ac_mpi(comm,
                                          generation,
                                          pixel_position_reciprocal,
                                          reciprocal_extent,
                                          intensities,
                                          M[generation],
                                          orientations=orientations)
        # phase
        ac_phased, support_, rho_ = phaser.phase_mpi(comm, generation, ac, support_, rho_)
        if rank == 0:
            save_output(generation, output, ac, rho_, orientations=None)
    
    print("elapsed time is %.2f" %((time.time() - start_time)/60.0))
    return


def main():

    # set up MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    # gather command line input and set up storage dictionary  
    args = parse_input()
    if rank == 0:
        if not os.path.isdir(args['output']):
            os.mkdir(args['output'])
    n_images_batch = int(args['n_images'] / size)
    
    # set resolution and volume size at each iteration
    if args['res_limit_ac'] is not None:
        args['niter'] = len(args['res_limit_ac'])
    if len(args['M']) == 1:
        args['M'] = args['niter'] * args['M']
    if len(args['M']) != args['niter']:
        print("Error: M array does not match number of MTIP iterations")
        sys.exit()

    # load subset of data onto each rank and bin if requested
    data = load_h5(args['input'], start=rank*n_images_batch, end=(rank+1)*n_images_batch)
    if args['bin_factor']!=1:
        for key in ['intensities', 'pixel_position_reciprocal']:
            data[key] = bin_data(data[key], args['bin_factor'], data['det_shape'])
        data['reciprocal_extent'] = np.linalg.norm(data['pixel_position_reciprocal'], axis=0).max()
        data['pixel_index_map'] = bin_pixel_index_map(data['pixel_index_map'], args['bin_factor'])
        data['det_shape'] = data['pixel_index_map'].shape[:3]

    # run MTIP to reconstruct density from simulated diffraction images
    run_mtip_mpi(comm, data, args['M'], args['output'], aligned=args['aligned'], 
                 n_iterations=args['niter'], res_limit_ac=args['res_limit_ac'], a_params=args['a_params'])


if __name__ == '__main__':
    main()
