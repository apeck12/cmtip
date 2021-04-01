import argparse, time, os
import numpy as np
import skopi as sk
import h5py
from mpi4py import MPI

import cmtip.alignment as alignment
from cmtip.prep_data import load_h5, clip_data
from cmtip.reconstruct import save_output
import cmtip.phasing as phaser
from cmtip.autocorrelation import autocorrelation_mpi as autocorrelation

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser(description="Reconstruct an SPI dataset using the MTIP algorithm with MPI parallelization.")
    parser.add_argument('-i', '--input', help='Input h5 file containing intensities and exp information.')
    parser.add_argument('-m', '--M', help='Cubic length of reconstruction volume', required=True, type=int)
    parser.add_argument('-o', '--output', help='Path to output directory', required=True, type=str)
    parser.add_argument('-t', '--n_images', help='Total number of images to process', required=True, type=int)
    parser.add_argument('-n', '--niter', help='Number of MTIP iterations', required=False, type=int, default=10)
    parser.add_argument('-a', '--aligned', help='Alignment from reference quaternions', action='store_true')

    return vars(parser.parse_args())


def run_mtip_mpi(comm, data, M, output, aligned=True, n_iterations=10):
    """
    Run MTIP algorithm (though without orientation matching for now).
    
    :param comm: MPI intracommunicator instance
    :param data: dictionary containing images, pixel positions, orientations, etc.
    :param M: length of cubic autocorrelation volume
    :param output: path to output directory
    :param aligned: if False use ground truth quaternions
    :param n_iterations: number of MTIP iterations to run, default=10
    """  
    print("Running MTIP")
    start_time = time.time()
    rank = comm.rank
    
    # alignment parameters
    nclip, n_ref = 144, 3000

    # iteration 0: ac_estimate is unknown
    generation = 0
    orientations = None
    if aligned:
        print("Using ground truth quaternions")
        orientations = data['orientations']

    ac = autocorrelation.solve_ac_mpi(comm, 
                                      generation,
                                      data['pixel_position_reciprocal'],
                                      data['reciprocal_extent'],
                                      data['intensities'],
                                      M,
                                      orientations=orientations)
    ac_phased, support_, rho_ = phaser.phase_mpi(comm, generation, ac)
    if rank == 0:
        save_output(generation, output, ac, rho_, orientations=None)

    # iterations 1-n_iterations: ac_estimate from phasing
    for generation in range(1, n_iterations):
        # align slices using clipped data
        if not aligned:
            pixel_position_reciprocal = clip_data(data['pixel_position_reciprocal'], nclip)
            intensities = clip_data(data['intensities'], nclip)
            orientations = alignment.match_orientations(generation,
                                                        pixel_position_reciprocal,
                                                        data['reciprocal_extent'],
                                                        intensities,
                                                        ac_phased.astype(np.float32),
                                                        n_ref)
        else:
            print("Using ground truth quaternions")
            orientations = data['orientations']

        # solve for autocorrelation
        ac = autocorrelation.solve_ac_mpi(comm,
                                          generation,
                                          data['pixel_position_reciprocal'],
                                          data['reciprocal_extent'],
                                          data['intensities'],
                                          M,
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

    # reconstruct density from simulated diffraction images
    data = load_h5(args['input'], start=rank*n_images_batch, end=(rank+1)*n_images_batch)
    run_mtip_mpi(comm, data, args['M'], args['output'], aligned=args['aligned'], n_iterations=args['niter'])


if __name__ == '__main__':
    main()
