import argparse, time, os
import numpy as np
import skopi as sk
import h5py

import cmtip.phasing as phaser
from cmtip.autocorrelation import autocorrelation
import cmtip.alignment as alignment
from cmtip.prep_data import *

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser(description="Reconstruct an SPI dataset using the MTIP algorithm.")
    parser.add_argument('-i', '--input', help='Input h5 file containing intensities and exp information.')
    parser.add_argument('-m', '--M', help='Cubic length of reconstruction volume', required=True, type=int)
    parser.add_argument('-o', '--output', help='Path to output directory', required=True, type=str)
    parser.add_argument('-n', '--niter', help='Number of MTIP iterations', required=False, type=int, default=10)
    parser.add_argument('-t', '--n_images', help='Total number of images to process', required=False, type=int)
    parser.add_argument('-b', '--bin_factor', help='Factor by which to bin data', required=False, type=int, default=1)
    parser.add_argument('-g', '--use_gpu', help='Use cufinufft for GPU-accelerated NUFFT calculations', action='store_true')
    parser.add_argument('-a', '--aligned', help='Alignment from reference quaternions', action='store_true')
    parser.add_argument('-cpt', '--checkpoint', help='Intermediate checkpoint file', required=False, type=str)

    return vars(parser.parse_args())


def run_mtip(data, M, output, aligned=True, n_iterations=10, use_gpu=False, checkpoint=load_checkpoint()):
    """
    Run MTIP algorithm.
    
    :param data: dictionary containing images, pixel positions, orientations, etc.
    :param M: length of cubic autocorrelation volume
    :param output: path to output directory
    :param aligned: if False use ground truth quaternions
    :param n_iterations: number of MTIP iterations to run, default=10
    :param use_gpu: boolean; if True, use cufinufft for NUFFT calculations
    :param checkpoint: dictionary containing intermediate results
    """  
    print("Running MTIP")
    start_time = time.time()
    
    # alignment parameters
    n_ref, res_limit = 5000, 9
    if aligned:
        print("Using ground truth quaternions")
        checkpoint['orientations'] = data['orientations']

    # iteration 0: ac_estimate is unknown
    if checkpoint['generation'] == 0:
        ac = autocorrelation.solve_ac(checkpoint['generation'],
                                      data['pixel_position_reciprocal'],
                                      data['reciprocal_extent'],
                                      data['intensities'],
                                      M,
                                      orientations=checkpoint['orientations'],
                                      use_gpu=use_gpu)
        checkpoint['ac_phased'], checkpoint['support_'], checkpoint['rho_'] = phaser.phase(checkpoint['generation'], ac)
        save_checkpoint(0, output, checkpoint)
    
    # iterations 1-n_iterations: ac_estimate from phasing
    for generation in range(checkpoint['generation']+1, n_iterations):
        # align slices using clipped data
        if not aligned:
            pixel_position_reciprocal = clip_data(data['pixel_position_reciprocal'],
                                                  data['pixel_position_reciprocal'],
                                                  res_limit)
            intensities = clip_data(data['intensities'],
                                    data['pixel_position_reciprocal'], res_limit)
            checkpoint['orientations'] = alignment.match_orientations(generation,
                                                                      pixel_position_reciprocal,
                                                                      data['reciprocal_extent'],
                                                                      intensities,
                                                                      checkpoint['ac_phased'].astype(np.float32),
                                                                      n_ref,
                                                                      use_gpu=use_gpu)
        # solve for autocorrelation
        ac = autocorrelation.solve_ac(generation,
                                      data['pixel_position_reciprocal'],
                                      data['reciprocal_extent'],
                                      data['intensities'],
                                      M,
                                      orientations=checkpoint['orientations'].astype(np.float32),
                                      use_gpu=use_gpu,
                                      ac_estimate=checkpoint['ac_phased'].astype(np.float32))
        # phase
        checkpoint['ac_phased'], checkpoint['support_'], checkpoint['rho_'] = phaser.phase(generation, 
                                                                                           ac, 
                                                                                           checkpoint['support_'], 
                                                                                           checkpoint['rho_'])
        save_checkpoint(generation, output, checkpoint)

    print("elapsed time is %.2f" %((time.time() - start_time)/60.0))
    return


def main():

    # gather command line input and set up storage dictionary
    args = parse_input()
    if not os.path.isdir(args['output']):
        os.mkdir(args['output'])

    # load data and bin if requested
    if args['n_images'] is not None:
        data = load_h5(args['input'], start=0, end=args['n_images'])
    else:
        data = load_h5(args['input'])

    if args['bin_factor']!=1:
        for key in ['intensities', 'pixel_position_reciprocal']:
            data[key] = bin_data(data[key], args['bin_factor'], data['det_shape'])
        data['reciprocal_extent'] = np.linalg.norm(data['pixel_position_reciprocal'], axis=0).max()
        data['pixel_index_map'] = bin_pixel_index_map(data['pixel_index_map'], args['bin_factor'])
        data['det_shape'] = data['pixel_index_map'].shape[:3]

    # load intermediate checkpoint file (or gather dummy values)
    checkpoint = load_checkpoint(args['checkpoint'])

    # reconstruct density from simulated diffraction images 
    run_mtip(data, args['M'], args['output'], aligned=args['aligned'], 
             n_iterations=args['niter'], use_gpu=args['use_gpu'], checkpoint=checkpoint)


if __name__ == '__main__':
    main()
