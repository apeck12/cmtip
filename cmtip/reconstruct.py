import argparse, time, os
import numpy as np
import skopi as sk
import h5py

import cmtip.phasing as phaser
from cmtip.autocorrelation import autocorrelation
import cmtip.alignment as alignment
from cmtip.prep_data import load_h5, clip_data

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser(description="Reconstruct an SPI dataset using the MTIP algorithm.")
    parser.add_argument('-i', '--input', help='Input h5 file containing intensities and exp information.')
    parser.add_argument('-m', '--M', help='Cubic length of reconstruction volume', required=True, type=int)
    parser.add_argument('-o', '--output', help='Path to output directory', required=True, type=str)
    parser.add_argument('-n', '--niter', help='Number of MTIP iterations', required=False, type=int, default=10)
    parser.add_argument('-a', '--aligned', help='Alignment from reference quaternions', action='store_true')

    return vars(parser.parse_args())


def save_output(generation, output, ac, rho_, orientations=None):
    """
    Save output from each MTIP iteration.

    :param generation: mtip iteration
    :param output: output directory
    :param ac: 3d array of autocorrelation
    :param rho_: 3d array of fft shifted density
    :param orientations: array of quaternions 
    """
    rho_unshifted = np.fft.ifftshift(rho_)
    phaser.maps.save_mrc(os.path.join(output, "density%i.mrc" %generation), rho_unshifted)
    phaser.maps.save_mrc(os.path.join(output, "ac%i.mrc" %generation), ac)
    
    if orientations is not None:
        np.save(os.path.join(output, "orientations%i.npy" %generation), orientations)
    return


def run_mtip(data, M, output, aligned=True, n_iterations=10):
    """
    Run MTIP algorithm.
    
    :param data: dictionary containing images, pixel positions, orientations, etc.
    :param M: length of cubic autocorrelation volume
    :param output: path to output directory
    :param aligned: if False use ground truth quaternions
    :param n_iterations: number of MTIP iterations to run, default=10
    """  
    print("Running MTIP")
    start_time = time.time()
    
    # alignment parameters
    nclip, n_ref = 144, 3000

    # iteration 0: ac_estimate is unknown
    generation = 0
    orientations = None
    if aligned:
        print("Using ground truth quaternions")
        orientations = data['orientations']

    ac = autocorrelation.solve_ac(generation,
                                  data['pixel_position_reciprocal'],
                                  data['reciprocal_extent'],
                                  data['intensities'],
                                  M,
                                  orientations=orientations)
    ac_phased, support_, rho_ = phaser.phase(generation, ac)
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
        ac = autocorrelation.solve_ac(generation,
                                      data['pixel_position_reciprocal'],
                                      data['reciprocal_extent'],
                                      data['intensities'],
                                      M,
                                      orientations=orientations.astype(np.float32),
                                      ac_estimate=ac_phased.astype(np.float32))
        # phase
        ac_phased, support_, rho_ = phaser.phase(generation, ac, support_, rho_)
        save_output(generation, output, ac, rho_, orientations)

    print("elapsed time is %.2f" %((time.time() - start_time)/60.0))
    return


def main():

    # gather command line input and set up storage dictionary
    args = parse_input()
    if not os.path.isdir(args['output']):
        os.mkdir(args['output'])

    # reconstruct density from simulated diffraction images
    data = load_h5(args['input'])
    run_mtip(data, args['M'], args['output'], aligned=args['aligned'], n_iterations=args['niter'])


if __name__ == '__main__':
    main()
