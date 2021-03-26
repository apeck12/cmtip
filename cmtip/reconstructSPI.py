import argparse, time, os
import numpy as np
import skopi as sk
import h5py

import cmtip.phasing as phaser
from cmtip.autocorrelation import autocorrelation
import cmtip.alignment as alignment

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


def load_h5(input_file, load_ivol=False):
    """
    Load h5 input file.

    :param input_file: input h5 file
    :param data: dict containing contents of input h5 file
    """
    print("Loading data from %s" %input_file)  
    data = dict()
    
    # retrieve attributes
    with h5py.File(input_file, 'r') as f:
        n_images = f.attrs['n_images']
        n_pixels = f.attrs['n_pixels']
        reciprocal_extent = f.attrs['reciprocal_extent']
        
    data['intensities'] = np.zeros((n_images, 1, n_pixels, n_pixels)).astype(np.float32)
    data['orientations'] = np.zeros((n_images, 4)).astype(np.float32)
    data['pixel_position_reciprocal'] = np.zeros((3, 1, n_pixels, n_pixels)).astype(np.float32)
    if load_ivol:
        data['volume'] = np.zeros((151,151,151), dtype=np.complex128) 
    
    with h5py.File(input_file, 'r') as f:
        for key in data.keys():
            data[key][:] = f[key][:]
            
    data['n_images'], data['n_pixels'] = n_images, n_pixels
    data['reciprocal_extent'] = reciprocal_extent

    return data


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


def clip_data(arr, n_remove):
    """
    Clip each image in a series of images with a clipping border n_remove wide.
    
    :param arr: input array to be clipped of shape (n_images,1,n_pixels,n_pixels)
    :param n_remove: width of border in pixels to be removed
    :return c_arr: clipped array 
    """
    c_arr = np.zeros((arr.shape[0],1,arr.shape[2]-2*n_remove,arr.shape[3]-2*n_remove))
    
    for i in range(arr.shape[0]):
        c_arr[i][:] = arr[i][0][n_remove:-n_remove,n_remove:-n_remove]

    return c_arr.astype(np.float32)


def run_mtip(data, M, output, aligned=True, n_iterations=10):
    """
    Run MTIP algorithm (though without orientation matching for now).
    
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
