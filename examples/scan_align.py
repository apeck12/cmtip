import numpy as np
import time, os, argparse
from cmtip.prep_data import *
from cmtip.alignment import align, errors

"""
Examine alignment error as a function of different parameters:
1. trimming vs clipping data (resolution to corner vs edge, circular vs square mask)
2. number of reference images
3. high-resolution limit of data to keep
4. order of resolution-weighting (1/s^order)
by performing orientation matching against the ideal autocorrelation.
"""

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser(description="Examine alignment errors as a function of parameters.")
    parser.add_argument('-i', '--input', help='Input h5 file containing intensities and exp information.')
    parser.add_argument('-n', '--n_ref', help='Number of reference slices', required=True, nargs='+', type=int)
    parser.add_argument('-r', '--resolution', help='Resolution of data to keep', required=True, nargs='+', type=int)
    parser.add_argument('-w', '--weight', help='Order of inverse resolution weighting', required=True, nargs='+', type=int)
    parser.add_argument('-o', '--output', help='Path to output directory', required=True, type=str)
    parser.add_argument('-t', '--n_images', help='Total number of images to process', required=True, type=int)
    parser.add_argument('-s', '--symmetry', help='Symmetry-factor of particle', required=False, type=int, default=1)
    parser.add_argument('-b', '--bin_factor', help='Factor by which to bin data', required=False, type=int, default=1)

    return vars(parser.parse_args())


def assess_alignment(data, n_ref, res_limit=0, weight=0):
    """
    Perform alignment using data to given resolution and n_ref reference slices through 
    ideal autocorrelation and compute error.

    :param data: dictionary from simulation
    :param n_ref: number of reference orientations
    :param res_limit: highest resolution data to keep; if 0, use all data
    :param weight: order for inverse resolution-weighting; if 0, uniform weights
    :return trim_error: error array for trimmed data
    :return clip_error: errory array for clipped data
    """
    for clip in [True, False]:
        if clip:
            pixel_position_reciprocal, intensities = clip_dataset(data['pixel_position_reciprocal'], 
                                                                  data['intensities'], res_limit)
        else:
            pixel_position_reciprocal, intensities = trim_dataset(data['pixel_index_map'], 
                                                                  data['pixel_position_reciprocal'], 
                                                                  data['intensities'], 
                                                                  data['det_shape'], 
                                                                  res_limit)        
        calc_quat = align.match_orientations(0,
                                             pixel_position_reciprocal,
                                             data['reciprocal_extent'],
                                             intensities,
                                             data['ac'],
                                             n_ref,
                                             order=weight)
        if clip:
            clip_error = errors.alignment_error(calc_quat, data['orientations'])
        else:
            trim_error = errors.alignment_error(calc_quat, data['orientations'])
        
    return trim_error, clip_error


def main():

    # gather command line input and set up storage dictionary
    args = parse_input()
    if not os.path.isdir(args['output']):
        os.mkdir(args['output'])

    # simulate images and reconstruct
    data = load_h5(args['input'], load_ivol=True, end=args['n_images'])
    ivol = np.square(np.abs(data['volume']))
    data['ac'] = np.fft.fftshift(np.abs(np.fft.ifftn(ivol))).astype(np.float32) # acf is IFT(FT(I))

    # optionally bin data
    if args['bin_factor']!=1:
        for key in ['intensities', 'pixel_position_reciprocal']:
            data[key] = bin_data(data[key], args['bin_factor'], data['det_shape'])
        data['reciprocal_extent'] = np.linalg.norm(data['pixel_position_reciprocal'], axis=0).max()
        data['pixel_index_map'] = bin_pixel_index_map(data['pixel_index_map'], args['bin_factor'])
        data['det_shape'] = data['pixel_index_map'].shape[:3]

    # scan over alignment parameters 
    for nr in args['n_ref']:
        for res in args['resolution']:
            for w in args['weight']:
                start_time = time.time()

                trim_err, clip_err = assess_alignment(data, nr, res, w)
                trim_err_mean = errors.mean_error(trim_err, symmetry=args['symmetry'])
                clip_err_mean = errors.mean_error(clip_err, symmetry=args['symmetry'])
                print(f"Trim error for {nr} orientations, data to {res} Angstrom, s-weighting order {w} is: {trim_err_mean}")
                print(f"Clip error for {nr} orientations, data to {res} Angstrom, s-weighting order {w} is {clip_err_mean}")
                np.save(os.path.join(args['output'], "error_trim_n%ir%iw%i.npy" %(nr,res,w)), trim_err)
                np.save(os.path.join(args['output'], "error_clip_n%ir%iw%i.npy" %(nr,res,w)), clip_err)

                print("elapsed time is %.2f" %((time.time() - start_time)/60.0))

if __name__ == '__main__':
    main()
