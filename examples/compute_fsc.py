import numpy as np
import time, os, argparse
import glob, mrcfile
from cmtip.phasing import *

"""
Compute FSC for a set of MTIP solutions by comparison to the reference density. For 
each density map, both the solution and its inversion (due to handedness ambiguity) 
are tested, and the higher resolution score is kept.
"""

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser(description="Compute FSC relative to refrence from a series of MRC files.")
    parser.add_argument('-i', '--input', help='Input PDB files in glob-readable format', required=True, type=str)
    parser.add_argument('-o', '--output', help='Output for saving resolutions array', required=True, type=str)
    parser.add_argument('-p', '--pdb_file', help='PDB file of reference structure', required=True, type=str)
    parser.add_argument('-m', '--M', help='Cubic length of reconstruction volume', required=True, type=int)
    parser.add_argument('-r', '--resolution', help='Corner resolution of autocorrelation', required=True, type=float)
    parser.add_argument('-s', '--sigma', help='Sigma for Gaussian-filtering volumes prior to alignment', required=True, type=float)
    parser.add_argument('-d', '--spacing', help='d-spacing of FSC curve in per Angstrom', required=False, default=0.01, type=float)

    return vars(parser.parse_args())


def calculate_res(ref_density, est_density, sigma, max_res, spacing, n_iterations=10, tol=0.01):
    """
    Estimate resolution of the estimated density by aligning it to the reference density
    and determining where the FSC between the aligned volumes falls to 0.5.

    :param ref_density: reference density map
    :param est_density: estimated density map, same dimensions as ref_density
    :param sigma: sigma by which to Gaussian filter volumes during alignment
    :param max_res: corner resolution of AC from which est_density was derived in Angstrom
    :param spacing: d-spacing for computing the FSC curve in per Ansgtrom
    :param n_iteration: max number of alignment iterations, default=10
    :param tol: tolerance in degrees for determining when alignment has converged, default=0.01
    :return est_res: estimated resolution based on FSC=0.5 in Angstrom
    """
    rot_density = align_volumes(ref_density, est_density, 
                                sigma=sigma, n_iterations=n_iterations, tol=tol)
    rs, fsc, est_res = compute_fsc(ref_density, rot_density, 1e10/max_res, spacing)

    return est_res


def main():
    
    # retrieve command line and compute reference
    args = parse_input()
    ref_ac, ref_density = compute_reference(args['pdb_file'], args['M'], 1e10/args['resolution'])

    # retrieve filenames, set up resolution array to be saved
    fnames = glob.glob(args['input'])
    resolutions = np.zeros(len(fnames))

    for i,fn in enumerate(fnames):
        solution = mrcfile.open(fn).data
        res1 = calculate_res(ref_density, solution, args['sigma'], args['resolution'], args['spacing'])

        # check inverted density map
        solution = invert_handedness(fn)
        res2 = calculate_res(ref_density, solution, args['sigma'], args['resolution'], args['spacing'])

        resolutions[i] = np.min(np.array([res1, res2]))

    print(f"No. density maps: {len(fnames)}")
    print(f"mean resolution: {np.mean(resolutions):.2f}")
    print(f"min. resolution: {np.min(resolutions):.2f}")

    np.save(args['output'], resolutions)
    return


if __name__ == '__main__':
    main()
