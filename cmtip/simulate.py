import argparse, time, os
import numpy as np
import skopi as sk
import h5py

from skopi.util import asnumpy

"""
Simulate a simple SPI dataset on either a SimpleSquare or LCLSDetector. Either intensities 
(default) or photons (intensities+Poission noise) are computed, and then corrected for the
solid angle and polarization effects.
"""

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser(description="Simulate a simple SPI dataset.")
    parser.add_argument('-b', '--beam_file', help='Beam file', required=True, type=str)
    parser.add_argument('-p', '--pdb_file', help='Pdb file', required=True, type=str)
    parser.add_argument('-d', '--det_info', help='Detector info. Either (n_pixels, length, distance) for SimpleSquare'+
                        'or (det_type, geom_file, distance) for LCLSDetectors. det_type could be pnccd, for instance',
                        required=True, nargs=3)
    parser.add_argument('-m', '--num_particles', help='Number of particle per shot', default=1, type=int)
    parser.add_argument('-n', '--n_images', help='Number of slices to compute', required=True, type=int)
    parser.add_argument('-q', '--quantize', help='If true, compute photons rather than intensities', action='store_true')
    parser.add_argument('-s', '--increase_factor', help='Scale factor by which to increase beam fluence', required=False, default=1, type=float)
    parser.add_argument('-o', '--output', help='Path to output directory', required=False, type=str)

    return vars(parser.parse_args())


def setup_experiment(args):
    """
    Set up experiment class.
    
    :param args: dict containing beam, pdb, and detector info
    :return exp: SPIExperiment object
    """

    # If 'num_particles' is not in args, initialize it in args
    # and set it equal to 1.
    if 'num_particles' not in args.keys():
        args['num_particles'] = 1


    beam = sk.Beam(args['beam_file'])
    if args['increase_factor'] != 1:
        beam.set_photons_per_pulse(args['increase_factor']*beam.get_photons_per_pulse())
    
    particle = sk.Particle()
    particle.read_pdb(args['pdb_file'], ff='WK')

    if args['det_info'][0].isdigit():
        n_pixels, det_size, det_dist = args['det_info']
        det = sk.SimpleSquareDetector(int(n_pixels), float(det_size), float(det_dist), beam=beam) 
    elif args['det_info'][0] == 'pnccd':
        det = sk.PnccdDetector(geom=args['det_info'][1])
        det.distance = float(args['det_info'][2])
    elif args['det_info'][0] == 'cspad':
        det = sk.CsPadDetector(geom=args['det_info'][1])
        det.distance = float(args['det_info'][2])
    else:
        print("Detector type not recognized. Must be pnccd, cspad, or SimpleSquare.")
    
    exp = sk.SPIExperiment(det, beam, particle, n_part_per_shot=args['num_particles'])
    exp.set_orientations(sk.get_random_quat(args['n_images'] * args['num_particles']))
    
    return exp

def simulate_writeh5(args):
    """
    Simulate diffraction images and save to h5 file.
    :param args: dictionary of command line input
    """
    print("Simulating diffraction images")
    start_time = time.time()

    # set image type
    if args['quantize']:
        itype = 'photons'
    else:
        itype = 'intensities'

    # set up experiment and create h5py file
    exp = setup_experiment(args)
    f = h5py.File(args["output"], "w")

    # store useful experiment arrays
    f.create_dataset("pixel_position_reciprocal", data=np.moveaxis(asnumpy(exp.det.pixel_position_reciprocal), -1, 0)) # s-vectors in m-1
    f.create_dataset("volume", data=asnumpy(exp.volumes[0])) # reciprocal space volume, 151 pixels cubed
    f.create_dataset("pixel_index_map", data=asnumpy(exp.det.pixel_index_map)) # indexing map for reassembly
    f.create_dataset("orientations", data=asnumpy(exp._orientations)) # ground truth quaternions

    # simulate images and save to h5 file
    imgs = f.create_dataset(itype, shape=((args['n_images'],) + exp.det.shape))
    for num in range(args['n_images']):
        if itype == 'intensities':
            img = exp.generate_image_stack(return_intensities=True)
        else:
            img = exp.generate_image_stack(return_photons=True)
        imgs[num,:,:,:] = img #/ exp.det.polarization_correction / exp.det.solid_angle_per_pixel / 1e6 # scale for nufft bounds
        imgs[num,:,:,:] /= exp.det.polarization_correction / exp.det.solid_angle_per_pixel / 1e6 # scale for nufft bounds

    # save useful attributes
    f.attrs['reciprocal_extent'] = np.linalg.norm(asnumpy(exp.det.pixel_position_reciprocal), axis=-1).max() # max |s|
    f.attrs['n_images'] = args['n_images'] # number of simulated shots
    f.attrs['n_pixels_per_image'] = exp.det.pixel_num_total # number of total pixels per image
    f.attrs['det_shape'] = exp.det.shape # detector shape (n_panels, panel_num_x, panel_num_y)
    f.attrs['det_distance'] = float(args['det_info'][2]) # detector distance in meters

    f.close()

    print("Simulated dataset saved to %s" %args['output'])
    print("elapsed time is %.2f" %((time.time() - start_time)/60.0))

    return


def simulate_images(args):
    """
    Simulate diffraction images and return information needed to run M-TIP.
    :param args: dictionary of command line input
    :return data: dictionary of information needed to run M-TIP
    """
    print("Simulating diffraction images")
    start_time = time.time()

    # set up experiment
    exp = setup_experiment(args)
    
    # generate data dictionary and set image type
    data = dict()
    if args['quantize']:
        itype = 'photons'
    else:
        itype = 'intensities'

    # simulate shots
    data[itype] = np.zeros((args['n_images'],) + exp.det.shape)
    for num in range(args['n_images']):
        if itype == 'intensities':
            img = exp.generate_image_stack(return_intensities=True)
        else:
            img = exp.generate_image_stack(return_photons=True)
        data[itype][num,:,:,:] = img / exp.det.polarization_correction / exp.det.solid_angle_per_pixel / 1e6 # scale for nufft bounds

    # populate rest of dictionary
    data["orientations"] = exp._orientations # quaternions
    data['reciprocal_extent'] = np.linalg.norm(asnumpy(exp.det.pixel_position_reciprocal), axis=-1).max() # max |s|
    data['pixel_position_reciprocal'] = np.moveaxis(asnumpy(exp.det.pixel_position_reciprocal), -1, 0) # s-vectors in m-1
    data['n_images'] = args['n_images'] # number of simulated shots
    data['n_pixels_per_image'] = exp.det.pixel_num_total # number of total pixels per image
    data['volume'] = asnumpy(exp.volumes[0]) # reciprocal space volume
    data['pixel_index_map'] = exp.det.pixel_index_map # indexing map for reassembly
    data['det_shape'] = exp.det.shape # detector shape (n_panels, panel_num_x, panel_num_y)

    # flatten intensities and pixel_position_reciprocal arrays
    data[itype] = data[itype].reshape(args['n_images'],1,exp.det.pixel_num_total)
    data['pixel_position_reciprocal'] = data['pixel_position_reciprocal'].reshape(3,1,exp.det.pixel_num_total)

    print("elapsed time is %.2f" %((time.time() - start_time)/60.0))
    return data


def main():
    """
    Generate images and optionally save to h5 file.
    """

    # gather command line input 
    args = parse_input()

    # simulate images and save
    if args['output'] is not None:
        simulate_writeh5(args)
    else:
        sim_data = simulate_images(args)

if __name__ == '__main__':
    main()
