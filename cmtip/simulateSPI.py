import argparse, time, os
import numpy as np
import skopi as sk
import h5py

"""
Simulate an ultra-simple SPI datasets on a SimpleSquare detector and without any errors,
including quantization. The images returned are of the scattered intensities, corrected
for polarization and solid angle.
"""

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser(description="Simulate a simple SPI dataset.")
    parser.add_argument('-b', '--beam_file', help='Beam file', required=True, type=str)
    parser.add_argument('-p', '--pdb_file', help='Pdb file', required=True, type=str)
    parser.add_argument('-d', '--det_info', help='SimpleSquare detector settings: n_pixels, length, distance',
                        required=False, nargs=3, type=float)
    parser.add_argument('-n', '--n_images', help='Number of slices to compute', required=True, type=int)
    parser.add_argument('-o', '--output', help='Path to output directory', required=False, type=str)

    return vars(parser.parse_args())


def setup_experiment(args, increase_factor=1):
    """
    Set up experiment class.
    
    :param args: dict containing beam, pdb, and detector info
    :param increase_factor: factor by which to increase beam fluence
    :return exp: SPIExperiment object
    """
    beam = sk.Beam(args['beam_file'])
    if increase_factor != 1:
        beam.set_photons_per_pulse(increase_factor*beam.get_photons_per_pulse())
    
    particle = sk.Particle()
    particle.read_pdb(args['pdb_file'], ff='WK')

    n_pixels, det_size, det_dist = args['det_info']
    det = sk.SimpleSquareDetector(int(n_pixels), det_size, det_dist, beam=beam)     
    
    exp = sk.SPIExperiment(det, beam, particle)
    exp.set_orientations(sk.get_random_quat(args['n_images']))
    
    return exp


def simulate_writeh5(args):
    """
    Simulate diffraction images and save to h5 file.

    :param args: dictionary of command line input
    """
    print("Simulating diffraction images")
    start_time = time.time()

    # set up experiment and create h5py file
    exp = setup_experiment(args)
    f = h5py.File(args["output"], "w")

    # store useful experiment arrays
    f.create_dataset("pixel_position_reciprocal", data=np.moveaxis(exp.det.pixel_position_reciprocal, -1, 0)) # s-vectors in m-1 
    f.create_dataset("volume", data=exp.volumes[0]) # reciprocal space volume, 151 pixels cubed
    f.create_dataset("pixel_index_map", data=exp.det.pixel_index_map) # indexing map for reassembly
    f.create_dataset("orientations", data=exp._orientations) # ground truth quaternions

    # simulate images and save to h5 file
    imgs = f.create_dataset("intensities", shape=((args['n_images'],) + exp.det.shape))
    for num in range(args['n_images']):
        img = exp.generate_image_stack(return_intensities=True)
        imgs[num,:,:,:] = img / exp.det.polarization_correction / exp.det.solid_angle_per_pixel / 1e6 # scale to stay within nufft bounds

    # save useful attributes
    f.attrs['reciprocal_extent'] = np.linalg.norm(exp.det.pixel_position_reciprocal, axis=-1).max() # max |s|
    f.attrs['n_images'] = args['n_images'] # number of simulated shots
    f.attrs['n_pixels'] = int(args['det_info'][0]) # number of pixels per edge
    f.attrs['det_size'] = args['det_info'][1] # detector size in meters
    f.attrs['det_distance'] = args['det_info'][2] # detector distance in meters

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

    n_pixels = int(args['det_info'][0])
    data = dict()
    data['intensities'] = np.zeros((args['n_images'],) + (1,n_pixels,n_pixels))

    exp = setup_experiment(args)
    for num in range(args['n_images']):
        img = exp.generate_image_stack(return_intensities=True)
        data["intensities"][num,:,:,:] = img / exp.det.polarization_correction / exp.det.solid_angle_per_pixel / 1e6 # scale for nufft bounds

    data["orientations"] = exp._orientations # quaternions
    data['reciprocal_extent'] = np.linalg.norm(exp.det.pixel_position_reciprocal, axis=-1).max() # max |s|
    data['pixel_position_reciprocal'] = np.moveaxis(exp.det.pixel_position_reciprocal, -1, 0) # s-vectors in m-1
    data['n_images'] = args['n_images'] # number of simulated shots
    data['n_pixels'] = n_pixels # square length of detector
    data['volume'] = exp.volumes[0] # reciprocal space volume
    data['pixel_index_map'] = exp.det.pixel_index_map # indexing map for reassembly

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
