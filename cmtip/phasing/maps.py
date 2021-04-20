import numpy as np
import mrcfile, h5py
import skopi as sk

"""
Functions for visualizing results of phasing and comparing to reference.
"""

def save_mrc(savename, data, voxel_size=None):
    """
    Save Nd numpy array to path savename in mrc format.
    
    :param savename: path to which to save mrc file
    :param data: input numpy array
    :param voxel_size: voxel size for header, optional
    """

    mrc = mrcfile.new(savename, overwrite=True)
    mrc.header.map = mrcfile.constants.MAP_ID
    mrc.set_data(data.astype(np.float32))
    if voxel_size is not None:
        mrc.voxel_size = voxel_size
    mrc.close()
    return


def _retrieve_det_info(h5_file):
    """
    Retrieve the detector information from an input h5 file, assuming a
    SimpleSquare Detector type.

    :param h5_file: h5 file, with detector information stored as attributes
    :return det_info: tuple of (n_pixels, det_size, det_distance) 
    """
    with h5py.File(h5_file, 'r') as f:
        try:
            n_pixels = f.attrs['n_pixels']
        except:
            n_pixels = int(np.sqrt(f.attrs['n_pixels_per_image']))
        det_size = f.attrs['det_size']
        det_distance = f.attrs['det_distance']
    
    det_info = (n_pixels, det_size, det_distance)
    return det_info


def _setup_detector(det_info, beam_file):
    """
    Set up SimpleSquare detector.

    :param det_info: h5 file or tuple of (n_pixels, det_size, det_distance) 
    :param beam_file: path to beam file 
    :return det: instance of skopi's SimpleSquare Detector
    """    
    # set up Beam object
    beam = sk.Beam(beam_file)

    # instantiate Detector
    if type(det_info) is not tuple:
        det_info = _retrieve_det_info(det_info)
    det = sk.SimpleSquareDetector(int(det_info[0]), 
                                  float(det_info[1]), 
                                  float(det_info[2]), 
                                  beam=beam) 
    return det
    

def compute_reference(pdb_file, det_info, beam_file, M):
    """
    Compute reference autocorrelation and density maps to the resolution
    limit dictated by the detector settings. Currently a SimpleSquare
    detector is assumed.

    :param pdb_file: path to coordinates file
    :param det_info: h5 file or tuple of (n_pixels, det_size, det_distance)
    :param beam_file: path to beam file
    :param M: grid spacing
    :return ac: (M,M,M) array of the reference autocorrelation 
    :return density: (M,M,M) array of the reference density map
    """
    import skopi.gpu as pg
    
    # set up Particle object
    particle = sk.Particle()
    particle.read_pdb(pdb_file, ff='WK')

    # compute ideal diffraction volume
    det = _setup_detector(det_info, beam_file)
    mesh, voxel_length = det.get_reciprocal_mesh(M)
    cfield = pg.calculate_diffraction_pattern_gpu(mesh, particle, return_type='complex_field')
    ivol = np.square(np.abs(cfield))

    # compute autocorrelation and density maps
    ac = np.fft.fftshift(np.abs(np.fft.ifftn(ivol))) # acf is IFT(FT(I))
    density = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(cfield))).real

    return ac, density
    

def invert_handedness(density, output=None):
    """
    Flip the handedness of the input density file. Input can either be
    a 3d numpy array or the mrcfile containing the density to flip.
    
    :param density: 3d array of density volume or mrcfile
    :param output: mrcfile to write flipped density to, optional
    :return flipped: 3d array of flipped density 
    """
    # extract 3d array if mrcfile is supplied
    if type(density) == str:
        density = mrcfile.open(density).data
        
    # change handedness by multiplying phases by -1
    ft = np.fft.fftn(density)
    amps, phases = np.abs(ft), np.arctan2(ft.imag, ft.real)
    phases *= -1
    
    # reconstruct complex field and take inverse FT
    a,b = amps * np.cos(phases), amps * np.sin(phases)
    ft_flipped = a + b*1j
    flipped = np.fft.ifftn(ft_flipped).real # imaginary component should be zero
    
    # floor values
    threshold = density[density!=0].min()
    flipped[flipped<threshold] = 0
    
    if output is not None:
        save_mrc(output, flipped)
        
    return flipped
