import numpy as np
import mrcfile, h5py
import skopi as sk
import scipy.ndimage
import scipy.interpolate
from cmtip.alignment import errors

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


def get_reciprocal_mesh(voxel_number_1d, distance_reciprocal_max):
    """
    Get a centered, symetric mesh of given dimensions. 
    Altered from skopi.
    
    :param voxel_number_1d: number of voxel per axis
    :param distance_reciprocal_max: corner resolution of autocorrelation
    :return reciprocal_mesh: grid of reciprocal distances with max edge 
       resolution of distance_reciprocal_max
    """
    max_value = distance_reciprocal_max
    linspace = np.linspace(-max_value, max_value, voxel_number_1d)
    reciprocal_mesh_stack = np.asarray(
        np.meshgrid(linspace, linspace, linspace, indexing='ij'))
    reciprocal_mesh = np.moveaxis(reciprocal_mesh_stack, 0, -1)

    return reciprocal_mesh
    

def compute_reference(pdb_file, M, distance_reciprocal_max):
    """
    Compute reference autocorrelation and density maps to the resolution
    limit dictated by the detector settings. Currently a SimpleSquare
    detector is assumed.
    
    :param pdb_file: path to coordinates file
    :param M: grid spacing
    :param distance_reciprocal_max: corner resolution in meters
    :return ac: (M,M,M) array of the reference autocorrelation 
    :return density: (M,M,M) array of the reference density map
    """
    import skopi.gpu as pg
    
    # set up Particle object
    particle = sk.Particle()
    particle.read_pdb(pdb_file, ff='WK')

    # compute ideal diffraction volume
    mesh = get_reciprocal_mesh(M, distance_reciprocal_max)
    cfield = pg.calculate_diffraction_pattern_gpu(mesh, particle, return_type='complex_field')
    ivol = np.square(np.abs(cfield))

    # compute autocorrelation and density maps
    ac = np.fft.fftshift(np.abs(np.fft.ifftn(ivol))) # acf is IFT(FT(I))
    density = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(cfield))).real

    return ac, density
    

def rotate_volume(volume, R):
    """
    Rotate input volume. Code from: https://www.javaer101.com/es/article/50982769.html.
    
    :param volume: input volume
    :param R: rotation matrix
    :return volumeR: rotated input volume
    """

    # create meshgrid
    dim = volume.shape
    ax = np.arange(dim[0])
    ay = np.arange(dim[1])
    az = np.arange(dim[2])
    coords = np.meshgrid(ax, ay, az)

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz = np.vstack([coords[0].reshape(-1) - float(dim[0]) / 2,  # x coordinate, centered
                     coords[1].reshape(-1) - float(dim[1]) / 2,  # y coordinate, centered
                     coords[2].reshape(-1) - float(dim[2]) / 2])  # z coordinate, centered

    # apply transformation
    transformed_xyz = np.dot(R, xyz)

    # extract coordinates
    x = transformed_xyz[0, :] + float(dim[0]) / 2
    y = transformed_xyz[1, :] + float(dim[1]) / 2
    z = transformed_xyz[2, :] + float(dim[2]) / 2

    x = x.reshape((dim[1],dim[0],dim[2]))
    y = y.reshape((dim[1],dim[0],dim[2]))
    z = z.reshape((dim[1],dim[0],dim[2])) # reason for strange ordering: see next line

    # the coordinate system seems to be strange, it has to be ordered like this
    new_xyz = [y, x, z]

    # sample
    volumeR = scipy.ndimage.map_coordinates(volume, new_xyz, order=1)
    
    return volumeR


def score_orientations(ref_volume, volume, orientations):
    """
    Score a set of quaternions to find the one that best rotates the target onto the 
    reference volume.
    
    :param ref_volume: reference volume
    :param volume: target volume
    :param orientations: array of quaternions
    :return quat: quaternions that best rotates volume onto ref_volume
    """
    ccs = np.zeros(len(orientations))
    for i,q in enumerate(orientations):
        R = sk.quaternion2rot3d(q)
        r_volume = rotate_volume(volume, R)
        ccs[i] = np.corrcoef(r_volume.flatten(), ref_volume.flatten())[0,1]
    
    index = np.argmax(ccs)
    print(f"Index {index} has maximum CC score of {np.max(ccs)}")
    
    return orientations[index]


def align_volumes(ref_volume, volume, sigma=0, n_iterations=10, tol=0.05):
    """
    Align volume to ref_volume. Candidate orientations are scored by computing
    the cross-correlation between Gaussian-filtered versions of the volumes.
    
    :param ref_volume: reference volume
    :param volume: target volume
    :param sigma: sigma for Gaussian-filtering during alignment
    :param n_iterations: maximum number of iterations
    :param tol: convergence error in degrees (between iterations)
    :return r_volume: rotated volume
    """
    
    ref_volume_f = scipy.ndimage.gaussian_filter(ref_volume, sigma=sigma)
    volume_f = scipy.ndimage.gaussian_filter(volume, sigma=sigma)
    
    quats = np.zeros((n_iterations,4))
    orientations = sk.get_uniform_quat(720)
    quats[0] = score_orientations(ref_volume_f, volume_f, orientations)
    
    for n in range(1,n_iterations):
        orientations = sk.get_preferred_orientation_quat(1.0/n, 360, base_quat=quats[n-1])
        orientations = np.vstack((quats[n-1], orientations))
        quats[n] = score_orientations(ref_volume_f, volume_f, orientations)
        if errors.alignment_error(np.array(quats[n-1]), np.array(quats[n])) < tol:
            quats = quats[:n]
            break
        
    r_volume = rotate_volume(volume, sk.quaternion2rot3d(quats[-1]))
    return r_volume


def compute_fsc(volume1, volume2, distance_reciprocal_max, spacing):
    """
    Compute the FSC curve.
    
    :param volume1: first volume
    :param volume2: second volume
    :param distance_reciprocal_max: corner resolution
    :param spacing: spacing in per Angstrom for computing the FSC
    :return rshell: reciprocal space radius in per Angstrom
    :return fsc: fourier shell correlation values
    :return resolution: estimated resolution based on FSC=0.5 in Angstrom
    """
    
    mesh = get_reciprocal_mesh(volume1.shape[0], distance_reciprocal_max)
    smags = np.linalg.norm(mesh, axis=-1).flatten() * 1e-10
    r_spacings = np.arange(0, smags.max() / np.sqrt(3), spacing)
    
    ft1 = np.fft.fftshift(np.fft.fftn(volume1)).flatten()
    ft2 = np.conjugate(np.fft.fftshift(np.fft.fftn(volume2)).flatten())
    rshell, fsc = np.zeros(len(r_spacings)), np.zeros(len(r_spacings))
    
    for i,r in enumerate(r_spacings):
        indices = np.where((smags>r) & (smags<r+spacing))[0]
        numerator = np.sum(ft1[indices] * ft2[indices])
        denominator = np.sqrt(np.sum(np.square(np.abs(ft1[indices]))) * np.sum(np.square(np.abs(ft2[indices]))))
        rshell[i] = r + 0.5*spacing 
        fsc[i] = numerator.real / denominator
        
    f = scipy.interpolate.interp1d(fsc, rshell)
    try:
        resolution = 1.0/f(0.5)
        print("Estimated resolution from FSC: %.2f Angstrom" %(resolution))
    except ValueError:
        resolution = -1
        print("Resolution could not be estimated.")
        
    return rshell, fsc, resolution
    

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


def resize_volume(vol, M_next, re_orig, re_next):
    """
    Resize the volume from original shape to dimensions of M_next cubed,
    from the original to new reciprocal space extent.
    
    :param vol: 3d numpy array of volume to resize
    :param M_next: cubic length of new volume
    :param re_orig: reciprocal space extent of input volume
    :param re_next: reciprocal space extent of resized volume
    """
    # scale based on changes in resolution
    zfactor = re_next / re_orig
    temp = scipy.ndimage.zoom(vol, zfactor)
    hs_temp, hs = int(temp.shape[0]/2), int(vol.shape[0]/2)
    temp = temp[hs_temp-hs:hs_temp+hs+1,
                hs_temp-hs:hs_temp+hs+1,
                hs_temp-hs:hs_temp+hs+1]

    # center in correct volume size
    hs_next = int(M_next / 2)
    resized = np.zeros((M_next, M_next, M_next))
    resized[hs_next-hs:hs_next+hs+1,
            hs_next-hs:hs_next+hs+1,
            hs_next-hs:hs_next+hs+1] = temp
    
    return resized
