import numpy as np
import mrcfile
from geom import *
from scipy import ndimage

def save_mrc(output, data, voxel_size=None, header_origin=None):
    """
    Save numpy array as an MRC file.
    
    Parameters
    ----------
    output : string, default None
        if supplied, save the aligned volume to this path in MRC format
    data : numpy.ndarray
        image or volume to save
    voxel_size : float, default None
        if supplied, use as value of voxel size in Angstrom in the header
    header_origin : numpy.recarray
        if supplied, use the origin from this header object
    """
    mrc = mrcfile.new(output, overwrite=True)
    mrc.header.map = mrcfile.constants.MAP_ID
    mrc.set_data(data.astype(np.float32))
    if voxel_size is not None:
        mrc.voxel_size = voxel_size
    if header_origin is not None:
        mrc.header['origin']['x'] = float(header_origin['origin']['x'])
        mrc.header['origin']['y'] = float(header_origin['origin']['y'])
        mrc.header['origin']['z'] = float(header_origin['origin']['z'])
        mrc.update_header_from_data()
        mrc.update_header_stats()
    mrc.close()
    return

def rotate_volume(vol, quat):
    """
    Rotate copies of the volume by the given quaternions.
    
    Parameters
    ----------
    vol : numpy.ndarray, shape (n,n,n)
        volume to be rotated
    quat : numpy.ndarray, shape (n_quat,4)
        quaternions to apply to the volume
        
    Returns
    -------
    rot_vol : numpy.ndarray, shape (n_quat,n,n,n)
        rotated copies of volume
    """
    M = vol.shape[0]
    lincoords = np.arange(M)
    coords = np.meshgrid(lincoords,lincoords,lincoords)

    xyz = np.vstack([coords[0].reshape(-1) - int(M/2),
                     coords[1].reshape(-1) - int(M/2),
                     coords[2].reshape(-1) - int(M/2)])

    R = quaternion2rot3d(np.array(quat))
    transformed_xyz = np.dot(R,xyz) + int(M/2)
    
    new_xyz = [transformed_xyz[:,1,:].flatten(), 
               transformed_xyz[:,0,:].flatten(), 
               transformed_xyz[:,2,:].flatten()] 
    rot_vol = ndimage.map_coordinates(vol, new_xyz, order=1)
    rot_vol = rot_vol.reshape((quat.shape[0],M,M,M))
    
    return rot_vol

def center_volume(vol):
    """
    Apply translational shifts to center the density within the volume.
      
    Parameters
    ----------
    vol : numpy.ndarray, shape (n,n,n)
        volume to be centered
        
    Returns
    -------
    cen_vol : numpy.ndarray, shape (n,n,n)
        centered volume
    """
    old_center = np.array(ndimage.measurements.center_of_mass(vol))
    new_center = np.array(np.array(vol.shape)/2).astype(int)
    cen_vol = ndimage.shift(vol, -1*(old_center - new_center))
    return cen_vol
    
def pearson_cc(arr1, arr2):
    """
    Compute the Pearson correlation-coefficient between the input arrays.
    
    Parameters
    ----------
    arr1 : numpy.ndarray, shape (n_samples, n_points)
        input array
    arr2 : numpy.ndarray, shape (n_samples, n_points) or (1, n_points)
        input array to compute CC with
    
    Returns
    -------
    ccs : numpy.ndarray, shape (n_samples)
        correlation coefficient between paired sample arrays, or if
        arr2.shape[0] == 1, then between each sample of arr1 to arr2
    """
    vx = arr1 - arr1.mean(axis=-1)[:,None]
    vy = arr2 - arr2.mean(axis=-1)[:,None]
    numerator = np.sum(vx * vy, axis=1)
    denom = np.sqrt(np.sum(vx**2, axis=1)) * np.sqrt(np.sum(vy**2, axis=1))
    return numerator / denom

def score_deformations(mrc1, mrc2, warp):
    """
    Compute the Pearson correlation coefficient between the input volumes 
    after rotating or displacing the first volume by the given quaternions
    or translations.
    
    Parameters
    ----------
    mrc1 : numpy.ndarray, shape (n,n,n)
        volume to be warped
    mrc2 : numpy.ndarray, shape (n,n,n)
        volume to be held fixed
    warp : numpy.ndarray, shape (n_quat,4) or (n_trans,3)
        rotations or displacements to apply to mrc2
        
    Returns
    -------
    ccs : numpy.ndarray, shape (n_quat)
        correlation coefficients associated with warp
    """
    # deform one of the input volumes by rotation or displacement
    if warp.shape[-1] == 4:
        wmrc1 = rotate_volume(mrc1, warp)
    elif warp.shape[-1] == 3:
        print("Not yet implemented")
        return
    else:
        print("Warp input must be quaternions or translations")
        return
        
    # score each deformed volume using the Pearson CC
    wmrc1_flat = wmrc1.reshape(wmrc1.shape[0],-1)
    mrc2_flat = np.expand_dims(mrc2.flatten(), axis=0)
    ccs = pearson_cc(wmrc1_flat, mrc2_flat)
    return ccs

def scan_orientations_fine(mrc1, mrc2, opt_q, prev_score, n_iterations=10, n_search=420):
    """
    Perform a fine alignment search in the vicinity of the input quaternion 
    to align mrc1 to mrc2. 
    
    Parameters
    ----------
    mrc1 : numpy.ndarray, shape (n,n,n)
        volume to be rotated
    mrc2 : numpy.ndarray, shape (n,n,n)
        volume to be held fixed
    opt_q : numpy.ndarray, shape (1,4)
        starting quaternion to apply to mrc1 to align it with mrc2
    prev_score : float
        cross-correlation associated with alignment quat opt_q
    n_iterations: int, default 10
        number of iterations of alignment to perform
    n_search : int, default 420
        number of quaternions to score at each orientation
        
    Returns
    -------
    opt_q : numpy.ndarray, shape (4)
        quaternion to apply to mrc1 to align it with mrc2
    prev_score : float
        cross-correlation between aligned mrc1 and mrc2
    """
    # perform a series of fine alignment, ending if CC no longer improves
    sigmas = 2-0.2*np.arange(1,10)
    for n in range(1, n_iterations):
        quat = get_preferred_orientation_quat(n_search-1, float(sigmas[n-1]), base_quat=opt_q)
        quat = np.vstack((opt_q, quat))
        ccs = score_deformations(mrc1, mrc2, quat)
        if np.max(ccs) < prev_score:
            break
        else:
            opt_q = quat[np.argmax(ccs)]
        #print(torch.max(ccs), opt_q) # useful for debugging
        prev_score = np.max(ccs)     
    
    return opt_q, prev_score

def scan_orientations(mrc1, mrc2, n_iterations=10, n_search=420, nscs=1):
    """
    Find the quaternion and its associated score that best aligns volume mrc1 to mrc2. 
    Candidate orientations are scored based on the Pearson correlation coefficient. 
    First a coarse search is performed, followed by a series of increasingly fine 
    searches in angular space. To prevent getting stuck in a bad solution, the top nscs 
    solutions from the coarse grained search can be investigated.
    
    Parameters
    ----------
    mrc1 : numpy.ndarray, shape (n,n,n)
        volume to be rotated
    mrc2 : numpy.ndarray, shape (n,n,n)
        volume to be held fixed
    n_iterations: int, default 10
        number of iterations of alignment to perform
    n_search : int, default 420
        number of quaternions to score at each orientation
    nscs : int, default 1
        number of solutions from the coarse-grained search to investigate
        
    Returns
    -------
    opt_q : numpy.ndarray, shape (4)
        quaternion to apply to mrc1 to align it with mrc2
    score : float
        cross-correlation between aligned mrc1 and mrc2
    """
    # perform a coarse alignment to start
    quat = sk.get_uniform_quat(n_search) # update for skopi
    ccs = score_deformations(mrc1, mrc2, quat)
    ccs_order = np.argsort(ccs)[::-1]
    
    # scan the top solutions
    opt_q_list, ccs_list = np.zeros((nscs,4)), np.zeros(nscs)
    for n in range(nscs):
        start_q, start_score = quat[ccs_order[n]], ccs[ccs_order[n]]
        opt_q_list[n], ccs_list[n] = scan_orientations_fine(mrc1, mrc2, start_q, start_score, 
                                                            n_iterations=n_iterations, n_search=n_search)

    opt_q, score = opt_q_list[np.argmax(ccs_list)], np.max(ccs_list)
    return opt_q, score

def align_volumes(mrc1, mrc2, zoom=1, sigma=0, n_iterations=10, n_search=420, nscs=1, output=None, voxel_size=None):
    """
    Find the quaternion that best aligns volume mrc1 to mrc2. Volumes are
    optionally preprocessed by up / downsampling and applying a Gaussian
    filter.
    
    Parameters
    ----------
    mrc1 : numpy.ndarray, shape (n,n,n)
        volume to be rotated
    mrc2 : numpy.ndarray, shape (n,n,n)
        volume to be held fixed
    zoom : float, default 1
        if not 1, sample by which to up or downsample volume
    sigma : int, default 0
        sigma of Gaussian filter to apply to each volume
    n_iterations: int, default 10
        number of iterations of alignment to perform
    n_search : int, default 420
        number of quaternions to score at each orientation
    nscs : int, default 1
        number of solutions from the coarse-grained alignment search to investigate
    output : string, default None
        if supplied, save the aligned volume to this path in MRC format
    voxel_size : float, default None
        if supplied, use as value of voxel size in Angstrom for output
    
    Returns
    -------
    r_vol : numpy.ndarray, shape (n,n,n)
        copy of centered mrc1 aligned with centered mrc2 
    mrc2_original : umpy.ndarray, shape (n,n,n)
        copy of centered mrc2
    """    
    # center input volumes, then make copies
    mrc1, mrc2 = center_volume(mrc1), center_volume(mrc2)
    mrc1_original = mrc1.copy()
    mrc2_original = mrc2.copy()
    
    # optionally up/downsample volumes
    if zoom != 1:
        mrc1 = ndimage.zoom(mrc1, (zoom,zoom,zoom))
        mrc2 = ndimage.zoom(mrc2, (zoom,zoom,zoom))
    
    # optionally apply a Gaussian filter to volumes
    if sigma != 0:
        mrc1 = ndimage.gaussian_filter(mrc1, sigma=sigma)
        mrc2 = ndimage.gaussian_filter(mrc2, sigma=sigma)
    
    # evaluate both hands
    opt_q1, cc1 = scan_orientations(mrc1, mrc2, n_iterations, n_search, nscs=nscs)
    opt_q2, cc2 = scan_orientations(np.flip(mrc1, [0,1,2]), mrc2, n_iterations, n_search, nscs=nscs)
    if cc1 > cc2: 
        opt_q, cc_r, invert = opt_q1, cc1, False
    else: 
        opt_q, cc_r, invert = opt_q2, cc2, True
    print(f"Alignment CC after rotation is: {cc_r:.3f}")    
    
    # generate final aligned map
    if invert:
        print("Map had to be inverted")
        mrc1_original = np.flip(mrc1_original, [0,1,2])
    r_vol = rotate_volume(mrc1_original, np.expand_dims(opt_q, axis=0))[0]
    final_cc = pearson_cc(np.expand_dims(r_vol.flatten(), axis=0), np.expand_dims(mrc2_original.flatten(), axis=0))
    print(f"Final CC between unzoomed / unfiltered volumes is: {float(final_cc):.3f}")
    
    if output is not None:
        save_mrc(output, np.array(r_vol), voxel_size=voxel_size)
            
    return r_vol, mrc2_original
