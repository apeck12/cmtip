import numpy as np
import cmtip.nufft as nufft
import skopi as sk
from sklearn.metrics.pairwise import euclidean_distances

"""
Functions for determining image orientations by comparing to reference images
computed by slicing through the estimated diffraction volume (from the ac).
"""

def calc_eudist(model_slices, slices):
    """
    Calculate the Euclidean distance between reference and data slices.
    
    :param model_slices: reference images of shape (n_images, n_detector_pixels)
    :param slices: data images of shape (n_images, n_detector_pixels)
    """
    euDist = euclidean_distances(model_slices, slices)
    return euDist


def calc_argmin(euDist, n_images, n_refs, n_pixels):
    """
    Calculate the indices where the distance between the reference and
    data slices are minimized.
    
    :param model_slices:
    :param n_images: number of data images
    :param n_refs: number of reference images
    :param n_pixels: number of detector pixels
    """
    index = np.argmin(euDist, axis=0)
    return index


def nearest_neighbor(model_slices, slices):
    """
    Calculate the indices where the distance between the reference and
    data slices are minimized, using a nearest-neighbor approach that
    minimizes the Euclidean distance.
    
    :param model_slices: reference images of shape (n_images, n_detector_pixels)
    :param slices: data images of shape (n_images, n_detector_pixels)
    """
    euDist = calc_eudist(model_slices, slices)
    index = calc_argmin(euDist, 
                        slices.shape[0],
                        model_slices.shape[0],
                        slices.shape[1])
    return index


def compute_slices(orientations, pixel_position_reciprocal, reciprocal_extent, ac):
    """
    Compute slices through the diffraction volume estimated from the autocorrelation.
    
    :param orientations: array of quaternions
    :param pixel_position_reciprocal: pixels' reciprocal space positions
    :param reciprocal_extent: reciprocal space magnitude of highest resolution pixel
    :param ac: 3d array of autocorrelation
    :return model_slices: flattened array of requested model slices
    """
    # compute rotated reciprocal space positions
    rotmat = np.array([np.linalg.inv(sk.quaternion2rot3d(quat)) for quat in orientations])
    H, K, L = np.einsum("ijk,klm->jilm", rotmat, pixel_position_reciprocal)

    # scale and change type for compatibility with finufft
    H_ = H.astype(np.float32).flatten() / reciprocal_extent * np.pi 
    K_ = K.astype(np.float32).flatten() / reciprocal_extent * np.pi 
    L_ = L.astype(np.float32).flatten() / reciprocal_extent * np.pi 

    # compute model slices from the NUFFT of the autocorrelation
    model_slices = nufft.forward_cpu(ac, H_, K_, L_, support=None, use_recip_sym=True).real    
    
    return model_slices


def match_orientations(generation, 
                       pixel_position_reciprocal, 
                       reciprocal_extent, 
                       slices_,
                       ac,
                       n_ref_orientations,
                       true_orientations=None):
    """
    Determine orientations of the data images by matching to reference images
    computed by randomly slicing through the diffraction intensities esimated
    from the autocorrelation. Matching is done by minimizing nearest neighbors.
    
    :param generation: current iteration
    :param pixel_position_reciprocal: pixels' reciprocal space positions, array of shape
        (3,n_panels,n_pixels_per_panel)
    :param reciprocal_extent: reciprocal space magnitude of highest resolution pixel
    :param slices_: intensity data of shape (n_images,n_panels,n_pixels_per_panel)
    :param ac: 3d array of estimated autocorrelation
    :param n_ref_orientations: number of reference orientations to compute from autocorrelation
    :param true_orientations: quaternion orientations of slices_, used for debugging
    :return ref_orientations: array of quaternions matched to slices_
    """
    
    # generate reference images by slicing through autocorrelation
    n_det_pixels = pixel_position_reciprocal.shape[-1]
    ref_orientations = sk.get_uniform_quat(n_ref_orientations, True).astype(np.float32)
    
    model_slices = compute_slices(ref_orientations, pixel_position_reciprocal, reciprocal_extent, ac)
    model_slices = model_slices.reshape((n_ref_orientations, n_det_pixels))
    
    # for debugging purposes, add in slices that match exactly
    if true_orientations is not None:
        tmodel_slices = compute_slices(true_orientations, pixel_position_reciprocal, reciprocal_extent, ac)
        tmodel_slices = tmodel_slices.reshape((true_orientations.shape[0], n_det_pixels))
        model_slices = np.vstack((model_slices, tmodel_slices))
        
        shuffled = np.arange(model_slices.shape[0])
        np.random.shuffle(shuffled)
        
        ref_orientations = np.vstack((ref_orientations, true_orientations))
        ref_orientations = ref_orientations[shuffled]
        model_slices = model_slices[shuffled]
    
    # flatten each image in data
    slices_ = slices_.reshape((slices_.shape[0], n_det_pixels))
    
    # scale model_slices
    data_model_scaling_ratio = slices_.std() / model_slices.std()
    print(f"Data/Model std ratio: {data_model_scaling_ratio}.", flush=True)
    model_slices *= data_model_scaling_ratio
    
    # compute indices of matches between reference and data orientations
    index = nearest_neighbor(model_slices, slices_)

    return ref_orientations[index]
