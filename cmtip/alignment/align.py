import numpy as np
import cmtip.nufft as nufft
import skopi as sk
from sklearn.metrics.pairwise import euclidean_distances

"""
Functions for determining image orientations by comparing to reference images
computed by slicing through the estimated diffraction volume (from the ac).
"""

def generate_weights(pixel_position_reciprocal, order=0):
    """
    Generate weights: 1/(pixel_position_reciprocal)^order, to dictate
    the contribution of the high-resolution pixels.

    :param pixel_position_reciprocal: reciprocal space position of each pixel
    :param order: power, uniform weights if zero
    :return weights: resolution-based weight of each pixel
    """
    s_magnitudes = np.linalg.norm(pixel_position_reciprocal, axis=0) * 1e-10 # convert to Angstrom
    weights = 1.0 / (s_magnitudes ** order)
    weights /= np.sum(weights)
    
    return weights


def calc_eudist(model_slices, slices):
    """
    Calculate the Euclidean distance between reference and data slices.
    
    :param model_slices: reference images of shape (n_images, n_detector_pixels)
    :param slices: data images of shape (n_images, n_detector_pixels)
    :return euDist: matrix of distances of shape (no. model_slices, no. slices)
    """
    euDist = euclidean_distances(model_slices, slices)
    return euDist


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
                       nbatches=1,
                       order=0,
                       true_orientations=None):
    """
    Determine orientations of the data images by matching to reference images
    computed by randomly slicing through the diffraction intensities esimated
    from the autocorrelation. Matching is done by minimizing nearest neighbors.
    Optionally batch the calculation by dividing up the number of model slices.
    
    :param generation: current iteration
    :param pixel_position_reciprocal: pixels' reciprocal space positions, array of shape
        (3,n_panels,n_pixels_per_panel)
    :param reciprocal_extent: reciprocal space magnitude of highest resolution pixel
    :param slices_: intensity data of shape (n_images,n_panels,n_pixels_per_panel)
    :param ac: 3d array of estimated autocorrelation
    :param n_ref_orientations: number of reference orientations to compute from autocorrelation
    :param nbatches: number of batches to divide model slices into
    :param order: power for s-weighting, uniform weights if zero 
    :param true_orientations: quaternion orientations of slices_, used for debugging
    :return ref_orientations: array of quaternions matched to slices_
    """

    quot, remainder = divmod(n_ref_orientations, nbatches)
    if remainder != 0:
        n_ref_orientations = (quot + 1) * nbatches 
        quot, remainder = divmod(n_ref_orientations, nbatches)

    # flatten each image in data
    n_det_pixels = pixel_position_reciprocal.shape[-1]
    slices_ = slices_.reshape((slices_.shape[0], n_det_pixels))

    # get reference orientations 
    ref_orientations = sk.get_uniform_quat(n_ref_orientations, True).astype(np.float32)
    if true_orientations is not None:
        ref_orientations = np.vstack((ref_orientations[:-len(true_orientations)], true_orientations))
        np.random.shuffle(ref_orientations)

    # compute distances to data slices for each batch of model slices
    distances = np.zeros((n_ref_orientations, slices_.shape[0]))
    for nb in range(nbatches):
        start, end = nb*quot, (nb+1)*quot
        model_slices = compute_slices(ref_orientations[start:end], pixel_position_reciprocal, reciprocal_extent, ac)
        model_slices = model_slices.reshape((end - start, n_det_pixels))

        # scale model_slices
        data_model_scaling_ratio = slices_.std() / model_slices.std()
        print(f"Data/Model std ratio: {data_model_scaling_ratio}.", flush=True)
        model_slices *= data_model_scaling_ratio

        # compute indices of matches between reference and data orientations
        weights = generate_weights(pixel_position_reciprocal, order=order)
        distances[start:end,:] = calc_eudist(model_slices * weights, slices_ * weights)

    # determine which 
    index = np.argmin(distances, axis=0)
    return ref_orientations[index]


def match_orientations_batch(generation, 
                             pixel_position_reciprocal, 
                             reciprocal_extent, 
                             slices_,
                             ac,
                             n_ref_orientations,
                             nbatches=1,
                             batch_size=400,
                             order=0,
                             true_orientations=None):
    """
    Wrapper that batches image alignment by sequentially aligning batches of data
    images against the reference slices.

    :param generation: current iteration
    :param pixel_position_reciprocal: pixels' reciprocal space positions, array of shape
        (3,n_panels,n_pixels_per_panel)
    :param reciprocal_extent: reciprocal space magnitude of highest resolution pixel
    :param slices_: intensity data of shape (n_images,n_panels,n_pixels_per_panel)
    :param ac: 3d array of estimated autocorrelation
    :param n_ref_orientations: number of reference orientations to compute from autocorrelation
    :param nbatches: number of batches to divide model slices into 
    :param batch_size: number of data images per batch
    :param order: power for s-weighting, uniform weights if zero 
    :param true_orientations: quaternion orientations of slices_, used for debugging
    :return ref_orientations: array of quaternions matched to slices_
    """
    ref_orientations = np.zeros((slices_.shape[0],4))
    
    quot, remainder = divmod(slices_.shape[0], batch_size)
    if quot == 0:
        quot = 1
    
    for i in range(quot):
        start, end = i*batch_size, (i+1)*batch_size
        if i == quot - 1:
            end = slices_.shape[0]
            
        if true_orientations is not None:
            true_orientations[start:end]
        
        ref_orientations[start:end] = match_orientations(generation,
                                                         pixel_position_reciprocal,
                                                         reciprocal_extent,
                                                         slices_[start:end],
                                                         ac,
                                                         n_ref_orientations,
                                                         nbatches=nbatches,
                                                         order=order,
                                                         true_orientations=true_orientations)
    return ref_orientations
