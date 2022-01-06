import numpy as np
import h5py, sys, os
import cmtip.phasing as phaser

"""
Functions for data pre-processing -- loading, binning, cutting resolution,
reassembling data from panels to detector shape -- and for checkpointing
intermediate results during an MTIP run.
"""

def load_h5(input_file, start=0, end=None, load_ivol=False):
    """
    Load h5 input file of simulated data. If neither start nor end indices
    are given, load all images. If conformations and / or orientations are 
    available, load those as well. Data keys are flattened upon loading,
    and single precision is used.
    
    :param input_file: input h5 file
    :param start: index of the first image to load
    :param end: index of the last image to load
    :param load_ivol: whether to load the diffraction volume, optional
    :return data: dict containing contents of input h5 file
    """
    print("Loading data from %s" %input_file)  
    data, attrs, dsets = dict(), dict(), list()
    
    # retrieve attributes
    with h5py.File(input_file, 'r') as f:
        # retrieve attributes, first checking if stored in h5 file
        attributes = (list(f.attrs.keys()))
        for key in ['n_images', 'n_pixels_per_image', 'det_shape']:
            if key in attributes:
                if key == 'det_shape':
                    attrs[key] = tuple(f.attrs[key])
                else:
                    attrs[key] = f.attrs[key]

        # otherwise get this info from the dataset keys
        for key in list(f.keys()):
            if key == 'intensities' or key == 'photons':
                attrs['n_images'] = f[key].shape[0]
                attrs['det_shape'] = f[key].shape[1:]
                attrs['n_pixels_per_image'] = np.prod(np.array(attrs['det_shape']))
                dsets.append(key)
            if key == 'orientations' or key == 'conformations':
                dsets.append(key)
                    
    # handle case of loading subset of file
    if (end is None) or (end > attrs['n_images']):
        end = attrs['n_images']
    attrs['n_images'] = end - start
    
    # set up empty arrays to be populated with data with single precision
    data['intensities'] = np.zeros((attrs['n_images'],) + attrs['det_shape']).astype(np.float32)
    data['pixel_position_reciprocal'] = np.zeros((3,) + attrs['det_shape']).astype(np.float32)
    data['pixel_index_map'] = np.zeros(attrs['det_shape'] + (2,)).astype(int)
    if 'orientations' in dsets:
        data['orientations'] = np.zeros((attrs['n_images'], 4)).astype(np.float32)
    if 'conformations' in dsets:
        data['conformations'] = np.zeros(attrs['n_images']).astype(np.float32)
    if load_ivol:
        data['volume'] = np.zeros((151,151,151), dtype=np.complex128) 

    # retrieve data arrays
    with h5py.File(input_file, 'r') as f:
        for key in ['pixel_position_reciprocal', 'pixel_index_map']:
            try:
                data[key][:] = f[key][:]
            except ValueError:
                print(f"Detected a spinifel-style dataset, tranposing {key} axes")
                data[key] = np.moveaxis(f[key][:], -1, 0)
        if load_ivol:
            data['volume'][:] = f[key][:]
        for key in dsets:
            if key == 'photons' or key == 'intensities':
                data['intensities'][:] = f[key][start:end]
            else:
                try:
                    data[key][:] = f[key][start:end]
                except ValueError:
                    print(f"Detected a spinifel-style dataset, flattening one dimension of {key}")
                    data[key][:] = f[key][start:end].reshape(data[key].shape[0], data[key].shape[-1])

    # flatten each image and corresponding positions in reciprocal space
    data['intensities'] = data['intensities'].reshape(attrs['n_images'],1,attrs['n_pixels_per_image'])
    data['pixel_position_reciprocal'] = data['pixel_position_reciprocal'].reshape(3,1,attrs['n_pixels_per_image'])

    # compute resolution of corner pixel and store attributes
    data['reciprocal_extent'] = np.linalg.norm(data['pixel_position_reciprocal'], axis=0).max()
    for key in attrs.keys():
        data[key] = attrs[key]

    return data


def trim_dataset(pixel_index_map, pixel_position_reciprocal, intensities, det_shape, res_limit):
    """
    Trim data such that the corner pixel of images has the desired resolution.
    
    :param pixel_index_map: pixel index map of shape (n_panels, n_pixels_x, n_pixels_y, 2)
    :param pixel_position_reciprocal: reciprocal space positions of each pixel, of 
        shape (3, n_panels, n_pixels_per_image) in meters
    :param intensities: intensity data of shape (n_images, n_panels, n_pixels_per_image)
    :param det_shape: shape of detector (n_panels, n_pixels_x, n_pixels_y)
    :param res_limit: maximum resolution (corner pixel) to retain
        if 0, then data are returned untrimmed
    :return pixel_position_reciprocal: trimmed reciprocal space positions 
    :return intensities: trimmed intensity data
    """
    if res_limit==0:
        return pixel_position_reciprocal, intensities

    else:
        # if requested resolution exceeds maximum data resolution, return all data
        data_max_res = 1e10/np.linalg.norm(pixel_position_reciprocal, axis=0).max()
        if res_limit < data_max_res:
            print ("Warning: requested resolution exceeds data resolution.")
            return pixel_position_reciprocal, intensities

        # assemble into detector shape
        intensities = intensities.reshape((intensities.shape[0],) + det_shape)
        mask = np.expand_dims(np.ones_like(intensities[0]), axis=0)
        mask = assemble_image_stack_batch(mask, pixel_index_map)
        intensities = assemble_image_stack_batch(intensities, pixel_index_map)
        pixel_position_reciprocal = pixel_position_reciprocal.reshape((3,) + det_shape)
        pixel_position_reciprocal = assemble_image_stack_batch(pixel_position_reciprocal, 
                                                               pixel_index_map)

        # get assembled image shape and determine how many pixels to keep for res_limit
        assembled_shape = np.array([pixel_index_map[:,:,:,0].max()+1, 
                                    pixel_index_map[:,:,:,1].max()+1])
        center = np.array(0.5*assembled_shape).astype(int)
        
        # figure out correspondence between image dimension and resolution
        min_pix = 20
        img_width = range(min_pix,np.min(center))
        norms = np.linalg.norm(pixel_position_reciprocal, axis=0)
        res = np.array([1e10/norms[center[0]-i, center[1]-i] for i in img_width])
        nkeep = np.argmin(np.abs(res - res_limit)) + min_pix
        
        # trim data to requested resolution
        intensities = intensities[:,center[0]-nkeep:center[0]+nkeep,center[1]-nkeep:center[1]+nkeep]
        mask = mask[:,center[0]-nkeep:center[0]+nkeep,center[1]-nkeep:center[1]+nkeep]
        pixel_position_reciprocal = pixel_position_reciprocal[:,center[0]-nkeep:center[0]+nkeep,center[1]-nkeep:center[1]+nkeep]

        # disassemble images: flatten and remove masked regions
        intensities = intensities[:,np.where(mask==1)[1],np.where(mask==1)[2]]
        pixel_position_reciprocal = pixel_position_reciprocal[:,np.where(mask==1)[1],np.where(mask==1)[2]]
        intensities = np.expand_dims(intensities, axis=1)
        pixel_position_reciprocal = np.expand_dims(pixel_position_reciprocal, axis=1)

        return pixel_position_reciprocal, intensities


def clip_data(arr, pixel_position_reciprocal, res_limit):
    """
    Clip each image in a series of images such that any pixels beyond
    input resolution are discarded.
    
    :param arr: array to be clipped of shape (n_images, n_panels, n_pixels_per_image)
    :param pixel_position_reciprocal: reciprocal space positions of each pixel, of 
        shape (3, n_panels, n_pixels_per_image) in meters
    :param res_limit: highest resolution pixel to keep in Angstrom
    :return c_arr: clipped array of (n_images, n_panels, n_retained_pixels)
    """
    resolution = 1.0 / np.linalg.norm(pixel_position_reciprocal, axis=0) * 1e10
    indices = np.where(resolution[0]>res_limit)[0]    
    c_arr = arr[:,:,indices]
    return c_arr


def clip_dataset(pixel_position_reciprocal, intensities, res_limit):
    """
    Clip reciprocal space positions and intensities to requested resolution.
    In contrast to the trim_dataset function, the retained data will fall in a 
    sphere rather than coming from square detector images. This is a wrapper
    for the clip_data function.
    
    :param intensities: array of images of shape (n_images, n_panels, n_pixels_per_image)
    :param pixel_position_reciprocal: reciprocal space positions of each pixel, of 
        shape (3, n_panels, n_pixels_per_image) in meters
    :param res_limit: highest resolution pixel to keep in Angstrom
    :return pixel_position_reciprocal: clipped reciprocal space positions
    :return intensities: clipped intensity images
    """
    if res_limit==0:
        return pixel_position_reciprocal, intensities

    intensities = clip_data(intensities, pixel_position_reciprocal, res_limit)
    pixel_position_reciprocal = clip_data(pixel_position_reciprocal, 
                                          pixel_position_reciprocal,
                                          res_limit)
    
    return pixel_position_reciprocal, intensities


def bin_data(arr, bin_factor, det_shape=None):
    """
    Bin detector data by bin_factor through averaging. 
    
    :param arr: array shape (n_images, n_panels, panel_shape_x, panel_shape_y)
      or if det_shape is given of shape (n_images, 1, n_pixels_per_image) 
    :param bin_factor: how may fold to bin arr by
    :param det_shape: tuple of detector shape, optional
    :return arr_binned: binned data of same dimensions as arr
    """
    # reshape as needed
    if det_shape is not None:
        arr = np.array([arr[i].reshape(det_shape) for i in range(arr.shape[0])])
    
    # ensure that original shape is divisible by bin factor
    assert arr.shape[2] % bin_factor == 0
    assert arr.shape[3] % bin_factor == 0   
    
    # bin each panel of each image
    binned_arr = arr.reshape(arr.shape[0], 
                             arr.shape[1],
                             int(arr.shape[2] / bin_factor),
                             bin_factor,
                             int(arr.shape[3] / bin_factor),
                             bin_factor).mean(-1).mean(3)

    # if input data were flattened, reflatten
    if det_shape is not None:
        binned_arr = binned_arr.reshape((binned_arr.shape[0],1) + (np.prod(np.array(binned_arr.shape[1:])),))
    return binned_arr


def bin_pixel_index_map(arr, bin_factor):
    """
    Bin pixel_index_map by bin factor.
    
    :param arr: pixel_index_map of shape (n_panels, panel_shape_x, panel_shape_y, 2)
    :param bin_factor: how may fold to bin arr by
    :return binned_arr: binned pixel_index_map of same dimensions as arr
    """
    arr = np.moveaxis(arr, -1, 0)
    arr = np.minimum(arr[..., ::bin_factor, :], arr[..., 1::bin_factor, :])
    arr = np.minimum(arr[..., ::bin_factor], arr[..., 1::bin_factor])
    arr = arr // bin_factor
    
    return np.moveaxis(arr, 0, -1)


def assemble_image_stack_batch(image_stack, index_map):
    """
    Assemble the image stack to obtain a 2D pattern according to the index map.
    Modified from skopi.

    :param image_stack: [stack num, panel num, panel pixel num x, panel pixel num y]
    :param index_map: [panel num, panel pixel num x, panel pixel num y]
    :return: [stack num, 2d pattern x, 2d pattern y]
    """
    # get boundary
    index_max_x = np.max(index_map[:, :, :, 0]) + 1
    index_max_y = np.max(index_map[:, :, :, 1]) + 1
    # get stack number and panel number
    stack_num = image_stack.shape[0]
    panel_num = image_stack.shape[1]

    # set holder
    image = np.zeros((stack_num, index_max_x, index_max_y))

    # loop through the panels
    for l in range(panel_num):
        image[:, index_map[l, :, :, 0], index_map[l, :, :, 1]] = image_stack[:, l, :, :]

    return image


def save_checkpoint(generation, output, checkpoint):
    """
    Save an h5 file containing the results from an MTIP iteration and also
    the density map to an MRC file.
    
    :param generation: the current iteration of the MTIP pipeline
    :param output: output directory
    :param checkpoint: dictionary of arrays to save
    """    
    # save arrays needed to restart the MTIP pipeline
    f = h5py.File(os.path.join(output, f"checkpoint_g{generation}.h5"), "w")
    for key in checkpoint.keys():
        if key != 'generation':
            if checkpoint[key] is not None:
                f.create_dataset(key, data=checkpoint[key])
    f.close()
    
    # save the density map to a Chimera-compatible format
    if 'rho_' in checkpoint.keys():
        rho_unshifted = np.fft.ifftshift(checkpoint['rho_'])
        phaser.maps.save_mrc(os.path.join(output, f"density{generation}.mrc"), rho_unshifted)
    
    return


def load_checkpoint(input_file=None):
    """
    Load an intermediate checkpoint file or provide default values if 
    no checkpoint file is supplied.
    
    :param input_file: path to h5 checkpoint file
    :return checkpoint: dictionary containing output from an MTIP iteration
    """
    checkpoint = dict()
    if input_file is None:
        checkpoint['generation'] = 0
        checkpoint['orientations'] = None
    else:
        with h5py.File(input_file, 'r') as f:
            checkpoint['generation'] = int(input_file.split("/")[-1].split("g")[1][:-3])
            for key in f.keys():
                checkpoint[key] = f[key][:]
                
    return checkpoint
