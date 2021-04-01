import numpy as np
import h5py

def load_h5(input_file, start=0, end=None, load_ivol=False):
    """
    Load h5 input file of simulated data. If neither start nor end indices
    are given, load all images.

    :param input_file: input h5 file
    :param start: index of the first image to load
    :param end: index of the last image to load
    :param load_ivol: whether to load the diffraction volume, optional
    :return data: dict containing contents of input h5 file
    """
    print("Loading data from %s" %input_file)  
    data = dict()
    
    # retrieve attributes
    with h5py.File(input_file, 'r') as f:
        n_images = f.attrs['n_images']
        n_pixels = f.attrs['n_pixels']
        reciprocal_extent = f.attrs['reciprocal_extent']
    
    # handle case of loading subset of file
    if (end is None) or (end > n_images):
        end = n_images
    n_images = end - start
    
    # retrieve data
    data['intensities'] = np.zeros((n_images, 1, n_pixels, n_pixels)).astype(np.float32)
    data['orientations'] = np.zeros((n_images, 4)).astype(np.float32)
    data['pixel_position_reciprocal'] = np.zeros((3, 1, n_pixels, n_pixels)).astype(np.float32)
    if load_ivol:
        data['volume'] = np.zeros((151,151,151), dtype=np.complex128) 
    
    with h5py.File(input_file, 'r') as f:
        for key in data.keys():
            if key in ['orientations', 'intensities']:
                data[key][:] = f[key][start:end]
            else:
                data[key][:] = f[key][:]
            
    data['n_images'], data['n_pixels'] = n_images, n_pixels
    data['reciprocal_extent'] = reciprocal_extent

    return data


def clip_data(arr, n_remove):
    """
    Clip each image in a series of images with a clipping border n_remove pixels wide.
    
    :param arr: input array to be clipped of shape (n_images,1,n_pixels,n_pixels)
    :param n_remove: width of border in pixels to be removed
    :return c_arr: clipped array 
    """
    c_arr = np.zeros((arr.shape[0],1,arr.shape[2]-2*n_remove,arr.shape[3]-2*n_remove))
    
    for i in range(arr.shape[0]):
        c_arr[i][:] = arr[i][0][n_remove:-n_remove,n_remove:-n_remove]

    return c_arr.astype(np.float32)
