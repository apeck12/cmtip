import numpy as np
import mrcfile

"""
Functions for visualizing results of phasing and comparing to reference.

TO-DO: add functions for generating reference AC and density, scaling.
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

