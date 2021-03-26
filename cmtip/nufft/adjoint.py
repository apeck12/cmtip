import numpy as np
from cufinufft import cufinufft
import finufft as nfft 


"""
Functions for computing the (type 1) adjoint NUFFT: from a nonuniformly sampled
input to a gridded output. Since the adjoint calculation is used to estimate the
autocorrelation from diffraction images, the functions are hardcoded to compute
the inverse Fourier transform (isign=1).
"""


def adjoint_cpu(nuvect, H_, K_, L_, M, use_recip_sym=True, support=None):
    """
    Compute the adjoint NUFFT: from a nonuniform to uniform set of points. 
    The sign is set for this to be the inverse FT.
    
    :param nuvect: flattened data vector sampled in nonuniform space
    :param H_: H dimension of reciprocal space position
    :param K_: K dimension of reciprocal space position
    :param L_: L dimension of reciprocal space position
    :param M: cubic length of desired output array
    :param support: 3d object support array
    :param use_recip_sym: if True, discard imaginary component # name seems misleading
    :return ugrid: Fourier transform of nuvect, sampled on a uniform grid
    """
    
    # make sure that points lie within permissible finufft domain
    assert np.max(np.abs(np.array([H_, K_, L_]))) < 3*np.pi

    # Allocating space in memory and sovling NUFFT
    ugrid = np.zeros((M,)*3, dtype=np.complex64)
    nfft.nufft3d1(H_, K_, L_, nuvect, out=ugrid, eps=1.0e-15, isign=1)

    # Apply support if given
    if support is not None:
        ugrid *= support

    # Discard imaginary component
    if use_recip_sym:
        ugrid = ugrid.real

    return ugrid  / (M**3)


def adjoint_gpu(nuvect, H_, K_, L_, M, use_recip_sym=True, support=None):
    """
    Compute the adjoint NUFFT: from a nonuniform to uniform set of points.
    GPU version of the inverse Fourier transform calculation.
    
    :param nuvect: flattened data vector sampled in nonuniform space
    :param H_: H dimension of reciprocal space position
    :param K_: K dimension of reciprocal space position
    :param L_: L dimension of reciprocal space position
    :param M: cubic length of desired output array
    :param support: 3d object support array
    :param use_recip_sym: if True, discard imaginary component # name seems misleading
    :return ugrid: Fourier transform of nuvect, sampled on a uniform grid
    """
    
    from pycuda.gpuarray import GPUArray, to_gpu
    import pycuda.autoinit
    
    # make sure that points lie within permissible finufft domain
    assert np.max(np.abs(np.array([H_, K_, L_]))) < 3*np.pi

    # Preparing input and output arrays; both need to be of complex type
    nuvect_gpu = to_gpu(nuvect.astype(np.complex64).flatten())
    ugrid_gpu = to_gpu(np.zeros((M,)*3, dtype=np.complex64))
    
    # Solving NUFFT
    plan = cufinufft(1, (M,M,M), n_trans=1, isign=1, eps=1.0e-15, dtype=np.float32, gpu_method=1, gpu_device_id=0)
    plan.set_pts(to_gpu(H_), to_gpu(K_), to_gpu(L_))
    plan.execute(nuvect_gpu, ugrid_gpu)
    ugrid = ugrid_gpu.get()

    # Apply support if given
    if support is not None:
        ugrid *= support

    # Discard imaginary component
    if use_recip_sym:
        ugrid = ugrid.real

    return ugrid / (M**3)


