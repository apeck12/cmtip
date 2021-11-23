import numpy as np
import finufft as nfft 


"""
Functions for computing the forward (type 2) NUFFT Fourier transform: from grid 
data to nonuniformly sampled output. Since the forward calculation is used to 
estimate diffraction images by slicing through the autocorrelation the functions 
are hardcoded to compute the Fourier transform rather than its inverse (isign=-1).
"""


def forward_cpu(ugrid, H_, K_, L_, support, use_recip_sym):
    """
    Compute the forward NUFFT: from a uniform to nonuniform set of points.
    
    :param ugrid: 3d array with grid sampling
    :param H_: H dimension of reciprocal space position to evaluate
    :param K_: K dimension of reciprocal space position to evaluate
    :param L_: L dimension of reciprocal space position to evaluate
    :param support: 3d object support array
    :param use_recip_sym: if True, discard imaginary component # name seems misleading
    :return nuvect: Fourier transform of uvect sampled at nonuniform (H_, K_, L_)
    """
    
    # make sure that points lie within permissible finufft domain
    assert np.max(np.abs(np.array([H_, K_, L_]))) < 3*np.pi

    # Check if recip symmetry is met
    if use_recip_sym:
        assert np.all(np.isreal(ugrid))

    # Apply support if given, overwriting input array
    if support is not None:
        ugrid *= support 
        
    # Allocate space in memory and solve NUFFT
    nuvect = np.zeros(H_.shape, dtype=np.complex64)
    nfft.nufft3d2(H_, K_, L_, ugrid, out=nuvect, eps=1.0e-12, isign=-1)
    
    return nuvect 


def forward_gpu(ugrid, H_, K_, L_, support=None, use_recip_sym=True):
    """
    Compute the forward NUFFT: from a uniform to nonuniform set of points.
    GPU version of calculation
    
    :param ugrid: 3d array with grid sampling
    :param H_: H dimension of reciprocal space position to evaluate
    :param K_: K dimension of reciprocal space position to evaluate
    :param L_: L dimension of reciprocal space position to evaluate
    :param support: 3d object support array
    :param use_recip_sym: if True, discard imaginary component # name seems misleading
    :return nuvect: Fourier transform of uvect sampled at nonuniform (H_, K_, L_)
    """
    
    from pycuda.gpuarray import GPUArray, to_gpu
    import pycuda.autoinit
    from cufinufft import cufinufft
    
    # make sure that points lie within permissible finufft domain
    assert np.max(np.abs(np.array([H_, K_, L_]))) < 3*np.pi
    
    if support is not None:
        ugrid *= support
        
    if use_recip_sym is True:
        ugrid = ugrid.real

    # Preparing input and output arrays; both need to be of complex type
    ugrid_gpu = to_gpu(ugrid.astype(np.complex64))
    nuvect_gpu = to_gpu(np.zeros(H_.shape, dtype=np.complex64))
    
    # Solving NUFFT
    plan = cufinufft(2, ugrid.shape, n_trans=1, isign=-1, eps=1.0e-12, dtype=np.float32, gpu_method=1, gpu_device_id=0)
    plan.set_pts(to_gpu(L_), to_gpu(K_), to_gpu(H_))
    plan.execute(nuvect_gpu, ugrid_gpu)
    nuvect = nuvect_gpu.get()
    
    return nuvect
