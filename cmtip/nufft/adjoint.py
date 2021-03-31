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


def adjoint_cpu_parallel(comm, nuvect, H_, K_, L_, M, use_recip_sym=True, support=None):
    """
    Interface for parallelizing the adjoint calculation across CPUs, by making use
    of the fact that F(x+y) = F(x) + F(y), where F is the adjoint operation and the
    signal x+y corresponds to the nonuniformly sampled diffraction data.

    :param comm: MPI intracommunicator instance
    :param nuvect: flattened data vector sampled in nonuniform space
    :param H_: H dimension of reciprocal space position
    :param K_: K dimension of reciprocal space position
    :param L_: L dimension of reciprocal space position
    :param M: cubic length of desired output array
    :param support: 3d object support array
    :param use_recip_sym: if True, discard imaginary component # name seems misleading
    :return ugrid: Fourier transform of nuvect, sampled on a uniform grid
    """
    from mpi4py import MPI

    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        # interleave data such that sets of [H,K,L,I] will be passed to each rank
        d1 = np.vstack((H_, K_, L_, nuvect)).reshape((-1), order='F')
        d1 = d1.astype(np.float64)

        # determine splits for roughly equal batching, always divisible by 4
        quot, remainder = divmod(len(d1), size*4)
        split_size = int(quot) * 4 * np.ones(size).astype(int)
        split_size[-1] += int(remainder)
        split_disp = np.insert(np.cumsum(split_size), 0, 0)[0:-1].astype(int)

        # generate output ac array
        ugrid = np.zeros((M,M,M))

    else:
        # initialize variables on remaining cores; first for scatter, then for reduce
        d1, split, split_size, split_disp = None, None, None, None
        ugrid = None

    # scatter batches across cores
    split_size = comm.bcast(split_size, root = 0)
    split_disp = comm.bcast(split_disp, root = 0)
    d1_local = np.zeros(split_size[rank])
    comm.Scatterv([d1, split_size, split_disp, MPI.DOUBLE], d1_local, root=0)

    # reshape such that columns are [H, K, L, I] 
    quot, remainder = divmod(len(d1_local), 4)
    assert remainder == 0
    d1_local = d1_local.reshape(quot, 4).astype(np.float32)

    # compute partial adjoint and sum, converting to float64 for consistency with MPI.DOUBLE
    adjoint_partial = adjoint_cpu(d1_local[:,3], d1_local[:,0], d1_local[:,1], d1_local[:,2], M)
    adjoint_partial = adjoint_partial.astype(np.float64)

    # synchronize cores and compute sum of batched adjoint calculations
    comm.Barrier()
    comm.Reduce([adjoint_partial, MPI.DOUBLE], [ugrid, MPI.DOUBLE], op=MPI.SUM, root=0)

    return ugrid
    
