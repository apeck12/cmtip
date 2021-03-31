import numpy as np
import skopi as sk
import cmtip.nufft as nufft
import cmtip.autocorrelation as ac_base

from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, cg
from mpi4py import MPI

"""
Functions for estimating the autocorrelation from a series of diffraction images.
This version uses MPI to distribute the adjoint calculation across multiple CPUs.
"""

def setup_linops(comm, H, K, L, data,
                 ac_support, weights, x0,
                 M, Mtot, N, reciprocal_extent,
                 alambda, rlambda, flambda,
                 use_recip_sym=True):
    """Define W and d parts of the W @ x = d problem.

    W = al*A_adj*Da*A + rl*I  + fl*F_adj*Df*F
    d = al*A_adj*Da*b + rl*x0 + 0

    Where:
        A represents the NUFFT operator
        A_adj its adjoint
        I the identity
        F the FFT operator
        F_adj its atjoint
        Da, Df weights
        b the data
        x0 the initial guess (ac_estimate)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    H_ = H.flatten() / reciprocal_extent * np.pi 
    K_ = K.flatten() / reciprocal_extent * np.pi 
    L_ = L.flatten() / reciprocal_extent * np.pi 

    # generate "antisupport" -- this has zeros in central sphere, 1s outside
    F_antisupport = ac_base.gen_F_antisupport(M) 

    # Using upsampled convolution technique instead of ADA
    M_ups = M * 2
    ugrid_conv = nufft.adjoint_cpu_parallel(comm, np.ones_like(data), H_, K_, L_, 
                                            M_ups, use_recip_sym=use_recip_sym, support=None)

    if rank == 0:
        F_ugrid_conv_ = np.fft.fftn(np.fft.ifftshift(ugrid_conv)) 
    else:
        F_ugrid_conv_ = None
    F_ugrid_conv_ = comm.bcast(F_ugrid_conv_, root=0)

    def W_matvec(uvect):
        """Define W part of the W @ x = d problem."""
        uvect_ADA = ac_base.core_problem_convolution(uvect, M, F_ugrid_conv_, M_ups, ac_support)
        uvect_FDF = ac_base.fourier_reg(uvect, ac_support, F_antisupport, M, use_recip_sym=use_recip_sym)
        uvect = alambda*uvect_ADA + rlambda*uvect + flambda*uvect_FDF
        return uvect

    W = LinearOperator(
        dtype=np.complex64,
        shape=(Mtot, Mtot),
        matvec=W_matvec)
    nuvect_Db = data * weights 

    # not sure why this is necessary, but without it nonzero ranks get a uvect_ADb of None
    if rank != 0:
        uvect_ADb = None
    uvect_ADb = nufft.adjoint_cpu_parallel(comm, nuvect_Db, H_, K_, L_, M, 
                                           support=ac_support, use_recip_sym=use_recip_sym)
    uvect_ADb = comm.bcast(uvect_ADb, root=0)
    uvect_ADb = uvect_ADb.flatten()
    comm.Barrier()

    if np.sum(np.isnan(uvect_ADb)) > 0:
        print("Warning: nans in the adjoint calculation; intensities may be too large", flush=True)
    d = alambda*uvect_ADb + rlambda*x0

    return W, d


def solve_ac(comm,
             generation,
             pixel_position_reciprocal,
             reciprocal_extent,
             slices_,
             M,
             orientations=None,
             ac_estimate=None,
             use_recip_sym=True):
    """
    Estimate the autocorrelation by solving a sparse linear system that maximizes the 
    consistency of the projected images with the intensity model.
    
    :param comm: MPI intracommunicator instance
    :param generation: current iteration
    :param pixel_position_reciprocal: pixels' reciprocal space positions, array of shape
        (3,n_panels,panel_pixel_num_x,panel_pixel_num_y)
    :param reciprocal_extent: reciprocal space magnitude of highest resolution pixel
    :param slices_: intensity data of shape (n_images, n_panels, panel_pixel_num_x, panel_pixel_num_y)
    :param M: cubic length of autocorrelation mesh
    :param orientations: n_images quaternions, if available
    :param ac_estimate: 3d array of estimated autocorrelation, if available
    :param use_recip_sym: if True, discard imaginary component of ac estimate
    :return ac: 3d array solution for autocorrelation 
    """
    # set up MPI variables
    size = comm.Get_size()
    rank = comm.Get_rank()

    Mtot = M**3 # number of points on uniform grid
    N_images = slices_.shape[0] # number of images in input dataset
    N = np.prod(slices_.shape) # total number of data points in non-uniform reciprocal space
    #ac_avg = np.zeros((M,M,M)).astype(np.float64) # for final output

    # compute oriented reciprocal space positions for each data point
    if rank == 0:
        if orientations is None:
            orientations = sk.get_random_quat(N_images)
        H, K, L = ac_base.gen_nonuniform_positions(orientations, pixel_position_reciprocal)
        data = slices_.flatten().astype(np.float32)
        ac_avg = np.zeros((M,M,M)).astype(np.float64)
    else:
        H, K, L, data = None, None, None, None
        ac_avg = None

    # broadcast reciprocal space positions and intensity data to all ranks
    H = comm.bcast(H, root=0)
    K = comm.bcast(K, root=0)
    L = comm.bcast(L, root=0)
    data = comm.bcast(data, root=0)

    # generate or preprocess the given autocorrelation
    if ac_estimate is None:
        ac_support = np.ones((M,)*3)
        ac_estimate = np.zeros((M,)*3)
    else:        
        ac_smoothed = gaussian_filter(ac_estimate, 0.5)
        ac_support = (ac_smoothed > 1e-12).astype(np.float)
        ac_estimate *= ac_support
    weights = np.ones(N).astype(np.float32)

    # optimization parameters
    alambda = 1
    rlambda = Mtot / N / 1000
    flambda = 1e3
    maxiter = 100

    def callback(xk):
        callback.counter += 1
    callback.counter = 0

    # solve sparse linear system
    x0 = ac_estimate.flatten()
    W, d = setup_linops(comm, H, K, L, data,
                        ac_support, weights, x0,
                        M, Mtot, N, reciprocal_extent,
                        alambda, rlambda, flambda,
                        use_recip_sym=use_recip_sym)
    ret, info = cg(W, d, x0=x0, maxiter=maxiter, callback=callback)

    ac = ret.reshape((M,)*3)
    if use_recip_sym:
        assert np.all(np.isreal(ac))
    ac = ac.real.astype(np.float64) / np.float(size)

    print(f"Recovered AC in {callback.counter} iterations.", flush=True)

    comm.Barrier()
    comm.Reduce([ac, MPI.DOUBLE], [ac_avg, MPI.DOUBLE], op=MPI.SUM, root=0)
    return ac_avg
