import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, cg
from scipy.linalg import norm

import skopi as sk
import cmtip.autocorrelation as ac_base
import cmtip.nufft as nufft

try:
    from mpi4py import MPI
except ImportError:
    print("MPI seems unavailable")

"""
MPI version of the autocorrelation solver. Functions of this component of the MTIP algorithm
that do not require communication among ranks are called from the main autocorrelation script.
"""

def reduce_bcast(comm, vect):
    """
    Compute sum of input vect (ugrid_conv) from all ranks and broadcast.

    :param comm: MPI intracommunicator instance
    :param vect: uniform grid vector from each rank
    :return vect: sum of vect from all ranks
    """
    vect = np.ascontiguousarray(vect)
    reduced_vect = np.zeros_like(vect)
    comm.Reduce(vect, reduced_vect, op=MPI.SUM, root=0)
    vect = reduced_vect
    comm.Bcast(vect, root=0)
    return vect


def setup_linops_mpi(comm, H, K, L, data, ac_support, weights, x0, 
                     M, Mtot, N, reciprocal_extent, alambda, rlambda, flambda, use_gpu=False, use_recip_sym=True):
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
    # prepare for NUFFT calculation: flatten, limit range, convert dtype
    H_ = H.flatten().astype(np.float32) / reciprocal_extent * np.pi 
    K_ = K.flatten().astype(np.float32) / reciprocal_extent * np.pi 
    L_ = L.flatten().astype(np.float32) / reciprocal_extent * np.pi 

    # generate "antisupport" -- this has zeros in central sphere, 1s outside
    F_antisupport = ac_base.gen_F_antisupport(M) 

    # Using upsampled convolution technique instead of ADA
    M_ups = M * 2
    if use_gpu:
        ugrid_conv = nufft.adjoint_gpu(np.ones_like(data), H_, K_, L_,
                                       M_ups, use_recip_sym=use_recip_sym, support=None)
    else:
        ugrid_conv = nufft.adjoint_cpu(np.ones_like(data), H_, K_, L_, 
                                       M_ups, use_recip_sym=use_recip_sym, support=None)
    ugrid_conv = reduce_bcast(comm, ugrid_conv)
    F_ugrid_conv_ = np.fft.fftn(np.fft.ifftshift(ugrid_conv)) 

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
    if use_gpu:
        uvect_ADb = nufft.adjoint_gpu(nuvect_Db, H_, K_, L_, M,
                                      support=ac_support, use_recip_sym=use_recip_sym).flatten()
    else:
        uvect_ADb = nufft.adjoint_cpu(nuvect_Db, H_, K_, L_, M, 
                                      support=ac_support, use_recip_sym=use_recip_sym).flatten()
    uvect_ADb = reduce_bcast(comm, uvect_ADb)

    if np.sum(np.isnan(uvect_ADb)) > 0:
        print("Warning: nans in the adjoint calculation; intensities may be too large", flush=True)
    d = alambda*uvect_ADb + rlambda*x0

    return W, d


def solve_ac_mpi(comm, generation, pixel_position_reciprocal, reciprocal_extent,
                 slices_, M, orientations=None, ac_estimate=None, use_gpu=False, use_recip_sym=True):
    """
    Estimate the autocorrelation volume using MPI. 
    
    :param comm: MPI intracommunicator instance
    :param generation: current iteration
    :param pixel_position_reciprocal: pixels' reciprocal space positions, array of shape
        (3,n_panels,n_pixels_per_panel)
    :param reciprocal_extent: reciprocal space magnitude of highest resolution pixel
    :param slices_: intensity data of shape (n_images, n_panels, n_pixels_per_panel)
    :param M: cubic length of autocorrelation mesh
    :param orientations: n_images quaternions, if available
    :param ac_estimate: 3d array of estimated autocorrelation, if available
    :param use_gpu: boolean; if True use cufinufft rather than finufft
    :param use_recip_sym: if True, discard imaginary component of ac estimate
    :return ac: 3d array solution for autocorrelation from top-scoring rank
    """

    Mtot = M**3
    N_images = slices_.shape[0]
    N = np.prod(slices_.shape) # total number of data points

    # compute oriented reciprocal space positions for each data point
    if orientations is None:
        orientations = sk.get_random_quat(N_images)
    H, K, L = ac_base.gen_nonuniform_positions(orientations, pixel_position_reciprocal)
    data = slices_.flatten().astype(np.float32)

    if ac_estimate is None:
        ac_support = np.ones((M,)*3)
        ac_estimate = np.zeros((M,)*3)
    else:
        ac_smoothed = gaussian_filter(ac_estimate, 0.5)
        ac_support = (ac_smoothed > 1e-12).astype(np.float)
        ac_estimate *= ac_support
    weights = np.ones(N).astype(np.float32)

    alambda = 1
    #rlambda = Mtot / N / 1000
    #rlambda = Mtot/(N*comm.size) * 2**(comm.rank - comm.size/2)
    rlambda = Mtot/N * 2**(comm.rank - comm.size/2)
    #flambda = 1e3
    flambda = 1e5 * pow(10, comm.rank - comm.size//2)
    maxiter = 100

    def callback(xk):
        callback.counter += 1
    callback.counter = 0

    # solve sparse linear system
    x0 = ac_estimate.flatten()
    W, d = setup_linops_mpi(comm, H, K, L, data,
                            ac_support, weights, x0,
                            M, Mtot, N, reciprocal_extent,
                            alambda, rlambda, flambda,
                            use_gpu=use_gpu, use_recip_sym=use_recip_sym)
    ret, info = cg(W, d, x0=x0, maxiter=maxiter, callback=callback)

    # assess which rank to keep by analyzing converged solution and residuals
    v1 = norm(ret)
    v2 = norm(W*ret-d)

    summary = comm.gather((comm.rank, rlambda, v1, v2), root=0)
    if comm.rank == 0:
        ranks, lambdas, v1s, v2s = [np.array(el) for el in zip(*summary)]

        if generation == 0:
            # Expect non-convergence => weird results.
            # Heuristic: retain rank with highest lambda and high v1.
            idx = v1s >= np.mean(v1s)
            imax = np.argmax(lambdas[idx])
            iref = np.arange(len(ranks), dtype=int)[idx][imax]
        else:
            # Take corner of L-curve: min (v1+v2)
            iref = np.argmin(v1s+v2s)
        ref_rank = ranks[iref]

    else:
        ref_rank = -1
    ref_rank = comm.bcast(ref_rank, root=0)

    ac = ret.reshape((M,)*3)
    if use_recip_sym:
        assert np.all(np.isreal(ac))
    ac = np.ascontiguousarray(ac.real)
    
    print(f"Rank {comm.rank} got AC in {callback.counter} iterations.", flush=True)
    comm.Bcast(ac, root=ref_rank)
    if comm.rank == 0:
        print(f"Keeping result from rank {ref_rank}.")

    return ac


def solve_ac_mpi_mult(comm, generation, pixel_position_reciprocal, reciprocal_extent,
                      slices_, M, n_conformations, orientations=None, conformations=None, 
                      ac_estimate=None, use_gpu=False, use_recip_sym=True):
    """
    Estimate multiple autocorrelation volumes for each conformation with MPI acceleration. 
    
    :param comm: MPI intracommunicator instance
    :param generation: current iteration
    :param pixel_position_reciprocal: pixels' reciprocal space positions, array of shape
        (3,n_panels,n_pixels_per_panel)
    :param reciprocal_extent: reciprocal space magnitude of highest resolution pixel
    :param slices_: intensity data of shape (n_images, n_panels, n_pixels_per_panel)
    :param M: cubic length of autocorrelation mesh
    :param n_conformations: number of expected conformations
    :param orientations: n_images quaternions, if available
    :param conformations: n_images conformation assignments, if available
    :param ac_estimate: 4d array of autocorrelation volumes of shape (n_volumes,M,M,M)
    :param use_gpu: boolean; if True, use cufinufft rather than finufft
    :param use_recip_sym: if True, discard imaginary component of ac estimate
    :return ac_mult: 4d array of autocorrelation volumes of shape (n_volumes,M,M,M)
    """
    # randomly assign each image to a conformation if none supplied
    if conformations is None:
        conformations = np.random.randint(low=0, high=n_conformations, size=len(slices_))
        
    # generate random quaternions if none supplied
    if orientations is None:
        orientations = sk.get_random_quat(len(slices_))

    if ac_estimate is None:
        ac_estimate = n_conformations*[None]

    # for debugging purposes
    rank = comm.rank
    nc0, nc1 = len(np.where(conformations==0)[0]), len(np.where(conformations==1)[0])
    print(f"Rank {rank}: {nc0} images assigned to conf0, {nc1} images assigned to conf1")
    
    ac_mult = np.zeros((n_conformations,) + (M,M,M))
    for nc in range(n_conformations):            
        ac_mult[nc] = solve_ac_mpi(comm, 
                                   generation, 
                                   pixel_position_reciprocal, 
                                   reciprocal_extent,
                                   slices_[conformations==nc], 
                                   M, 
                                   orientations=orientations[conformations==nc], 
                                   ac_estimate=ac_estimate[nc], 
                                   use_gpu=use_gpu,
                                   use_recip_sym=use_recip_sym)

    return ac_mult
