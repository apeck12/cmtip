import numpy as np
import skopi as sk
import cmtip.nufft as nufft

from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator, cg

"""
Functions for estimating the autocorrelation from a series of diffraction images.
"""

def gen_nonuniform_positions(orientations, pixel_position_reciprocal):
    """
    Randomly rotate the reciprocal space vectors associated with each image. If 
    insufficient orientations are provided, reciprocal space positions are not
    rotated at all (but still broadcast to the correct shape).
    
    :param orientations: array of quaternions of shape (N_images, 4)
    :param pixel_position_reciprocal: array of pixels' reciprocal space vectors
    :return H,K,L: rotated reciprocal space vectors, each of shape 
        (N_images, n_panels,panel_pixel_num_x,panel_pixel_num_y)
    """
    if orientations.shape[0] > 0:
        rotmat = np.array([np.linalg.inv(sk.quaternion2rot3d(quat)) for quat in orientations])
    else:
        rotmat = np.array([np.eye(3)]) # changed
    H, K, L = np.einsum("ijk,klm->jilm", rotmat, pixel_position_reciprocal)
    # shape -> [N_images] x det_shape
    return H, K, L


def core_problem_convolution(uvect, M, F_ugrid_conv_, M_ups, ac_support, use_recip_sym=True):
    """
    Convolve data vector and input kernel of where data sample reciprocal 
    space in upsampled regime.
    
    :param uvect: data vector on uniform grid, flattened
    :param M: length of data vector along each axis
    :param F_ugrid_conv_: Fourier transform of convolution sampling array
    :param M_ups: length of data vector along each axis when upsampled
    :param ac_support: 2d support object for autocorrelation
    :param use_recip_sym: if True, discard imaginary componeent
    :return ugrid_conv_out: convolution of uvect and F_ugrid_conv_, flattened
    """
    if use_recip_sym:
        assert np.all(np.isreal(uvect))
    # Upsample
    ugrid = uvect.reshape((M,)*3) * ac_support
    ugrid_ups = np.zeros((M_ups,)*3, dtype=uvect.dtype)
    ugrid_ups[:M, :M, :M] = ugrid
    # Convolution = Fourier multiplication
    F_ugrid_ups = np.fft.fftn(np.fft.ifftshift(ugrid_ups))
    F_ugrid_conv_out_ups = F_ugrid_ups * F_ugrid_conv_
    ugrid_conv_out_ups = np.fft.fftshift(np.fft.ifftn(F_ugrid_conv_out_ups))
    # Downsample
    ugrid_conv_out = ugrid_conv_out_ups[:M, :M, :M]
    ugrid_conv_out *= ac_support
    if use_recip_sym:
        # Both ugrid_conv and ugrid are real, so their convolution
        # should be real, but numerical errors accumulate in the
        # imaginary part.
        ugrid_conv_out = ugrid_conv_out.real
    return ugrid_conv_out.flatten()


def gen_F_antisupport(M):
    """
    Generate an antisupport in Fourier space, which has zeros in the central
    sphere and ones in the high-resolution corners. 
    
    :param M: length of the cubic antisupport volume
    :return F_antisupport: volume that masks central region
    """
    # generate "antisupport" -- this has zeros in central sphere, 1s outside
    lu = np.linspace(-np.pi, np.pi, M)
    Hu_, Ku_, Lu_ = np.meshgrid(lu, lu, lu, indexing='ij')
    Qu_ = np.sqrt(Hu_**2 + Ku_**2 + Lu_**2)
    F_antisupport = Qu_ > np.pi 

    assert np.all(F_antisupport == F_antisupport[::-1, :, :])
    assert np.all(F_antisupport == F_antisupport[:, ::-1, :])
    assert np.all(F_antisupport == F_antisupport[:, :, ::-1])
    assert np.all(F_antisupport == F_antisupport[::-1, ::-1, ::-1])

    return F_antisupport


def fourier_reg(uvect, support, F_antisupport, M, use_recip_sym):
    """
    Generate the flattened matrix component that penalizes noise in the outer
    regions of reciprocal space, specifically outside the unit sphere of radius
    pi, where H_max, K_max, and L_max have been normalized to equal pi.
    
    :param uvect: data vector on uniform grid, flattened
    :param support: 3d support object for autocorrelation
    :param F_antisupport: support in Fourier space, unmasked at high frequencies
    :param M: length of data vector along each axis
    :param use_recip_sym: if True, discard imaginary component
    :return uvect: convolution of uvect and F_antisupport, flattened
    """
    ugrid = uvect.reshape((M,)*3) * support
    if use_recip_sym:
        assert np.all(np.isreal(ugrid))
    F_ugrid = np.fft.fftn(np.fft.ifftshift(ugrid))
    F_reg = F_ugrid * np.fft.ifftshift(F_antisupport)
    reg = np.fft.fftshift(np.fft.ifftn(F_reg))
    uvect = (reg * support).flatten()
    if use_recip_sym:
        uvect = uvect.real
    return uvect


def setup_linops(H, K, L, data,
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
    H_ = H.flatten() / reciprocal_extent * np.pi 
    K_ = K.flatten() / reciprocal_extent * np.pi 
    L_ = L.flatten() / reciprocal_extent * np.pi 

    # generate "antisupport" -- this has zeros in central sphere, 1s outside
    F_antisupport = gen_F_antisupport(M) 

    # Using upsampled convolution technique instead of ADA
    M_ups = M * 2
    ugrid_conv = nufft.adjoint_cpu(np.ones_like(data), H_, K_, L_, 
                                   M_ups, use_recip_sym=use_recip_sym, support=None)
    F_ugrid_conv_ = np.fft.fftn(np.fft.ifftshift(ugrid_conv)) #/ Mtot

    def W_matvec(uvect):
        """Define W part of the W @ x = d problem."""
        uvect_ADA = core_problem_convolution(uvect, M, F_ugrid_conv_, M_ups, ac_support)
        uvect_FDF = fourier_reg(uvect, ac_support, F_antisupport, M, use_recip_sym=use_recip_sym)
        uvect = alambda*uvect_ADA + rlambda*uvect + flambda*uvect_FDF
        return uvect

    W = LinearOperator(
        dtype=np.complex64,
        shape=(Mtot, Mtot),
        matvec=W_matvec)

    nuvect_Db = data * weights
    uvect_ADb = nufft.adjoint_cpu(nuvect_Db, H_, K_, L_, M, 
                                  support=ac_support, use_recip_sym=use_recip_sym).flatten()
    if np.sum(np.isnan(uvect_ADb)) > 0:
        print("Warning: nans in the adjoint calculation; intensities may be too large", flush=True)
    d = alambda*uvect_ADb + rlambda*x0

    return W, d


def solve_ac(generation,
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
    Mtot = M**3
    N_images = slices_.shape[0]
    N = np.prod(slices_.shape) # total number of data points

    # compute oriented reciprocal space positions for each data point
    if orientations is None:
        orientations = sk.get_random_quat(N_images)
    H, K, L = gen_nonuniform_positions(orientations, pixel_position_reciprocal)
    H, K, L = H.astype(np.float32), K.astype(np.float32), L.astype(np.float32)

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
    rlambda = Mtot/N / 1000
    flambda = 1e3
    maxiter = 100

    def callback(xk):
        callback.counter += 1
    callback.counter = 0

    # solve sparse linear system
    x0 = ac_estimate.flatten()
    W, d = setup_linops(H, K, L, data,
                        ac_support, weights, x0,
                        M, Mtot, N, reciprocal_extent,
                        alambda, rlambda, flambda,
                        use_recip_sym=use_recip_sym)
    ret, info = cg(W, d, x0=x0, maxiter=maxiter, callback=callback)
    ac = ret.reshape((M,)*3)
    if use_recip_sym:
        assert np.all(np.isreal(ac))
    ac = ac.real
    it_number = callback.counter

    print(f"Recovered AC in {it_number} iterations.", flush=True)

    return ac
