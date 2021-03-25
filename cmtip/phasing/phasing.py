import numpy as np
from scipy.ndimage import gaussian_filter

"""
Functions for phasing an oversampled autocorrelation.
"""

def save_mrc(savename, data, voxel_size=None):
    """
    Save Nd numpy array to path savename in mrc format.
    
    :param savename: path to which to save mrc file
    :param data: input numpy array
    :param voxel_size: voxel size for header, optional
    """
    import mrcfile

    mrc = mrcfile.new(savename, overwrite=True)
    mrc.header.map = mrcfile.constants.MAP_ID
    mrc.set_data(data.astype(np.float32))
    if voxel_size is not None:
        mrc.voxel_size = voxel_size
    mrc.close()
    return


def center_of_mass(rho_, hkl_, M):
    """
    Compute the object's center of mass.
    
    :param rho_: density array
    :param hkl_: coordinates array
    :param M: cubic length of density volume
    :return vect: vector center of mass in units of pixels
    """
    rho_ = np.abs(rho_)
    num = (rho_ * hkl_).sum(axis=(1, 2, 3))
    den = rho_.sum()
    return np.round(num/den * M/2)


def recenter(rho_, support_, M):
    """
    Recenter the density and support by shifting center of mass to origin.
    
    :param rho_: density array
    :param support_: object's support
    :param M: cubic length of density volume
    """
    ls = np.linspace(-1, 1, M+1)
    ls = (ls[:-1] + ls[1:])/2

    hkl_list = np.meshgrid(ls, ls, ls, indexing='ij')
    hkl_ = np.stack([np.fft.ifftshift(coord) for coord in hkl_list])
    vect = center_of_mass(rho_, hkl_, M)

    for i in range(3):
        shift = int(vect[i])
        rho_[:] = np.roll(rho_, -shift, i)
        support_[:] = np.roll(support_, -shift, i)


def create_support_(ac_, M, Mquat, generation):
    """
    Generate a support based on the region of high ACF signal (thresh_support_)
    inside the central quarter region of the full ACF volume (square_support_).
    
    :param ac_: autocorrelation volume, fftshifted
    :param M: cubic length of autocorrelation volume
    :param Mquat: cubic length of region of interest
    :param generation: current iteration
    """
    sl = slice(Mquat, -Mquat)
    square_support = np.zeros((M, M, M), dtype=np.bool_)
    square_support[sl, sl, sl] = 1
    square_support_ = np.fft.ifftshift(square_support)

    thresh_support_ = ac_ > 1e-2 * ac_.max()

    return np.logical_and(square_support_, thresh_support_)


def ER_loop(n_loops, rho_, amplitudes_, amp_mask_, support_, rho_max):
    """
    Wrapper for calling an iteration of Error Reduction (ER).
    """
    for k in range(n_loops):
        ER(rho_, amplitudes_, amp_mask_, support_, rho_max)


def HIO_loop(n_loops, beta, rho_, amplitudes_, amp_mask_, support_, rho_max):
    """
    Wrapper for calling an iteration of Hybrid Input Output (HIO).
    """
    for k in range(n_loops):
        HIO(beta, rho_, amplitudes_, amp_mask_, support_, rho_max)


def ER(rho_, amplitudes_, amp_mask_, support_, rho_max):
    """
    Perform one step of Error Reduction. This involves (1) updating the amplitudes 
    from the current estimated density with those given by the autocorrelation, and 
    (2) enforcing density positivity by setting any positions with negative density
    or outside the support to 0. Density values that exceed rho_max are thresholded.
    
    :param rho_: density estimate
    :param amplitudes_: amplitudes computed from the autocorrelation
    :param amp_mask_: amplitude mask, where the position of the DC term is masked
    :param support_: binary mask for object's support
    :param rho_mask: maximum permitted density value
    """
    rho_mod_, support_star_ = step_phase(rho_, amplitudes_, amp_mask_, support_)
    rho_[:] = np.where(support_star_, rho_mod_, 0)
    i_overmax = rho_mod_ > rho_max
    rho_[i_overmax] = rho_max


def HIO(beta, rho_, amplitudes_, amp_mask_, support_, rho_max):
    """
    Perform one step of Hybrid Input Output. This involves (1) updating the amplitudes 
    from the current estimated density with those given by the autocorrelation, and 
    (2) forcing any positions that don't satisfy the updated support constraint toward
    0 (rather than exactly 0, as in ER). 
    
    :param beta: feedback constant, affects rate of convergence 
    :param rho_: density estimate
    :param amplitudes_: amplitudes computed from the autocorrelation
    :param amp_mask_: amplitude mask, where the position of the DC term is masked
    :param support_: binary mask for object's support
    :param rho_mask: maximum permitted density value
    """
    rho_mod_, support_star_ = step_phase(rho_, amplitudes_, amp_mask_, support_)
    rho_[:] = np.where(support_star_, rho_mod_, rho_-beta*rho_mod_)
    i_overmax = rho_mod_ > rho_max
    rho_[i_overmax] += 2*beta*rho_mod_[i_overmax] - rho_max


def step_phase(rho_, amplitudes_, amp_mask_, support_):
    """
    Replace the amplitudes given by the density estimate with those computed from 
    the autocorrelation function, except for the amplitude of the central/max peak 
    of the Fourier transform. Then recalculate the estimated density and update the 
    support, with positivity of the density enforced for the latter.
    
    :param rho_: density estimate
    :param amplitudes_: amplitudes computed from the autocorrelation
    :param amp_mask_: amplitude mask, where the position of the DC term is masked
    :param support_: binary mask for object's support
    :return rho_mod_: updated density estimate 
    :return support_star_: updated support
    """
    rho_hat_ = np.fft.fftn(rho_)
    phases_ = np.angle(rho_hat_)
    rho_hat_mod_ = np.where(
        amp_mask_,
        amplitudes_ * np.exp(1j*phases_),
        rho_hat_)
    rho_mod_ = np.fft.ifftn(rho_hat_mod_).real
    support_star_ = np.logical_and(support_, rho_mod_>0)
    return rho_mod_, support_star_


def shrink_wrap(cutoff, sigma, rho_, support_):
    """
    Perform shrink-wrap algorithm to update the support for improved convergence.
    Specifically, the support is 'shrunk' to only include positions where the low-
    pass filtered density exceeds some threshold (given by the max_density*cutoff). 
    
    :param cutoff: fraction of maximum density value
    :param sigma: standard deviation of Gaussian to low-pass filter density with
    :param rho_: density estimate
    :param support_: object support
    """
    rho_abs_ = np.absolute(rho_)
    # By using 'wrap', we don't need to fftshift it back and forth
    rho_gauss_ = gaussian_filter(
        rho_abs_, mode='wrap', sigma=sigma, truncate=2)
    support_[:] = rho_gauss_ > rho_abs_.max() * cutoff

    
def phase(generation, ac, support_=None, rho_=None, n_iterations=10):
    """
    Solve the phase problem from the oversampled autocorrelation by performing cycles
    of ER, HIO, ER, and shrinkwrap. 
    
    :param generation: current iteration of M-TIP loop
    :param ac: 3d array of autocorrelation
    :param support_: initial object support
    :param rho_: initial density estimate
    :param n_iterations: number of cycles of ER/HIO/ER/shrinkwrap to perform
    :return ac_phased: updated autocorrelation estimate
    :return support_: updated support estimate
    :return rho_: updated density estimate
    """
    Mquat = int((ac.shape[0]-1)/4)#parms.Mquat
    M = ac.shape[0] #4*Mquat + 1
    Mtot = M**3

    ac_filt = gaussian_filter(np.maximum(ac.real, 0), mode='constant',
                              sigma=1, truncate=2)
    ac_filt_ = np.fft.ifftshift(ac_filt)

    intensities_ = np.abs(np.fft.fftn(ac_filt_))

    amplitudes_ = np.sqrt(intensities_)

    amp_mask_ = np.ones((M, M, M), dtype=np.bool_)
    amp_mask_[0, 0, 0] = 0  # Mask out central peak

    if support_ is None:
        support_ = create_support_(ac_filt_, M, Mquat, generation)

    if rho_ is None:
        rho_ = support_ * np.random.rand(*support_.shape)

    rho_max = np.infty

    nER = 10 #parms.nER
    nHIO = 5 #parms.nHIO

    for i in range(n_iterations):
        ER_loop(nER, rho_, amplitudes_, amp_mask_, support_, rho_max)
        HIO_loop(nHIO, 0.3, rho_, amplitudes_, amp_mask_, support_, rho_max)
        ER_loop(nER, rho_, amplitudes_, amp_mask_, support_, rho_max)
        shrink_wrap(5e-2, 1, rho_, support_)
    ER_loop(nER, rho_, amplitudes_, amp_mask_, support_, rho_max)

    recenter(rho_, support_, M)

    intensities_phased_ = np.abs(np.fft.fftn(rho_))**2

    ac_phased_ = np.abs(np.fft.ifftn(intensities_phased_))
    ac_phased = np.fft.fftshift(ac_phased_)

    return ac_phased, support_, rho_
