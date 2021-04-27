import numpy as np
import cmtip.phasing as phase_base

try:
    from mpi4py import MPI
except ImportError:
    print("MPI seems unavailable")


def phase_mpi(comm, generation, ac, support_=None, rho_=None, n_iterations=10, nER=50, nHIO=25):
    """
    Wrapper for phasing using MPI. Phasing is performed on the first rank only by 
    iterating through ER, HIO, and shrinkwrap. Results are broadcast to all ranks.

    :param comm: MPI intracommunicator instance
    :param generation: current iteration of M-TIP loop
    :param ac: 3d array of autocorrelation
    :param support_: initial object support
    :param rho_: initial density estimate
    :param n_iterations: number of cycles of ER/HIO/ER/shrinkwrap to perform
    :param nER: number of ER iterations to perform for each ER loop
    :param nHIO: number of HIO iterations to perofrm for each HIO loop
    :return ac_phased: updated autocorrelation estimate
    :return support_: updated support estimate
    :return rho_: updated density estimate
    """
    if comm.rank == 0:
        ac_phased, support_, rho_ = phase_base.phase(generation, ac, support_=support_, 
                                                     rho_=rho_, n_iterations=n_iterations,
                                                     nER=nER, nHIO=nHIO)
    else:
        ac_phased = np.zeros(ac.shape, order="F")
        support_, rho_ = None, None
    
    comm.Bcast(ac_phased, root=0)
    return ac_phased, support_, rho_


def resize_mpi(comm, support_, rho_, M_next, r_extent_orig, r_extent_next):
    """
    Wrapper for resizing the support and density volumes.

    :param comm: MPI intracommunicator instance
    :param support_: support estimate
    :param rho_: density estimate
    :param M_next: cubic length of density volume of next iteration
    :param r_extent_orig: reciprocal space extent of current iteration
    :param r_extent_next: reciprocal space extent of next iteration
    :return support_: resized support estimate
    :return rho_: resized density estimate
    """
    if comm.rank == 0:
        # shift DC component to center
        support = np.fft.ifftshift(support_)
        rho = np.fft.ifftshift(rho_)
        
        # resize volume
        support = phase_base.resize_volume(support, M_next, r_extent_orig, r_extent_next)
        rho = phase_base.resize_volume(rho, M_next, r_extent_orig, r_extent_next)  
        
        # shift DC component back to corner
        support_ = np.fft.fftshift(support)
        rho_ = np.fft.fftshift(rho)

    else:
        support_, rho_ = None, None
    
    return support_, rho_
