from mpi4py import MPI
import numpy as np
import cmtip.phasing as phase_base


def phase_mpi(comm, generation, ac, support_=None, rho_=None, n_iterations=10):
    """
    Wrapper for phasing using MPI. Phasing is performed on the first rank only by 
    iterating through ER, HIO, and shrinkwrap. Results are broadcast to all ranks.

    :param comm: MPI intracommunicator instance
    :param generation: current iteration of M-TIP loop
    :param ac: 3d array of autocorrelation
    :param support_: initial object support
    :param rho_: initial density estimate
    :param n_iterations: number of cycles of ER/HIO/ER/shrinkwrap to perform
    :return ac_phased: updated autocorrelation estimate
    :return support_: updated support estimate
    :return rho_: updated density estimate
    """
    if comm.rank == 0:
        ac_phased, support_, rho_ = phase_base.phase(generation, ac, support_=support_, 
                                                     rho_=rho_, n_iterations=n_iterations)
    else:
        ac_phased = np.zeros(ac.shape, order="F")
        support_, rho_ = None, None
    
    comm.Bcast(ac_phased, root=0)
    return ac_phased, support_, rho_
