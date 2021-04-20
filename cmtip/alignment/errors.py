import numpy as np


def alignment_error(q1,q2):
    """
    Compute the alignment error between the reference / ground truth quaternions
    and those computed by the alignment scheme. Error is the angular difference
    between pairs of quaternions, given by:

    theta = arccos( 2 * <q1, q2>**2 - 1)

    where <q1, q2> is the inner product of the quaternions.
    See: https://math.stackexchange.com/questions/90081/quaternion-distance.

    :param q1: reference / ground truth quaternions
    :param q2: quaternions estimated by alignment
    :param theta: error array in degrees
    """
    inner_product = np.sum(q1 * q2, axis=-1)
    cos_theta = 2*np.square(inner_product) - 1
    cos_theta = np.around(cos_theta, decimals=5) # ensure within arccos domain
    error = np.arccos(cos_theta)
    return np.rad2deg(error)


def mean_error(errors, symmetry=2):
    """
    Compute the mean alignment error, taking symmetry into account.
    
    :param errors: array of orientation errors
    :param symmetry: particle's symmetry fold
    :return mean_error: mean orientation error
    """
    interval=360/symmetry
    symm_vals = np.arange(0,180+interval,interval)
    symm_vals = np.tile(symm_vals, errors.shape[0]).reshape(errors.shape[0],len(symm_vals))
    
    symm_diffs = np.abs(errors - symm_vals.T)
    return np.mean(np.min(symm_diffs, axis=0))
