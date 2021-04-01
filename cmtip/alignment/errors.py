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
