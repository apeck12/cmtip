from cmtip.alignment import errors
import numpy as np

def test_alignment_error():
    """ Check that calculation of rotation between quaternions is correct. """

    # no rotation
    quat1 = np.array([1., 0., 0., 0.]) 
    # 180 degree rotation
    quatx = np.array([0., 1., 0., 0.]) 
    quaty = np.array([0., 0., 1., 0.]) 
    quatz = np.array([0., 0., 0., 1.])
    # 90 degree rotation
    quatx90 = np.array([1., 1., 0., 0.]) / np.sqrt(2)
    quaty90 = np.array([1., 0., 1., 0.]) / np.sqrt(2)
    quatz90 = np.array([1., 0., 0., 1.]) / np.sqrt(2)

    for q in [quatx,quaty,quatz]:
        assert errors.alignment_error(quat1, q) == 180
    for q in [quatx90,quaty90,quatz90]:
        assert errors.alignment_error(quat1, q) == 90
