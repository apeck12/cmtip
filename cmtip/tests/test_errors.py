import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cmtip.alignment import errors
from cmtip.phasing import *
import numpy as np
import skopi as sk
import os

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


class TestMapsFunctions(object):
    """
    Test the phasing algorithm starting from the ideal autocorrelation.
    In this case, different handedness will be indistinguishable in projection.
    """
    
    @classmethod
    def setup_class(cls):
        
        args = dict()
        args['beam_file'] = os.path.join(os.path.dirname(__file__), '../../examples/input/amo86615.beam')
        args['pdb_file'] = os.path.join(os.path.dirname(__file__), '../../examples/input/3iyf.pdb')
        args['resolution'] = 9.0 # in Angstrom
        args['M'] = 81 
        args['spacing'] = 0.01
        
        ref_ac, cls.ref_density = compute_reference(args['pdb_file'], args['M'], 1e10/args['resolution'])
        cls.args = args

    def test_rotate_volume(self):
        """ Check that volume rotation works. """

        q_tar = sk.get_random_quat(1)[0]
        tar_density = rotate_volume(self.ref_density, np.linalg.inv(sk.quaternion2rot3d(q_tar)))
        inv_density = rotate_volume(tar_density, sk.quaternion2rot3d(q_tar))
        assert np.corrcoef(self.ref_density.flatten(), inv_density.flatten())[0,1] > 0.85

        f, ((ax1,ax2,ax3)) = plt.subplots(1, 3, figsize=(9,3))
        hs = int(self.args['M']/2)

        ax1.imshow(self.ref_density[hs,:,:])
        ax2.imshow(tar_density[hs,:,:])
        ax3.imshow(inv_density[hs,:,:])
        
        ax1.set_title("Reference", fontsize=12)
        ax2.set_title("Rotated", fontsize=12)
        ax3.set_title("Unrotated", fontsize=12)

        f.savefig("test_rotate_volume.png", dpi=300, bbox_inches='tight')

    def test_compute_fsc(self):
        """ Check volume alignment and FSC calculation. """

        q_tar = sk.get_random_quat(1)[0]
        tar_density = rotate_volume(self.ref_density, np.linalg.inv(sk.quaternion2rot3d(q_tar)))
        inv_density = rotate_volume(tar_density, sk.quaternion2rot3d(q_tar))
        
        rs, fsc, res = compute_fsc(self.ref_density, inv_density, 1e10/self.args['resolution'], self.args['spacing'])
        assert res < self.args['resolution']*1.25

