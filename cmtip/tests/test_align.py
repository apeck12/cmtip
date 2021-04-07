import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import skopi as sk
import os

from cmtip.simulate import *
import cmtip.alignment as alignment

class TestAlignment(object):
    """
    Test the orientation matching component of the MTIP algorithm. 
    """
    @classmethod
    def setup_class(cls):
        
        args = dict()
        args['beam_file'] = os.path.join(os.path.dirname(__file__), '../../examples/input/amo86615.beam')
        args['pdb_file'] = os.path.join(os.path.dirname(__file__), '../../examples/input/3iyf.pdb')
        args['det_info'] = ("128", "0.08", "0.2")
        args['n_images'] = 3
        
        cls.data = simulate_images(args)
        ivol = np.square(np.abs(cls.data['volume']))
        cls.ac = np.fft.fftshift(np.abs(np.fft.ifftn(ivol))).astype(np.float32)

    def test_match_orientations(self):
        """ 
        Test that alignment.match_orientations finds the correct quaternions from
        a mix of correct and random quaternions.
        """
        quats = alignment.match_orientations(0, 
                                             self.data['pixel_position_reciprocal'], 
                                             self.data['reciprocal_extent'], 
                                             self.data['intensities'],
                                             self.ac,
                                             100,
                                             true_orientations=self.data['orientations'])
    
        assert np.all(quats - self.data['orientations'])==0
    
    def test_compute_slices(self):
        """
        Check that alignment.compute_slices retrieves correctly oriented slices by
        computing the NUFFT from the ideal autocorrelation. Plot for visualization.
        """
        # quantitative check -- threshold takes into account interpolation error
        cslices = alignment.compute_slices(self.data['orientations'],
                                           self.data['pixel_position_reciprocal'], 
                                           self.data['reciprocal_extent'], 
                                           self.ac)
        cslices = cslices.reshape(len(self.data['orientations']), self.data['n_pixels_per_image'])
        
        cc_vals = list()
        for i in range(3):
            for j in range(3):
                cc_vals.append(np.corrcoef(cslices[i], self.data['intensities'][j][0].flatten())[0,1])
        cc_vals = np.array(cc_vals).reshape(3,3)

        assert np.allclose(np.diag(cc_vals),[1],atol=1e-3) # matching images
        assert not np.allclose(np.triu(cc_vals, k=1),[1],atol=1e-3) # mismatched images

        # also plot for visual inspection
        f, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(9,6))

        for i,ax in enumerate([ax1,ax2,ax3]):
            ax.imshow(cslices[i].reshape(self.data['det_shape'][1:]), vmax=cslices.mean())
            ax.set_title("Orientation %i" %i, fontsize=12)
        for i,ax in enumerate([ax4,ax5,ax6]):
            ax.imshow(self.data['intensities'][i][0].reshape(self.data['det_shape'][1:]), vmax=self.data['intensities'].mean())

        for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
            ax.set_xticks([])
            ax.set_yticks([])

        ax1.set_ylabel("Sliced image", fontsize=12)
        ax4.set_ylabel("Reference image", fontsize=12)
    
        f.savefig("test_compute_slices.png", dpi=300, bbox_inches='tight')
