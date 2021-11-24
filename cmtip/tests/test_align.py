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
        args['increase_factor'] = 1
        args['quantize'] = False

        cls.data = simulate_images(args)
        ivol = np.square(np.abs(cls.data['volume']))
        cls.ac = np.fft.fftshift(np.abs(np.fft.ifftn(ivol))).astype(np.float32)

        # data for multiple conformations test
        args['det_info'] = ("128", "0.08", "0.04")
        args['n_images'] = 2

        args['pdb_file'] = os.path.join(os.path.dirname(__file__), '../../examples/input/2cex_a.pdb')
        cls.data_c1 = simulate_images(args)
        ivol = np.square(np.abs(cls.data_c1['volume']))
        cls.ac_c1 = np.fft.fftshift(np.abs(np.fft.ifftn(ivol))).astype(np.float32)

        args['pdb_file'] = os.path.join(os.path.dirname(__file__), '../../examples/input/2cex_b.pdb')
        cls.data_c2 = simulate_images(args)
        ivol = np.square(np.abs(cls.data_c2['volume']))
        cls.ac_c2 = np.fft.fftshift(np.abs(np.fft.ifftn(ivol))).astype(np.float32)

    def test_match_orientations(self):
        """ 
        Test that alignment.match_orientations finds the correct quaternions from
        a mix of correct and random quaternions.
        """
        for use_gpu in [True, False]:
            quats = alignment.match_orientations(0, 
                                                 self.data['pixel_position_reciprocal'], 
                                                 self.data['reciprocal_extent'], 
                                                 self.data['intensities'],
                                                 self.ac,
                                                 100,
                                                 nbatches=2,
                                                 use_gpu=use_gpu,
                                                 true_orientations=self.data['orientations'])
    
            assert np.all(quats - self.data['orientations'])==0
    
    def test_compute_slices(self):
        """
        Check that alignment.compute_slices retrieves correctly oriented slices by
        computing the NUFFT from the ideal autocorrelation. Plot for visualization.
        """
        # quantitative check -- threshold takes into account interpolation error
        for use_gpu,tag in zip([True, False],['gpu','cpu']):
            cslices = alignment.compute_slices(self.data['orientations'],
                                               self.data['pixel_position_reciprocal'], 
                                               self.data['reciprocal_extent'], 
                                               self.ac,
                                               use_gpu=use_gpu)
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
    
            f.savefig(f"test_compute_slices_{tag}.png", dpi=300, bbox_inches='tight')

    def test_match_orientations_mult(self):
        """
        Test that alignment.match_orientations_mult finds the correct quaternions and
        conformations from a mix of correct and random quaternions and two ac volumes.
        """
        # compile and scramble images from each dataset
        slices_ = np.vstack((self.data_c1['intensities'], self.data_c2['intensities']))
        reindex = np.arange(len(slices_))
        np.random.shuffle(reindex)
        slices_ = slices_[reindex]

        # compile true orientations; assemble true conformation assignments
        true_orientations = np.vstack((self.data_c1['orientations'], self.data_c2['orientations']))
        true_confs = np.zeros(len(slices_)).astype(int)
        true_confs[int(len(slices_)/2):] = 1
        true_confs = true_confs[reindex]

        # perform alignment to each autocorrelation volume
        for use_gpu in [True, False]:
            quats, confs = alignment.match_orientations_mult(0, 
                                                             self.data_c1['pixel_position_reciprocal'], 
                                                             self.data_c1['reciprocal_extent'],
                                                             slices_,
                                                             [self.ac_c1, self.ac_c2],
                                                             50,
                                                             nbatches=2,
                                                             order=-2,
                                                             use_gpu=use_gpu,
                                                             true_orientations=true_orientations)

            assert np.all(true_orientations[reindex] - quats) == 0
            assert np.all(true_confs - confs) == 0
