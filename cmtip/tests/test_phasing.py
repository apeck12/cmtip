import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import skopi as sk
import os
import scipy.ndimage

from cmtip.phasing import compute_reference, phase
from scipy.ndimage import gaussian_filter

class TestPhase(object):
    """
    Test the phasing algorithm starting from the ideal autocorrelation.
    """
    
    @classmethod
    def setup_class(cls):
        
        args = dict()
        args['beam_file'] = os.path.join(os.path.dirname(__file__), '../../examples/input/amo86615.beam')
        args['pdb_file'] = os.path.join(os.path.dirname(__file__), '../../examples/input/3iyf.pdb')
        args['det_info'] = (128, 0.08, 0.2)
        args['M'] = 81 
        
        ref_ac, ref_density = compute_reference(args['pdb_file'], args['det_info'], args['beam_file'], args['M'])
        cls.ref_density = gaussian_filter(ref_density, sigma=0.8) # reference will have higher res than phased
        ac_phased, support_, rho_ = phase(0, ref_ac, nER=60, nHIO=30)
        cls.est_density = np.fft.ifftshift(rho_)
        
    def test_phased_vs_ref(self):
        """ Check that the phased result matches reference by comparing projections. """
        
        for i in range(3):
            for j in range(3):
                if i>= j:
                    ref_proj = np.sum(self.ref_density, axis=i)
                    est_proj = np.sum(self.est_density, axis=j)
                    cc_proj = np.corrcoef(ref_proj.flatten(), est_proj.flatten())[0,1]
                    if i == j: # corresponding slices match
                        assert cc_proj > 0.9
                    elif i!=2 and j!=2: # additional matches due to this protein's symmetry
                        assert cc_proj > 0.9
                    else:
                        assert cc_proj < 0.9
                        
        f, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2, 3, figsize=(9,6))

        for i,ax in enumerate([ax1,ax2,ax3]):
            ax.imshow(np.sum(self.ref_density, axis=i))
        for i,ax in enumerate([ax4,ax5,ax6]):
            ax.imshow(np.sum(self.est_density, axis=i))
            
        ax1.set_ylabel("Reference", fontsize=12)
        ax4.set_ylabel("Phased estimate", fontsize=12)

        f.savefig("test_phased_vs_ref.png", dpi=300, bbox_inches='tight')

