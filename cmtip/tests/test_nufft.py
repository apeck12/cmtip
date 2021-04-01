import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import skopi as sk
import os

from cmtip.simulate import *
import cmtip.nufft as nufft

class TestNUFFT(object):
    """
    Test the non-uniform fast Fourier transform calculations. 
    """
    @classmethod
    def setup_class(cls):
        
        args = dict()
        args['beam_file'] = os.path.join(os.path.dirname(__file__), '../../examples/input/amo86615.beam')
        args['pdb_file'] = os.path.join(os.path.dirname(__file__), '../../examples/input/3iyf.pdb')
        args['det_info'] = (128, 0.08, 0.2)
        args['n_images'] = 1
        
        data = simulate_images(args)
        cls.ivol = np.square(np.abs(data['volume'].astype(np.float32))) * 1e-6
        cls.ac = np.fft.fftshift(np.abs(np.fft.ifftn(cls.ivol))).astype(np.float32)

        # set up reciprocal space positions for uniform grid
        cls.M = cls.ac.shape[0]
        ls = np.linspace(-1, 1, cls.M+1)
        ls = (ls[:-1] + ls[1:])/2
        hkl_list = np.meshgrid(ls, ls, ls, indexing='ij')
        H_, K_, L_ = np.pi*hkl_list[0].flatten(), np.pi*hkl_list[1].flatten(), np.pi*hkl_list[2].flatten()
        cls.H_, cls.K_, cls.L_ = H_.astype(np.float32), K_.astype(np.float32), L_.astype(np.float32)

        cls.atol = 1e-4 # absolute tolerance between calculations

    def test_forward_uniform(self):
        """Compare forward calculations to each other and np.fft library on a uniform grid."""

        calc_cpu = nufft.forward_cpu(self.ac, self.H_, self.K_, self.L_, support=None, use_recip_sym=True).real
        calc_gpu = nufft.forward_gpu(self.ac, self.H_, self.K_, self.L_, support=None, use_recip_sym=True).real
        calc_np = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(self.ac)).real) 

        assert np.mean(np.abs(calc_np.flatten() - calc_cpu)) < self.atol
        assert np.mean(np.abs(calc_np.flatten() - calc_gpu)) < self.atol
        assert np.mean(np.abs(calc_cpu - calc_gpu)) < self.atol

        # plot for visual inspection
        f, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(12,3))

        hs = int(self.M/2)
        ax1.imshow(self.ivol[hs], vmax=self.ivol.mean())
        ax2.imshow(calc_np[hs], vmax=self.ivol.mean())
        ax3.imshow(calc_gpu.reshape(self.ivol.shape)[hs], vmax=self.ivol.mean())
        ax4.imshow(calc_cpu.reshape(self.ivol.shape)[hs], vmax=self.ivol.mean())

        ax1.set_title("Diffraction volume")
        ax2.set_title("np.fft estimate")
        ax3.set_title("cufinufft forward")
        ax4.set_title("finufft forward")

        for ax in [ax1,ax2,ax3,ax4]:
            ax.axis('off')

        f.savefig("test_forward_uniform.png", dpi=300, bbox_inches='tight')
        return

    def test_adjoint_uniform(self):
        """Compare adjoint calculations to each other and np.fft library on a uniform grid."""

        calc_cpu = nufft.adjoint_cpu(self.ivol.flatten(), self.H_, self.K_, self.L_, self.M) 
        calc_gpu = nufft.adjoint_gpu(self.ivol.flatten(), self.H_, self.K_, self.L_, self.M) 

        assert np.mean(np.abs(self.ac - calc_cpu)) < self.atol
        assert np.mean(np.abs(self.ac - calc_gpu)) < self.atol
        assert np.mean(np.abs(calc_cpu - calc_gpu)) < self.atol

        f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(9,3))

        hs = int(self.M/2)
        ax1.imshow(self.ac[hs], vmax=5*self.ac.mean())
        ax2.imshow(calc_cpu[hs], vmax=5*self.ac.mean())
        ax3.imshow(calc_gpu.reshape(self.ac.shape)[hs], vmax=5*self.ac.mean())

        ax1.set_title("np.fft estimate")
        ax2.set_title("cufinufft adjoint")
        ax3.set_title("finufft adjoint cpu")

        for ax in [ax1,ax2,ax3]:
            ax.axis('off')

        f.savefig("test_adjoint_uniform.png", dpi=300, bbox_inches='tight')
        return
