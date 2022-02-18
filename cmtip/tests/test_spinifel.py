import numpy as np
import scipy.signal
import cmtip.nufft as nufft

class TestNUFFT(object):
    """
    Test the non-uniform fast Fourier transform calculations.
    """
    @classmethod
    def setup_class(cls):
        # set up mock array to transform
        cls.M = 81
        mask = scipy.signal.tukey(cls.M)
        mask = mask[:,np.newaxis,np.newaxis] * mask[np.newaxis,:,np.newaxis] * mask[np.newaxis,np.newaxis,:]
        cls.ivol = (mask * np.random.uniform(low=0,high=100,size=(cls.M,cls.M,cls.M))).astype(np.float32)
        cls.ac = np.fft.fftshift(np.abs(np.fft.ifftn(cls.ivol))).astype(np.float32)

        # set up reciprocal space positions for uniform grid
        ls = np.linspace(-1, 1, cls.M+1)
        ls = (ls[:-1] + ls[1:])/2
        hkl_list = np.meshgrid(ls, ls, ls, indexing='ij')
        H_, K_, L_ = np.pi*hkl_list[0].flatten(), np.pi*hkl_list[1].flatten(), np.pi*hkl_list[2].flatten()
        cls.H_, cls.K_, cls.L_ = H_.astype(np.float32), K_.astype(np.float32), L_.astype(np.float32)
        
        cls.atol = 1e-2 # may require some adjustment

    def test_forward(self):
        """
        Check that the forward NUFFTs yield the same result as each other and as numpy.fft.
        """
        calc_cpu = nufft.forward_cpu(self.ac, self.H_, self.K_, self.L_, support=None, use_recip_sym=True).real
        calc_gpu = nufft.forward_cpu(self.ac, self.H_, self.K_, self.L_, support=None, use_recip_sym=True).real
        calc_np = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(self.ac)).real) 
        
        assert np.allclose(calc_gpu, calc_cpu, atol=self.atol)
        assert np.allclose(calc_cpu, calc_np.flatten(), atol=self.atol)
        
    def test_adjoint(self):
        """
        Check that the adjoint NUFFT calls produce the same result as each other.
        """
        calc_cpu = nufft.adjoint_cpu(self.ivol.flatten(), self.H_, self.K_, self.L_, self.M) 
        calc_gpu = nufft.adjoint_gpu(self.ivol.flatten(), self.H_, self.K_, self.L_, self.M) 

        assert np.allclose(calc_cpu, calc_gpu, atol=self.atol)
        
