import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import cholesky as chol

class GPTIE():
    """
    Gaussian-process Transport of Intensity Equation solver a focal-series based
    phase reconstruction algorithm.

    Example
    -------
    >>> from GPTIE import GPTIE
    >>> gptie = GPTIE(fs)   # Run GP-fit: where fs is a focal-series
    >>> phase_gp = gptie()  # Invert TIE: uses 4 2D-FFT to solve TIE.

    References
    ----------
    Z. Jingshan, R. A. Claus, L. Tian, and L. Waller. “Transport of intensity
    phase imaging by intensity spectrum fitting of exponentially spaced defocus
    planes”, Opt. Express , 22:10661–10674 (2014).

    Authors
    -------
    Alberto Eljarrat and Christoph T. Koch.
    Institute für Physik. Humboldt Universität zu Berlin.

    Adapted from the original Matlab software by Jingshan Zhong and coworkers
    (see: www.dauwels.com  or  www.laurawaller.com).
    """
    def __init__(self, fs, *args, **kw):
        """
        Parameters
        ----------
        fs : Hyperspy Image
         Dataset containing a focal series.

        *args and **kwargs are passed to the "fit" method.
        """
        self.fit(fs, *args, **kw)

    def __call__(self, unpad=False, eps=1e-6):
        """
        Will run the TIE inversion, in the GPU or CPU, depending on how the
        parameters were set-up, and return the resulting phase with or without
        padding. This result is also stored in the "phase" attribute.

        Parameters
        ----------
        unpad : bool
         Pad / Unpad switch for the returned wave. Set to False by defect.

        Returns
        -------
        phase: Hyperspy Image
         Gradient-process TIE phase.
        """
        self.run(eps)
        return self._return_phase(unpad)

    def _return_phase(self, unpad=False):
        """
        This method returns the current phase, with or without padding depending
        on a boolean parameter unpad.
        """
        self.phase.data = self.phdata.real

        if unpad:
             phase = self.phase.unset_padding()
             return phase
        else:
            phase = self.phase
            return phase

    def fit(self, fs, Nsl=50, defoci=None):
        """
        Gaussian Process Fit routine. First it sets Nsl bins in reciprocal-space
        The GP regression is obtained for each frequency within one bin, shares
        the same frequency thresholds and hyper-parameters...

        Parameters
        ----------
        fs : Hyperspy Image
         Mandatory! A focal series with all the appropriate parameters.

        defoci : array
         Optional, use it to explicitly set defoci values. Must have a dimension
         equal to the navigation axis of fs.
        """
        Ividmeas = fs.data.copy()
        wlen     = fs.metadata.ModImage.wave_length
        Nz, Ny, Nx = fs.data.shape
        k2, kx, ky = fs.get_fourier_space('both')

        if defoci is None:
            defoci = fs.axes_manager.navigation_axes[0].axis

        k2norm = np.fft.fftshift(k2 * wlen/(8.*np.pi**2))
        k2cof = np.linspace(0,1,Nsl)**2 * k2norm.max()

        FrqtoSc = np.linspace(1.2,1.1,Nsl)      # trade off of noise and accuracy
        p = Nz / (defoci.max()- defoci.min())   # average data on unit space

        # Calculate GP hyperparameters
        SigmafStack = np.zeros(Nsl)
        SigmanStack = np.zeros(Nsl)
        SigmalStack = np.zeros(Nsl)
        for k in range(Nsl):
            # Init Sigman and Sigmaf
            Sigman = 1.0e-9
            Sigmaf = 1.0

            # Calculate Sigmal
            f1 = k2cof[k]
            sc = f1 * FrqtoSc[k]
            a  = 2 * (np.pi*sc)**2
            b = np.log((p*(2*np.pi)**0.5) / Sigman)

            fu2 = lambda x: (a*np.exp(x) - 0.5*x - b)
            # Needs a high initial guess to
            # reach the correct result
            x = fsolve(fu2,100.)
            Sigmal = np.exp(x)

            SigmafStack[k] = Sigmaf
            SigmanStack[k] = Sigman
            SigmalStack[k] = Sigmal

        # GP-regression
        # Recover partial phase images for each Nsl hyperparamter
        dIdzStack   = np.zeros((Nsl, Ny, Nx))
        #CoeffStack  = np.zeros((Nsl, Nz))
        #Coeff2Stack = np.zeros((Nsl, Nz))

        for k in range(Nsl):
            Sigmal = SigmalStack[k]
            Sigman = SigmanStack[k]
            Sigmaf = SigmafStack[k]

            # Regression params
            VectorOne = np.ones(Nz)
            KZ        = np.dot(VectorOne[:,None],defoci[:,None].T) - \
                        np.dot(VectorOne[None,:].T,defoci[None,:]).T
            K = Sigmaf * np.exp(-0.5/Sigmal*KZ**2)
            L = np.linalg.cholesky(K + Sigman*np.eye(Nz)).T
            KZ2= - defoci

            # First derivative: D / L / L' is more stable to the matrix inversion?
            D = - KZ2 * Sigmaf * np.exp(-0.5/Sigmal*KZ2**2) / Sigmal
            Coeff = np.linalg.lstsq(L, np.linalg.lstsq(L.T, D)[0])[0]
            # Robert P. says check: Coeff = np.linalg.solve(K + Sigman*np.eye(Nz))

            # Smoothin Regression: not used later?
            #D2 = Sigmaf * np.exp(-0.5/Sigmal*KZ2**2)
            #Coeff2 = np.linalg.lstsq(L, np.linalg.lstsq(L.T, D2)[0])[0]

            dIdz = np.sum(Ividmeas * Coeff[:,None,None],0)

            dIdzStack[k, :, :] = dIdz
            #CoeffStack[k, :]   = Coeff   # derivative
            #Coeff2Stack[k, :]  = Coeff2  # smoothing (not used?)

        # Combine phase
        # Cutoff is the cutoff area in CosH
        F = lambda x : np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x)))
        Ft = lambda x : np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x)))

        dIdzC_fft = np.zeros((Ny,Nx))
        Maskf = np.zeros((Ny,Nx))
        f0=0
        f1=0
        for k in range(Nsl):
            dIdz = dIdzStack[k,:,:]
            dIdz_fft = F(dIdz)
            f1 = k2cof[k]
            Maskf[(k2norm <= f1) * (k2norm > f0)] = 1
            f0=f1
            # Only update in the update area
            dIdzC_fft = dIdzC_fft + dIdz_fft * Maskf
            # Zero-out the mask for next usage
            Maskf.fill(0.)

        # Save result times wave-number, for TIE: dzI * ko
        self.dzI = np.real(Ft(dIdzC_fft)) * 2.*np.pi / wlen

        Nzh = np.ceil(Nz/2).astype('int') - 1
        iIo = 1. / fs.data[Nzh,:,:]

        # Save some parameters for laters
        self.phase = fs._get_signal_signal()
        self.phase.metadata = fs.metadata.copy()
        self.k2  = k2
        self.iIo = iIo
        self.k = (ky, kx)
        self.phdata = self.phase.data
        self.k2_cut_off_freqs = k2cof * (8.*np.pi**2) / wlen

    def run(self, eps):
        """
        Solves the MFTIE. The result is stored on an attribute
        "self.phase" containing a numpy array.
        """
        ky, kx = self.k
        nabla_minus_2 = 1. / (self.k2+eps)
        iIo = self.iIo

        # Apply Laplacian inversions
        nm2_dz =  nabla_minus_2 * np.fft.fft2(self.dzI)

        # Do divergence of the gradient in reciprocal-space
        grad_x = np.fft.fft2( np.fft.ifft2(kx[None,:] * nm2_dz) * iIo )
        grad_y = np.fft.fft2( np.fft.ifft2(ky[:,None] * nm2_dz) * iIo )
        lm2_dz = nabla_minus_2 * (kx[None,:]*grad_x + ky[:,None]*grad_y)

        self.phdata = np.real(np.fft.ifft2(lm2_dz)).astype('float32')
        self.phdata = self.phdata
