import numpy as np

## check gpu functionality! ##

global gpu_func
gpu_func = True

try:
    __import__("pycuda")
except ImportError:
    gpu_func = False
else:
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import skcuda.fft as cu_fft
    from skcuda import misc
    misc.init()

    def to_gpu_c(somedata):
        # all my complex data
        return gpuarray.to_gpu(somedata.astype('complex64'))
    def to_gpu_f(somedata):
        # all my float data
        return gpuarray.to_gpu(somedata.astype('float32'))

## check gpu functionality! ##

class MFTIE():
    """
    Multi-focus Transport of Intensity Equation solver, a focal-series based
    phase reconstruction algorithm. Will retrieve the phase from a focal series
    deterministically through the solution of the multi-focus transport of
    intensity equation (MF-TIE). A focal series must be provided such that:
    A reference plane (in-focus image) is found at middle of the navigation axis
    and the defoci values are symmetrically distributed around this reference;

        [-df1, -df2, ... df=0 ... df2, df1]

    Example
    -------
    >>> from MFTIE import MFTIE
    >>> mftie = MFTIE(fs)   # Init MF-fit: sets inputs for TIE
    >>> phase_mf = mftie()  # Invert TIE: uses 4 2D-FFT to solve TIE.

    Parameters can be updated by using specialized "set" methods:
    >>> mftie.set_inverse_laplacian(2.,0.2,0.05)

    The code can run in a GPU, if available:
    >>> mftie = MFTIE(fs, True)   # Init will try to use GPU...
    >>> phase_mf = mftie()        # If available, this step is the sped-up!

    Authors
    -------
    Alberto Eljarrat and Christoph T. Koch.
    Institute für Physik. Humboldt Universität zu Berlin.

    No commercial use or modification of this code
    is allowed without permission of the authors.

    """

    def __init__(self, fs, using_gpu=False, *args, **kw):
        """
        Parameters
        ----------
        fs : Hyperspy Image
         This is a focal series.

        using_gpu : bool
         Sets the GPU/CPU functionality.

        *args and **kwargs are passed to the "set_inverse_laplacian" method.
        """
        if (using_gpu and not gpu_func):
            # Not so fast ...
            self.using_gpu = - using_gpu
            raise Exception('GPU functionality not available !')
        elif using_gpu:
            # Set-up GPU Machinery
            self.using_gpu = using_gpu
        else:
            # User selects CPU
            self.using_gpu = using_gpu

        self.set_derivatives(fs)
        self.set_inverse_laplacian(*args, **kw)

    def __call__(self, unpad=False):
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
         Multi-focus TIE phase.
        """
        if self.using_gpu:
            self.run_gpu()
            return self._return_phase(unpad)
        else:
            self.run_cpu()
            return self._return_phase(unpad)

    def _return_phase(self, unpad=False):
        """
        This method returns the current phase, with or without padding depending
        on a boolean parameter unpad.
        """
        if self.using_gpu:
            self.phase.data = self.phdata.real.get()
        else:
            self.phase.data = self.phdata.real

        if unpad:
             phase = self.phase.unset_padding()
             return phase
        else:
            phase = self.phase
            return phase

    def set_derivatives(self, fs, defoci=None, nref=None):
        """
        Parameters
        ----------
        fs : Hyperspy Image
         Mandatory! A focal series with all the appropriate parameters.

        defoci : array
         Optional, use it to explicitly set defoci values. Must have a dimension
         equal to the navigation axis of fs.

        nref : number
         Optional, use it to explicitly set the refractive index. This value can
         also be set implicitly from metadata in fs.
         """
        # Real space parameters
        wnum = 2.*np.pi / fs.metadata.ModImage.wave_length
        Nz, Ny, Nx = fs.data.shape
        if defoci is None:
            defoci = fs.axes_manager.navigation_axes[0].axis
        # Refractive index
        if nref is None:
            if fs.metadata.has_item('ModImage.refractive_index'):
                nref = fs.metadata.ModImage.refractive_index
            else:
                nref = 1.

        # K-space parameters
        k2, kx, ky = fs.get_fourier_space('both')

        # In the middle of the focal series is the reference plane
        Nz_half = np.ceil(Nz/2).astype('int') # actually, in Nz_half-1
        adef = np.abs(defoci[Nz_half:])

        # Defocus differences: ko.dI/dz
        diff_data = (fs.data[Nz_half:,:,:] - fs.data[Nz_half-2::-1,:,:]) / \
                      (2.*adef[:,None,None])
        diff_data = wnum * nref * diff_data

        Nz = Nz_half-1
        iIo = 1. / fs.data[Nz,:,:]

        # Set some parameters
        self.phase = fs._get_signal_signal()
        self.phase.metadata = fs.metadata.copy()
        self.k2  = k2
        self.adef = adef
        self.wnum = wnum
        self.nref = nref

        # Allocate things ...
        if self.using_gpu:
            # ... in GPU!
            # - The inputs
            [kx, ky] = np.meshgrid(kx, ky)
            self.k = [to_gpu_f(np.repeat(ki[None,:,:], Nz, 0)) for ki in (ky, kx)]
            self.iIo = to_gpu_f(np.repeat(iIo[None,:,:],Nz,0))
            self.dzI = to_gpu_c(diff_data)
            self.shape= (Nz, Ny, Nx)

            # - The outputs
            self.phdata = gpuarray.empty((Ny, Nx), np.complex64)

            # - The plans for FFT
            self.pft3dcc = cu_fft.Plan((Ny, Nx), np.complex64, np.complex64, Nz)
            self.pft2dcc = cu_fft.Plan((Ny, Nx), np.complex64, np.complex64)
        else:
            # ... in CPU!
            self.k = (ky, kx)
            self.iIo = iIo[None,:,:]
            self.dzI = diff_data.astype(np.complex64)
            self.shape = (Nz, Ny, Nx)
            self.phdata = self.phase.data

    def set_dzI(self, diff_data):
        if self.using_gpu:
            self.dzI = to_gpu_c(diff_data)
        else:
            self.dzI = diff_data.astype('complex64')

    def set_inverse_laplacian(self, lothres=1., hithres=0.25, alpha=None,
                              SNR=32., do_norm=True, filter_type='butter',
                              Nbutter=10, himethod='chi'):
        """
        Calculates the inverse_laplacian filter in CPU and sends it to GPU.
        The filter contains band-pass-filters* and a convergence envelope that
        are different for each defocus. The cut-off frequencies are set to
        reasonable values by default (see "print_k2_thres" method). However
        these values can be altered using the following input parameters.

        Parameters
        ----------
        lothres : Number
            Sets the high-pass cut-off frequencies that control the impact of
            noise to MFTIE. A value of 1 set the last cut-off* to coincide with
            the lowest spatial-frequency. Note that if the filter_type parameter
            is set to "tikho", then lothres just sets Tikhonov filters.

        hithres : Number
            Sets the low-pass cut-off frequencies that control the impact of
            non-linear effects to MFTIE. Two options are available, using
            keyword "himethod2 (see below). In a nutshell, one method compares
            the aberration-function (Chi), and other compares the phase-transfer
            -funtion (PTF), the former being the less conservative method.

        alpha : number
            Convergence angle for envelope function, in radians. It is None by
            default. In this case, the values is taken from metadata and used if
            not also equal to None. If both are None, or do_norm=False, the
            envelope function is not used.

        do_norm : bool
            Set/unset Laplacian normalization, True by default. Note that MFTIE
            needs Laplacian normalization to obtain good results, but
            three-plane TIE can be performed without it.

        filter_type : string
            One of the following: ['gauss', 'butter', 'tikho']. Sets the type of
            RSD* to use. The Gaussian filter has fixed slope, but the
            Butterwoth can be adjusted using the Nbutter parameter. The Tikhonov
            variant is completely different as it uses all frequency range but
            adds a regularization constant to q**2.

        Nbutter : int
            Sets the Butterwoth filter slope, set it to a number from 2-10. By
            defect it is set to 10 (sharp slope).

        Notes
        -----
        * AKA Reciprocal-space donuts (RSD). Note that the RSD donuts are set
        to have a fixed noise contribution to RMSE.
        """
        nref = self.nref
        wnum = self.wnum
        adef = self.adef
        k2   = self.k2
        # Compute Detector size, noise-to-signal ratio, three-plane noise ...
        Lkdet  = np.mean([2.*np.pi/(np.sqrt(dk2i)) for dk2i in (k2[1,0], k2[0,1])])
        NSRlin = 10.**(-SNR/10.)
        Anoise = np.pi*(nref*wnum*NSRlin*Lkdet/(adef*2.*np.pi))**2
        k2min  = np.min([self.k2[0,1], self.k2[1,0]])
        rmse2n = Anoise[-1] / k2min
        # Set High and Low threshold values in k-space
        if himethod is 'ptf':    # aka Nugent method: constant sensitivity
            k2Hi = 2. * nref * wnum * np.sqrt(6. * hithres) / adef
        elif himethod is 'chi':  # compare Chi2 functions
            k2Hi = wnum*nref*np.sqrt(8*wnum*nref*hithres/adef)
        k2Lo = k2Hi / (lothres*rmse2n*k2Hi/Anoise + 1.)
        # Save threshold info
        self.k2_hi_thres = k2Hi
        self.k2_lo_thres = k2Lo
        self.Anoise      = Anoise
        self.rmse2n      = rmse2n * lothres
        self.parameters  = [lothres,hithres,alpha,SNR]

        # Calculate laplacian
        if filter_type is 'gauss':
            nabla_minus_2 = (1. / k2) * np.exp(- k2 / k2Hi[:,None,None]) * \
                                  (1. - np.exp(- k2 / k2Lo[:,None,None]))
        elif filter_type is 'butter':
            nabla_minus_2 = (1. / k2) * (1./(1.+(k2/k2Hi[:,None,None])**Nbutter)) * \
                                     (1.-1./(1.+(k2/k2Lo[:,None,None])**Nbutter))
        elif filter_type is 'tikho':
            nabla_minus_2 = k2 / (k2 + k2Lo[:,None,None])**2 * \
                                np.exp(- k2 / k2Hi[:,None,None])
        nabla_minus_2[:, 0, 0] = 1.

        # Normalization function
        if do_norm:
            # should we use also convergence?
            if (alpha is None):
                if self.phase.metadata.has_item('ModImage.convergence_semiangle'):
                    alpha = self.phase.metadata.ModImage.convergence_semiangle

            if alpha is not None:
                envelope = np.exp(- k2 * ((0.5*adef*alpha)**2)[:,None,None])
                fnorm = (np.sum(nabla_minus_2 * envelope, 0) * k2)[None,:,:]
                fnorm[fnorm < 1.] = 1.
                nabla_minus_2 /= np.sqrt(fnorm)
            else:
                fnorm = np.sum(nabla_minus_2, 0) * k2
                fnorm[fnorm < 1.] = 1.
                nabla_minus_2 /= np.sqrt(fnorm[None,:,:])

        # Allocate nabla...
        if self.using_gpu:
            # ... on the GPU!
            self.inverse_laplacian = to_gpu_f(nabla_minus_2)
        else:
            # ... on the CPU!
            self.inverse_laplacian = nabla_minus_2

    def print_k2_thres(self):
        """
        Nice print-out of the k2 thresholds.
        """
        dz = self.adef
        kM = np.sqrt(self.k2_hi_thres)
        km = np.sqrt(self.k2_lo_thres)
        Adz = self.Anoise
        kmin = np.sqrt(np.min([self.k2[0,1], self.k2[1,0]]))
        rmse2n = self.rmse2n
        SNR = self.parameters[3]
        print(u' \u03b4z \t\t  qH \t\t  qL \t\t  A\u03b4z')
        print('--------------------------------------------------------')
        for dzi, kMi, kmi, Adzi in zip(dz, kM, km, Adz):
            print('%.3f   \t %.5f   \t %.5f   \t %.5f' % (dzi,kMi,kmi,Adzi))
        print('========================================================')
        print(u'Reciprocal Space Donuts (RSD\u2122):')
        print('qMinim =', np.round(kmin,5))
        print('RMSE2n =',np.round(rmse2n,5), 'for SNR = ', SNR)

    def run_gpu(self):
        """
        Solves the MFTIE on GPU. The result is stored on an attribute
        "self.phase" containing a GPU array.
        """
        # Extract pre-allocated GPU arrays
        # extract inputs
        iIo  = self.iIo
        nm2 = self.inverse_laplacian
        dzI = self.dzI
        ky, kx = self.k
        Nz, Ny, Nx = self.shape

        # create outputs
        ft_dzI = gpuarray.empty((Nz, Ny, Nx), np.complex64)
        gradx  = gpuarray.empty((Nz, Ny, Nx), np.complex64)
        grady  = gpuarray.empty((Nz, Ny, Nx), np.complex64)

        # extract plans
        ft3dcc = self.pft3dcc
        ft2dcc = self.pft2dcc

        # Do the math!
        # FT(dzI)
        cu_fft.fft(dzI, ft_dzI, ft3dcc)
        # IFT(k*nm2*...)
        cu_fft.ifft((ft_dzI*nm2)*kx, gradx, ft3dcc, True)
        cu_fft.ifft((ft_dzI*nm2)*ky, grady, ft3dcc, True)
        # FT(... / Io)
        cu_fft.fft(gradx*iIo, gradx, ft3dcc)
        cu_fft.fft(grady*iIo, grady, ft3dcc)
        # Sum_z(nm2*(k*...))
        Slapl = misc.sum((nm2*(kx*gradx+ky*grady)).reshape(Nz, Ny*Nx),0).reshape(Ny, Nx)
        # IFT(...)
        cu_fft.ifft(Slapl, self.phdata, ft2dcc, True)

    def run_cpu(self):
        """
        Solves the MFTIE on CPU. The result is stored on an attribute
        "self.phase" containing a numpy array.
        """
        ky, kx = self.k
        nabla_minus_2 = self.inverse_laplacian
        iIo = self.iIo

        # Apply Laplacian inversions
        nm2_dz =  nabla_minus_2 * np.fft.fft2(self.dzI)

        # Do divergence of the gradient in reciprocal-space
        grad_x = np.fft.fft2( np.fft.ifft2(kx[None,:] * nm2_dz) * iIo )
        grad_y = np.fft.fft2( np.fft.ifft2(ky[:,None] * nm2_dz) * iIo )
        lm2_dz = np.sum(nabla_minus_2 * (kx[None,:]*grad_x + ky[:,None]*grad_y), 0)

        self.phdata = np.real(np.fft.ifft2(lm2_dz)).astype('float32')
