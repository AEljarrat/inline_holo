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

class CTFSim():
    """
    Simulation of focal series using non-linear Optics. This module simulates
    propagation of a given wave to diferent focus positions along the optical
    axis in an imaging system. The algorithm uses a flux-preserving formulation
    including a contrast transfer function (CTF) and a spatial coherence
    envelope. Thus, it is mandatory to input a wave and defocus values. The
    module gets the CTF and the the spatial coherence envelope (if a convergence
    semi-angle is given).

    Authors
    -------
    Alberto Eljarrat and Christoph T. Koch.
    Institute für Physik. Humboldt Universität zu Berlin.

    No commercial use or modification of this code
    is allowed without permission of the authors.
    """

    def __init__(self, wave, defoci, alpha=None, using_gpu=False, *args, **kw):
        """
        Parameters
        ----------
        wave : Hyperspy Image
         This is a complex wave-function (set amplitude and phase).

        defoci : numpy array
         Set the defocus values for the simulation.

        alpha : number
         Convergence-semi angle, use this to set a spatial coherence envelope.

        using_gpu : bool
         Sets the GPU/CPU functionality

        *args and **kwargs are passed to the "set_contrast_transfer" method of
        "wave" input parameter (see doc there for more info).

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

        self.Nz = len(defoci)
        self.set_wave(wave)
        self.set_contrast_transfer(defoci, *args, **kw)
        self.set_convergence_envelope(alpha)

    def __call__(self, unpad=False, complex_wave=False):
        """
        Will run the simulation, in the GPU or CPU, depending on how the
        parameters were set-up, and return the resulting focal series with or
        without padding. This result is also stored in the "focal_series"
        attribute.

        Parameters
        ----------
        unpad : bool
         Pad / Unpad switch for the returned wave. Set to False by defect.

        complex_wave : bool
         Return the propagated waves instead. Set to false by defect.

        Returns
        -------
        focal_series : Hyperspy Image
         Dataset containing the intensity of the propagated wave at positions
         set by the defocus values.
        """
        if self.using_gpu:
            self.run_gpu(complex_wave)
            return self._return_focal_series(unpad, complex_wave)
        else:
            self.run_cpu(complex_wave)
            return self._return_focal_series(unpad, complex_wave)

    def _return_focal_series(self, unpad=False, complex_wave=False):
        """
        This method returns the current focal series, with or without padding
        depending on a boolean parameter unpad.
        """
        if self.using_gpu:
            self.focal_series.data = self.fsdata.get()
        else:
            self.focal_series.data = self.fsdata

        if not complex_wave:
            self.focal_series.data = self.focal_series.data.real

        if unpad:
            fs = self.focal_series.unset_padding()
            return fs
        else:
            fs = self.focal_series
            return fs

    def set_complex_wave(self, amplitude, phase):
        """
        Use it to build a complex wave from amplitude and phase. Padding must be
        set in advance for both signals and should coincide. Calls "set_wave".

        Parameters
        ----------
        amplitude : Hyperspy Image
         The amplitude of complex wave.

        phase : Hyperspy Image
         The phase of complex wave.

        """
        wave = amplitude.deepcopy()
        wave.metadata = amplitude.metadata.copy()
        wave.data = np.sqrt(amplitude.data) * np.exp(1j*phase.data)
        self.set_wave(wave)

    def set_wave(self, wave):
        """
        Use it to set directly the wave.

        Parameters
        ----------
        wave : Hyperspy Image
         A complex wave with all the appropriate parameters.
        """
        # TODO: trigget set_contrast_transfer if wave changes something important
        self.wave = wave

        if self.using_gpu:
            self.wdata = to_gpu_c(wave.data)
        else:
            self.wdata = wave.data.astype('complex64')

    def set_contrast_transfer(self, defoci, *args, **kw):
        """
        Use it to set the contrast transfer function. The get_contrast_transfer
        method is called, see ModImage class for a complete list of arguments.

        Parameters
        ----------
        defoci : numpy array, list or tuple
         Sets the defoci from the
        """
        wave = self.wave
        # Optionally, set defoci from list:
        if not isinstance(defoci, np.ndarray):
            defoci = np.array(defoci)
        Nz = len(defoci)

        # call method to get CTF
        ctf = wave.get_contrast_transfer(defoci, *args, **kw)
        simdata = np.zeros_like(ctf.data)

        # Prepare input/output objects
        Nx, Ny = wave.axes_manager.signal_shape
        self.Nz = Nz
        self.shape= (Nz, Ny, Nx)
        self.k2 = wave.get_fourier_space()
        self.defoci = defoci
        self.contrast_transfer = ctf

        # Send tp GPU if needed
        if self.using_gpu:
            self.ctfd = to_gpu_c(ctf.data)
            self.fsdata = to_gpu_c(simdata)
            # - The plans for FFT
            self.pft2dcc = cu_fft.Plan((Ny, Nx), np.complex64, np.complex64)
            self.pft3dcc = cu_fft.Plan((Ny, Nx), np.complex64, np.complex64, Nz)
        else:
            self.ctfd = ctf.data.astype('complex64')
            self.fsdata = simdata

        # Correctly store scalings in simulation result object
        fs = ctf.deepcopy()
        fs.data = simdata

        # Get the spatial coordinates from the input wave
        fs.axes_manager.signal_axes[0].scale  = wave.axes_manager.signal_axes[0].scale
        fs.axes_manager.signal_axes[0].offset = wave.axes_manager.signal_axes[0].offset
        fs.axes_manager.signal_axes[1].scale  = wave.axes_manager.signal_axes[1].scale
        fs.axes_manager.signal_axes[1].offset = wave.axes_manager.signal_axes[1].offset
        fs.axes_manager[0].offset = defoci[0]
        fs.axes_manager[0].scale  = defoci[1] - defoci[0]

        fs.get_dimensions_from_data()

        # now set defoci, for non-linear vectors!
        fs.axes_manager[0].axis = defoci

        # Store the used parameters
        fs.metadata = ctf.metadata.copy()
        self.focal_series = fs

    def set_convergence_envelope(self, alpha):
        """
        Sets the spatial-coherence envelope function. The convergence semiangle,
        alpha, is set by input preferentially or, if possible, by metadata.

        Parameters
        ----------
        alpha : float
         To set the convergence semi-angle, in radians. If None, will try to
         read from metadata.
        """
        k2 = self.k2
        defoci = self.defoci
        metadata = self.wave.metadata
        if (alpha is None):
            if metadata.has_item('ModImage.convergence_semiangle'):
                alpha = metadata.ModImage.convergence_semiangle
            else:
                alpha = 0.

        envelope = np.exp(-k2*((0.5*defoci*alpha)**2)[:,None,None])

        if self.using_gpu:
            self.Esdata = to_gpu_f(envelope)
        else:
            self.Esdata = envelope.astype('float32')

        self.focal_series.metadata.set_item('ModImage.convergence_semiangle', alpha)

    def run_gpu(self, complex_wave):
        """
        Does CTFSim on GPU. The result is stored on an attribute
        "self.fsdata" containing a GPU array.
        """
        # create outputs
        Nz, Ny, Nx = self.shape
        ftwave = gpuarray.empty((Ny, Nx), np.complex64)
        # extract plans
        ft2dcc = self.pft2dcc
        ft3dcc = self.pft3dcc

        # Propagate the initial wave to simulate defocused waves
        # Psi(x,y,z) = convolve[Psi(x,y,0), CTF(x,y,z)]
        cu_fft.fft(self.wdata, ftwave, ft2dcc)
        for kk in range(Nz):
            self.fsdata[kk,:,:] = self.ctfd[kk,:,:] * ftwave
        cu_fft.ifft(self.fsdata, self.fsdata, ft3dcc, True)
        if not complex_wave:
            # Use the intensities, Isim = |Psi|**2
            self.fsdata = self.fsdata * self.fsdata.conj()
            if self.focal_series.metadata.has_item('ModImage.convergence_semiangle'):
                # Convolve with spatial-coherence envelope
                # Isim = convolve[Isim, Es]
                cu_fft.fft(self.fsdata, self.fsdata, ft3dcc)
                cu_fft.ifft(self.fsdata*self.Esdata, self.fsdata, ft3dcc, True)

    def run_cpu(self, complex_wave):
        """
        Does CTFSim on CPU. The result is stored on an attribute
        "self.fsdata" containing a numpy array.
        """
        ctf = self.ctfd
        wdata = self.wdata
        waves = ctf * np.fft.fft2(wdata)[None, :, :]
        if complex_wave:
            # use the wave in real-space representation
            fsdata = np.fft.ifft2(np.squeeze(waves))
        else:
            # use the intensities
            fsdata = np.abs(np.fft.ifft2(np.squeeze(waves)))**2

        # Use convergence envelope?
        uno = self.focal_series.metadata.has_item(
                        'ModImage.convergence_semiangle')
        dos = complex_wave is False
        if uno and dos:
            fsdata = np.fft.ifft2(np.fft.fft2(fsdata)*self.Esdata)
            fsdata = fsdata.real.astype('float32')
        self.fsdata = fsdata
