import numpy as np
from tqdm import trange
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
    from pycuda.elementwise import ElementwiseKernel
    from skcuda import misc
    misc.init()

    def to_gpu_c(somedata):
        # all my complex data
        return gpuarray.to_gpu(somedata.astype('complex64'))
    def to_gpu_f(somedata):
        # all my float data
        return gpuarray.to_gpu(somedata.astype('float32'))
    def to_gpu_b(somedata):
        # all my bool data
        return gpuarray.to_gpu(somedata)

## check gpu functionality! ##

class GS():
    """
    Gerchberg-Saxton (GS) algorithm for non-linear wave refinement. GS algorithm
    uses a loop of propagation and back-propagation operations of this wave
    convolved with a contrast transfer function (CTF) and a spatial coherence
    envelope that model the imaging system. The refinement works by combining
    experimental intensities with the propagated waves. Thus, it is mandatory
    to provide a (padded) focal series; An initial wave and CTF should also be
    provided, but can be initialized from input (more below).

    Example
    -------
    >>> from GS import GS
    >>> gsax = GS(fs)   # Init: sets inputs for GS   # Init: sets inputs for GS
    >>> wave = gsax()   # Iterate

    Parameters can be also updated by using specialized "set" methods:
    >>> gsax.set_initial_wave(wave_other)

    The code can run in a GPU, if available:
    >>> gsax = GS(using_gpu=True)   # Init will try to use GPU...
    >>> wave = mftie()              # if available, this step is the sped-up!


    Authors
    -------
    Alberto Eljarrat and Christoph T. Koch.
    Institute für Physik. Humboldt Universität zu Berlin.

    No commercial use or modification of this code
    is allowed without permission of the authors.

    """
    def __init__(self, fs, wave=None, ctf=None, alpha=None, using_gpu=False,
                 *args, **kw):
        """
        Parameters
        ----------
        fs : Hyperspy Image
         Dataset containing an experimental focal series and including some
         pad region.

        wave : Hyperspy Image
         Image containing a wave with same signal dimensions as the input focal
         series. The wave is expected in its real-space representation, a 2D
         complex function. If not provided the algorithm is initialized with the
         in-focus image and zero phase.

        ctf : Hyperspy Image
         Dataset containing the contrast transfer function with the same shape as
         the input focal series. The CTF is expected in its reciprocal-space
         representation, a 2D complex function. If not provided, the
         "fs.get_contrast_transfer(*args, **kw)" result is used.

        alpha : number
         Convergence-semi angle, use this to set a spatial coherence envelope.

        using_gpu : bool
         Sets the GPU/CPU functionality

        *args, **kw passed to the "set_contrast_transfer" method.

        """
        if (using_gpu and not gpu_func):
            # Not so fast ...
            self.using_gpu = - using_gpu
            raise Exception('GPU functionality not available !')
        elif using_gpu:
            # Set-up GPU Machinery
            self.using_gpu = using_gpu
            # EWK to avoid phase wrapping
            self.cuphase = ElementwiseKernel(
            "pycuda::complex<float> *a,pycuda::complex<float> *b,float *c",
            "c[i] = c[i] + arg(a[i]/b[i]);")
        else:
            # User selects CPU
            self.using_gpu = using_gpu

        if wave is None:
            # Get amplitude from middle image
            amp = fs.data[self.Nz_half,:,:]
            wave = fs._get_signal_signal()
            wave.metadata = fs.metadata.deepcopy()
            wave.data = np.sqrt(amp) * np.exp(1j*np.zeros_like(amp))

        self.set_initial_wave(wave)
        self.set_focal_series(fs)
        self.set_contrast_transfer(ctf, *args, **kw)
        self.set_convergence_envelope(alpha)

    def __call__(self, Niters=5, unpad=False):
        """
        Will run the reconstruction, in the GPU or CPU, depending on how the
        parameters were set-up, and return the resulting wave with or without
        padding. This result is also stored in the "new_wave" attribute.

        Parameters
        ----------
        Niters : int
         Numer of GS-loop iterations. Set to 5 by defect.

        unpad : bool
         Pad / Unpad switch for the returned wave. Set to False by defect.

        Returns
        -------
        new_wave: Hyperspy Image
         Gerchberg-Saxton refined complex wave.

        Notes
        -----
        The flux-preserving G-S algorithm contains an additional convolution
        step if a convergence envelope is used. This is activated by the
        presence of said envelope. In case a calculation without the
        additional convolution should run after one (or several ones) with it,
        the convergence envelope attribute must be destroyed. Use, for instance:

        >>> del self.Esdata
        """
        if self.using_gpu:
            self.run_gpu(Niters)
        else:
            self.run_cpu(Niters)
        return self._return_new_wave(unpad)

    def _return_new_wave(self, unpad=False):
        """
        This method returns the current wave, with or without padding depending
        on a boolean parameter unpad.
        """
        if self.using_gpu:
            self.new_wave.data = self.wdata.get()
        else:
            self.new_wave.data = self.wdata

        if unpad:
            return self.new_vave.unset_padding()
        else:
            return self.new_wave

    def _return_new_phase(self, unpad=False):
        """
        This method returns the current phase, with or without padding depending
        on a boolean parameter unpad.
        """
        new_phase = self.new_phase.deepcopy()
        if self.using_gpu:
            new_phase.data = self.phase_data.get()
        else:
            new_phase.data = self.phase_data.copy()

        if unpad:
            return new_phase.unset_padding()
        else:
            return new_phase

    def set_focal_series(self, fs, defoci=None):
        """
        Setting the focal series also sets important parameters as the padding
        and defocus values.

        Parameters
        ----------
        fs : Hyperspy Image
         Mandatory! A focal series with all the appropriate parameters. Padding
         is read from metadata.

        defoci : array
         Optional, use it to explicitly set defoci values. Must have a dimension
         equal to the navigation axis of fs.
         """
        # Real space parameters
        Nz, Ny, Nx = fs.data.shape
        if defoci is None:
            defoci = fs.axes_manager.navigation_axes[0].axis

        Nz = len(defoci)
        self.zdim = defoci
        self.k2 = fs.get_fourier_space()
        self.Nz = Nz

        # In the middle of the focal series is the reference plane
        Nz_half = np.ceil(Nz/2).astype('int') - 1

        # Pad mask
        Npy, Npx = fs.metadata.Signal.pad_tuple
        mask = np.zeros((Nz,Ny,Nx), dtype=np.bool)
        mask[:,Npy[0]:(Ny-Npy[1]), Npx[0]:(Nx-Npx[1])] = True

        # Set some parameters
        self.new_wave = fs._get_signal_signal()
        self.new_wave.metadata = fs.metadata.deepcopy()
        self.shape = (Nz, Ny, Nx)

        # Allocate things ...
        if self.using_gpu:
            # ... in GPU!
            self.Iexp  = to_gpu_f(fs.data)
            self.mask = to_gpu_b(mask)

            # - The plans for FFT
            self.pft3dcc = cu_fft.Plan((Ny, Nx), np.complex64, np.complex64, Nz)
            self.pft2dcc = cu_fft.Plan((Ny, Nx), np.complex64, np.complex64)
        else:
            # ... in CPU!
            self.Iexp = fs.data.astype('complex64')
            self.mask = mask

    def set_initial_complex_wave(self, amplitude, phase):
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
        wave.metadata = amplitude.metadata.deepcopy()
        wave.data = np.sqrt(amplitude.data) * np.exp(1j*phase.data)
        self.set_input_wave(wave, phase.data.copy())

    def set_initial_wave(self, wave, phase_data=None):
        """
        Use it to directly set the wave. It writes to an attribute "wdata" that
        is overwritten everytime a G-S calculation runs. This allows to call G-S
        refinements consecutively. Nevertheless, the attribute "wave" will
        contain the original input wave.

        Parameters
        ----------
        wave : Hyperspy Image
         A complex wave with all the appropriate parameters.

        phase_data : numpy array
         Use this to set the phase data separately to avoid wrapping issues.
        """
        # TODO: trigget set_contrast_transfer if wave changes something important
        self.wave = wave

        # The phase data is stored separately to avoid wrapping issues
        if phase_data is None:
            phase_data = np.angle(wave.data)

        if self.using_gpu:
            self.wdata = to_gpu_c(wave.data)
            self.phase_data = to_gpu_f(phase_data)
        else:
            self.wdata = wave.data.astype('complex64')
            self.phase_data = phase_data.copy()

    def set_contrast_transfer(self, ctf, *args, **kw):
        """
        Use it to set the contrast transfer function. If a ctf image is not
        provided, the get_contrast_transfer method is called with defoci set by
        the focal series navigation axes. More info in the docs therein.
        """
        if ctf is None:
            ctf = self.wave.get_contrast_transfer(self.zdim, *args, **kw)

        if self.using_gpu:
            self.ctfd = to_gpu_c(ctf.data)
        else:
            self.ctfd = ctf.data.astype('complex64')

    def set_convergence_envelope(self, alpha):
        """
        Sets the spatial-coherence envelope function. The convergence semiangle,
        alpha, is set by input preferentially or, if possible, by metadata.

        Parameters
        ----------
        alpha : float or None
         To set the convergence semi-angle, in radians. If None, will try to
         read from metadata.

        Note
        ----
        This version also has sets an ElementwiseKernel for combining the
        experimental intensities and simulated waves in each GS iteration.
        """
        k2 = self.k2
        metadata = self.new_wave.metadata
        if (alpha is None):
            if metadata.has_item('ModImage.convergence_semiangle'):
                alpha = metadata.ModImage.convergence_semiangle
            else:
                alpha = 0.

        envelope = np.exp( -k2 * ((0.5*self.zdim*alpha)**2)[:,None,None] )

        if self.using_gpu and (alpha != 0):
            self.Esdata = to_gpu_f(envelope)
            # EWK for wave combo
            self.cuwave = ElementwiseKernel(
            "float *a,pycuda::complex<float> *b,float *c,pycuda::complex<float> *d",
            "const pycuda::complex<float> j(0.0,1.0); \
             d[i] = (abs(b[i])+sqrt(a[i])-sqrt(c[i])) * exp(j*arg(b[i]));")
        elif self.using_gpu and (alpha == 0):
            # No spatial-coherence envelope, but we need
            # EWK for wave combo because GPU
            self.cuwave = ElementwiseKernel(
            "float *a,pycuda::complex<float> *b,pycuda::complex<float> *c",
            "const pycuda::complex<float> j(0.0,1.0); \
             c[i] = sqrt(a[i]) * exp(j*arg(b[i]));")
        else:
            self.Esdata = envelope.astype('float32')

        self.new_wave.metadata.set_item('ModImage.convergence_semiangle', alpha)

    def run_gpu(self, Niters):
        """
        Run G-S on GPU. The result is overwritten on the attribute "self.wdata"
        containing a pycuda array.
        """
        Nz, Ny, Nx = self.shape
        # Allocate output data
        wdata = gpuarray.empty((Ny, Nx), np.complex64)
        sim   = gpuarray.empty((Nz, Ny, Nx), np.complex64)
        Isim  = gpuarray.empty((Nz, Ny, Nx), np.complex64)
        for io in trange(Niters):
            # Propagate the initial wave to simulate defocused waves
            # Psi(x,y,z) = convolve[Psi(x,y,0), CTF(x,y,z)]
            cu_fft.fft(self.wdata, wdata, self.pft2dcc)
            for kk in range(Nz):
                sim[kk,:,:] = self.ctfd[kk,:,:] * wdata
            cu_fft.ifft(sim, sim, self.pft3dcc, True)
            if hasattr(self, 'Esdata'):
                # Use the intensities, Isim = |Psi|**2
                # Convolve with spatial-coherence envelope
                # Isim = convolve[Isim, Es]
                Isim = sim * sim.conj()
                cu_fft.fft(Isim, Isim, self.pft3dcc)
                cu_fft.ifft(Isim*self.Esdata, Isim, self.pft3dcc, True)
                # Combine experimental and simulated amplitudes with simulated phase
                # Psi' = [abs(Psi)+sqrt(Iexp)-sqrt(Isim)]*exp[i*arg(Psi)]
                self.cuwave(self.Iexp, sim, Isim.real, Isim)
            else:
                # Combine experimental amplitudes with simulated phase
                # Psi' = [sqrt(Iexp)]*exp[i*arg(Psi)]
                self.cuwave(self.Iexp, sim, Isim)
            sim = gpuarray.if_positive(self.mask, Isim, sim)
            # then back-propagate to the exit plane and take average
            # Psi(x,y,0) = < convolve[Psi, CTF*] >_z
            cu_fft.fft(sim, sim, self.pft3dcc)
            sim = sim * self.ctfd.conj()
            cu_fft.ifft(sim, sim, self.pft3dcc, True)
            wdata = misc.mean(sim.reshape(Nz,Nx*Ny),0).reshape(Ny,Nx)
            # update phase and wave
            self.cuphase(wdata, self.wdata, self.phase_data)
            self.wdata = wdata.copy()

    def run_cpu(self, Niters):
        """
        Run G-S on CPU. The result is overwritten on the attribute "self.wdata"
        containing a numpy array.
        """
        Nz, Ny, Nx = self.shape
        for io in trange(Niters):
            # Propagate the initial wave to simulate defocused waves
            # Psi(x,y,z) = convolve[Psi(x,y,0), CTF(x,y,z)]
            Psi = np.fft.ifft2(np.fft.fft2(self.wdata)[None,:,:]*self.ctfd)
            if hasattr(self, 'Esdata'):
                # Use the intensities, Isim = |Psi|**2
                # Convolve with spatial-coherence envelope
                # Isim = convolve[Isim, Es]
                Psim = np.fft.ifft2(np.fft.fft2(Psi * Psi.conj()) * self.Esdata)
                # Combine experimental and simulated amplitudes with simulated phase
                # Psi' = [abs(Psi)+sqrt(Iexp)-sqrt(Isim)]*exp[i*arg(Psi)]
                Psim = (np.abs(Psi)+np.sqrt(self.Iexp)-np.sqrt(Psim.real)) * \
                        np.exp(1j*np.angle(Psi))
            else:
                # Combine experimental amplitudes with simulated phase
                # Psi' = [sqrt(Iexp)]*exp[i*arg(Psi)]
                Psim = np.sqrt(self.Iexp) * np.exp(1j*np.angle(Psi))
            Psi[self.mask] = Psim[self.mask]
            # then back-propagate to the exit plane and take average
            # Psi(x,y,0) = < convolve[Psi, CTF*] >_z
            wdata = np.mean(np.fft.ifft2(np.fft.fft2(Psi)*self.ctfd.conj()),0)
            # Update phase and wave
            self.phase_data = self.phase_data + np.angle(wdata/self.wdata)
            self.wdata = wdata.copy()
