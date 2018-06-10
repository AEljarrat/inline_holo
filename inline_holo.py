""" Inline holography in Python !

This is the main file, it defines a modified hyperspy image class with new
methods useful for inline holography. The more complex algorithms for image
simulation and wavefront reconstruction can be accessed through wrappers, with
only limited functionality. Complete functionality is achieved by separately
instantiating the corresponding module.

Also included; a validation class object that can be used to run checks on the
Image data and results, including fourier ring correlation, chi-2 and rvalue
tests.

Example
-------
from inline_holo import ModifiedImage as MI
img1 = MI(some_numpy_array)
img2 = MI(some_hyperspy_Signal2D)
hs.plot.plot_images([img1, img2])

from inline_holo import validation
val = validation(img1, img2, unpad=False)
frc = val.run_fourier_ring_correlation()
frc.plot()


Authors
-------
Alberto Eljarrat and Christoph T. Koch.
Institute für Physik. Humboldt Universität zu Berlin.

No commercial use or modification of this code
is allowed without permission of the authors.
"""
import numpy as np

# hyperspy dependency
from hyperspy.signals import Signal1D as Signal
from hyperspy.signals import Signal2D as Image

# import modules
from CTFSim import CTFSim
from MFTIE import MFTIE
from GPTIE import GPTIE
from GS import GS

# Useful routines for binary integration
def integrate_binary_real(sdata, inv):
    '''
    Binarized counting integral. For real valued datasets.
    sdata and inv have the same dimensions.

    Parameters
    ----------
    sdata : numpy array
     Real image data we want to integrate.
    inv : numpy array
     Binarized map of the image used for the binary counting.

    Returns
    -------
    idata : numpy array
     The integral result.
    '''
    idata = np.bincount(inv, weights=sdata.ravel())
    return idata

def integrate_binary_comp(sdata, inv):
    '''
    Binarized counting integral. For complex valued datasets.
    sdata and inv have the same dimensions.

    Parameters
    ----------
    sdata : numpy array
     Complex image data we want to integrate.
    inv : numpy array
     Binarized map of the image used for the binary counting.

    Returns
    -------
    idata : numpy array
     The integral result.
    '''
    idata = np.bincount(inv, weights=sdata.real.ravel()) + \
            1j * np.bincount(inv, weights=sdata.imag.ravel())
    return idata

class ModifiedImage(Image):
    """
    Modification of the hyperspy Image class that adapts to the needs of focal
    series theoretical and experimental analysis. It contains methods that
    seamlessly adapt hyperspy Image-Objects to numpy pad functionality. A method
    to generate a CTF based on a non-linear imaging model compatible with
    Rayleigh integral formulation (high numerical apertures and refractive
    indices other than 1). It contains simple methods that allow to test in-line
    multi-focus holography algorithms, such as transport of intensity equation
    and gradient-flipping.
    """
    def __init__(self, *erps, **kw):
        Image.__init__(self, *erps, **kw)

    def set_padding(self, pad_width=None, *erps, **kw):
        """
        Pad a hyperspy Image of navigation dimension <= 1. This is a wrapper for
        numpy.pad function. Additionally, as the padding information is stored
        in metadata, it is possible to reverse the effect of this function with
        a call to the unset_padding method.

        Parameters
        ----------
        pad_width : tuple of tuples
         Number of values padded to the edges of each axis.
          ((before_1, after_1), ... (before_N, after_N))
         unique pad widths for each axis. ((before, after),) yields same before
         and after pad for each axis. (pad,) or int is a shortcut for
         before = after = pad width for all axes.

        *erps, **kwerps are passed to numpy.pad function. See docs therein...

        Returns
        -------
        ModImage: hyperspy Image
            A new ModifiedImage object, with a number of padded pixels,
            indicated in pad_tuple parameter in metadata.
        """
        if pad_width is None:
            # If padding == None:
            # Same signal is returned, nothing else done
            return self
        # Store padded data and exit
        s = self.deepcopy()

        if self.axes_manager.navigation_dimension == 0:
            paddata = np.pad(self.data, pad_width, *erps, **kw)
            s.data = paddata
            s.get_dimensions_from_data()

        elif self.axes_manager.navigation_dimension == 1:
            # First dimension is navigation
            paddata = np.stack([np.pad(data,pad_width, *erps, **kw)
                       for data in self.data], 0)
            s.data = paddata
            s.get_dimensions_from_data()
            s.axes_manager.navigation_axes[0].axis = self.axes_manager.navigation_axes[0].axis.copy()

        pad_tuple = pad_width
        s.metadata.set_item('Signal.pad_tuple', pad_tuple)
        return s

    def unset_padding(self):
        """
        This method returns an un-padded copy of a padded Image by reading the
        value from metadata. See the set_padding function for more information.
        """
        if self.metadata.Signal.has_item('pad_tuple'):
            Npy, Npx = self.metadata.Signal.pad_tuple
        else:
            # If no padding was done, return the same signal
            return self
        Nx, Ny = self.axes_manager.signal_shape
        s=self.deepcopy()
        del s.metadata.Signal.pad_tuple
        if self.axes_manager.navigation_dimension == 0:
            s.data = s.data[Npy[0]:(Ny-Npy[1]), Npx[0]:(Nx-Npx[1])]
            s.get_dimensions_from_data()
        elif self.axes_manager.navigation_dimension > 0:
            s.data = s.data[..., Npy[0]:(Ny-Npy[1]), Npx[0]:(Nx-Npx[1])]
            s.get_dimensions_from_data()
            # copy in case of non-linear defoci
            s.axes_manager.navigation_axes[0].axis = self.axes_manager.navigation_axes[0].axis.copy()
        return s

    def get_real_space(self, shifts=None):
        """
        Returns the real space coordinate vectors for this image.

        Parameters
        ----------
        shifts : None or iterable
         Shift the image coordinates according to the given amounts, in pixel
         or scaled units depending if an int or float value are used. By default
         equal to None, the origin set by the axes_manager is used.

        Returns
        -------
        coords : list
         With two values.
        """
        scales = [axi.scale for axi in self.axes_manager.signal_axes]
        coords = [axi.axis.copy() for axi in self.axes_manager.signal_axes]

        if shifts is not None:
            shifts = list(shifts)
            if not len(shifts) == 2:
                raise "The number of shift values should be 2"
            for io in range(2):
                if type(shifts[io]) is int:
                    shifts[io] *= scales[io]
                coords[io] -= shifts[io]
        return coords

    def get_fourier_space(self, retstr='square'):
        """
        Returns meshgrids for Fourier transforms of the image. Can be selected
        by choosing the parameter "retstr". Choosing from k2 = kx^2+ky^2, these
        vectors, or both. See below for more info.

        Parameters
        ----------
        retstr : one of these strings: 'square', 'vectors', 'both'
         Return string options: Can be 'square' (the method returns |k|^2),
         'vectors' (it returns [kx, ky]), or 'both' (it returns [|k|^2, kx, ky])

        Return
        ------
        ndarray : numpy array or list of
         Meshgrids in the reciprocal-space that corresponds to the real-space
         scales, di, as 2.*pi / (Ni*di), where i = x, y. If the retstr='vectors'
         or 'both' a list of two or three arrays is returned, respectively.
        """
        # Get scales
        Nx, Ny = self.axes_manager.signal_shape
        dx, dy = [axi.scale for axi in self.axes_manager.signal_axes]

        # Calculate K-space
        dky = 2.*np.pi / (Ny*dy)
        dkx = 2.*np.pi / (Nx*dx)
        ky1D = dky*(-np.floor(Ny/2.)+np.arange(Ny))
        kx1D = dkx*(-np.floor(Nx/2.)+np.arange(Nx))
        kx = np.fft.ifftshift(kx1D)
        ky = np.fft.ifftshift(ky1D)
        # Custom output depending on "retstr"
        if retstr is 'vectors':
            return kx, ky
        else:
            [kxm, kym] = np.meshgrid(kx, ky)
            k2 = kxm**2+kym**2
            if retstr is 'square':
                return k2
            elif retstr is 'both':
                return k2, kx, ky
            else:
                raise ValueError('Parameter retstr not recognized: '+retstr)

    def get_digitized_radius(self, bin_size=None, shifts=None):
        '''
        Obtain a digitized radial mesh of the image useful for binary counting.

        Parameters
        ----------
        bin_size : None, int, float
         Bin size for the digitized mesh, in scaled units when a float number is
         used. If an int is used, the image is divided into that number of
         regions. By default equal to None, a bin size equal to the pixel size
         is used.
        shifts : None or iterable
         Shift the origin of the image coordinates according to the given
         amounts. See ``self.get_real_space`` for more info.

        Returns
        -------
        radius : Signal2D
         Digited radius.
        '''
        # self is a HS image
        xx, yy = self.get_real_space(shifts=shifts)
        radius = np.sqrt(xx[:, None]**2. + yy[None,:]**2.)

        # binarization
        scales = [axi.scale for axi in self.axes_manager.signal_axes]
        if bin_size is None:
            bin_size = np.sqrt(np.sum(np.power(scales, 2)))
        elif type(bin_size) is int:
            bin_size = radius.max() / bin_size

        bins = np.arange(radius.min(), radius.max()+1.1*bin_size, bin_size)
        ret = self._get_signal_signal()
        ret.data = bins[np.digitize(radius, bins)]
        return ret

    def get_digitized_angle(self, bin_size=None, shifts=None):
        '''
        Obtain a digitized angular mesh of the image useful for binary counting.

        Parameters
        ----------
        bin_size : None, int, float
         Bin size for the digitized mesh, in scaled units when a float number is
         used. If an int is used, the image is divided into that number of
         regions. By default equal to None, a bin size equal to the pixel size
         is used.
        shifts : None or iterable
         Shift the origin of the image coordinates according to the given
         amounts. See ``self.get_real_space`` for more info.

        Returns
        -------
        angle : Signal2D
         Digitized angle.
        '''
        # self is a HS image
        xx, yy = self.get_real_space(shifts=shifts)
        angle = np.angle(xx[:, None] + 1j* yy[None,:])

        # binarization
        scales = [axi.scale for axi in self.axes_manager.signal_axes]
        if bin_size is None:
            bin_size = np.sqrt(np.sum(np.power(scales, 2)))
        elif type(bin_size) is int:
            bin_size = 2.*np.pi / bin_size

        bins = np.arange(angle.min(), angle.max()+1.1*bin_size, bin_size)
        ret = self._get_signal_signal()
        ret.data = bins[np.digitize(angle, bins)]
        return ret

    def integrate_binary(self, bin_mask, normalize=True, *args, **kwargs):
        '''
        Image integration with binarized mask. The binarization is achieved by
        first finding the unique elements of the provided array and adding the
        values of the image(s) in the corresponding coordinates.

        Parameters
        ----------
        bin_mask : Signal2D
         A 2D signal with the same shape as the image(s), the coordinates of
         unique elements in the data array are found by calling ``np.unique``.
        normalize : bool
         Controls wether we obtain an absolute sum or a normalized integral,
         the last one being the default setting.

        Returns
        -------
        integral : signal
         With signal dimension axis equal to the radial mesh bins used for the
         integration and a single navigation dimension with size corresponding
         to the navigation dimension of the original signal but flattened, if
         the original signal had one.
        '''
        dst, inv, cts = np.unique(bin_mask.data,
                                  return_inverse=True,
                                  return_counts=True)

        # Complex data support
        if np.iscomplexobj(self.data):
            integrator = integrate_binary_comp
        else:
            integrator = integrate_binary_real

        integral_signal = self.map(integrator,
                                   inv=inv,
                                   inplace=False,
                                   *args,
                                   **kwargs)

        integral_signal = Signal(integral_signal.data)

        # Normalization
        if normalize:
            integral_signal.data = integral_signal.data / cts

        # Set the axis
        integral_signal.axes_manager.signal_axes[0].axis = dst

        # Also the scales, but beware, the axis is regular only if the mask was!
        nbins = np.unique(np.round(np.diff(dst), 5)).size
        if nbins == 1:
            # regular mask
            integral_signal.axes_manager.signal_axes[0].offset = dst[0]
            integral_signal.axes_manager.signal_axes[0].scale  = dst[1] - dst[0]

        return integral_signal

    def integrate_radial(self, bin_size=None, shifts=None, *args, **kwargs):
        '''
        Integrate the image in a radial mesh. The mesh is calculated from
        the pixel coordinates of the image, with an optional translation to the
        center. The integration is performed by binary counting using
        ``self.integrate_binary``. Optional parameters may be provided to use a
        specific coordinate set, see the documentation in this method.

        *args and **kwargs are passed to ``self.get_digitized_radius``, for more
        information see also the docs therein.

        Returns
        -------
        radial_integral : Signal1D
         With signal dimension axis equal to the radial mesh bins used for the
         integration and same navigation dimension as the original signal.
        '''
        radius = self.get_digitized_radius(bin_size=bin_size, shifts=shifts)
        radial_integral = self.integrate_binary(bin_mask=radius,*args,**kwargs)
        return radial_integral

    def integrate_angular(self, bin_size=None, shifts=None, *args, **kwargs):
        '''
        Integrate the image in an angular mesh. The mesh is calculated from
        the pixel coordinates of the image, with an optional translation to the
        center. The integration is performed by binary counting using
        ``self.integrate_binary``. Optional parameters may be provided to use a
        specific coordinate set, see the documentation in this method.

        *args and **kwargs are passed to ``self.get_digitized_angle``, for more
        information see also the docs therein.

        Returns
        -------
        angular_integral : Signal1D
         With signal dimension axis equal to the radial mesh bins used for the
         integration and same navigation dimension as the original signal.
        '''
        angle = self.get_digitized_angle(bin_size=bin_size, shifts=shifts)
        angle_integral = self.integrate_binary(bin_mask=angle, *args, **kwargs)
        return angle_integral

    def plot(self, polar=False, fftshift=False, *erps, **kwerps):
        """
        A bad cover version of Image.plot method, useful to produce two images
        when data is a complex image.

        Parameters
        ----------
        polar : bool
         Controls whether polar or cartesian representation is used. It is set
         to cartesian (False) by defect.
        fftshift : bool
         Will use "np.fft.fftshift" on the image dimensions prior to plot. Unset
         (False) by defect.
        *erps, **kwerps are passed to Image.plot.
        """
        if fftshift:
            s = self.deepcopy()
            s.data = np.fft.fftshift(self.data, (-2,-1))
        else:
            s = self
        if np.iscomplexobj(s.data):
            if polar:
                amp = s.deepcopy()
                amp.data = np.absolute(s.data)
                amp.metadata.General.title += ', Modulus'
                Image.plot(amp, *erps, **kwerps)

                arg = s.deepcopy()
                arg.data = np.angle(s.data)
                arg.metadata.General.title += ', Argument'
                Image.plot(arg, *erps, **kwerps)
            else:
                real = s.deepcopy()
                real.data = np.real(s.data)
                real.metadata.General.title += ', real part'
                Image.plot(real, *erps, **kwerps)

                imag = s.deepcopy()
                imag.data = np.imag(s.data)
                imag.metadata.General.title += ', imag part'
                Image.plot(imag, *erps, **kwerps)
        else:
            Image.plot(s, *erps, **kwerps)

    def get_contrast_transfer(self, defoci=None, angles=None, wlen=None,
                              nref=None, NA=None, fsmooth=None):
        """
        Obtain the contrast transfer function (CTF) to model an imaging system.
        The CTF is given in Fourier representation. Imaging systems with high
        numerical aperture, NA > 1, are supported. It is possible to also obtain
        smooth apertures setting the fsmooth parameter. Such smoothing is
        performed up to fsmooth times the size of the aperture in Fourier space.

        Parameters
        ----------
        defoci : None, list, tuple or numpy vector
         Set single or multiple defocus values, used to generate a single or
         multidimensional CTF, respectively. A new (1st) dimension is added with
         one CTF for each value of defoci. In case the parameter is not set
         and if possible, the defoci are read from the 1st navigation axis.
        angles : None, list, tuple or numpy vector
         Set single or multiple astigmatism-angle values, used in combination
         with the defocus defined above. A new (2nd) dimension is added with
         one CTF for each value of this angle. In case the parameter is not set
         and if possible, the angles are read from the the 1st previously unread
         navigation axis (see above).

        ! The following parameters are equal to None by default. In this case,
          the values are read from ModImage metadata and used, if not None.
          If both are None, typically the value is set to 1. The exception is
          fsmooth, if equal to None the aperture smoothing is simply not used.

        wlen : number
         Wave-length for the incident radiation.
        nref : number
         Refractive index for the imaging system.
        NA : number
         Numerical aperture for the imaging system.
        fsmooth : number
         Cosine-bell smoothing parameter.

        Returns
        -------
         ModifiedImage    A Hyperspy Image, with stored calculation parameters

        """
        # Read defocus and angle values from input or set them from the axes
        # Notice the method stops if defoci is not present
        ax_count = 0
        if defoci is None:
            try:
                defoci = self.axes_manager.navigation_axes[ax_count].axis
                ax_count += 1
            except IndexError:
                print('No defocus value given! Exiting...')
                return
        else:
            if not isinstance(defoci, np.ndarray):
                defoci = np.array(defoci)
        # Set astigmatism from input or from the navigation axis, if present
        if angles is None:
            try:
                angles = self.axes_manager.navigation_axes[ax_count].axis
            except IndexError:
                pass
        else:
            if not isinstance(angles, np.ndarray):
                angles = np.array(angles)

        # Get scales: if wave_length is not set in
        # metadata, lambda = 1. is used!
        Nx, Ny = self.axes_manager.signal_shape
        dx, dy = [axi.scale for axi in self.axes_manager.signal_axes]
        k2, kx, ky = self.get_fourier_space('both')
        [kx, ky] = np.meshgrid(kx, ky)
        if wlen is None:
            if self.metadata.has_item('ModImage.wave_length'):
                wlen = self.metadata.ModImage.wave_length
            else:
                wlen = 1.
        wnum = 2.*np.pi/wlen

        # The wave-number depends in input
        # if refractive index is found
        # a rescaled version is used.
        if nref is None:
            if self.metadata.has_item('ModImage.refractive_index'):
                nref = self.metadata.ModImage.refractive_index
            else:
                nref = 1.
        wnum = wnum * nref

        # The Aberrations (Chi) function:
        # Change to Astigmatism mode if angles is present
        if angles is not None:
            ka2 = kx * np.cos(angles[:,None,None]) + ky * np.sin(angles[:,None,None])
            ka2 = ka2*ka2
            Chi = np.sqrt(wnum**2-ka2) - 0.5*(np.sqrt(wnum**2-k2) + wnum)
        if angles is None:
            Chi = np.sqrt(wnum**2-k2) - wnum

        # Same for the APERTURE:
        # if only parameter NA is found, a sharp aperture is set
        # if parameter fsmooth is also found, we use a smooth one
        if NA is None:
            if self.metadata.has_item('ModImage.numerical_aperture'):
                NA = self.metadata.ModImage.numerical_aperture
            else:
                NA = 1.
        # Set hard aperture: NA re-scaled by nref
        k2Aperture = (NA*wnum/nref)**2
        aperture = np.zeros((Ny,Nx))
        aperture[k2 <= k2Aperture] = 1
        # Now smooth it optionally ...
        if fsmooth is None:
            if self.metadata.has_item('ModImage.f_smooth_aperture'):
                fsmooth = self.metadata.ModImage.f_smooth_aperture
        if fsmooth is not None:
            # set smooth aperture using cosine-bell smoothing
            ind = (k2 > k2Aperture) * (k2 < (1.+fsmooth)*k2Aperture)
            aperture[ind] = 0.5 * (1. + np.cos(np.pi*(k2[ind]-k2Aperture) / \
                                        (fsmooth*k2Aperture)))

        # Calculate CTF:
        if Chi.ndim == 2:
            CTF = np.exp(1j * Chi * defoci[:,None,None]) * aperture
        elif Chi.ndim == 3:
            CTF = np.exp(1j * Chi * defoci[:,None,None,None]) * aperture
        CTF = np.squeeze(CTF)

        # Store the CTF
        # The Depth axis are the defoci
        # The signal axis are in Fourier-space
        CTF = np.nan_to_num(CTF).astype('complex64')
        ctf = ModifiedImage(CTF)

        # The 1st and 2nd Nav-axis are defocus and Astig.-angles, respectively
        depthdims = ctf.axes_manager.navigation_dimension
        if (depthdims == 1) and (angles is None):
            ctf.axes_manager[0].offset = defoci[0]
            ctf.axes_manager[0].scale  = defoci[1] - defoci[0]
        elif (depthdims == 1):
            ctf.axes_manager[0].offset = angles[0]
            ctf.axes_manager[0].scale  = angles[1] - angles[0]
        elif (depthdims == 2):
            ctf.axes_manager[0].offset = defoci[0]
            ctf.axes_manager[0].scale  = defoci[1] - defoci[0]
            ctf.axes_manager[1].offset = angles[0]
            ctf.axes_manager[1].scale  = angles[1] - angles[0]
        # K-space
        for io, axi in enumerate(ctf.axes_manager.signal_axes[::-1]):
            axi.axis   = k2.take(0,-io)
            axi.offset = axi.axis[0]
            axi.scale  = axi.axis[1]-axi.axis[0]
        ctf.get_dimensions_from_data()

        # Store the used parameters
        ctf.metadata = self.metadata.deepcopy()
        ctf.metadata.set_item('ModImage.wave_length', wlen)
        ctf.metadata.set_item('ModImage.refractive_index', nref)
        ctf.metadata.set_item('ModImage.numerical_aperture', NA)
        ctf.metadata.set_item('ModImage.f_smooth_aperture', fsmooth)

        # IMP! set again defocus, for non-linear defoci vectors, or...
        # the astigmatism info that might be useful in the future
        if angles is None:
            ctf.axes_manager[0].axis = defoci
        else:
            ctf.axes_manager[-3].axis = angles
            ctf.metadata.set_item('ModImage.astigmatic_defocus', defoci.squeeze())

        return ctf

# Useful routines for loading data
def numpy2ModI(npdata, dsig=(1., 1.), osig=(0., 0.), dnav=(1., ), onav=(0., )):
    """
    Helper method to instantiate a ModifiedImage class object from a numpy array
    The last 2 dimensions of the numpy array are identified as Image (Signal2D)
    dimensions. Any other dimensions are identified with navigation dimensions,
    following the hyperspy conventions.

    Parameters
    ----------
    npdata : ndarray
     Numpy data to generate ModImage, thus npdata.ndim >= 2.; Additionally,
     npdata.ndim - 2  = Nnav (see below).
    dsig : tuple
     With 2 values indicating the scale of the signal axis.
    osig : tuple
     With 2 values indicating the origin of the signal axis.
    dsig : tuple
     With Nnav values indicating the scale of the navigation axis.
    osig : tuple
     With Nnav values indicating the origin of the signal axis.

    Returns
    -------
    img : ModifiedImage
     A modified hyperspy Signal2D class object.

    """
    img = ModifiedImage(npdata)
    for io, axi in enumerate(img.axes_manager.signal_axes):
        axi.offset = osig[io]
        axi.scale  = dsig[io]
    if npdata.ndim > 2:
        for io, axi in enumerate(img.axes_manager.navigation_axes):
            axi.offset = onav[io]
            axi.scale  = dnav[io]
    img.get_dimensions_from_data()
    return img

# Module wrappers here
def simulate_CTF(wave, defoci, alpha=None, using_gpu=False, *erps, **kwerps):
    """
    Simulation of focal series using non-linear Optics. The named parameters
    are used to set the calculations using module CTFSim. Anonymous params
    (*erps and **kwerps) are used to calculate a contrast transfer function,
    through a call to the get_contrast_transfer method.

    Parameters
    ----------
    wave : Hyperspy Image
     The input wave that is projected. Generally, this is a complex wave.
    defoci : numpy array
     Set the defocus values for the simulation.
    alpha : number
     Convergence-semi angle, use this to set a spatial coherence envelope.
    using_gpu : bool
     Sets the GPU/CPU functionality

    Returns
    -------
    focal_series : Hyperspy Image
     Dataset containing the intensity of the propagated wave at positions
     set by the defocus values.
    """
    ctsim = CTFSim(wave, zdef, alpha, using_gpu, *erps, **kwerps)
    fs = ctsim()
    return fs

def multi_focus_TIE(fs, using_gpu=False, printing=False, *erps, **kwerps):
    """
    Will retrieve the phase from a focal series in a deterministic fashion
    using the multi-focus transport of intensity (MFTIE) method. The named
    parameters are passed to the __init__ method of module MFTIE. Anonymous
    parameters (*erps and **kwerps) are used to calculate the inverse
    Laplacian filters (more info in module MFTIE).

    Parameters
    ----------
    fs : Hyperspy Image
     This is a focal series.
    using_gpu : bool
     Sets the GPU/CPU functionality.
    printing : bool
     Print information about the MFTIE filters.

    Returns
    -------
    phase: Hyperspy Image
     Multi-focus TIE phase.
    """
    mftie = MFTIE(fs, using_gpu, *erps, **kwerps)
    phase = mftie()
    if printing:
        mftie.print_k2_thres()
    return phase

def gaussian_process_TIE(fs, *erps, **kwerps):
    """
    Will retrieve the phase from a focal series in a deterministic fashion
    using the gaussian-process transport of intensity (GPTIE) method. The
    anonymous parameters (*erps and **kwerps) are passed to the GPTIE module

    Parameters
    ----------
    fs : Hyperspy Image
     This is a focal series.

    Returns
    -------
    phase: Hyperspy Image
     Gaussian-process TIE phase.

    Reference
    ---------
    Adapted from the original Matlab software by Jingshan Zhong and coworkers
    (see: www.dauwels.com  or  www.laurawaller.com).
    """
    gptie = GPTIE(fs, *erps, **kwerps)
    phase = gptie()
    return phase

def multi_focus_GS(fs, Niters=5, init_wave=None, ctf=None, alpha=None, using_gpu=False,
                   init_phase=None, *erps, **kwerps):
    """
    Will refine a wave guess from a focal series in an iterative fashion
    using a version of the Gerchberg-Saxton (GS) loop. The named parameters
    are passed to the __init__ method of module GS. Anonymous parameters
    are optionally used to calculate a contrast transfer function, in the case
    this is not given, through fs.get_contrast_transfer(*erps and **kwerps).

    Parameters
    ----------
    fs : hyperspy Image
     Dataset containing an experimental focal series and including some
     pad region.
    Niters : int
     Number of iterations for the GS loop.
    init_wave : Hyperspy Image
     Image containing a wave with same signal dimensions as the input focal
     series. The wave is expected in its real-space representation, a 2D
     complex function. If not provided the algorithm is initialized with the
     in-focus image and zero init_phase.
    init_phase : Hyperspy Image
     As above, but for the phase. Provide the phase separately in case its
     in absolute larger that pi and phase wrapping is a problem.
    ctf : Hyperspy Image
     Dataset containing the contrast transfer function with the same shape as
     the input focal series. The CTF is expected in its reciprocal-space
     representation, a 2D complex function. If not provided, the
     "fs.get_contrast_transfer(*erps, **kwerps)" result is used.
    alpha : number
     Convergence-semi angle, use this to set a spatial coherence envelope.
    using_gpu : bool
     Sets the GPU/CPU functionality

    Returns
    -------
    wave : hyperspy Image
     Multi-focus TIE phase.
    phase : hyperspy Image
     In case init_phase is given, it is used to compute the final phase.
     This gives the correct GS refined phase if wrapping is a problem.
    """
    gsref = GS(fs, init_wave, ctf, alpha, using_gpu, *erps, **kwerps)

    if init_phase is None:
        wave = gsref(Niters)
        return wave

    if init_phase is not None:
        # provide phase information for wrapping awareness
        gsref.set_initial_wave(gsref.wave, init_phase.data)
        wave = gsref(Niters)
        phase = gsref._return_new_phase()
        return wave, phase

class validation():
    """
    Perform validation tests using the provided images
    Several options are available by combinations of the funtions and
    parameters below, thus, the images must have compatible dimensions, at
    least when unpadded (If as by default, both images are unpadded previous
    to any calculation). See below for more info.

    Authors
    -------
    Alberto Eljarrat, Johannes Müller and Christoph T. Koch.
    Institute für Physik. Humboldt Universität zu Berlin.

    No commercial use or modification of this code
    is allowed without permission of the authors.
    """
    def __init__(self, observed, expected, unpad=True):
        """
        Parameters
        ----------
        observed : Hyperspy Image, numpy ndarray
         Dataset containing the values corresponding to the observations.

        expected : Hyperspy Image, numpy ndarray
         Expectation in the statistical sense.

        unpad : bool
         Set to False to override unset_padding previous to any calculation.
         In this case, observed and expected must have the same unpadded signal
         dimensions.

        Notes
        -----
        The parameter names, observed and expected, indicate the role of the
        datesets provided, in the statistical sense.
        """
        self.obs = self._set_image(observed, unpad)
        self.exp = self._set_image(expected, unpad)

    def _set_image(self, imdata, unpad):
        if unpad is True:
            # Unset pad, only ModImages
            return imdata.unset_padding()
        else:
            # what if numpy array?
            if isinstance(imdata, np.ndarray):
                return ModifiedImage(imdata)
            else:
                return imdata

    def run_chi2(self, silent_run=False):
        """
        Perform a chi square test.

        Returns
        -------
        chi2 : Hyperspy Image
         With the same signal and navigation dimensions as (the unpadded verions
         of, if unpad = True) self and expected

        """
        self.chi2 = (self.obs - self.exp)**2.
        self.chi2.data /= self.exp.data**2.
        if not silent_run:
            return self.chi2

    def run_rvalue(self, silent_run=False):
        """
        Perform a r-value test. Only positive data (intensities).

        Returns
        -------
        rvalue : float
         Only one number.
        """
        odata = np.sqrt(self.obs.data)
        edata = np.sqrt(self.exp.data)
        self.rvalue = np.sqrt((odata - edata)**2.).sum()
        self.rvalue /= edata.sum()
        if not silent_run:
            return self.rvalue

    def run_rmse_check(self):
        """
        Measure RMSE^2 in real and reciprocal space.

        Returns
        -------
        rmse_ij, rmse_kl  : images
         RMSE^2 in real and reciprocal space, respectively.
        """

        # Shift the intensities to compare
        obs = self.obs.data - self.obs.data.mean()
        exp = self.exp.data - self.exp.data.mean()

        # Experimental RMSE2 in real space
        rmse_real = (obs - exp)
        rmse_real = rmse_real**2

        # Experimental RMSE2 in reciprocal space
        ftdiff  = np.fft.fft2(obs.data,norm='ortho') - np.fft.fft2(exp.data,norm='ortho')
        rmse_fourier = np.real(np.abs(ftdiff)**2.)

        # Return
        rmse_ij = self.obs.deepcopy()
        rmse_ij.data = rmse_real

        rmse_kl = ModifiedImage(rmse_fourier)
        return rmse_ij, rmse_kl

    def run_fourier_ring_correlation(self, silent_run=False, d2d=False):
        """
        Perform Fourier ring correlation (FRC) test. At the time, only the FRC
        for 2D images has been implemented. Because this algorithm uses FFT to
        compare images, it is important to enforce periodic boundary conditions,
        e.g. by the use of padding.

        Parameters
        ----------
        d2d : bool
         Set to True to obtain an image with the ring values in reciprocal space
         that could be useful for plotting.

        Returns
        -------
        frc : Hyperspy signal
         Contains the FRC and the unitary distance bins used for the calculation.

        """
        # This is done in the fourier plane
        exp_ft = np.fft.fft2(self.exp.data)
        obs_ft = np.fft.fft2(self.obs.data)

        # Get correlations
        exp_obs_cc = np.real(exp_ft * obs_ft.conj())
        exp_sc = np.abs(exp_ft)**2
        obs_sc = np.abs(obs_ft)**2

        # Binarize the spatial frequencies
        # using rings at the same distances
        # in fourier space (esoteric way, like ModifiedImage.integrate_radial
        # but taking fftshift into account)

        N = exp_ft.shape
        dist_x, dist_y = [np.minimum(np.arange(Ni), np.arange(Ni, 0, -1))
                          for Ni in N]
        dist_x *= dist_x
        dist_y *= dist_y
        # Distances in 2D-Fourier space
        dist_2d = np.sqrt(dist_x[:, None] + dist_y)
        dist_2d = np.round(dist_2d)
        # Binning and calculation
        distances, inv = np.unique(dist_2d, return_inverse=True)
        xcorr = np.bincount(inv, weights = exp_obs_cc.real.ravel())
        selfcorr  = np.bincount(inv, weights = exp_sc.ravel())
        selfcorr1 = np.bincount(inv, weights = obs_sc.ravel())
        FSC = xcorr / np.sqrt(selfcorr*selfcorr1)
        # Generate output
        self.frc = Signal(FSC)
        k2 = self.obs.get_fourier_space()
        self.frc.axes_manager[0].scale = np.sqrt(k2[0,1])
        self.frc.axes_manager[0].offset = 0.
        self.frc.get_dimensions_from_data()
        self.frc.metadata.General.title = 'FRC'
        # Save also the distances
        self.frc_d2d = ModifiedImage(np.fft.fftshift(dist_2d))
        if not silent_run:
            if not d2d:
                return self.frc
            else:
                return self.frc, self.frc_d2d

# TODO: implement FRC using digitized width
""""
width = 1.
bins = np.arange(q.min(), q.max(), width)
idx = np.digitize(q, bins)
dst, inv, cts = np.unique(idx, return_inverse=True, return_counts=True)
...
"""

# Useful routines for validation
def fs_validation(fs, phase, amplitude, init_wave=None, alpha=None):
    '''
    Run focal-series validation of the given phase and amplitude data. First,
    these images are used to simulate through-focus images at the same defocus
    values as the provided focal-series, fs.

    Parameters
    ----------
    fs : ModifiedImage
    phase : ModifiedImage
    amplitude : ModifiedImage
    init_wave : None or ModifiedImage
    alpha : None or ModifiedImage

    Returns
    -------
    chi2 : Signal1D
    rval : number
    '''
    # Do a focal series simulation with defocus from fs
    defoci = fs.axes_manager[0].axis
    ctsim = CTFSim(amplitude, defoci, alpha, True)
    ctsim.set_complex_wave(amplitude, phase)
    if init_wave is not None:
        # reset the init wave!
        ctsim.set_wave(wavi)
    fssim = ctsim()
    # Run validation
    val = validation(fs, fssim, unpad=True)
    chi2 = val.run_chi2()
    rval = val.run_rvalue()
    return chi2, rval
