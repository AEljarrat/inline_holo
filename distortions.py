import numpy as np
from scipy.ndimage import map_coordinates
from hyperspy.signals import ComplexSignal2D

def _get_nm_from_string(Pnm):
    """
    Return radial and accimuthal indices (n and m) from string input. The input
    must be of the form Pnm, where P={A, B} and n and m comply with the rules
    for triangular distortion coefficients (more below).

    Parameters
    ----------
    Pnm : string
     A string argument of the form "Anm" or "Bnm". In other words, a combination
     of one of two letters; A or B; and two numbers. The letter sets the parity
     of the coefficient. A = even, B = odd. The radial, "n", and accimuthal,
     "m", indices are set by two numbers, following some rules;
        1) m <= n;
        2) m==0 only if n is even.
        3) m and n share parity.

    Returns
    -------
    n, m : integer numbers
     The radial and accimuthal indices.
    """

    warn = 'Distortion string not recognized, '

    # First check
    if len(Pnm) is not 3:
        raise ValueError(warn+Pnm+'. Must by of the form {A, B}nm.')

    parity = Pnm[0]
    n = int(Pnm[1])
    m = int(Pnm[2])

    # More checks
    if m > n:
        raise ValueError(warn+Pnm+'. m > n')

    if (m == 0) and bool(n % 2):
        raise ValueError (warn+Pnm+'. m==0 with odd n')

    if (m % 2) != (n % 2):
        raise ValueError (warn+Pnm+'. m and n have different parity')

    if parity is 'B':
        m *= -1
    elif parity is not 'A':
        raise ValueError(warn+Pnm+'. Must by of the form {A, B}nm.')

    if (m == 0) and (parity is 'B'):
        raise ValueError(warn+Pnm+'. m==0 with odd parity')

    return n, m

def _get_unit_circle(N=64, fill_value=np.nan):
    """
    The unit circle has a radius of 1. Outer values are filled using the
    provided value.

    Parameters
    ----------
    N : int
     Number of pixels to use. Default 64.
    fill_value : number
     Provide fill value.

    Returns
    -------
    rho, phi : numpy array
     Radial and Accimuthal coordinates.
    """
    xx = np.linspace(-1., 1., N)
    yy = np.linspace(-1., 1., N)
    P_mesh = (xx[:,None] + 1j*yy[None,:])

    rho = np.float32(np.abs(P_mesh))
    phi = np.float32(np.angle(P_mesh))

    outside = rho > 1.
    rho[outside] = fill_value
    phi[outside] = fill_value
    return rho, phi

class Distortion():
    """
    Distortion polynomial module
    ----------------------------
    For Zernike-like distortions. These transforms are described as a polynomial
    expansion using polar coordinates. The polynomials are determined by a
    radial and accimuthal index. The indices follow a triangular distribution,
    which is bound to some selection rules. This is described in detail in the
    documentation of the `~.set_coefficients` method.

    """
    def __init__(self, *args, **kwargs):
        """
        **kwargs passed to `~.set_coefficients` method.

        Example
        -------
        Init the first 10 distortion coefficients:
        > kwdist = {
        ... 'A00' : 1+1j, 'B11' : 1+1j, 'A11' : 1+1j, 'B22' : 1+1j,
        ... 'A20' : 1+1j, 'A22' : 1+1j, 'B33' : 1+1j, 'B31' : 1+1j,
        ... 'A31' : 1+1j, 'A33' : 1+1j}
        > kdist = Distortions(**kwdist)
        """

        self._coeffs = {}
        self.set_coefficients(**kwargs)

    def __repr__(self):
        # TODO: format string
        out = self.get_coefficients()
        return out.__repr__()

    def __call__(self, image=None, *args, **kwargs):
        """
        Using the current polynomial expansion, calling will either project the
        displacement in the unit circle, or to a provided image, distorting it.

        Parameters
        ----------
        image : {None, hyperspy Signal2D}
         A hyperspy image to be distorted. If None is provided, the distortion
         displacement is projected into the unit circle. Depending on this input
         *args, **kwargs are passed to `_get_unit_circle` function or to the
         `~._distort_image` method.

        Returns
        -------
        out : hyperspy Signal2D
         If no image was provided, a complex signal, with the X and Y
         displacement is returned. In the other case, a distorted image is
         returned.
        """

        if image is None:

            # show displacements in the unit circle
            rho, phi = _get_unit_circle(*args, **kwargs)
            D = self._get_displacement(rho, phi)
            return ComplexSignal2D(D)

        # apply distortions to this image
        new_img_data = self._distort_image(image, *args, **kwargs)

        return image.__class__(new_img_data)

    def set_coefficients(self, **kwargs):
        """
        Flexible tool for setting correct distortion coefficients Anm or Bnm.
        The parity is set by a letter: A for even (cosine) and B for odd (sine).
        Radial and accimuthal indices are set by two numbers. More info below.

        Parameters
        ----------
        **kwargs : float
         All keyword arguments have to be of the form "Anm" or "Bnm". In other
         words, a combination of one of two letters; A or B; and two numbers.
         The letter sets the parity of the coefficient. A = even, B = odd.
         The radial, "n", and accimuthal, "m", indices are set by two numbers.
         These indices must comply with some rules;
          m <= n;
          m==0 only if n is even.
          m and n share parity.

        Note
        ----
        The coefficient list is stored in `~._coeffs`. This list is sorted
        according to the OSA standard index for Zernike polynomials.
        """

        coeffs = self._coeffs.copy()
        for ki in kwargs:

            # process string input
            n, m = _get_nm_from_string(ki)

            # compute OSA index
            p = (n*(n+2)+m) // 2

            # get value
            value  = complex(kwargs[ki])

            # (over)write the value if its not zero
            if value == 0j:
                if ki in coeffs:
                    del coeffs[ki]
            else:
                coeffs[ki] = [n, m, p, value]

        # sort the coefficients following p
        self._coeffs = {}
        for key, value in sorted(coeffs.items(), key=lambda x: x[1][2]):
            self._coeffs[key] = value

    def get_coefficients(self):
        """
        Method to obtain a distortion coefficient dictionary.

        Returns
        -------
        out : dict
         A dictionary with the distortion coefficients that can be used to init
         another distortion object.
        """
        out={}
        for key in self._coeffs.keys():
            out[key] = self._coeffs[key][3]
        return out

    def save_coefficients(self, name):
        """
        Save the coefficients to a human-readable text file.

        Parameters
        ----------
        name : string
         A valid file name for the saved document.
        """
        out = self.get_coefficients()
        with open(name, 'w') as f:
            for key in out:
                value = out[key]
                f.write("%s %.4e %.4e\n" % (key, value.real, value.imag))

    def load_coefficients(self, name):
        """
        Load the coefficients from a human-readable text file. This file
        contains one line for each distortion coefficient, with three terms
        separated by a space. The first term declares the distortion polynomial
        to use, such as "A11" or "B22". The following two terms are the real and
        imaginary parts of the coefficient. More information below.

        Parameters
        ----------
        name : string
         A valid file name for a properly formated document.

        General format
        --------------
        `{A, B}nm Real-coefficient Imag-coefficient`

        Example
        -------
        distortion.txt:
        >>> A00 9.2999e-06 4.0513e-06
        >>> B11 1.8369e-06 5.8968e-06
        >>> A11 7.9964e-06 7.7446e-06
        >>> B22 1.9602e-06 1.2024e-06
        >>> A20 8.6922e-06 4.7277e-07
        >>> A22 7.8818e-06 3.3797e-06
        >>> B33 4.5008e-06 4.5870e-06
        >>> B31 2.4208e-06 9.2781e-06
        >>> A31 6.5722e-06 2.8047e-06
        >>> A33 5.1478e-06 8.2145e-06
        """
        indict = {}
        with open(name, 'r') as f:
            rl = f.readlines()
            for line in rl:
                line = line.partition(' ')
                key = line[0]
                line = line[2].partition(' ')
                value = float(line[0])+1j*float(line[2])
                indict[key] = value
        self.set_coefficients(**indict)

    def get_n_m_Cp(self):
        """
        Utility method to parse `~.coeffs` into useful numpy arrays for the
        radial and accimuthal indices, plus the distortion coefficients.

        Returns
        -------
        n, m, Cp : numpy arrays
         The radial and accimuthal indices and the distortion coefficients for
         this transform.
        """
        n = []
        m = []
        Cp  = []
        for key in self._coeffs.keys():
            nmpCp = self._coeffs[key]
            n += [ nmpCp[0], ]
            m += [ nmpCp[1], ]
            Cp  += [ nmpCp[-1], ]
        n = np.array(n)
        m = np.array(m)
        Cp  = np.array(Cp)
        return n, m, Cp

    def _get_polynomials(self, rho, phi, n, m, Cp):

        odd = m < 0
        evn = ~ odd

        # from radial
        radial = (rho[...,None]**n)/(n+1)
        K = np.zeros_like(radial, dtype=np.complex64)

        K[...,evn] =    radial[...,evn]*np.cos(phi[...,None]*m[evn])   # Anm
        K[...,odd] = 1j*radial[...,odd]*np.sin(phi[...,None]*m[odd])   # Bnm

        return K

    def _get_displacement(self, rho, phi):

        n, m, Cp = self.get_n_m_Cp()
        K = self._get_polynomials(rho, phi, n, m, Cp)

        # the total displacement is obtained
        D = np.dot(K, Cp)

        return D

    def _distort_image(self, img, gpu=False, *args, **kwargs):
        '''
        Use this to interpolate in the CPU from some coordinates. Works on a
        single image.
        '''

        # Mesh corresponds to image shape and optical axis in the centre
        Ny, Nx = img.axes_manager.signal_shape

        xx = np.arange(Nx) - Nx*0.5
        yy = np.arange(Ny) - Ny*0.5

        R = (xx[:,None] + 1j*yy[None,:])

        rho = np.float32(np.abs(R))
        phi = np.float32(np.angle(R))

        if gpu is False:
            # update image coordinates with displacement
            D = self._get_displacement(rho, phi)
            R = R - D + 0.5*(Nx+1j*Ny)

            # map interpolation
            r_map = np.stack([R.real, R.imag], 0)
            return map_coordinates(img.data, r_map, *args, **kwargs)
        else:
            pass
