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

class Distortions():
    """
    Distortion polynomial module
    ----------------------------

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
        > kdist = Krivanek(**kwdist)
        """

        self._coeffs = {}
        self.set_coefficients(**kwargs)

    def __repr__(self):
        # TODO: format string
        out = self.get_coefficients()
        return out.__repr__()

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
        Load the coefficients from a human-readable text file.

        Parameters
        ----------
        name : string
         A valid file name for a properly formated document.
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
        Parse `~.coeffs` into useful numpy arrays.
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

class Krivanek(Distortions):
    """
    Krivanek polynomial
    -------------------
    $ K_{nm} = \rho^n / {n+1} e^{im\phi} = A_{nm} + i B_{nm} $
    """

    def __call__(self, img=None, gpu=False, *args, **kwargs):
        """
        Apply distortion to image or unit circle.
        """

        if img is None:

            # show displacements in the unit circle
            rho, phi = _get_unit_circle(*args, **kwargs)
            D = self.get_displacement(rho, phi)
            return ComplexSignal2D(D)

        if not gpu:

            # apply distortions to this image
            new_img_data = self._apply_distortion_cpu(img, *args, **kwargs)

            return img.__class__(new_img_data)

    def get_displacement(self, rho, phi):

        n, m, Cp = self.get_n_m_Cp()
        odd = m < 0
        evn = ~ odd

        # from radial
        radial = (rho[...,None]**n)/(n+1)
        K = np.zeros_like(radial, dtype=np.complex64)

        K[...,evn] =    radial[...,evn]*np.cos(phi[...,None]*m[evn])   # Anm
        K[...,odd] = 1j*radial[...,odd]*np.sin(phi[...,None]*m[odd])   # Bnm

        # the total displacement is obtained
        D = np.dot(K, Cp)

        return D

    def _apply_distortion_cpu(self, img, *args, **kwargs):
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

        # update image coordinates with displacement
        D = self.get_displacement(rho, phi)

        R += 0.5*(Nx+1j*Ny)
        R -= D

        # map interpolation
        r_map = np.stack([R.real, R.imag], 0)
        return map_coordinates(img.data, r_map, *args, **kwargs)
