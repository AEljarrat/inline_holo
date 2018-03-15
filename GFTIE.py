""" Warning! this module is experimental... 
"""
import numpy as np
from tqdm import tqdm

def GFTIE(fs, phase, Niters=5, Rlp=5, dChi2=1e-6, delta=0.9, beta=0.85,
          defoci=None):
    """
    Use a gradient flipping regularization to refine a MFTIE phase result.
    Gradient flipping works by reversing the sign of the delta percentile from
    the total number of pixels in the absolute value gradient [1, 2].
    The information is used to update the through-focus derivative only at low
    spatial frequencies.

    Parameters
    ----------
    fs : ModifiedImage
     Input focal-series.
    phase : ModifiedImage
     Input phase.
    Niters : int
     Maximum number of GF iterations.
    Rlp : float
     Used to set the low-pass filtering.
    dChi2 : float
     Chi-square precision parameter. The GF runs until the Chi-square
     phase change between iterations is smaller than this number.
    delta : float
     Used to calculate the flip threshold for the gradient as delta times
     the maximum value of that gradient. It has to be between [0-1].
    beta : float
     Factor applied with the flipping. Set it close but smaller than 1.

    Returns
    -------
    flip_dzI : Hyperspy Image
     Regularized through-focus derivative that can be used to invert the TIE +
     using the multi-focus TIE algorithm.

    Example
    -------
    >>> init_mftie = MFTIE(fs, True, 1., 0.25, 0.003, SNR)
    >>> init_phase = init_mftie(False)
    >>> flip_dzI   = GFTIE(fs, init_phase)
    >>> flip_mftie = MFTIE(fs, True, 1e5, 0.25, 0.003, SNR)
    >>> flip_mftie.set_dzI(flip_dzI)
    >>> flip_phase = flip_mftie(False)
    >>> hs.plot.plot_images([init_phase, flip_phase])

    References
    ----------
    [1] A. Parvizi, E. Van den Broek, and C. T. Koch. “Recovering low spatial
    frequencies in wavefront sensing based on intensity measurements”, Advanced
    Structural and Chemical Imaging, 2:3 (2016)
    [2] A. Parvizi, W. Van den Broek, and C. T. Koch. “Gradient flipping
    algorithm: introducing non-convex constraints in wavefront reconstructions
    with the transport of intensity equation”, Opt. Express , 248:8344 (2016)
    """

    # Real space parameters
    wnum = 2.*np.pi / fs.metadata.ModImage.wave_length
    inv_wnum = 1. / wnum
    Nz, Ny, Nx = fs.data.shape
    Nz_half = np.ceil(Nz/2).astype('int')
    if defoci is None:
        defoci = fs.axes_manager.navigation_axes[0].axis
    adef = np.abs(defoci[Nz_half:])

    # K-space parameters
    k2, kx, ky = fs.get_fourier_space('both')
    kvec = (kx[None,:], ky[:,None])

    # In the middle of the focal series is the reference plane
    Nz_half = np.ceil(Nz/2).astype('int') # actually, in Nz_half-1

    # Pad play:
    m_pad = np.ones([Ny, Nx], dtype=bool)
    # Select mask for pad area of padded image
    if fs.metadata.Signal.has_item('pad_tuple'):
        Npy, Npx = fs.metadata.Signal.pad_tuple
        m_pad[Npy[0]:(Ny-Npy[1]), Npx[0]:(Nx-Npx[1])] = False
    else:
        raise ValueError('No pad_tuple found!')

    # Calculate inverse laplacian
    nabla_minus_2 = (1. / np.repeat(k2[None,...], len(adef), 0) )
    nabla_minus_2[:, 0, 0] = 1.

    # Calculate low-pass filters
    k2gauss = np.zeros_like(adef)
    k2gauss.fill(2. * np.pi / Rlp)
    hlopass = np.exp(- k2 / k2gauss[:,None,None])
    hhipass = 1. - hlopass
    # ! these filters do not need further normalization

    # Apply hi-pass filter to experimental images, only once
    Dexp = (fs.data[Nz_half:,:,:] - fs.data[Nz_half-2::-1,:,:]) / \
           (2.*adef[:,None,None])
    ft_Dexp = hhipass * np.fft.fft2(Dexp)
    hp_Dexp = np.fft.ifft2(ft_Dexp).real

    # Initialize the phase
    phdata = phase.data.copy()
    sigaxs = phase.axes_manager.signal_axes[::-1]
    daxi = [axi.scale for axi in sigaxs]

    # Initialize G(phase) = grad(phase)
    G = np.gradient(phdata, *daxi, edge_order=2, axis=(-2,-1))
    G = [gi[None,...].repeat(len(adef),0) for gi in G]
    G = np.stack(G)

    # Initialize reference plane, I0
    I0 = fs.data[Nz_half-1,:,:]

    # Initialize D
    D = np.zeros_like(Dexp)

    # Precalculate chi-2 normalization
    chi2norm = 1. / np.sum(np.abs(Dexp[:, ~m_pad]), axis=None)

    iters = 1
    keep_flippin = True
    chi2old = 10.
    G_list = [G.copy(),]

    print('Starting GF; Rlp = '+str(np.round(Rlp, 4))+' um')
    pbar_desc = '\n'
    pbar = tqdm(pbar_desc, total=Niters)

    while (keep_flippin is True) and (iters <= Niters):
        # G-Flippin':
        # Calculate Pdelta for all pixels in each defocus
        Pdelta = np.percentile(np.abs(G), delta*100., (0,2,3))

        # Apply them
        flipi = np.abs(G) < Pdelta[None,:,None,None]
        G[flipi] = - beta * G[flipi]

        # Compute D(G) = - div(I0*G) / ko
        Dflip = - inv_wnum * (
          np.gradient(G[0,...]*I0[None,...], daxi[0], edge_order=2, axis=-2) + \
          np.gradient(G[1,...]*I0[None,...], daxi[1], edge_order=2, axis=-1))

        # Update D, will only be saved in pad area
        D[:, m_pad] = Dflip[:, m_pad]

        # A: Update D, in measured area
        D[:, ~m_pad] = hp_Dexp[:, ~m_pad] + \
            np.fft.ifft2(hlopass*np.fft.fft2(Dflip)).real[:, ~m_pad]

        # Chi-2 error in the measured area
        chi2 = np.sum((D[:, ~m_pad] - Dexp[:, ~m_pad])**2, axis=None) * chi2norm

        delta_chi2 = np.abs(chi2-chi2old) / chi2old

        pbar.update(1)
        pbar.desc = 'Chi2 = ' + str(chi2) + '; dChi2 = ' + str(delta_chi2) + pbar_desc

        if (delta_chi2 > dChi2):
            iters += 1
            chi2old = chi2
            # Update G for next iter
            # Invert Gradient, with sum in between
            nm2_D = 1j * nabla_minus_2 * np.fft.fft2(D)
            G = np.stack(
                    [wnum * np.fft.ifft2(ki[None,...] * nm2_D).real / I0[None,...] \
                     for ki in kvec[::-1]])
            G_list.append(G.copy())
        else:
            print("Gradient Flipped !")
            keep_flippin = False

    # Return scaled divergence, can be used together with MFTIE.
    #return wnum*D, G_list, k2gauss
    return wnum*D
