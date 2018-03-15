def GFTIE(self, phase=None, dChi2=1e-3, delta=0.75, beta=0.97, Niters=100,
          Rlp=None, return_phase=True, lothres=1., hithres=0.4, alpha=None,
          do_norm=True, filter_type='butter', Nbutt=10, SNR=32., defoci=None,
          nref=None, *args, **kw):
    """
    Will use gradient flipping to refine a given phase, if none given, will
    precalculate the phase using MF-TIE. Gradient flipping works by
    reversing the sign of the delta percentile from the total number of
    pixels in the absolute value gradient vector G [1]. This version
    supports multi-focus focal series.

    Parameters
    ----------
    phase : Hyperspy Image
        None by default. If not provided, the multi_focus_TIE method will be
        called to initialize it, with *args and **kwargs parameters.
    dChi2 : float
        Chi-square precision parameter. The GF runs until the Chi-square
        phase change between iterations is smaller than this number.
    delta : float
        Used to calculate the flip threshold for the gradient as delta times
        the maximum value of that gradient. It has to be between [0-1].
    beta : float
        Factor applied with the flipping. Set it close but smaller than 1.
    Niters : int
        Maximum number of GF iterations.
    lothres, hithres, alpha, do_norm, filter_type, Nbutt, SNR
        Parameters used to generate a multi-focus inverse laplacian operator
        as donuts with low- and high-pass filters set to minimize non-linear
        and noise induced artifacts.

    Returns
    -------
    new_phase: Hyperspy Image
        Like the input phase (or the result from MF-TIE if not provided),
        but hopefully with a sizable amount of sensible flipping to it.

    References
    ----------
        [1] A. Parvizi, W. Van den Broek and C.T. Koch ASCI (2016) 2:3 also,
            Optics express 24 (2016) 8344.
    """
    # Real space parameters
    wnum = 2.*np.pi / self.metadata.ModImage.wave_length
    inv_wnum = 1. / wnum
    Nz, Ny, Nx = self.data.shape
    Nz_half = np.ceil(Nz/2).astype('int')
    if defoci is None:
        defoci = self.axes_manager.navigation_axes[0].axis
    adef = np.abs(defoci[Nz_half:])

    # K-space parameters
    k2, kx, ky = self.get_fourier_space('both')
    kvec = (kx[None,:], ky[:,None])

    # In the middle of the focal series is the reference plane
    Nz_half = np.ceil(Nz/2).astype('int') # actually, in Nz_half-1

    # set refractive index
    if nref is None:
        if self.metadata.has_item('ModImage.refractive_index'):
            nref = self.metadata.ModImage.refractive_index
        else:
            nref = 1.

    # Pad play 1:
    m_pad = np.ones([Ny, Nx], dtype=bool)
    # Select mask for pad area of padded image
    if self.metadata.Signal.has_item('pad_tuple'):
        Npy, Npx = self.metadata.Signal.pad_tuple
        m_pad[Npy[0]:(Ny-Npy[1]), Npx[0]:(Nx-Npx[1])] = False
    else:
        raise ValueError('No pad_tuple found!')

    # Set High and Low threshold values in k-space
    mean_size = np.mean([2.*np.pi/(np.sqrt(dk2i)) for dk2i in (k2[1,0], k2[0,1])])
    sigma_over_mu = 10.**(-SNR/10.)
    Anoise = np.pi*(wnum*sigma_over_mu*mean_size/(adef*2.*np.pi))**2
    k2min  = np.min([k2[0,1], k2[1,0]])
    rmse2n = Anoise[-1] / k2min
    k2Hi = wnum*nref*np.sqrt(8*wnum*nref*hithres/adef)
    k2Lo = k2Hi / (lothres*rmse2n*k2Hi/Anoise + 1.)

    # Calculate laplacian
    if filter_type is 'gauss':
        nabla_minus_2_tie = (1. / k2) * np.exp(- k2 / k2Hi[:,None,None]) * \
                              (1. - np.exp(- k2 / k2Lo[:,None,None]))
    elif filter_type is 'butter':
        nabla_minus_2_tie = (1. / k2) * (1./(1.+(k2/k2Hi[:,None,None])**Nbutt)) * \
                                 (1.-1./(1.+(k2/k2Lo[:,None,None])**Nbutt))
    nabla_minus_2_tie[:, 0, 0] = 1.
    nabla_minus_2_gfp = nabla_minus_2_tie.copy()

    # Normalization function
    if do_norm:
        # should we use also convergence?
        if (alpha is None):
            if self.metadata.has_item('ModImage.convergence_semiangle'):
                alpha = self.metadata.ModImage.convergence_semiangle

        if alpha is not None:
            envelope = np.exp(- k2 * ((0.5*adef*alpha)**2)[:,None,None])
            fnorm = (np.sum(nabla_minus_2_tie * envelope, 0) * k2)[None,:,:]
            fnorm[fnorm < 1.] = 1.
            #nabla_minus_2_gfp /= np.abs(fnorm)
            nabla_minus_2_tie /= np.sqrt(fnorm)
        else:
            fnorm = np.sum(nabla_minus_2_tie, 0) * k2
            fnorm[fnorm < 1.] = 1.
            #nabla_minus_2_gfp /= np.abs(fnorm[None,:,:])
            nabla_minus_2_tie /= np.sqrt(fnorm[None,:,:])

    # Set Low-pass filters for GF-ing using Rlp
    #k2gauss = (2. * np.pi / Rlp)**2.
    #h_lp_str = 'Rlp = '+str(np.round(Rlp,4))+' um'
    #hlopass = np.exp(- k2 / k2gauss)
    #hhipass = 1. - hlopass # these do not need further normalization

    # A: Set Low-pass filters for GF-ing using Rlp
    rmse2n = Anoise[-1] / (2. * np.pi / Rlp)**2.
    k2gauss = k2Hi / (rmse2n*k2Hi/Anoise + 1.) # like k2Lo, but lothres=1
    k2gauss.fill(k2gauss[-1])
    Rlp_print = 2.*np.pi / np.sqrt(k2gauss[-1])
    h_lp_str = 'Rlp = '+str(np.round(Rlp_print,4))+' um'
    hlopass = np.exp(- k2 / k2gauss[:,None,None])
    hhipass = 1. - hlopass # these do not need further normalization

    # GF init
    # Apply hi-pass filter to experimental images, only once
    #Dexp = (self.data[Nz_half:,:,:] - self.data[Nz_half-2::-1,:,:]) / \
    #       (2.*adef[:,None,None])
    #ft_Dexp = hhipass[None,:,:] * np.fft.fft2(Dexp)
    #hp_Dexp = np.fft.ifft2(ft_Dexp).real

    # A: Apply hi-pass filter to experimental images, only once
    Dexp = (self.data[Nz_half:,:,:] - self.data[Nz_half-2::-1,:,:]) / \
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
    I0 = self.data[Nz_half-1,:,:]

    # Initialize D
    D = np.zeros_like(Dexp)

    # Precalculate chi-2 normalization
    chi2norm = 1. / np.sum(np.abs(Dexp[:, ~m_pad]), axis=None)

    iters = 1
    keep_flippin = True
    chi2old = 10.
    G_list = [G.copy(),]
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

        # Update D, in measured area
        #D[:, ~m_pad] = hp_Dexp[:, ~m_pad] + \
        #    np.fft.ifft2(hlopass*np.fft.fft2(Dflip)).real[None, ~m_pad]

        # A: Update D, in measured area
        D[:, ~m_pad] = hp_Dexp[:, ~m_pad] + \
            np.fft.ifft2(hlopass*np.fft.fft2(Dflip)).real[:, ~m_pad]

        # Chi-2 error in the measured area
        chi2 = np.sum((D[:, ~m_pad] - Dexp[:, ~m_pad])**2, axis=None) * chi2norm

        delta_chi2 = np.abs(chi2-chi2old) / chi2old

        clear_output(True)
        print("GF with", h_lp_str, ", Iter = ", iters)
        print("Chi2 = ", chi2, ", dChi2 = ", delta_chi2)

        if (delta_chi2 > dChi2):
            iters += 1
            chi2old = chi2
            # Update G for next iter
            # Invert Gradient, with sum in between
            nm2_D = 1j * nabla_minus_2_gfp * np.fft.fft2(D)
            G = np.stack(
                    [wnum * np.fft.ifft2(ki[None,...] * nm2_D).real / I0[None,...] \
                     for ki in kvec[::-1]])
            G_list.append(G.copy())
        else:
            print("Gradient Flipped !")
            keep_flippin = False

    if return_phase:
        # Get new phase from D, by solving MFTIE
        nm2_D = nabla_minus_2_tie * np.fft.fft2(D)
        Grad = [np.fft.fft2(np.fft.ifft2(ki * nm2_D) / I0 ) for ki in kvec]
        lm2_dz = np.sum(nabla_minus_2_tie * (kvec[0]*Grad[0]+kvec[1]*Grad[1]),0)
        phase_gf = wnum * np.real(np.fft.ifft2(lm2_dz)).astype('float32')
        phase_gf = self._get_signal_signal(phase_gf)
        phase_gf.metadata = self.metadata.copy()
        return phase_gf
    else:
        # Return scaled divergence, can be used together with MFTIE.
        return wnum*D, G_list, k2gauss
