""" This module includes some scripts to work with the ModifiedImage class.
The objective of the module is to incubate functions to maybe later develop
functionalities of the ModifiedImage class. Or just slacker programming :P """

import numpy as np

def plot_non_linear(self):
    """ To plot ModImages with non-linear defocus axis."""
    s = self.deepcopy()
    axi = s.axes_manager[0]
    axi.axis = axi.scale * np.arange(axi.size) + axi.offset
    s.get_dimensions_from_data()
    s.plot()

def shifter(self, unset_padding=True):
    """
    Shifts the intensities of an image to make the mean == 0

    Parameters
    ----------
    unset_padding : bool
     Decide if the pad area should be excluded. Only works if the unset_padding
     method is available to self.
    """
    if unset_padding:
        s = self.unset_padding()
    else:
        s = self.deepcopy()
    s.data = s.data - s.data.mean()
    return s

def get_middle(self, axis=0):
    """Returns the middle index of the first or provided axis"""
    return np.ceil(self.data.shape[axis]/2.).astype('int') - 1

def add_noise_gauss(data, SNR, seed=None):
    """ Additive uncorrelated gaussian noise.
    Loosely follows: SNR = 10.* log10(mean(signal)/s(noise))
    """
    if seed:
        np.random.seed(seed)
    N = data.shape
    mean_signal = data.mean()
    sigma_snr = mean_signal / 10.**(SNR/10.)
    noise = np.random.normal(0., sigma_snr, np.prod(N)).reshape(N)
    return noise

def add_noise_poiss(data, SNR, seed=None):
    # determine number of counts depending on SNR
    Nsnr = np.exp(SNR/5.) / data.mean()
    # random name generator baby
    rs=np.random.RandomState(seed=seed)
    return rs.poisson(data * Nsnr) / Nsnr
