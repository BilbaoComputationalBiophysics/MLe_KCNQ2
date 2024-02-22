from copy import deepcopy
import numpy as np
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter1d

def regular_mti(F, max_window=31, alpha=1e-6):

    iterations = (max_window - 1) // 2
    MTI = F / (F + alpha)

    for _ in range(iterations):

        MTI_prev = deepcopy(MTI)

        for position in range(len(F)):

            if position == 0:
                MTI[position] = MTI_prev[:2].mean()
            
            elif position == len(F):
                MTI[position] = MTI_prev[-2:].mean()

            else:
                MTI[position] = MTI_prev[position-1:position+2].mean()
                    
    return MTI

def gaussian_convolution_vfi(F, sigma: int=5, max_window: int=31, alpha: float=1e-7) -> np.array:
    """Computes the VFI for a sequence with allele frequencies given by F.
    """
    
    F_norm = F / (F + alpha)
    center = max_window // 2

    x = np.arange(max_window)
    gaussian_kernel = np.exp(-0.5 * ((x - x[center]) / (sigma))**2)
    gaussian_kernel /= gaussian_kernel.sum()

    VFI = gaussian_filter1d(F_norm, sigma=sigma)

    return VFI
