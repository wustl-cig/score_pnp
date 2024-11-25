from deepinv.optim import BaseOptim
from deepinv.models import DRUNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import create_iterator
import numpy as np

def get_PnPADMM_params(noise_level_img, max_iter, S1_noise_level_scaling_factor, S2_noise_level_scaling_factor, lamb):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    # max_iter = 8
    # max_iter = 50
    # s1 = 49.0 / 255.0
    s1 = noise_level_img * S1_noise_level_scaling_factor
    # s1 = 2555.0 / 255.0
    s2 = noise_level_img * S2_noise_level_scaling_factor
    sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), max_iter).astype(
        np.float32
    )
    stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2

    return lamb, list(sigma_denoiser), list(lamb * stepsize), max_iter

# def get_PnPADMM_params(noise_level_img):
#     r"""
#     Default parameters for the DPIR Plug-and-Play algorithm.

#     :param float noise_level_img: Noise level of the input image.
#     """
#     # ! BEST super-resolution vp socre 25,55.0/255.0, 0.8/0.23
#     # max_iter = 8
#     max_iter = 25
#     # s1 = 55.0 / 255.0
#     s1 = 20.0 / 255.0
#     # s1 = noise_level_img
#     s2 = noise_level_img
#     sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), max_iter).astype(
#         np.float32
#     )
#     print(f"sigma_denoiser: {sigma_denoiser}")
    
    
#     stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2
#     # lamb = 0.5 / 0.23
#     lamb = 0.8 / 0.23
#     return lamb, list(sigma_denoiser), list(lamb * stepsize), max_iter