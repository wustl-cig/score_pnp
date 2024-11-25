import numpy as np


def get_GSPnP_params(problem, noise_level_img):
    r"""
    Default parameters for the GSPnP Plug-and-Play algorithm.

    :param str problem: Type of inverse-problem problem to solve. Can be ``deblur``, ``super-resolution``, or ``inpaint``.
    :param float noise_level_img: Noise level of the input image.
    """
    if problem == "deblur1":
        max_iter = 500
        sigma_denoiser = 1.8 * noise_level_img
        lamb = 0.1
        # sigma_denoiser = 2.0 * noise_level_img
        # lamb = 0.065
    elif problem == "deblur":
        max_iter = 30
        sigma_denoiser = 1.8 * noise_level_img
        lamb = 0.065
        # BEST: 1.8 / 0.065
        # sigma_denoiser = 2.0 * noise_level_img
        # lamb = 0.065
    elif problem == "super-resolution1":
        max_iter = 500
        sigma_denoiser = 2.0 * noise_level_img
        lamb = 0.065
    # ! HERE I ADD THE MODIFICATION HERE
    elif problem == "super-resolution":
        max_iter = 30
        sigma_denoiser = 2.0 * noise_level_img
        lamb = 0.065
    elif problem == "inpaint":
        max_iter = 100
        sigma_denoiser = 10.0 / 255
        lamb = 0.1
    else:
        raise ValueError("parameters unknown with this degradation")
    stepsize = 1 / lamb

    max_iter = 25
    # max_iter = 50
    # s1 = 49.0 / 255.0
    s1 = noise_level_img * 6.0
    # s1 = 2555.0 / 255.0
    s2 = noise_level_img * 2.0
    sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), max_iter).astype(
        np.float32
    )
    # stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2
    # stepsize
    # lamb = 0.065
    # lamb = 0.5 / 0.23
    return lamb, list(sigma_denoiser), stepsize, max_iter
    # return lamb, sigma_denoiser, stepsize, max_iter
