from deepinv.optim import BaseOptim
from deepinv.models import DRUNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import create_iterator
import numpy as np

def get_params(noise_level_img, max_iter, S1_noise_level_scaling_factor, S2_noise_level_scaling_factor, lamb, iterative_algorithms, denoiser_network_type):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    # max_iter = 50
    # s1 = 49.0 / 255.0
    # S2_noise_level_scaling_factor = S1_noise_level_scaling_factor
    # if denoiser_network_type == 'cnn':
    #     s1 = 2./255.
    # else:
    #     s1 = 2./255.
    #     # s1 = noise_level_img * S1_noise_level_scaling_factor
        
    # s2 = s1
    if denoiser_network_type == 'vp_score_fix':
        s1 = S1_noise_level_scaling_factor/255.0
        s2 = S1_noise_level_scaling_factor/255.0
    else:
        if denoiser_network_type == 'cnn':
            if iterative_algorithms == 'dpir':
                s1 = S1_noise_level_scaling_factor/255.0
                s2 = noise_level_img#/255.0
            else:
                s1 = S1_noise_level_scaling_factor/255.0
                s2 = S2_noise_level_scaling_factor/255.0
        else:
            s1 = S1_noise_level_scaling_factor/255.0
            s2 = S2_noise_level_scaling_factor/255.0
    
    if (s1 != s2) and (denoiser_network_type == 'cnn') and (iterative_algorithms != 'dpir'):
        raise ValueError("Noise is scheduled on not scalable denoiser")

    if s2 > s1:
        raise ValueError("s1 is smaller than s2")
    
    sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), max_iter).astype(
        np.float32
    )
    
    # raise ValueError(f"sigma_denoiser: {sigma_denoiser}\nlen(sigma_denoiser): {len(sigma_denoiser)}")

    if iterative_algorithms == 'dpir':
        # stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2
        # return None, list(sigma_denoiser), list(lamb * stepsize), max_iter
        # max_iter = 8
        # s1 = 49.0 / 255.0
        # s2 = noise_level_img
        # s1 = S1_noise_level_scaling_factor/ 255.0
        # s2 = S2_noise_level_scaling_factor/ 255.0
        # if s2 > s1:
        #     raise ValueError("s1 is smaller than s2")
        # s2 = noise_level_img
        # sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), max_iter).astype(
        #     np.float32
        # )
        # sigma_denoiser = np.maximum(sigma_denoiser, 0.01)
        # raise ValueError(f"S1_noise_level_scaling_factor: {S1_noise_level_scaling_factor} \n S2_noise_level_scaling_factor: {S2_noise_level_scaling_factor}\n s1: {s1}, s2: {s2}")
        stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2
        # lamb = 1 / 0.23
        lamb = 1 / (lamb)
        return None, list(sigma_denoiser), list(lamb * stepsize), max_iter
    elif iterative_algorithms in ['pnpadmm', 'pnpista', 'pnpfista']:
        # stepsize = ((sigma_denoiser)**2) / lamb
        # if denoiser_network_type == 'vp_score_anneal':
        stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2
        lamb = 1 / (lamb)
        stepsize = list(lamb*stepsize)
        return lamb, list(sigma_denoiser), stepsize, max_iter
    elif iterative_algorithms == 'red':
        if denoiser_network_type == 'vp_score_anneal':
            stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2
            lamb = 1 / (lamb)
            stepsize = list(lamb*stepsize)
        else:
            # lamb = 1 / (lamb)
            # stepsize = lamb
            stepsize = 1 / (lamb)
            # stepsize = lamb
        return lamb, list(sigma_denoiser), stepsize, max_iter

def get_params_10202024(noise_level_img, max_iter, S1_noise_level_scaling_factor, S2_noise_level_scaling_factor, lamb, iterative_algorithms, denoiser_network_type):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    # max_iter = 50
    # s1 = 49.0 / 255.0
    S2_noise_level_scaling_factor = S1_noise_level_scaling_factor
    if denoiser_network_type == 'cnn':
        if noise_level_img * S1_noise_level_scaling_factor >= 50./255.:
            s1 = 50./255.
        else:
            s1 = noise_level_img * S1_noise_level_scaling_factor
    else:
        s1 = noise_level_img * S1_noise_level_scaling_factor
        
    s2 = s1

    sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), max_iter).astype(
        np.float32
    )

    if iterative_algorithms == 'dpir':
        stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2
        return None, list(sigma_denoiser), list(lamb * stepsize), max_iter
    elif iterative_algorithms in ['pnpadmm', 'pnpista', 'pnpfista']:
        stepsize = ((sigma_denoiser)**2) / lamb
        # stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2
        # return lamb, list(sigma_denoiser), list(lamb * stepsize), max_iter
        return lamb, list(sigma_denoiser), list(stepsize), max_iter
    elif iterative_algorithms == 'red':
        stepsize = 1/lamb
        return lamb, list(sigma_denoiser), stepsize, max_iter

def get_params1(noise_level_img, max_iter, S1_noise_level_scaling_factor, S2_noise_level_scaling_factor, lamb, iterative_algorithms, denoiser_network_type):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    # max_iter = 50
    # s1 = 49.0 / 255.0
    if denoiser_network_type == 'cnn':
        if noise_level_img * S1_noise_level_scaling_factor >= 50./255.:
            s1 = 50./255.
        else:
            s1 = noise_level_img * S1_noise_level_scaling_factor
    else:
        s1 = noise_level_img * S1_noise_level_scaling_factor
    # s1 = 2555.0 / 255.0
    if noise_level_img * S2_noise_level_scaling_factor >= s1:
        s2 = s1
    else:
        s2 = noise_level_img * S2_noise_level_scaling_factor
    sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), max_iter).astype(
        np.float32
    )
    if iterative_algorithms == 'dpir':
        stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2
        return None, list(sigma_denoiser), list(lamb * stepsize), max_iter
    elif iterative_algorithms == 'pnpadmm':
        stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2
        return lamb, list(sigma_denoiser), list(lamb * stepsize), max_iter
    elif iterative_algorithms == 'red':
        stepsize = 1/lamb
        return lamb, list(sigma_denoiser), stepsize, max_iter

def get_DPIR_params(noise_level_img, max_iter, S1_noise_level_scaling_factor, S2_noise_level_scaling_factor, lamb):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    # max_iter = 50
    # s1 = 49.0 / 255.0
    s1 = noise_level_img * S1_noise_level_scaling_factor
    # s1 = 2555.0 / 255.0
    s2 = noise_level_img * S2_noise_level_scaling_factor
    sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), max_iter).astype(
        np.float32
    )
    stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2

    return list(sigma_denoiser), list(lamb * stepsize), max_iter

def get_DPIR_params1(noise_level_img):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    # max_iter = 8
    max_iter = 25
    # max_iter = 50
    # s1 = 49.0 / 255.0
    s1 = 180.0 / 255.0
    # s1 = 2555.0 / 255.0
    s2 = noise_level_img
    sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), max_iter).astype(
        np.float32
    )
    stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2
    lamb = 1 / 0.23
    # lamb = 0.5 / 0.23
    return list(sigma_denoiser), list(lamb * stepsize), max_iter


class DPIR(BaseOptim):
    r"""
    Deep Plug-and-Play (DPIR) algorithm for image restoration.

    The method is based on half-quadratic splitting (HQS) and a PnP prior with a pretrained denoiser :class:`deepinv.models.DRUNet`.
    The optimization is stopped early and the noise level for the denoiser is adapted at each iteration.
    See :ref:`sphx_glr_auto_examples_plug-and-play_demo_PnP_DPIR_deblur.py` for more details on the implementation,
    and how to adapt it to your specific problem.

    This method uses a standard :math:`\ell_2` data fidelity term.

    The DPIR method is described in Zhang, K., Zuo, W., Gu, S., & Zhang, L. (2017). "Learning deep CNN denoiser prior for image restoration"
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).

    :param float sigma: Standard deviation of the measurement noise, which controls the choice of the
        rest of the hyperparameters of the algorithm. Default is ``0.1``.
    """

    def __init__(self, sigma=0.1, device="cuda"):
        prior = PnP(denoiser=DRUNet(pretrained="download", device=device))
        sigma_denoiser, stepsize, max_iter = get_DPIR_params(sigma)
        params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser}
        super(DPIR, self).__init__(
            create_iterator("HQS", prior=prior, F_fn=None, g_first=False),
            max_iter=max_iter,
            data_fidelity=L2(),
            prior=prior,
            early_stop=False,
            params_algo=params_algo,
        )
