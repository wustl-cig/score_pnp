"""
DPIR method for PnP image deblurring.
====================================================================================================

This example shows how to use the DPIR method to solve a PnP image deblurring problem. The DPIR method is described in
the following paper:
Zhang, K., Zuo, W., Gu, S., & Zhang, L. (2017). 
Learning deep CNN denoiser prior for image restoration. 
In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).
"""

import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.models import DRUNet, DnCNN
from deepinv.sampling import DiffPIR, DPS
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import optim_builder
from deepinv.training import test
from torchvision import transforms
from deepinv.optim.dpir import get_params
from deepinv.utils.demo import load_dataset, load_degradation
from tqdm import tqdm
import numpy as np
import os
from torchmetrics.image.fid import FrechetInceptionDistance as FID

from deepinv.optim.prior import RED
from deepinv.utils.parameters import get_GSPnP_params

from deepinv.loss.metric import PSNR, SSIM, LPIPS
from torchvision.utils import save_image

from util.data import ImageDataset
from util.diffusion_utils import compute_alpha, get_betas, find_nearest_del

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_vp_model
from data.dataloader import get_dataset, get_dataloader
from util.tweedie_utility import tween_noisy_training_sample, get_memory_free_MiB, mkdir_exp_recording_folder,clear_color, mask_generator, load_yaml, get_noiselevel_alphas_timestep
from util.logger import get_logger
import torchvision.transforms.functional as F

def second_red(noise_level_img, max_iter, denoising_strength_sigma_at_begin, denoising_strength_sigma_at_end, lamb, gamma, zeta,
                           denoiser_network_type, save_image_dir, dataset_name, iterative_algorithms, operation,
                           scale_factor, batch_size, kernel_index, kernel_dir, img_size, plot_images, plot_convergence_metrics, gpu, device, pretrained_check_point, dataset_dir, 
                           diffusion_config=None):

    essential_parameter_dict = {"noise_level_img": noise_level_img, "max_iter": max_iter, "denoising_strength_sigma_at_begin": denoising_strength_sigma_at_begin, "denoising_strength_sigma_at_end": denoising_strength_sigma_at_end, "lamb": lamb, "gamma": gamma, "denoiser_network_type": denoiser_network_type, "save_image_dir": save_image_dir, "iterative_algorithms": iterative_algorithms, "operation": operation, "scale_factor": scale_factor, "kernel_index": kernel_index}

    BASE_DIR = Path(save_image_dir)
    DATA_DIR = BASE_DIR / "measurements"
    torch.manual_seed(0)
    
    # ------------
    # (Step 1) Declare dataset
    # ------------
    val_transform = transforms.Compose(
        [
            transforms.CenterCrop(img_size), 
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    if dataset_name == "imagenet":
        dataset = ImageDataset(os.path.join(dataset_dir),
                os.path.join(dataset_dir, 'imagenet_val.txt'),
                image_size=img_size,
                normalize=False)
    else:
        raise ValueError("Given dataset is not yet implemented.")
    n_images_max = len(dataset)

    # ------------
    # (Step 2) Declare model
    # ------------
    if (denoiser_network_type == "score") and (diffusion_config is not None):
        score = create_vp_model(**diffusion_config)
        score = score.to(device)
        score.eval()
    elif denoiser_network_type == "dncnn":
        dncnn_denoiser = DnCNN(pretrained=pretrained_check_point, device=device)

    elif denoiser_network_type == "drunet":
        drunet_denoiser = DRUNet(pretrained=pretrained_check_point, device=device)

    else:
        raise ValueError("Given noise perturbation type is not existing.")
    # ------------
    # (Step 3) Inverse problem setup & data processing
    # ------------
    if operation == "deblur":
        DEBLUR_KER_DIR = Path(kernel_dir)
        kernel_torch = load_degradation(name="Levin09.npy", data_dir = DEBLUR_KER_DIR, index=kernel_index, download=False)
        kernel_torch = kernel_torch.unsqueeze(0).unsqueeze(0)  # add batch and channel dimensions
        # --------------------------------------------------------------------------------
        # We use the BlurFFT class from the physics module to generate a dataset of blurred images.
        # Use parallel dataloader if using a GPU to fasten training,
        # otherwise, as all computes are on CPU, use synchronous data loading.
        n_channels = 3  # 3 for color images, 1 for gray-scale images
        p = dinv.physics.BlurFFT(
            img_size=(n_channels, img_size, img_size),
            filter=kernel_torch,
            device=device,
            noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
        )
    else:
        raise ValueError("Given operation is not yet implemented.")

    data_preprocessing = dinv.datasets.generate_dataset_in_memory(
        train_dataset=dataset,
        physics=p,
        device=device,
        train_datapoints=n_images_max,
        batch_size=batch_size,
        supervised=True,
    )
    dataset = dinv.datasets.InMemoryDataset(data_store=data_preprocessing, train=True)
    data_fidelity = L2()
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False
    )
    
    if iterative_algorithms == "red":
        # ------------
        # (Step 4) Customized parameter setup for PnP
        # ------------
        # ! HERE IS POTENTIAL IMPROVEMENT I CAN MAKE.
        early_stop = False  # Stop algorithm when convergence criteria is reached
        # smaller than thres_conv
        thres_conv = 1e-5
        backtracking = True
        use_bicubic_init = False  # Use bicubic interpolation to initialize the algorithm
        
        sigma_denoiser = np.logspace(np.log10(denoising_strength_sigma_at_begin/255.), np.log10(denoising_strength_sigma_at_end/255.), max_iter).astype(np.float32)
        tau = 1/(gamma)
        
        params_algo = {
            "stepsize": tau,
            "g_param": sigma_denoiser,
            "lambda": gamma,
        }
        # The GSPnP prior corresponds to a RED prior with an explicit `g`.
        # We thus write a class that inherits from RED for this custom prior.
        class GSPnP(RED):
            r"""
            Gradient-Step Denoiser prior.
            """
            def __init__(self, is_diffusion_model,diffusion_config, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.explicit_prior = True

            def g(self, x, *args, **kwargs):
                r"""
                Computes the prior :math:`g(x)`.

                :param torch.tensor x: Variable :math:`x` at which the prior is computed.
                :return: (torch.tensor) prior :math:`g(x)`.
                """
                return self.denoiser.potential(x, *args, **kwargs)

        method = "GSPnP"
        denoiser_name = "gsdrunet"

        # ! HERE my customization: path of pretrained
        if denoiser_network_type in ["score"]:
            prior = GSPnP(denoiser=score, is_diffusion_model = True, diffusion_model_type = denoiser_network_type, diffusion_config=diffusion_config, device = device)
        elif denoiser_network_type == "dncnn":
            prior = GSPnP(denoiser=dncnn_denoiser, is_diffusion_model = False, diffusion_config = False)
        elif denoiser_network_type == "drunet":
            prior = GSPnP(denoiser=drunet_denoiser, is_diffusion_model = False, diffusion_config = False)
        else:
            raise ValueError("Check the denoiser_network_type")

        # This function is given by the deepinv we want to output the intermediate PGD update to finish with a denoising step.
        def custom_output(X):
            return X["est"][1]

        model = optim_builder(
            iteration="PGD",
            prior=prior,
            g_first=True,
            data_fidelity=data_fidelity,
            params_algo=params_algo,
            early_stop=early_stop,
            max_iter=max_iter,
            crit_conv="cost",
            thres_conv=thres_conv,
            backtracking=backtracking,
            get_output=custom_output,
            verbose=False,
        )
        
    else:
        raise ValueError("Check the iterative_algorithms")

    model.eval()

    folder_name = iterative_algorithms + f"_{denoiser_network_type}"
    save_folder = BASE_DIR / folder_name
    
    
    metric_log = test(
        model=model,
        test_dataloader=dataloader,
        physics=p,
        metrics=[dinv.loss.PSNR(), dinv.loss.SSIM(), dinv.loss.LPIPS(device = device)],
        device=device,
        plot_images=plot_images,
        save_folder=save_folder,
        plot_convergence_metrics=plot_convergence_metrics,
        verbose=True,
        essential_parameter_dict = essential_parameter_dict
    )
    average_psnr_input = metric_log['PSNR no learning']
    average_ssim_input = metric_log['SSIM no learning']
    average_lpips_input = metric_log['LPIPS no learning']
    average_psnr_recon = metric_log['PSNR']
    average_ssim_recon = metric_log['SSIM']
    average_lpips_recon = metric_log['LPIPS']
    
    formatted_recon_psnr_avg = f"{average_psnr_recon:.4f}"#.zfill(4)
    formatted_recon_ssim_avg = f"{average_ssim_recon:.4f}"#.zfill(4)
    formatted_recon_lpips_avg = f"{average_lpips_recon:.4f}"#.zfill(4)
    formatted_input_psnr_avg = f"{average_psnr_input:.4f}"#.zfill(4)
    formatted_input_ssim_avg = f"{average_ssim_input:.4f}"#.zfill(4)
    formatted_input_lpips_avg = f"{average_lpips_input:.4f}"#.zfill(4)
    
    print(f"# ------------")
    print(f"# {iterative_algorithms}({denoiser_network_type})- configuration: num_iters: {max_iter} / gamma: {gamma}")
    print(f"# [Input] PSNR: {average_psnr_input} / SSIM: {average_ssim_input} / LPIPS: {average_lpips_input}")
    print(f"# [Recon] PSNR: {average_psnr_recon} / SSIM: {average_ssim_recon} / LPIPS: {average_lpips_recon}")
    print(f"# Check out experiment at {save_folder}")
    print(f"# ------------")