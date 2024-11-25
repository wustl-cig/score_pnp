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
from util.diffusion_utils import compute_alpha, get_alphas, get_betas, find_nearest_del

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_vp_model
from data.dataloader import get_dataset, get_dataloader
from util.tweedie_utility import tween_noisy_training_sample, get_memory_free_MiB, mkdir_exp_recording_folder,clear_color, mask_generator, load_yaml, get_noiselevel_alphas_timestep
from util.logger import get_logger
import torchvision.transforms.functional as F

def second_diffpir(noise_level_img, max_iter, denoising_strength_sigma_at_begin, denoising_strength_sigma_at_end, lamb, gamma, zeta,
                           denoiser_network_type, save_image_dir, dataset_name, iterative_algorithms, operation,
                           scale_factor, batch_size, kernel_index, kernel_dir, img_size, plot_images, plot_convergence_metrics, gpu, device, pretrained_check_point, dataset_dir, 
                           diffusion_config=None):

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
    score = create_vp_model(**diffusion_config)
    score = score.to(device)
    score.eval()

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
    
    model = dinv.models.DiffUNet(large_model=True, pretrained=pretrained_check_point).to(device)

    model.eval()

    folder_name = iterative_algorithms + f"_{denoiser_network_type}"
    save_folder = BASE_DIR / folder_name
    
    # DiffPIR 
    T = 1000
    alphas = get_alphas(num_train_timesteps=T, device = device)
    sigmas = torch.sqrt(1.0 - alphas) / alphas.sqrt()
    
    diffusion_steps = 100  # Maximum number of iterations of the DiffPIR algorithm
    num_steps = diffusion_steps
    
    lambda_ = lamb  # Regularization parameter

    rhos = lambda_ * (noise_level_img**2) / (sigmas**2)

    # get timestep sequence
    seq = np.sqrt(np.linspace(0, T**2, diffusion_steps))
    seq = [int(s) for s in list(seq)]
    seq[-1] = seq[-1] - 1
    

    if not isinstance(dataloader, list):
        dataloader = [dataloader]

    current_iterators = [iter(loader) for loader in dataloader]

    batches = min([len(loader) - loader.drop_last for loader in dataloader])

    model.eval()

    psnr_func = PSNR()
    ssim_func = SSIM()
    fid_func = FID()
    lpips_func = LPIPS(device = device)
    
    sum_input_psnr = 0
    sum_input_ssim = 0
    sum_input_lpips = 0
    sum_recon_psnr = 0
    sum_recon_ssim = 0
    sum_recon_lpips = 0
    
    pbar = tqdm(range(batches), ncols=150)
    
    with torch.no_grad():
        for i in pbar:
            data_batch = [next(iterator) for iterator in current_iterators]
            
            for data in data_batch:
                if (type(data) is not tuple and type(data) is not list) or len(data) != 2:
                    raise ValueError(
                        "If online_measurements=False, the dataloader should output a tuple (x, y)"
                    )

                x_gt, y = data
                x_gt = [s.to(device) for s in x_gt] if isinstance(x_gt, (list, tuple)) else x_gt.to(device)
                y = p(x_gt.to(device))
                y = y.to(device)
                
                num_train_timesteps = 1000  # Number of timesteps used during training

                betas = get_betas(num_train_timesteps=num_train_timesteps, device=device)

                diffusion_steps = 100

                batch_size = 1

                seq = np.sqrt(np.linspace(0, T**2, diffusion_steps))
                seq = [int(s) for s in list(seq)]
                seq[-1] = seq[-1] - 1

                x0 = x_gt * 2.0 - 1.0
                y_dps = p(x0.to(device))
                x = y
                
                diff_model = DiffPIR(
                    model=model,
                    data_fidelity=data_fidelity,
                    max_iter = 100,
                    sigma = noise_level_img,
                    zeta = zeta,
                    lambda_ = lamb
                ) 

                x = diff_model(x, p)
                
                input_psnr = psnr_func(x_net = y, x = x_gt).item()
                input_ssim = ssim_func(x_net = y, x = x_gt).item()
                input_lpips = lpips_func(x_net = y, x = x_gt).item()
                recon_psnr = psnr_func(x_net = x, x = x_gt).item()
                recon_ssim = ssim_func(x_net = x, x = x_gt).item()
                recon_lpips = lpips_func(x_net = x, x = x_gt).item()
                sum_input_psnr += input_psnr
                sum_input_ssim += input_ssim
                sum_input_lpips += input_lpips
                sum_recon_psnr += recon_psnr
                sum_recon_ssim += recon_ssim
                sum_recon_lpips += recon_lpips
                
                formatted_noise_level_img = f"{noise_level_img:.3f}".zfill(4)
                formatted_recon_psnr = f"{recon_psnr:.3f}"#.zfill(4)
                formatted_recon_ssim = f"{recon_ssim:.3f}"#.zfill(4)
                formatted_recon_lpips = f"{recon_lpips:.4f}"#.zfill(4)
                formatted_input_psnr = f"{input_psnr:.3f}"#.zfill(4)
                formatted_input_ssim = f"{input_ssim:.3f}"#.zfill(4)
                formatted_input_lpips = f"{input_lpips:.4f}"#.zfill(4)
                formatted_lamb = f"{lamb:.4f}"#.zfill(4)
                formatted_zeta = f"{zeta:.4f}"#.zfill(4)
                
                title = f"{operation}_{iterative_algorithms}_iters_{max_iter}_mnoise_{formatted_noise_level_img}_denoiser_{denoiser_network_type}_DiffPIR_lamb_{formatted_lamb}_zeta_{formatted_zeta}_kernel_{kernel_index}_inputpsnr_{formatted_input_psnr}_inputssim_{formatted_input_ssim}_inputlpips_{formatted_input_lpips}_reconpsnr_{formatted_recon_psnr}_reconssim_{formatted_recon_ssim}_reconlpips_{formatted_recon_lpips}"
                gt_title = f"idx_{i}"
                
                recon_folder_name = f"{save_folder}/Reconstruction/"
                input_folder_name = f"{save_folder}/Measurement/"
                gt_folder_name = f"{save_folder}/Ground truth/"
                Path(recon_folder_name).mkdir(parents=True, exist_ok=True)
                Path(input_folder_name).mkdir(parents=True, exist_ok=True)
                Path(gt_folder_name).mkdir(parents=True, exist_ok=True)
                
                if plot_images == True:
                    save_image(x, recon_folder_name + f"{title}.png")
                    save_image(x_gt, gt_folder_name + f"{gt_title}.png")
                    save_image(y, input_folder_name + f"{title}.png")
                
                pbar.set_postfix({'input_psnr': f"{input_psnr:.2f}", 'recon_psnr': f"{recon_psnr:.2f}"}, refresh=False)
            
    average_psnr_input = sum_input_psnr / batches
    average_ssim_input = sum_input_ssim / batches
    average_lpips_input = sum_input_lpips / batches
    average_psnr_recon = sum_recon_psnr / batches
    average_ssim_recon = sum_recon_ssim / batches 
    average_lpips_recon = sum_recon_lpips / batches
    
    formatted_recon_psnr_avg = f"{average_psnr_recon:.4f}"#.zfill(4)
    formatted_recon_ssim_avg = f"{average_ssim_recon:.4f}"#.zfill(4)
    formatted_recon_lpips_avg = f"{average_lpips_recon:.4f}"#.zfill(4)
    formatted_input_psnr_avg = f"{average_psnr_input:.4f}"#.zfill(4)
    formatted_input_ssim_avg = f"{average_ssim_input:.4f}"#.zfill(4)
    formatted_input_lpips_avg = f"{average_lpips_input:.4f}"#.zfill(4)
    
    avg_title = f"{operation}_{iterative_algorithms}_mnoise_{formatted_noise_level_img}_denoiser_{denoiser_network_type}_kernel_{kernel_index}_inputpsnr_{formatted_input_psnr_avg}_inputssim_{formatted_input_ssim_avg}_inputlpips_{formatted_input_lpips_avg}_reconpsnr_{formatted_recon_psnr_avg}_reconssim_{formatted_recon_ssim_avg}_reconlpips_{formatted_recon_lpips_avg}"

    os.makedirs(save_folder, exist_ok=True)

    print(f"# ------------")
    print(f"# {iterative_algorithms}({denoiser_network_type})- configuration: num_iters: {max_iter} / lambda: {lamb} / zeta: {zeta}")
    print(f"# [Input] PSNR: {average_psnr_input} / SSIM: {average_ssim_input} / LPIPS: {average_lpips_input}")
    print(f"# [Recon] PSNR: {average_psnr_recon} / SSIM: {average_ssim_recon} / LPIPS: {average_lpips_recon}")
    print(f"# Check out experiment at {save_folder}")
    print(f"# ------------")
