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

def second_experiment(noise_level_img, max_iter, denoising_strength_sigma_at_begin, denoising_strength_sigma_at_end, lamb, gamma, zeta,
                           denoiser_network_type, save_image_dir, dataset_name, iterative_algorithms, operation,
                           scale_factor, batch_size, kernel_index, kernel_dir, img_size, plot_images, plot_convergence_metrics, gpu, device, pretrained_check_point, dataset_dir, 
                           diffusion_config=None):

    essential_parameter_dict = {"noise_level_img": noise_level_img, "max_iter": max_iter, "denoising_strength_sigma_at_begin": denoising_strength_sigma_at_begin, "denoising_strength_sigma_at_end": denoising_strength_sigma_at_end, "lamb": lamb, "denoiser_network_type": denoiser_network_type, "save_image_dir": save_image_dir, "iterative_algorithms": iterative_algorithms, "operation": operation, "scale_factor": scale_factor, "kernel_index": kernel_index}

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
    
    if iterative_algorithms == "dpir":
        # ------------
        # (Step 4) Customized parameter setup for PnP
        # ------------
        # _, sigma_denoiser, stepsize, max_iter = get_params(noise_level_img=noise_level_img, max_iter=max_iter, denoising_strength_sigma_at_begin=denoising_strength_sigma_at_begin, denoising_strength_sigma_at_end=denoising_strength_sigma_at_end, lamb=lamb, iterative_algorithms=iterative_algorithms, denoiser_network_type = denoiser_network_type)
        # params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser}
        sigma_denoiser = np.logspace(np.log10(denoising_strength_sigma_at_begin/255.), np.log10(denoising_strength_sigma_at_end/255.), max_iter).astype(np.float32)
        tau = ((sigma_denoiser / max(0.01, noise_level_img)) ** 2)*(1/(lamb))
        params_algo = {"stepsize": tau, "g_param": sigma_denoiser}
        
        

        early_stop = False  # Do not stop algorithm with convergence criteria
        if denoiser_network_type in ["score"]:
            prior = PnP(denoiser=score, is_diffusion_model = True, diffusion_model_type = denoiser_network_type, diffusion_config=diffusion_config, device = device)
        elif denoiser_network_type == "drunet":
            prior = PnP(denoiser=drunet_denoiser)
        else:
            raise ValueError("Check the denoiser_network_type")

        # instantiate the algorithm class to solve the IP problem.
        model = optim_builder(
            iteration="HQS",
            prior=prior,
            data_fidelity=data_fidelity,
            early_stop=early_stop,
            max_iter=max_iter,
            verbose=True,
            crit_conv = "cost",
            params_algo=params_algo,
        )
    elif iterative_algorithms == "pnpista":
        # ------------
        # (Step 4) Customized parameter setup for PnP
        # ------------
        # lamb, sigma_denoiser, stepsize, max_iter = get_params(noise_level_img=noise_level_img, max_iter=max_iter, denoising_strength_sigma_at_begin=denoising_strength_sigma_at_begin, denoising_strength_sigma_at_end=denoising_strength_sigma_at_end, lamb=lamb, iterative_algorithms=iterative_algorithms, denoiser_network_type = denoiser_network_type)
        # params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser}
        sigma_denoiser = np.logspace(np.log10(denoising_strength_sigma_at_begin/255.), np.log10(denoising_strength_sigma_at_end/255.), max_iter).astype(np.float32)
        tau = ((sigma_denoiser / max(0.01, noise_level_img)) ** 2)*(1/(lamb))
        params_algo = {"stepsize": tau, "g_param": sigma_denoiser}

        early_stop = False  # Do not stop algorithm with convergence criteria
        if denoiser_network_type in ["score"]:
            prior = PnP(denoiser=score, is_diffusion_model = True, diffusion_model_type = denoiser_network_type, diffusion_config=diffusion_config, device = device)
        elif denoiser_network_type == "dncnn":
            prior = PnP(denoiser=dncnn_denoiser)
        elif denoiser_network_type == "drunet":
            prior = PnP(denoiser=drunet_denoiser)
        else:
            raise ValueError("Check the denoiser_network_type")

        model = optim_builder(
            iteration="ISTA",
            prior=prior,
            data_fidelity=data_fidelity,
            early_stop=early_stop,
            max_iter=max_iter,
            verbose=True,
            params_algo=params_algo,
            crit_conv = "cost",
            g_first = False
        )
    elif iterative_algorithms == "pnpfista":
        # ------------
        # (Step 4) Customized parameter setup for PnP
        # ------------
        # lamb, sigma_denoiser, stepsize, max_iter = get_params(noise_level_img=noise_level_img, max_iter=max_iter, denoising_strength_sigma_at_begin=denoising_strength_sigma_at_begin, denoising_strength_sigma_at_end=denoising_strength_sigma_at_end, lamb=lamb, iterative_algorithms=iterative_algorithms, denoiser_network_type = denoiser_network_type)
        # params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser}
        # raise ValueError(f"params_algo: {params_algo}")
        # ------------
        # (Step 4) Customized parameter setup for PnP
        # ------------
        # lamb, sigma_denoiser, stepsize, max_iter = get_params(noise_level_img=noise_level_img, max_iter=max_iter, denoising_strength_sigma_at_begin=denoising_strength_sigma_at_begin, denoising_strength_sigma_at_end=denoising_strength_sigma_at_end, lamb=lamb, iterative_algorithms=iterative_algorithms, denoiser_network_type = denoiser_network_type)
        # params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser}
        sigma_denoiser = np.logspace(np.log10(denoising_strength_sigma_at_begin/255.), np.log10(denoising_strength_sigma_at_end/255.), max_iter).astype(np.float32)
        tau = ((sigma_denoiser / max(0.01, noise_level_img)) ** 2)*(1/(lamb))
        params_algo = {"stepsize": tau, "g_param": sigma_denoiser}


        early_stop = False  # Do not stop algorithm with convergence criteria
        if denoiser_network_type in ["score"]:
            prior = PnP(denoiser=score, is_diffusion_model = True, diffusion_model_type = denoiser_network_type, diffusion_config=diffusion_config, device = device)
        elif denoiser_network_type == "dncnn":
            prior = PnP(denoiser=dncnn_denoiser)
        elif denoiser_network_type == "drunet":
            prior = PnP(denoiser=drunet_denoiser)
        else:
            raise ValueError("Check the denoiser_network_type")

        model = optim_builder(
            iteration="FISTA",
            prior=prior,
            data_fidelity=data_fidelity,
            early_stop=early_stop,
            max_iter=max_iter,
            verbose=True,
            params_algo=params_algo,
            crit_conv = "cost",
            g_first = False
        )

    elif iterative_algorithms == "pnpadmm":
        # ------------
        # (Step 4) Customized parameter setup for PnP
        # ------------
        # lamb, sigma_denoiser, stepsize, max_iter = get_params(noise_level_img=noise_level_img, max_iter=max_iter, denoising_strength_sigma_at_begin=denoising_strength_sigma_at_begin, denoising_strength_sigma_at_end=denoising_strength_sigma_at_end, lamb=lamb, iterative_algorithms=iterative_algorithms, denoiser_network_type = denoiser_network_type)
        # params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser}
        sigma_denoiser = np.logspace(np.log10(denoising_strength_sigma_at_begin/255.), np.log10(denoising_strength_sigma_at_end/255.), max_iter).astype(np.float32)
        if denoiser_network_type == "dncnn":
            tau = 1/(lamb)
        else:
            tau = ((sigma_denoiser / max(0.01, noise_level_img)) ** 2)*(1/(lamb))
        params_algo = {"stepsize": tau, "g_param": sigma_denoiser}
        # raise ValueError("Fix the code")
        # print(f"noise_level_img: {noise_level_img}\n max_iter: {max_iter}\n denoising_strength_sigma_at_begin: {denoising_strength_sigma_at_begin}\n denoising_strength_sigma_at_end: {denoising_strength_sigma_at_end}\n lamb: {lamb}\n iterative_algorithms: {iterative_algorithms}\n denoiser_network_type: {denoiser_network_type}")

        early_stop = False  # Do not stop algorithm with convergence criteria
        if denoiser_network_type in ["score"]:
            prior = PnP(denoiser=score, is_diffusion_model = True, diffusion_model_type = denoiser_network_type, diffusion_config=diffusion_config, device = device)
        elif denoiser_network_type == "dncnn":
            prior = PnP(denoiser=dncnn_denoiser)
        elif denoiser_network_type == "drunet":
            prior = PnP(denoiser=drunet_denoiser)
        else:
            raise ValueError("Check the denoiser_network_type")

        # instantiate the algorithm class to solve the IP problem.
        model = optim_builder(
            iteration="ADMM",
            prior=prior,
            data_fidelity=data_fidelity,
            early_stop=early_stop,
            max_iter=max_iter,
            verbose=True,
            params_algo=params_algo,
            # backtracking = True,
            crit_conv = "cost",
            g_first = False
        )

    elif iterative_algorithms == "red":
        # ------------
        # (Step 4) Customized parameter setup for PnP
        # ------------
        # ! HERE IS POTENTIAL IMPROVEMENT I CAN MAKE.
        early_stop = False  # Stop algorithm when convergence criteria is reached
        # smaller than thres_conv
        thres_conv = 1e-5
        backtracking = True
        use_bicubic_init = False  # Use bicubic interpolation to initialize the algorithm
        # load specific parameters for GSPnP
        # lamb, sigma_denoiser, stepsize, max_iter = get_params(operation, noise_level_img)
        # lamb, sigma_denoiser, stepsize, max_iter = get_params(noise_level_img=noise_level_img, max_iter=max_iter, denoising_strength_sigma_at_begin=denoising_strength_sigma_at_begin, denoising_strength_sigma_at_end=denoising_strength_sigma_at_end, lamb=lamb, iterative_algorithms=iterative_algorithms, denoiser_network_type = denoiser_network_type)
        
        sigma_denoiser = np.logspace(np.log10(denoising_strength_sigma_at_begin/255.), np.log10(denoising_strength_sigma_at_end/255.), max_iter).astype(np.float32)
        tau = 1/(lamb)
        # params_algo = {"stepsize": tau, "g_param": sigma_denoiser}
        # raise ValueError(f"max_iter: {max_iter}")
        params_algo = {
            "stepsize": tau,
            "g_param": sigma_denoiser,
            "lambda": lamb,
        }
        # raise ValueError(f"params_algo: {params_algo}")
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
        
    elif iterative_algorithms == "diffpir":
        model = dinv.models.DiffUNet(large_model=True, pretrained=pretrained_check_point).to(device)
    
    else:
        raise ValueError("Check the iterative_algorithms")

    model.eval()

    folder_name = iterative_algorithms + f"_{denoiser_network_type}"
    save_folder = BASE_DIR / folder_name
    
    
    if iterative_algorithms in ["dpir", "pnpista", "pnpfista", "pnpadmm", "red"]:
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
        print(f"# {iterative_algorithms}({denoiser_network_type})- configuration: num_iters: {max_iter} / lambda: {lamb} / zeta: {zeta}")
        print(f"# [Input] PSNR: {average_psnr_input} / SSIM: {average_ssim_input} / LPIPS: {average_lpips_input}")
        print(f"# [Recon] PSNR: {average_psnr_recon} / SSIM: {average_ssim_recon} / LPIPS: {average_lpips_recon}")
        print(f"# Check out experiment at {save_folder}")
        print(f"# ------------")

    elif iterative_algorithms == "dps":
        # print(f"Hello world")
        if not isinstance(dataloader, list):
            dataloader = [dataloader]

        current_iterators = [iter(loader) for loader in dataloader]

        batches = min([len(loader) - loader.drop_last for loader in dataloader])

        model.eval()

        psnr_func = PSNR()
        ssim_func = SSIM()
        lpips_func = LPIPS(device = device)
        
        sum_input_psnr = 0
        sum_input_ssim = 0
        sum_input_lpips = 0
        sum_recon_psnr = 0
        sum_recon_ssim = 0
        sum_recon_lpips = 0
        
        pbar = tqdm(range(batches), ncols=150)
        
        for i in pbar:
            # Gather data from each iterator in current_iterators
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
                # dps_dc_stepsize = 0.4
                dps_dc_stepsize = lamb


                betas = get_betas(num_train_timesteps=num_train_timesteps, device=device)

                # num_steps = 50
                num_steps = 1000

                skip = num_train_timesteps // num_steps

                batch_size = 1
                eta = 1.0

                seq = range(0, num_train_timesteps, skip)
                seq_next = [-1] + list(seq[:-1])
                time_pairs = list(zip(reversed(seq), reversed(seq_next)))

                # measurement
                x0 = x_gt * 2.0 - 1.0
                y_dps = p(x0.to(device))

                # initial sample from x_T
                x = torch.randn_like(x0)
                
                diff_model = DPS(
                    model=model,
                    data_fidelity=data_fidelity,
                    max_iter = 1000,
                    device = device
                ) 
                
                # x = diff_model(y=y, physics=p, x_init = x)
                # x = diff_model(y=y, physics=p)
                
                xs = [x]
                x0_preds = []

                for i, j in tqdm(time_pairs):
                    t = (torch.ones(batch_size) * i).to(device)
                    next_t = (torch.ones(batch_size) * j).to(device)

                    at = compute_alpha(betas, t.long())
                    at_next = compute_alpha(betas, next_t.long())

                    xt = xs[-1].to(device)

                    with torch.enable_grad():
                        xt.requires_grad_()

                        # 1. denoising step
                        # we call the denoiser using standard deviation instead of the time step.
                        aux_x = xt / 2 + 0.5
                        x0_t = 2 * model(aux_x, (1 - at).sqrt() / at.sqrt() / 2) - 1
                        x0_t = torch.clip(x0_t, -1.0, 1.0)  # optional

                        # 2. likelihood gradient approximation
                        # l2_loss = data_fidelity(x0_t, y, p).sqrt().sum()
                        l2_loss = data_fidelity(x0_t, y_dps, p).sqrt().sum()

                    norm_grad = torch.autograd.grad(outputs=l2_loss, inputs=xt)[0]
                    norm_grad = norm_grad.detach()

                    sigma_tilde = ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt() * eta
                    c2 = ((1 - at_next) - sigma_tilde**2).sqrt()

                    # 3. noise step
                    epsilon = torch.randn_like(xt)

                    # 4. DDPM(IM) step
                    xt_next = (
                        (at_next.sqrt() - c2 * at.sqrt() / (1 - at).sqrt()) * x0_t
                        + sigma_tilde * epsilon
                        + c2 * xt / (1 - at).sqrt()
                        - norm_grad*dps_dc_stepsize
                    )

                    x0_preds.append(x0_t.to("cpu"))
                    xs.append(xt_next.to("cpu"))
                    
                
                recon = xs[-1]

                # plot the results
                x_recon = recon / 2 + 0.5
                x_recon = x_recon.to(device)
                
                recon = recon.to(device)

                # Computing metrics
                input_psnr = psnr_func(x_net = y, x = x_gt).item()
                input_ssim = ssim_func(x_net = y, x = x_gt).item()
                input_lpips = lpips_func(x_net = y, x = x_gt).item()
                recon_psnr = psnr_func(x_net = x_recon, x = x_gt).item()
                # recon1_psnr = psnr_func(x_net = recon, x = x_gt).item()
                # print(f"torch.max(x_recon): {torch.max(x_recon)} / torch.min(x_recon): {torch.min(x_recon)} / torch.mean(x_recon): {torch.mean(x_recon)}")
                # print(f"torch.max(x_gt): {torch.max(x_gt)} / torch.min(x_gt): {torch.min(x_gt)} / torch.mean(x_gt): {torch.mean(x_gt)}")
                # print(f"torch.max(y): {torch.max(y)} / torch.min(y): {torch.min(y)} / torch.mean(y): {torch.mean(y)}")
                # raise ValueError(f"recon_psnr: {recon_psnr} / recon1_psnr: {recon1_psnr}")
                recon_ssim = ssim_func(x_net = x_recon, x = x_gt).item()
                recon_lpips = lpips_func(x_net = x_recon, x = x_gt).item()
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
                
                title = f"{operation}_{iterative_algorithms}_iters_{max_iter}_mnoise_{formatted_noise_level_img}_denoiser_{denoiser_network_type}_dpsStepsize_{lamb}_kernel_{kernel_index}_inputpsnr_{formatted_input_psnr}_inputssim_{formatted_input_ssim}_inputlpips_{formatted_input_lpips}_reconpsnr_{formatted_recon_psnr}_reconssim_{formatted_recon_ssim}_reconlpips_{formatted_recon_lpips}"
                gt_title = f"idx_{i}"
                # print(f"save_folder: {save_folder}")
                
                recon_folder_name = f"{save_folder}/Reconstruction/"
                input_folder_name = f"{save_folder}/Measurement/"
                gt_folder_name = f"{save_folder}/Ground truth/"
                Path(recon_folder_name).mkdir(parents=True, exist_ok=True)
                Path(input_folder_name).mkdir(parents=True, exist_ok=True)
                Path(gt_folder_name).mkdir(parents=True, exist_ok=True)
                
                if plot_images == True:
                    save_image(x_recon, recon_folder_name + f"{title}.png")
                    save_image(x_gt, gt_folder_name + f"{gt_title}.png")
                    save_image(y, input_folder_name + f"{title}.png")
                
                # pbar.set_postfix({'input_psnr':input_psnr,'recon_psnr': recon_psnr}, refresh=False)
                pbar.set_postfix({'input_psnr': f"{input_psnr:.2f}", 'recon_psnr': f"{recon_psnr:.2f}"}, refresh=False)
                # break
            # break

                
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
        with open(f'{save_folder}/{avg_title}.txt', 'w') as file:
            file.write(f"PSNR Reconstruction: {average_psnr_recon}\n")
            file.write(f"SSIM Reconstruction: {average_ssim_recon}\n")
            file.write(f"PSNR Input: {average_psnr_input}\n")
            file.write(f"SSIM Input: {average_ssim_input}\n")
            file.write(f"LPIPS Reconstruction: {average_lpips_recon}\n")
            file.write(f"LPIPS Input: {average_lpips_input}\n")
        
    elif iterative_algorithms == "diffpir":
        # DiffPIR
        def get_alphas(num_train_timesteps, device, beta_start=0.1 / 1000, beta_end=20 / 1000):
            betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
            betas = torch.from_numpy(betas).to(device)
            alphas = 1.0 - betas
            alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)  # This is \overline{\alpha}_t
            return torch.tensor(alphas_cumprod)
        
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
        print(f"# {iterative_algorithms}({denoiser_network_type})")
        print(f"# [Input] PSNR: {average_psnr_input} / SSIM: {average_ssim_input} / LPIPS: {average_lpips_input}")
        print(f"# [Recon] PSNR: {average_psnr_recon} / SSIM: {average_ssim_recon} / LPIPS: {average_lpips_recon}")
        print(f"# Check out experiment at {save_folder}")
        print(f"# ------------")
    
    else:
        raise ValueError("Check the iterative_algorithms")