# sweep.py
import os
import argparse
import multiprocessing
import itertools
from itertools import product
import pandas as pd
import torch
from joblib import hash
from pathlib import Path
from datetime import datetime
import random
import importlib
import yaml
from second_experiment import second_experiment
from second_diffpir import second_diffpir
from second_pnpadmm import second_pnpadmm
from second_red import second_red
from second_dpir import second_dpir
import time
import json
from util.tweedie_utility import mkdir_exp_recording_folder, save_param_dict
from util.img_utils import clear_color
import matplotlib.pyplot as plt
from data.dataloader import get_dataset, get_dataloader
from torchvision import transforms
import torchvision.transforms.functional as F
from pathlib import Path

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    multiprocessing.set_start_method(
        "spawn"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_config', type=str, default="configs/dpir_score.yaml")
    args = parser.parse_args()
    
    # ------------
    # (Prep step 1) Obtain necessary variable from config for experiment
    # ------------
    task_config = load_yaml(args.task_config)
    img_size = task_config['model']['image_size']
    gpu = task_config['machinesetup']['gpu_idx']
    device_str = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    save_dir = task_config['machinesetup']['save_dir']
    batch_size = 1
    dataset_name = task_config['data']['name'].lower()
    dataset_dir = task_config['data']['dir']
    # (1-1) Inverse problem configurations
    operation = task_config['measurement']['operator']['name'].lower()
    noise_level_img = task_config['measurement']['noise']['sigma']
    kernel_index = task_config['measurement']['operator']['kernel_index']
    kernel_dir = task_config['measurement']['operator']['kernel_dir']
    # (1-2) PnP algorithm configurations
    plot_images = task_config['plugandplay_hyperparams']['save_image']
    denoiser_network_type = task_config['plugandplay_hyperparams']['denoiser_network_type'].lower()
    max_iter = task_config['plugandplay_hyperparams']['max_iter']
    denoising_strength_sigma_at_begin = task_config['plugandplay_hyperparams']['denoising_strength_sigma_at_begin']
    denoising_strength_sigma_at_end = task_config['plugandplay_hyperparams']['denoising_strength_sigma_at_end']
    lamb = task_config['plugandplay_hyperparams']['lambda_']
    gamma = task_config['plugandplay_hyperparams']['gamma']
    zeta = task_config['plugandplay_hyperparams']['zeta']
    iterative_algorithms = task_config['plugandplay_hyperparams']['iterative_algorithms']
    pretrained_check_point = task_config['model']['pretrained_check_point']
    # (1-3) Diffusion model configurations
    diffusion_config = task_config['model'] if denoiser_network_type == 'score' else None

    scale_factor = "Only needed for super-reoslution"
    plot_convergence_metrics = False
    
    METHOD_LIST = {
        'diffpir': second_diffpir,
        'dpir': second_dpir,
        'red': second_red,
        'pnpadmm': second_pnpadmm,
    }
    
    # result = second_experiment(
    result = METHOD_LIST[iterative_algorithms](
        noise_level_img=noise_level_img,
        max_iter=max_iter,
        denoising_strength_sigma_at_begin=denoising_strength_sigma_at_begin,
        denoising_strength_sigma_at_end=denoising_strength_sigma_at_end,
        lamb=lamb,
        gamma = gamma,
        zeta=zeta,
        denoiser_network_type=denoiser_network_type,
        save_image_dir=save_dir,
        dataset_name=dataset_name,
        iterative_algorithms=iterative_algorithms,
        operation=operation,
        scale_factor=scale_factor,
        batch_size=batch_size,
        kernel_index=kernel_index,
        kernel_dir = kernel_dir,
        img_size=img_size,
        plot_images=plot_images,
        plot_convergence_metrics=plot_convergence_metrics,
        gpu=gpu,
        device = device,
        diffusion_config = diffusion_config,
        pretrained_check_point = pretrained_check_point,
        dataset_dir = dataset_dir
    )

if __name__ == "__main__":
    main()