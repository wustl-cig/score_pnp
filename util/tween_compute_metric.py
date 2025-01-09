from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch
import os
from tqdm import tqdm
from util.tools import normalize_np, clear_color, clear  # Assuming tools.py is in the same directory
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def tween_compute_metrics(reconstructed, reference, loss_fn, gpu, mode = None):
    """Compute PSNR, LPIPS, and DC distance between the reconstructed and reference images."""
    # Ensure the images are in the [0, 1] range for PSNR calculation
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)  
    
    reconstructed_np = normalize_np(reconstructed.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
    reference_np = normalize_np(reference.squeeze().detach().cpu().numpy().transpose(1, 2, 0))

    # PSNR
    psnr_value = peak_signal_noise_ratio(reference_np, reconstructed_np)
    # MSE 
    # mse_value = mean_squared_error(reference_np, reconstructed_np)
    
    reconstructed = torch.from_numpy(reconstructed_np).permute(2, 0, 1).to(device)
    reference = torch.from_numpy(reference_np).permute(2, 0, 1).to(device)
    reconstructed = reconstructed.view(1, 3, 256, 256) * 2. - 1.
    reference = reference.view(1, 3, 256, 256) * 2. - 1.

    if mode == "tau_tuning":
        lpips_value = -1
    else:
        lpips_value = -1
        lpips_value = loss_fn(reconstructed, reference).item()
    
    return psnr_value, lpips_value

def new_tween_compute_metrics(reconstructed, reference, loss_fn, gpu, mode = None):
    """Compute PSNR, LPIPS, and DC distance between the reconstructed and reference images."""
    # Ensure the images are in the [0, 1] range for PSNR calculation
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)  
    
    reconstructed_np = normalize_np(reconstructed.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
    reference_np = normalize_np(reference.squeeze().detach().cpu().numpy().transpose(1, 2, 0))

    # PSNR
    psnr_value = peak_signal_noise_ratio(reference_np, reconstructed_np)
    # # SNR
    # snr_value = # TODO
    # SNR (Signal-to-Noise Ratio)
    noise = reference_np - reconstructed_np
    signal_power = np.mean(np.square(reference_np))
    noise_power = np.mean(np.square(noise))
    snr_value = 10 * np.log10(signal_power / noise_power)

    # MSE 
    mse_value = mean_squared_error(reference_np, reconstructed_np)
    
    reconstructed = torch.from_numpy(reconstructed_np).permute(2, 0, 1).to(device)
    reference = torch.from_numpy(reference_np).permute(2, 0, 1).to(device)
    reconstructed = reconstructed.view(1, 3, 256, 256) * 2. - 1.
    reference = reference.view(1, 3, 256, 256) * 2. - 1.

    return psnr_value, snr_value, mse_value

# Example usage:
if __name__ == "__main__":
    print(end='')

