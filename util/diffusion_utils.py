import numpy as np
import torch

def get_betas(num_train_timesteps, device, beta_start=0.1 / 1000, beta_end=20 / 1000):
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas = torch.from_numpy(betas).to(device)
    return betas

def get_alphas(num_train_timesteps, device, beta_start=0.1 / 1000, beta_end=20 / 1000):
        betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        betas = torch.from_numpy(betas).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)  # This is \overline{\alpha}_t
        return torch.tensor(alphas_cumprod)

# [DPS] Utility function to let us easily retrieve \bar\alpha_t
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def find_nearest_del(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx