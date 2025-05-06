"""
This file contains helper functions used in this project
"""
import numpy as np
import torch
import os
import shutil
import random
from tifffile import imwrite
from collections import defaultdict
import pathlib
from torchvision import datasets, transforms
from data.dataloader import get_dataset, get_dataloader
from torchvision.utils import save_image
# from guided_diffusion.gaussian_diffusion import tween_p_mean_variance
import time
from datetime import datetime
from pathlib import Path
import json
import math
import matplotlib.pyplot as plt

from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch
import os
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import yaml

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def clear(x):
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(x)

def clear_color(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    
    # Handle batch input (n, 3, 128, 128)
    if x.ndim == 4:  # If the input has a batch dimension (n, 3, 128, 128)
        output = []
        for i in range(x.shape[0]):  # Iterate over the batch
            img = x[i].detach().cpu().squeeze().numpy()
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
                img = np.repeat(img, 3, axis=0)
            output.append(normalize_np(np.transpose(img, (0, 1, 2))))
        return np.stack(output)
    
    # Handle single image input (1, 3, 128, 128) or (3, 128, 128)
    x = x.detach().cpu().squeeze().numpy()
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
        x = np.repeat(x, 3, axis=0)
    return normalize_np(np.transpose(x, (1, 2, 0)))

def clear_color1(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.detach().cpu().squeeze().numpy()
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
        x = np.repeat(x, 3, axis=0)
    return normalize_np(np.transpose(x, (1, 2, 0)))


def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img


def get_tween_sampleidx(num_steps, max_value=999, last_time_step=0):
    """
    Returns the intersect indices for a given step size within the range 0 to max_value.
    
    Parameters:
    step (int): The step size to divide the range.
    max_value (int): The maximum value of the range (inclusive). Default is 999.
    
    Returns:
    list: A list of intersect indices.
    """
    if num_steps < 2:
        return [0] if num_steps == 1 else []
    
    step = max_value / (num_steps - 1)
    indices = [round(i * step) for i in range(num_steps)]
    
    # Ensure the last element is exactly max_value
    indices[-1] = max_value
    
    # Ensure all indices are greater than or equal to last_time_step
    indices = [max(idx, last_time_step) for idx in indices]
    
    # Remove duplicates if any
    indices = list(sorted(set(indices)))
    
    for i in range(len(indices)):
        if indices[i] < last_time_step:
            indices[i] = last_time_step
    
    return indices

def get_memory_free_MiB(gpu_index, t):
    # pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    # mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    memory = torch.cuda.memory_allocated(device=gpu_index)
    print(f"Memory({t}): {memory // 1024 ** 2}")
    # print(f"Memory({t}): {mem_info.free // 1024 ** 2}")
    # return mem_info.free // 1024 ** 2


"""
! need to be cleaned soon
"""
def tween_noisy_training_sample(img_shape, traindata_config, traindata_dir,
                                img_transform, tween_steps_idx, device, sampler,
                                save_tween_noisy_image_sequence, save_dir, model):
    """
    Get obtain noisy training sample which is for Tween
    (1) Access to training set folder
    (2) Randomly pick one image
    (2) According to the idx_tween_steps, define the cumulative noise.
    (3) Add the noise on the training_set
    """
    train_dataset = get_dataset(**traindata_config, transforms=img_transform)
    loader = get_dataloader(train_dataset, batch_size=1, num_workers=0, train=False)

    # randomly pick one training data sample
    random_idx = random.randint(0, len(train_dataset) - 1)
    random_ref_img = train_dataset[random_idx]

    # Ensure the random image is in the correct shape
    assert random_ref_img.shape != img_shape

    random_ref_img = random_ref_img.to(device)

    # TODO: According to the time, I should add the noise computation on the random_ref_img
    # Introduce noise to the image
    
    output_img_list = []
    for i in range(0, len(tween_steps_idx)):
        # print(f"tween_steps_idx[i]: {tween_steps_idx[i]}")
        noisy_image = sampler.q_sample(x_start = random_ref_img, t = tween_steps_idx[i]).unsqueeze(0)
        output_img_list.append(noisy_image)
        
        if save_tween_noisy_image_sequence == True:
            noisy_dir = save_dir + "/tween_noisy_sequence/noisy_image/"
            denoiser_output_dir = save_dir + "/tween_noisy_sequence/denoiser_output/"
            input_minus_denoiser_dir = save_dir + "/tween_noisy_sequence/input_minus_denoiser_image/"
            denoiser_mean_dir = save_dir + "/tween_noisy_sequence/denoiser_mean/"
            denoiser_pred_xstart_dir = save_dir + "/tween_noisy_sequence/denoiser_pred_xstart/"
            x_hat_mmse_dir = save_dir + "/tween_noisy_sequence/x_hat_mmse/"

            check_and_mkdir(noisy_dir)
            check_and_mkdir(denoiser_output_dir)
            check_and_mkdir(input_minus_denoiser_dir)
            check_and_mkdir(denoiser_mean_dir)
            check_and_mkdir(denoiser_pred_xstart_dir)
            check_and_mkdir(x_hat_mmse_dir)
            # check_and_mkdir(save_dir + "/tween_noisy_sequence/noisy_image/normalized/")
            # check_and_mkdir(save_dir + "/tween_noisy_sequence/denoiser_output/normalized/")
            # check_and_mkdir(save_dir + "/tween_noisy_sequence/input_minus_denoiser_image/normalized/")

            noisy_image_saving_dir = os.path.join(noisy_dir, f'noisy_image{tween_steps_idx[i]}.png')
            denoiser_output_image_saving_dir = os.path.join(denoiser_output_dir, f'denoiser_output{tween_steps_idx[i]}.png')
            input_minus_denoiser_image_saving_dir = os.path.join(input_minus_denoiser_dir,f'input_minus_denoiser{tween_steps_idx[i]}.png')
            denoiser_mean_image_saving_dir = os.path.join(denoiser_mean_dir, f'denoiser_mean{tween_steps_idx[i]}.png')
            denoiser_pred_xstart_image_saving_dir = os.path.join(denoiser_pred_xstart_dir, f'pred_xstart{tween_steps_idx[i]}.png')
            x_hat_mmse_image_saving_dir = os.path.join(x_hat_mmse_dir, f'x_hat_mmse{tween_steps_idx[i]}.png')
            
            # normalized_noisy_image_saving_dir = os.path.join(save_dir + "/tween_noisy_sequence/noisy_image/normalized/",
            #                         f'noisy_image{tween_steps_idx[i]}.png')
            # normalized_denoiser_output_saving_dir = os.path.join(save_dir + "/tween_noisy_sequence/denoiser_output/normalized/",
            #                         f'denoiser_output{tween_steps_idx[i]}.png')
            # normalized_input_minus_denoiser_saving_dir = os.path.join(save_dir + "/tween_noisy_sequence/input_minus_denoiser_image/normalized/",
            #                         f'denoiser_image{i}.png')

            model_dictionary = sampler.tween_p_mean_variance(model = model, x = noisy_image, t = torch.tensor([tween_steps_idx[i]] * noisy_image.shape[0], device=device))
            model_output = model_dictionary['model_output']
            model_mean = model_dictionary['mean']
            model_pred_xstart = model_dictionary['pred_xstart']
            model_x_hat_mmse = model_dictionary['x_hat_MMSE']
            input_minus_denoiser = noisy_image - model_output

            denormalize = transforms.Compose([
            transforms.Normalize((-1, -1, -1), (2, 2, 2))  # Reverse the normalization
            ])

            denormalized_noisy_image = denormalize(noisy_image.detach().cpu())
            denormalized_denoiser_output_image = denormalize(model_output.detach().cpu())
            denormalized_denoiser_mean_image = denormalize(model_mean.detach().cpu())
            denormalized_denoiser_pred_xstart_image = denormalize(model_pred_xstart.detach().cpu())
            denormalized_input_minus_denoiser_image = denormalize(input_minus_denoiser.detach().cpu())
            denormalized_x_hat_mmse_image = denormalize(model_x_hat_mmse.detach().cpu())
            # Save the image
            save_image(denormalized_noisy_image, noisy_image_saving_dir)
            save_image(denormalized_denoiser_output_image, denoiser_output_image_saving_dir)
            save_image(denormalized_input_minus_denoiser_image, input_minus_denoiser_image_saving_dir)
            save_image(denormalized_denoiser_mean_image, denoiser_mean_image_saving_dir)
            save_image(denormalized_denoiser_pred_xstart_image, denoiser_pred_xstart_image_saving_dir)
            save_image(denormalized_x_hat_mmse_image, x_hat_mmse_image_saving_dir)
            # save_image(noisy_image, normalized_noisy_image_saving_dir)
            # save_image(model_output, normalized_denoiser_output_saving_dir)
            # save_image(input_minus_denoiser, normalized_input_minus_denoiser_saving_dir)

    
    return output_img_list[-1]
    
    # # Interpolate between the original image and the noisy image
    # # tween_steps_idx should be a value between 0 and 1 indicating the interpolation amount
    # tween_factor = tween_steps_idx

    # noisy_image = (1 - tween_factor) * random_ref_img + tween_factor * noise

    # return noisy_image
    

# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# def minmax_normalization(input_):
#     """
#     This functon normalize the input to range of 0-1

#     input_: the input
#     """
#     if isinstance(input_, np.ndarray):
#         input_ = (input_ - np.amin(input_)) / (np.amax(input_) - np.amin(input_))

#     elif isinstance(input_, torch.Tensor):
#         # * pylint: disable=no-member
#         input_ = (input_ - torch.min(input_)) / (torch.max(input_) - torch.min(input_))
#         # * pylint: enable=no-member

#     else:
#         raise NotImplementedError("expected numpy or torch array")

#     return input_

# from ray import tune

def mkdir_exp_recording_folder(save_dir, measurement_operator_name, dataset_name, iterative_algorithms, sampling_strategy = None):
    """
    save_dir example: /project/cigserver5/export1/p.youngil/experiment/Diffusion_Model/sweep_results
    measurement_operator_name example: inpainting
    """
    current_time = time.time()
    current_date = datetime.now().strftime("%m%d%Y")
    current_hour_minute = datetime.now().strftime("%H%M")
    if sampling_strategy == None:
        # unique_name = f"{current_date}_{current_hour_minute}_exp_{dataset_name}_{iterative_algorithms}_{measurement_operator_name}"
        unique_name = f"{current_date}_{current_hour_minute}_{dataset_name}_{measurement_operator_name}"
    else:
        unique_name = f"{current_date}_{current_hour_minute}_{sampling_strategy}_{dataset_name}_{iterative_algorithms}_{measurement_operator_name}"
    result_file = Path(save_dir) / unique_name / "results.csv"
    os.makedirs(Path(save_dir) / unique_name, exist_ok=True)
    result_dir = Path(save_dir) / unique_name
    return result_dir, result_file

# def torch_complex_normalize(x):
#     x_angle = torch.angle(x)
#     x_abs = torch.abs(x)

#     x_abs -= torch.min(x_abs)
#     x_abs /= torch.max(x_abs)

#     x = x_abs * np.exp(1j * x_angle)

#     return x


# def strip_empties_from_dict(data):
#     new_data = {}
#     for k, v in data.items():
#         if isinstance(v, dict):
#             v = strip_empties_from_dict(v)

#         if v not in (None, str(), list(), dict(),):
#             new_data[k] = v
#     return new_data


# def ray_tune_override_config_from_param_space(config, param_space):
#     for k in param_space:
#         if isinstance(param_space[k], dict):
#             ray_tune_override_config_from_param_space(config[k], param_space[k])

#         else:
#             config[k] = param_space[k]

#     return config


# def get_last_folder(path):
#     return pathlib.PurePath(path).name


# def convert_pl_outputs(outputs):
#     outputs_dict = defaultdict(list)

#     for i in range(len(outputs)):
#         for k in outputs[i]:
#             outputs_dict[k].append(outputs[i][k])

#     log_dict, img_dict = {}, {}
#     for k in outputs_dict:
#         try:
#             tmp = torch.Tensor(outputs_dict[k]).detach().cpu()

#             log_dict.update({
#                 k: tmp
#             })

#         except Exception:
#             if outputs_dict[k][0].dim() == 2:
#                 tmp = torch.stack(outputs_dict[k], 0).detach().cpu()
#             else:
#                 tmp = torch.cat(outputs_dict[k], 0).detach().cpu()

#             if tmp.dtype == torch.complex64:
#                 tmp = torch.abs(tmp)

#             img_dict.update({
#                 k: tmp
#             })

#     return log_dict, img_dict


def check_and_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def copy_code_to_path(src_path=None, file_path=None):
    if (file_path is not None) and (src_path is not None):
        check_and_mkdir(file_path)

        max_code_save = 100  # only 100 copies can be saved
        for i in range(max_code_save):
            code_path = os.path.join(file_path, 'code%d/' % i)
            if not os.path.exists(code_path):
                shutil.copytree(src=src_path, dst=code_path)
                break


def merge_child_dict(d, ret, prefix=''):

    for k in d:
        if k in ['setting', 'test']:
            continue

        if isinstance(d[k], dict):
            merge_child_dict(d[k], ret=ret, prefix= prefix + k + '/')
        else:
            ret.update({
                prefix + k: d[k]
            })

    return ret


def write_test(save_path, log_dict=None, img_dict=None):

    if log_dict:

        cvs_data = torch.stack([log_dict[k] for k in log_dict], 0).numpy()
        cvs_data = np.transpose(cvs_data, [1, 0])

        cvs_data_mean = cvs_data.mean(0)
        cvs_data_mean.shape = [1, -1]

        cvs_data_std = cvs_data.std(0)
        cvs_data_std.shape = [1, -1]

        cvs_data_min = cvs_data.min(0)
        cvs_data_min.shape = [1, -1]

        cvs_data_max = cvs_data.max(0)
        cvs_data_max.shape = [1, -1]

        num_index = cvs_data.shape[0]
        cvs_index = np.arange(num_index) + 1
        cvs_index.shape = [-1, 1]

        cvs_data_with_index = np.concatenate([cvs_index, cvs_data], 1)

        cvs_header = ''
        for k in log_dict:
            cvs_header = cvs_header + k + ','

        np.savetxt(os.path.join(save_path, 'metrics.csv'), cvs_data_with_index,
                   delimiter=',', fmt='%.5f', header='index,' + cvs_header)
        np.savetxt(os.path.join(save_path, 'metrics_mean.csv'), cvs_data_mean,
                   delimiter=',', fmt='%.5f', header=cvs_header)
        np.savetxt(os.path.join(save_path, 'metrics_std.csv'), cvs_data_std,
                   delimiter=',', fmt='%.5f',  header=cvs_header)
        np.savetxt(os.path.join(save_path, 'metrics_min.csv'), cvs_data_min,
                   delimiter=',', fmt='%.5f', header=cvs_header)
        np.savetxt(os.path.join(save_path, 'metrics_max.csv'), cvs_data_max,
                   delimiter=',', fmt='%.5f', header=cvs_header)

        print("==========================")
        print("HEADER:", cvs_header)
        print("MEAN:", cvs_data_mean)
        print("STD:", cvs_data_std)
        print("MAX:", cvs_data_max)
        print("MIN:", cvs_data_min)
        print("==========================")

    if img_dict:

        for k in img_dict:

            imwrite(os.path.join(save_path, k + '.tiff'), data=np.array(img_dict[k]), imagej=True)


def get_save_path_from_config(config):
    save_path = os.path.join(config['setting']['exp_path'], config['setting']['exp_folder'])
    check_and_mkdir(save_path)

    return save_path



model_dir = "/project/cigserver5/export1/p.youngil/pretrained_models/Diffusion_Model/lpips/"
os.environ['TORCH_HOME'] = model_dir
# loss_fn_vgg = lpips.LPIPS(net='vgg').to('cuda')
# loss_fn_vgg = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to('cuda')

# def tween_compute_metrics(reconstructed, reference, loss_fn, gpu, mode = None):
#     """Compute PSNR, LPIPS, and DC distance between the reconstructed and reference images."""
#     # Ensure the images are in the [0, 1] range for PSNR calculation
#     #device = "cuda" if torch.cuda.is_available() else "cpu"
#     device_str = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
#     device = torch.device(device_str)  
    
#     reconstructed_np = normalize_np(reconstructed.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
#     reference_np = normalize_np(reference.squeeze().detach().cpu().numpy().transpose(1, 2, 0))

#     # PSNR
#     psnr_value = peak_signal_noise_ratio(reference_np, reconstructed_np, data_range=1)
#     # MSE 
#     # mse_value = mean_squared_error(reference_np, reconstructed_np)
    
#     reconstructed = torch.from_numpy(reconstructed_np).permute(2, 0, 1).to(device)
#     reference = torch.from_numpy(reference_np).permute(2, 0, 1).to(device)
#     reconstructed = reconstructed.view(1, 3, 256, 256) * 2. - 1.
#     reference = reference.view(1, 3, 256, 256) * 2. - 1.

#     if mode == "tau_tuning":
#         lpips_value = -1
#     else:
#         lpips_value = -1
#         lpips_value = loss_fn(reconstructed, reference).item()
    
#     return psnr_value, lpips_value
def save_param_dict(param_dict, file_path):
    with open(file_path, 'w') as f:
        json.dump(param_dict, f, indent=4)

def compute_metrics(reconstructed, reference, loss_fn, gpu, mode = None):
    """Compute PSNR, LPIPS, and DC distance between the reconstructed and reference images."""
    # Ensure the images are in the [0, 1] range for PSNR calculation
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)  
    
    print(f"reference.max(): {reference.max()}")
    print(f"reference.mean(): {reference.mean()}")
    print(f"reference.min(): {reference.min()}")
    
    reconstructed_np = normalize_np(reconstructed.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
    reference_np = normalize_np(reference.squeeze().detach().cpu().numpy().transpose(1, 2, 0))

    print(f"reference_np.max(): {reference_np.max()}")
    print(f"reference_np.mean(): {reference_np.mean()}")
    print(f"reference_np.min(): {reference_np.min()}")

    # PSNR
    psnr_value = peak_signal_noise_ratio(reference_np, reconstructed_np, data_range=1)
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

# # Example usage:
# if __name__ == "__main__":
#     print(end='')


def random_sq_bbox(img, mask_shape, randomize_box = True, fix_location_factor = None, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    if randomize_box == True:
        t = np.random.randint(margin_height, maxt)
        l = np.random.randint(margin_width, maxl)
    else:
        t = int((1-fix_location_factor[1])*(margin_height + maxt))
        l = int((fix_location_factor[0])*(margin_width + maxl))

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t+h, l:l+w] = 0

    return mask, t, t+h, l, l+w


class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, randomize_box=True, fix_location_factor = None, margin=(16, 16)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin
        self.randomize_box = randomize_box
        self.fix_location_factor = fix_location_factor

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        if l != h:
            mask_h = np.random.randint(l, h)
            mask_w = np.random.randint(l, h)
        else:
            mask_h = l
            mask_w = h
        mask, t, tl, w, wh = random_sq_bbox(img,
                              mask_shape=(mask_h, mask_w),
                              image_size=self.image_size,
                              margin=self.margin,
                              randomize_box = self.randomize_box,
                              fix_location_factor = self.fix_location_factor)
        return mask, t, tl, w, wh

    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask, t, th, w, wl = self._retrieve_box(img)
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask
        
# ================
# Helper function
# ================

def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)

def extract_and_expand_value(value, time, target):
    # array = torch.from_numpy(array).to(target.device)[time].float()
    array = torch.tensor(value).float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)

def expand_as(array, target):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    elif isinstance(array, np.float):
        array = torch.tensor([array])
   
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)

    return array.expand_as(target).to(target.device)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_schedule(t, smallest_noise_level,largest_noise_level, start=-3, end=3, tau=1.0, clip_min=1e-9):
    # A gamma function based on sigmoid function.
    v_start = sigmoid(start / tau)
    v_end = sigmoid(end / tau)
    output = sigmoid((t * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return np.clip(output, clip_min, 1.)*(largest_noise_level-smallest_noise_level) + smallest_noise_level

def get_noiselevel_alphas_timestep1(beta_at_clean, num_iters, schedule_name = "denoiser", num_diffusion_timesteps = 1000, last_time_step = 0, save_plot = False, save_root=None):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    # print(f"beta_at_clean: {beta_at_clean}")
    # print(f"num_diffusion_timesteps: {num_diffusion_timesteps}")
    """
        # TODO: 0814
        (1) Interpolate alphas to be very tightly almost continuously.
                I need to decide how many interpolation I will take.
        (2) Alphas array corresponding to the noise level (time) also needed
        (3) This one is possible implementation for simplified langevin & DDPM and all others I guess
        
        matching index should also created which has the same length as num_tween-steps
    """
    scale = 1000 / num_diffusion_timesteps
    assert scale == 1
    

    beta_start = scale * beta_at_clean
    beta_end = scale * 0.02
    beta_array = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64) # 999 to 0
    alpha_array = 1 - beta_array
    alphas_array = np.cumprod(alpha_array, axis=0)

    # ! HERE might require num_iters with certain values
    # num_iters = 2000
    # num_iters = 1000
    
    
    if schedule_name == "denoiser":
        discrete_steps = num_iters
        extended_length = num_iters
    elif schedule_name == "plotting":
        discrete_steps = num_iters
        extended_length = num_iters
        # assert discrete_steps >= extended_length
    elif schedule_name == "gentle_exponential":
        discrete_steps = num_iters
        extended_length = num_iters
        # assert discrete_steps >= extended_length
    else:
        # discrete_steps = 500000
        discrete_steps = 1000000
        plot_steps = discrete_steps
        extended_length = num_iters
        assert discrete_steps >= extended_length
    # extended_length = 5000
    new_indices = np.linspace(0, len(alphas_array) - 1, discrete_steps)
    
    """ before 11042024
    extended_alphas_array = np.interp(new_indices, np.arange(len(alphas_array)), alphas_array)
    denoiser_noise_sigma_array = np.sqrt((1-alphas_array)/(alphas_array))
    extended_denoiser_noise_sigma_array = np.sqrt((1-extended_alphas_array)/(extended_alphas_array))
    extended_denoiser_time_array = np.linspace(0, num_diffusion_timesteps - 1, discrete_steps)
    extended_time_array = np.linspace(0, num_diffusion_timesteps - 1, extended_length)
    """
    # extended_denoiser_noise_sigma_array = np.sqrt((1-extended_alphas_array)/(extended_alphas_array))
    denoiser_noise_sigma_array = np.sqrt((1-alphas_array)/(alphas_array))
    extended_denoiser_noise_sigma_array = np.interp(new_indices, np.arange(len(alphas_array)), denoiser_noise_sigma_array)
    extended_alphas_array = 1/(1+np.square(extended_denoiser_noise_sigma_array))
    # extended_alphas_array = np.interp(new_indices, np.arange(len(alphas_array)), alphas_array)
    extended_denoiser_time_array = np.linspace(0, num_diffusion_timesteps - 1, discrete_steps)
    extended_time_array = np.linspace(0, num_diffusion_timesteps - 1, extended_length)
    
    if schedule_name == "linear":
        extended_noise_sigma_array = (np.linspace(extended_denoiser_noise_sigma_array[0], extended_denoiser_noise_sigma_array[-1], extended_length, dtype=np.float64))#[::-1]

        if save_plot == True:
            sigma_list = [extended_noise_sigma_array[::-1], extended_denoiser_noise_sigma_array[::-1]]
            plot_and_save_sigma_tendency(save_path=save_root, sigmas_list=sigma_list, line_name = ["linear", "denoiser"], title=r'Noise sigma', ylabel=r'Noise sigma', plot_name="noise_sigma.png", tween_step_size=None, final_noise_time=None)
            # raise ValueError()

        assert extended_noise_sigma_array[0] < extended_noise_sigma_array[-1]
        assert (extended_noise_sigma_array[-1] == extended_denoiser_noise_sigma_array[-1])
        tolerance = 100
        time_idx_list = []
        time_list = []

        min_distance_time_idx = 0
        for i, value_a in enumerate(extended_noise_sigma_array):
            # print(f"[i]: {i}")
            matching_indicator = 0
            matching_time = 0
            min_distance = 10000
            
            for j in range(min_distance_time_idx, len(extended_denoiser_noise_sigma_array)):
                
                value_b = extended_denoiser_noise_sigma_array[j]
            # for j, value_b in enumerate(extended_denoiser_noise_sigma_array):
                if np.isclose(value_a, value_b, atol=tolerance):
                    if abs(value_a - value_b) < min_distance:
                        min_distance = abs(value_a - value_b)
                        min_distance_time_idx = j
                        min_distance_time = value_b
                        matching_indicator = 1
                        # print(f"[i: {i}] / [j: {j}] / [min_distance_time: {min_distance_time}]")
                        # print(f"[j:{j}] abs(value_a - value_b): {abs(value_a - value_b)}")
                    # index = j
                    # matching_indicator = 1
                    # matching_time = j
                    # break
                    else:
                        break
            # print(f"[i]: {i} / [j]: {j} / [min_distance_time]: {min_distance_time}")
            

            if min_distance_time <= last_time_step:
                min_distance_time = last_time_step
                    
            if matching_indicator == 1:
                # print(f"extended_time_array[min_distance_time]: {extended_time_array[min_distance_time]}")
                if extended_denoiser_time_array[min_distance_time_idx] <= last_time_step:
                    time_idx_list.append(min_distance_time_idx)
                    time_list.append(last_time_step)
                else:
                    time_idx_list.append(min_distance_time_idx)
                    time_list.append(extended_denoiser_time_array[min_distance_time_idx])
                # print(f"[i: {i}] min_distance_time: {min_distance_time} / min_distance: {min_distance}")
            elif matching_indicator != 1:
                time_idx_list.append(i)
                time_list.append(time_list[-1])
            else:
                raise ValueError("Check the implementation")
            
                
        time_idx_array = np.array(time_idx_list)
        time_array = np.array(time_list)
        extended_alphas_array = extended_alphas_array[time_idx_list]
        extended_noise_sigma_array = extended_denoiser_noise_sigma_array[time_idx_list]
      
        return extended_noise_sigma_array, extended_alphas_array, time_array, time_idx_array
    
    elif schedule_name == "sigmoid":
        start = 0
        end = 3
        # tau = 0.7
        tau = 0.5
        clip_min = 1e-9
        
        extended_time_array_for_sigmoid = np.linspace(0, 1, extended_length)
        
        extended_noise_sigma_array = sigmoid_schedule(extended_time_array_for_sigmoid, smallest_noise_level = extended_denoiser_noise_sigma_array[0],largest_noise_level = extended_denoiser_noise_sigma_array[-1], start=start, end=end, tau=tau, clip_min=clip_min)
        extended_noise_sigma_array = extended_noise_sigma_array[::-1]
        

        assert extended_noise_sigma_array[0] < extended_noise_sigma_array[-1]
        assert (extended_noise_sigma_array[-1] == extended_denoiser_noise_sigma_array[-1])
        tolerance = 100
        time_idx_list = []
        time_list = []

        min_distance_time_idx = 0
        for i, value_a in enumerate(extended_noise_sigma_array):
            # print(f"[i]: {i}")
            matching_indicator = 0
            matching_time = 0
            min_distance = 10000
            
            for j in range(min_distance_time_idx, len(extended_denoiser_noise_sigma_array)):
                
                value_b = extended_denoiser_noise_sigma_array[j]
                if np.isclose(value_a, value_b, atol=tolerance):
                    if abs(value_a - value_b) < min_distance:
                        min_distance = abs(value_a - value_b)
                        min_distance_time_idx = j
                        min_distance_time = value_b
                        matching_indicator = 1
                    else:
                        break
            # print(f"[i]: {i} / [j]: {j} / [min_distance_time]: {min_distance_time}")

            if min_distance_time <= last_time_step:
                min_distance_time = last_time_step
                    
            if matching_indicator == 1:
                # print(f"extended_time_array[min_distance_time]: {extended_time_array[min_distance_time]}")
                if extended_denoiser_time_array[min_distance_time_idx] <= last_time_step:
                    time_idx_list.append(min_distance_time_idx)
                    time_list.append(last_time_step)
                else:
                    time_idx_list.append(min_distance_time_idx)
                    time_list.append(extended_denoiser_time_array[min_distance_time_idx])
                # print(f"[i: {i}] min_distance_time: {min_distance_time} / min_distance: {min_distance}")
            elif matching_indicator != 1:
                time_idx_list.append(i)
                time_list.append(time_list[-1])
            else:
                raise ValueError("Check the implementation")
            
        time_idx_array = np.array(time_idx_list)
        time_array = np.array(time_list)
        extended_alphas_array = extended_alphas_array[time_idx_list]
        extended_noise_sigma_array = extended_denoiser_noise_sigma_array[time_idx_list]

        return extended_noise_sigma_array, extended_alphas_array, time_array, time_idx_array
    
    elif schedule_name == "denoiser":
        assert int(extended_time_array[-1]) == num_diffusion_timesteps-1
        time_idx_array =  np.linspace(0, extended_length - 1, extended_length).astype(int)
        time_array = extended_denoiser_time_array
        time_array = np.where(extended_denoiser_time_array <= last_time_step, last_time_step, extended_denoiser_time_array)
        return extended_denoiser_noise_sigma_array, extended_alphas_array, time_array, time_idx_array

    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def get_named_noise_sigma(beta_at_clean, schedule_name = "linear", num_diffusion_timesteps = 1000):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        # beta_start = scale * 0.0001
        beta_start = scale * beta_at_clean
        # beta_start = scale * 0.000001
        beta_end = scale * 0.02
        beta_array = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
        alpha_array = 1 - beta_array
        alphas_array = np.cumprod(alpha_array, axis=0)
        noise_sigma_array = np.sqrt((1-alphas_array)/(alphas_array))
        noise_sigma_array = noise_sigma_array[::-1]
        # raise ValueError(f"scale: {scale}\nnum_diffusion_timesteps: {num_diffusion_timesteps}\nbeta_start:{beta_start}\nbeta_end:{beta_end}")
        return noise_sigma_array
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def get_noiselevel_alphas_timestep(noise_level_to_get_time, beta_at_clean, denoiser_network_type, num_diffusion_timesteps = 1000, last_time_step = 0, previous_time_idx_in_list = -1):
    if denoiser_network_type == "vp_score":
        scale = 1000 / num_diffusion_timesteps
        assert scale == 1
        # ------------
        # (Prep step 1) Define diffusion noise schedule
        # ------------
        beta_start = scale * beta_at_clean
        beta_end = scale * 0.02
        beta_array = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64) # 999 to 0
        alpha_array = 1 - beta_array
        alphas_array = np.cumprod(alpha_array, axis=0)
        
        # ------------
        # (Prep step 2) Define discrete_steps to finely divide the diffusion timeline T, enabling accurate mapping from a given noise level to its corresponding time step.
        # ------------
        discrete_steps = 1000000
        extended_length = discrete_steps
        min_distance_time_idx = 0 
        assert discrete_steps >= extended_length
        new_indices = np.linspace(0, len(alphas_array) - 1, discrete_steps)
        extended_alphas_array = np.interp(new_indices, np.arange(len(alphas_array)), alphas_array)
        # ------------
        # (Prep step 3) The line below is defining the noise schedule given diffusion scheduling.
        # ------------
        denoiser_noise_sigma_array = np.sqrt((1-alphas_array)/(alphas_array))
        extended_denoiser_noise_sigma_array = np.sqrt((1-extended_alphas_array)/(extended_alphas_array))
        extended_denoiser_time_array = np.linspace(0, num_diffusion_timesteps - 1, discrete_steps)
        extended_time_array = np.linspace(0, num_diffusion_timesteps - 1, extended_length)
        
        matching_indicator = 1
        matching_time = 0
        tolerance = 10
        min_distance = 10000

        # ------------
        # (Prep step 4) In PnP or RED, where the denoising noise schedule may vary gradually, the else block improves efficiency by reusing previous_time_idx_in_list to avoid scanning the entire schedule. The if block is suitable for fixed noise levels.
        # ------------
        if previous_time_idx_in_list == -1:
            for i in range(min_distance_time_idx, len(extended_denoiser_noise_sigma_array)):
                looped_noise = extended_denoiser_noise_sigma_array[i]
                if np.isclose(looped_noise, noise_level_to_get_time, atol=tolerance):
                    if abs(looped_noise - noise_level_to_get_time) < min_distance:
                        min_distance = abs(looped_noise - noise_level_to_get_time)
                        min_distance_time_idx = i
                        min_distance_time = extended_denoiser_time_array[i]
                        min_alphas = extended_alphas_array[i]
                        matching_indicator = 1
                    else:
                        break
        else:
            for i in range(previous_time_idx_in_list, -1, -1):
                looped_noise = extended_denoiser_noise_sigma_array[i]
                if np.isclose(looped_noise, noise_level_to_get_time, atol=tolerance):
                    if abs(looped_noise - noise_level_to_get_time) < min_distance:
                        min_distance = abs(looped_noise - noise_level_to_get_time)
                        min_distance_time_idx = i
                        min_distance_time = extended_denoiser_time_array[i]
                        min_alphas = extended_alphas_array[i]
                        matching_indicator = 1
                    else:
                        break
        
        assert int(extended_time_array[-1]) == num_diffusion_timesteps-1

        time_idx_array =  np.linspace(0, extended_length - 1, extended_length).astype(int)
        time_array = extended_denoiser_time_array
        time_array = np.where(extended_denoiser_time_array <= last_time_step, last_time_step, extended_denoiser_time_array)
        
        return min_distance_time, min_alphas, min_distance_time_idx
    else:
        raise ValueError("Not yet to be implemented")
    

    
# ! HERE, I add my function
def get_noiselevel_alphas_timestep_array(noise_level_array_to_get_time, beta_at_clean, denoiser_network_type, num_diffusion_timesteps = 1000, last_time_step = 0):
    if denoiser_network_type == "vp_score":
        scale = 1000 / num_diffusion_timesteps
        assert scale == 1

        beta_start = scale * beta_at_clean
        beta_end = scale * 0.02
        beta_array = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64) # 999 to 0
        alpha_array = 1 - beta_array
        alphas_array = np.cumprod(alpha_array, axis=0)

        # ! HERE might require num_tween_steps with certain values
        # num_tween_steps = 2000
        # num_tween_steps = 1000
        
        # discrete_steps = num_tween_steps
        discrete_steps = 1000000
        extended_length = discrete_steps
        assert discrete_steps >= extended_length

        new_indices = np.linspace(0, len(alphas_array) - 1, discrete_steps)
        
        """ BEFORE 11042024
        extended_alphas_array = np.interp(new_indices, np.arange(len(alphas_array)), alphas_array)
        denoiser_noise_sigma_array = np.sqrt((1-alphas_array)/(alphas_array))
        extended_denoiser_noise_sigma_array = np.sqrt((1-extended_alphas_array)/(extended_alphas_array))
        """
        denoiser_noise_sigma_array = np.sqrt((1-alphas_array)/(alphas_array))
        extended_denoiser_noise_sigma_array = np.interp(new_indices, np.arange(len(alphas_array)), denoiser_noise_sigma_array)
        extended_alphas_array = 1/(1+np.square(extended_denoiser_noise_sigma_array))
        
        extended_denoiser_time_array = np.linspace(0, num_diffusion_timesteps - 1, discrete_steps)
        extended_time_array = np.linspace(0, num_diffusion_timesteps - 1, extended_length)
        
        min_distance = 10000
        matching_indicator = 1
        matching_time = 0
        tolerance = 100
        time_list = []
        
        min_distance_time_idx = int(discrete_steps*0.75)
        
        # print(f"extended_denoiser_noise_sigma_array[::15]: {extended_denoiser_noise_sigma_array[::15]}")
        
        for i, looped_input_noise in enumerate(noise_level_array_to_get_time):
            
            for j in range(min_distance_time_idx, len(extended_denoiser_noise_sigma_array)):
                looped_noise = extended_denoiser_noise_sigma_array[j]
                if np.isclose(looped_noise, looped_input_noise, atol=tolerance):
                    if abs(looped_noise - looped_input_noise) < min_distance:
                        min_distance = abs(looped_noise - looped_input_noise)
                        min_distance_time_idx = j
                        min_distance_time = extended_denoiser_time_array[j]
                        min_alphas = extended_alphas_array[j]
                        matching_indicator = 1
                    else:
                        break
                    
            if matching_indicator == 1:
                # print(f"extended_time_array[min_distance_time]: {extended_time_array[min_distance_time]}")
                if extended_denoiser_time_array[min_distance_time_idx] <= last_time_step:
                    time_list.append(last_time_step)
                else:
                    time_list.append(min_distance_time)
                # print(f"[i: {i}] min_distance_time: {min_distance_time} / min_distance: {min_distance}")
            elif matching_indicator != 1:
                time_list.append(time_list[-1])
            else:
                raise ValueError("Check the implementation")
        
        # for j, looped_noise in enumerate(extended_denoiser_noise_sigma_array):
        #     # print(f"looped_noise: {looped_noise}")
        #     # print(f"noise_level_to_get_time: {noise_level_to_get_time}")
        #     if np.isclose(looped_noise, noise_level_to_get_time, atol=tolerance):
        #         if abs(looped_noise - noise_level_to_get_time) < min_distance:
        #             min_distance = abs(looped_noise - noise_level_to_get_time)
        #             min_distance_time_idx = j
        #             min_distance_time = extended_denoiser_time_array[j]
        #             min_alphas = extended_alphas_array[j]
        #             matching_indicator = 1
        #         else:
        #             break
        # raise ValueError(f"min_distance_time: {min_distance_time} / min_distance_time_idx: {min_distance_time_idx}")
        
        # extended_time_array
        assert int(extended_time_array[-1]) == num_diffusion_timesteps-1

        time_array = np.array(time_list)
        # time_idx_array =  np.linspace(0, extended_length - 1, extended_length).astype(int)
        # time_array = extended_denoiser_time_array
        # time_array = np.where(extended_denoiser_time_array <= last_time_step, last_time_step, extended_denoiser_time_array)
        
        # raise ValueError(f"min_alphas: {min_alphas}")
        
        # return extended_denoiser_noise_sigma_array, extended_alphas_array, time_array, time_idx_array
        return time_array, min_alphas
    else:
        raise ValueError("Not yet to be implemented")
    
    
def p_mean_variance(model, x, t, alphas):
    raise ValueError()
    model_output = model(x, t)
    # raise ValueError()
    # In the case of "learned" variance, model will give twice channels.
    if model_output.shape[1] == 2 * x.shape[1]:
        model_output, model_var_values = torch.split(model_output, x.shape[1], dim=1)
    else:
        model_var_values = model_output
    
    # sqrt_one_minus_alphas_cumprod = torch.sqrt(1-alphas)
    # sqrt_alphas_cumprod = torch.sqrt(alphas)
        
    # sqrt_one_minus_alphas_coef = extract_and_expand(sqrt_one_minus_alphas_cumprod, 0, x)
    # sqrt_alphas_coef = extract_and_expand(sqrt_alphas_cumprod, 0, x)
    # x_hat_MMSE = (x - sqrt_one_minus_alphas_coef*model_output)/(sqrt_alphas_coef)
    log_gradient_x_i = - torch.sqrt((1)/(1-alphas)) * model_output # * By equation 39 in overleaf
    log_gradient_x_i = log_gradient_x_i*torch.sqrt(alphas)
    x_hat_MMSE = x + ((1-alphas)/(alphas))*log_gradient_x_i#*torch.sqrt(alphas)

    return {'model_output': model_output,
            'x_hat_MMSE': x_hat_MMSE}
    
def extract_and_expand_value(value, time, target):
    # array = torch.from_numpy(array).to(target.device)[time].float()
    array = torch.tensor(value).float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)

def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)
