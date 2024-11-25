from abc import ABC, abstractmethod
import torch
from util.tweedie_utility import clear_color,mask_generator
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import imageio

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    # def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
    #     """
    #     # Original DPS implementation
    #     if self.noiser.__name__ == 'gaussian':
    #         difference = measurement - self.operator.forward(x_0_hat, **kwargs)
    #         norm = torch.linalg.norm(difference)
    #         norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
    #     """
    #     if self.noiser.__name__ == 'gaussian':
    #         difference = measurement - self.operator.forward(x_0_hat, **kwargs)
    #         norm = torch.linalg.norm(difference)
    #         norm2 = torch.linalg.norm(difference)**2
    #         norm_grad = torch.autograd.grad(outputs=norm2, inputs=x_prev)[0]
    #         norm_grad = norm_grad * (1/norm) * 0.5
            
    #     elif self.noiser.__name__ == 'poisson':
    #         Ax = self.operator.forward(x_0_hat, **kwargs)
    #         difference = measurement-Ax
    #         norm = torch.linalg.norm(difference) / measurement.abs()
    #         norm = norm.mean()
    #         norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

    #     else:
    #         raise NotImplementedError
             
    #     return norm_grad, norm
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            # difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            # print(f"x_prev:{x_prev}")
            # print(f"x_0_hat:{x_0_hat}")

            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            # tween_difference = self.operator.forward(x_prev, **kwargs) - self.operator.forward(x_0_hat, **kwargs)
            # tween_norm = torch.linalg.norm(tween_difference)
            # norm = torch.linalg.norm(difference)
            norm = torch.linalg.norm(difference)
            norm2 = torch.linalg.norm(difference)**2
            # print(f"norm: {norm}")
            # print(f"x_0_hat.shape:{x_0_hat.shape}")
            # print(f"x_prev.shape:{x_prev.shape}")
            norm_grad = (torch.autograd.grad(outputs=norm2, inputs=x_prev)[0])
            norm_grad = norm_grad * (0.5)# * (1/norm)
            
            dummy = None
            
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError

        return norm_grad, norm, dummy
    
    # def ve_tween_grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
    #     if self.noiser.__name__ == 'gaussian':
    #         # difference = measurement - self.operator.forward(x_0_hat, **kwargs)
    #         # print(f"x_prev:{x_prev}")
    #         # print(f"x_0_hat:{x_0_hat}")
    #         # difference = measurement - self.operator.forward(x_0_hat, **kwargs)
    #         difference = measurement - self.operator.forward(x_0_hat, **kwargs)
    #         # difference =self.operator.forward(a, **kwargs) - self.operator.forward(x_0_hat, **kwargs)
    #         # print(f"x_0_hat.shape: {x_0_hat.shape}")
    #         # print(f"x_prev.shape: {x_prev.shape}")
    #         # print(f"difference: {difference}")
    #         # tween_difference = self.operator.forward(x_prev, **kwargs) - self.operator.forward(x_0_hat, **kwargs)
    #         # tween_norm = torch.linalg.norm(tween_difference)
    #         # norm = torch.linalg.norm(difference)
    #         norm = torch.linalg.norm(difference)
    #         norm2 = torch.linalg.norm(difference)**2
    #         # print(f"norm2: {norm2}")
    #         # norm2 = norm2
    #         # print(f"norm: {norm}")
    #         # x_prev = torch.tensor(x_prev, requires_grad=True)
    #         norm_grad = (torch.autograd.grad(outputs=norm2, inputs=x_prev)[0])
    #         norm_grad = norm_grad * (0.5)# * (1/norm)
            
    #         # input_file_path = os.path.join("/home/research/chicago/Diffusion_Model", f"mnoise.png")
    #         # plt.imsave(input_file_path, clear_color(measurement))


            
    #         # norm_del = torch.linalg.norm(norm_grad)
    #         # print(f"norm_grad: {norm_del}")
    #         dummy = None
            
            
    #         # print(f"tween_norm: {tween_norm}")
    #         # print(f"norm: {norm}")
        
        # elif self.noiser.__name__ == 'poisson':
        #     Ax = self.operator.forward(x_0_hat, **kwargs)
        #     difference = measurement-Ax
        #     norm = torch.linalg.norm(difference) / measurement.abs()
        #     norm = norm.mean()
        #     norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        # else:
        #     raise NotImplementedError
             
        # return norm_grad, norm, dummy

   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    # def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
    #     norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
    #     # ! HERE
    #     x_t -= norm_grad * self.scale
    #     # x_t -= norm_grad * (self.scale/norm)
    #     # print(f"self.scale: {self.scale}")
    #     return x_t, norm
    
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm, tween_norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        # norm_grad = norm_grad# * self.scale
        # norm_grad = norm_grad# * norm # * self.scale
        # x_t = x_t - norm_grad * self.scale
        return norm_grad, norm, tween_norm

    # def ve_tween_conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
    #     norm_grad, norm, tween_norm = self.ve_tween_grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
    #     # norm_grad = norm_grad# * self.scale
    #     # norm_grad = norm_grad# * norm # * self.scale
    #     # x_t = x_t - norm_grad * self.scale
    #     return norm_grad, norm, tween_norm
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm
