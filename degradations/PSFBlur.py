import random
import numpy as np
import torch
import kornia
import os
from albumentations.core.transforms_interface import ImageOnlyTransform


def apply_custom_filter(inp_img, kernel_):
    return kornia.filters.filter2d(inp_img, kernel_, normalized=True)

def augment_kernel(kernel, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    Rotate kernels (or images)
    '''
    if mode == 0:
        return kernel
    elif mode == 1:
        return np.flipud(np.rot90(kernel))
    elif mode == 2:
        return np.flipud(kernel)
    elif mode == 3:
        return np.rot90(kernel, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(kernel, k=2))
    elif mode == 5:
        return np.rot90(kernel)
    elif mode == 6:
        return np.rot90(kernel, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(kernel, k=3))

class PSFBlur(ImageOnlyTransform):
    """
    Применяет реалистичное размытие с использованием предопределенных
    ядер PSF (Point Spread Function).
    Этот класс корректно работает с файлами, содержащими ядра разных размеров.
    """
    def __init__(self, kernels_path, p=0.5):
        super().__init__(p=p)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, kernels_path)
    
        self.kernels = np.load(file_path, allow_pickle=True)

    def apply(self, img, **params):
        idx = random.randint(0, len(self.kernels) - 1)
        kernel = self.kernels[idx].astype(np.float64)
        
        kernel_size = kernel.shape[0]
        
        kernel = augment_kernel(kernel, mode=random.randint(0, 7)).copy()
        
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum

        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        
        tkernel = torch.from_numpy(kernel).view(1, kernel_size, kernel_size).type(torch.FloatTensor)
        
        blurry_tensor = apply_custom_filter(img_tensor, tkernel)
        
        blurry_tensor = torch.clamp(blurry_tensor, 0., 1.)
        blurry_np = blurry_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        return (blurry_np * 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("kernels_path",)
