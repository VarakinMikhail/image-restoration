import cv2
import random
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from .PSFBlur import PSFBlur

class BilateralBlur(ImageOnlyTransform):
    def __init__(
        self, 
        d_limit=(15, 25),
        sigma_color_limit=(150, 250),
        sigma_space_limit=(150, 250),
        p=0.5
    ):
        super().__init__(p=p)
        
        self.d_limit = d_limit
        self.sigma_color_limit = sigma_color_limit
        self.sigma_space_limit = sigma_space_limit

    def apply(self, image, **params):
        d = random.choice(range(self.d_limit[0], self.d_limit[1] + 1))
        sigma_color = random.uniform(self.sigma_color_limit[0], self.sigma_color_limit[1])
        sigma_space = random.uniform(self.sigma_space_limit[0], self.sigma_space_limit[1])

        try:
            return cv2.bilateralFilter(image, int(d), float(sigma_color), float(sigma_space))
        except Exception as e:
            print(f"Error applying bilateral filter: {e}")
            return image

    def get_transform_init_args_names(self):
        return ("d_limit", "sigma_color_limit", "sigma_space_limit")


def create_degradation_pipeline():
    transform = A.Compose([
        A.OneOf([
            PSFBlur(kernels_path='kernels.npy', p=1),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 1.5), p=1),
            A.Defocus(radius=(1, 2), alias_blur=(0.1, 0.2), p=1),
            A.MotionBlur(blur_limit=(5, 13), p=1),
            A.RingingOvershoot(blur_limit=(13, 17), cutoff=[1.5, 2], p = 1),
            A.MedianBlur(blur_limit=(3, 3), p = 1),
            BilateralBlur(d_limit=(3, 33), sigma_color_limit=(10, 100), sigma_space_limit=(10, 100), p = 1)
        ], p=0.16),

        A.OneOf([
            PSFBlur(kernels_path='kernels.npy', p=1),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 1.5), p=1),
            A.Defocus(radius=(1, 2), alias_blur=(0.1, 0.2), p=1),
            A.MotionBlur(blur_limit=(5, 13), p=1),
            A.RingingOvershoot(blur_limit=(13, 17), cutoff=[1.5, 2], p = 1),
            A.MedianBlur(blur_limit=(3, 3), p = 1),
            BilateralBlur(d_limit=(3, 33), sigma_color_limit=(10, 100), sigma_space_limit=(10, 100), p = 1)
        ], p=0.16),

        A.OneOf([
            A.UnsharpMask(blur_limit = 27, sigma_limit = 0, alpha = (1, 1), treshold = 0, p = 0.8),
            A.Sharpen(alpha = (0.1, 0.25), lightness = (0.5, 1), method = 'kernel', p = 0.2)
        ], p=0.05),

        A.OneOf([
            A.Downscale(scale_range=(0.1, 0.95), interpolation_pair={'downscale': cv2.INTER_AREA, 'upscale': cv2.INTER_CUBIC}, p=0.5),
            A.Downscale(scale_range=(0.1, 0.95), interpolation_pair={'downscale': cv2.INTER_AREA, 'upscale': cv2.INTER_LINEAR}, p=0.2),
            A.Downscale(scale_range=(0.1, 0.95), interpolation_pair={'downscale': cv2.INTER_AREA, 'upscale': cv2.INTER_LANCZOS4}, p=0.3),
        ], p=0.5),

        A.OneOf([
            A.UnsharpMask(blur_limit = 27, sigma_limit = 0, alpha = (1, 1), treshold = 0, p = 0.8),
            A.Sharpen(alpha = (0.1, 0.25), lightness = (0.5, 1), method = 'kernel', p = 0.2)
        ], p=0.05),

        A.OneOf([
            A.MultiplicativeNoise(multiplier=(0.88, 1.12), elementwise=True, p=1),
            A.MultiplicativeNoise(multiplier=(0.89, 1.11), elementwise=True, p=1),
            A.MultiplicativeNoise(multiplier=(0.90, 1.10), elementwise=True, p=1),
            A.MultiplicativeNoise(multiplier=(0.91, 1.09), elementwise=True, p=1),
            A.MultiplicativeNoise(multiplier=(0.92, 1.08), elementwise=True, p=1),
            A.MultiplicativeNoise(multiplier=(0.93, 1.07), elementwise=True, p=1),
            A.MultiplicativeNoise(multiplier=(0.94, 1.06), elementwise=True, p=1),
            A.MultiplicativeNoise(multiplier=(0.95, 1.05), elementwise=True, p=1),
            A.MultiplicativeNoise(multiplier=(0.96, 1.04), elementwise=True, p=1),
            A.MultiplicativeNoise(multiplier=(0.97, 1.03), elementwise=True, p=1),
            A.MultiplicativeNoise(multiplier=(0.98, 1.02), elementwise=True, p=1),
            A.MultiplicativeNoise(multiplier=(0.99, 1.01), elementwise=True, p=1)
        ], p=0.3),
        
        A.OneOf([
            A.GaussNoise(std_range=(0, 0.2), p=0.8),
            A.ShotNoise(scale_range=(0.04, 0.08), p = 0.2),
            A.Dithering(method='random', n_colors=16, color_mode='per_channel', noise_range=[-0.12, 0.12], p = 0.03),
            A.Dithering(method='random', n_colors=32, color_mode='per_channel', noise_range=[-0.12, 0.12], p = 0.03),
            A.Dithering(method='random', n_colors=64, color_mode='per_channel', noise_range=[-0.12, 0.12], p = 0.03),
            A.Dithering(method='random', n_colors=128, color_mode='per_channel', noise_range=[-0.12, 0.12], p = 0.03),
            A.Dithering(method='random', n_colors=16, color_mode='per_channel', noise_range=[-0.05, 0.05], p = 0.03),
            A.Dithering(method='random', n_colors=32, color_mode='per_channel', noise_range=[-0.05, 0.05], p = 0.03),
            A.Dithering(method='random', n_colors=64, color_mode='per_channel', noise_range=[-0.05, 0.05], p = 0.03),
            A.Dithering(method='random', n_colors=128, color_mode='per_channel', noise_range=[-0.05, 0.05], p = 0.03)
        ], p=0.3),

        A.OneOf([
            A.UnsharpMask(blur_limit = 27, sigma_limit = 0, alpha = (1, 1), treshold = 0, p = 0.8),
            A.Sharpen(alpha = (0.1, 0.25), lightness = (0.5, 1), method = 'kernel', p = 0.2)
        ], p=0.05),

        A.Posterize(num_bits=(4, 7), p = 0.1),

        A.OneOf([
            A.ImageCompression(
                quality_range=(15, 95),
                compression_type='jpeg',
                p=0.8
            ),
            A.ImageCompression(
                quality_range=(10, 95),
                compression_type='webp',
                p=0.2
            ),
        ], p=0.3),
        
    ],  save_applied_params=True, seed=42)
    return transform
