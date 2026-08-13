from scipy.ndimage import gaussian_filter
from skimage.restoration import richardson_lucy
import numpy as np
from scipy.signal import convolve2d


def apply_richardson_lucy(img, psf, iterations=10):
    """
    Applies Richardson-Lucy deconvolution to a single image.

    Args:
        img (np.ndarray): Input image, shape (H, W) or (H, W, C).
        psf (np.ndarray): Point Spread Function.
        iterations (int): Number of RLD iterations.

    Returns:
        np.ndarray: Deconvolved image.
    """
    if img.ndim == 3 and img.shape[-1] in (1, 3):  # Multi-channel
        return np.stack([richardson_lucy(img[..., c], psf, iterations=iterations) for c in range(img.shape[-1])], axis=-1)
    else:  # Grayscale
        return richardson_lucy(img, psf, iterations=iterations)


def generate_psf(size=13, sigma=2.0):
    """
    Create a Gaussian PSF kernel.
    """
    psf = np.zeros((size, size))
    psf[size//2, size//2] = 1
    return gaussian_filter(psf, sigma=sigma)
