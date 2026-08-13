import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.restoration import richardson_lucy, wiener, denoise_nl_means, estimate_sigma
import bm3d


def wiener_filter(img, psf=None, noise=0.01):
    """
    Apply Wiener deconvolution to a single- or multi-channel image.
    Args:
        img (np.ndarray): Image as numpy array in range [0, 1]
        psf (np.ndarray): Point spread function
        noise (float): Noise power for Wiener filter
    Returns:
        np.ndarray: Denoised image
    """
    if psf is None:
        psf = np.ones((5, 5)) / 25

    if img.ndim == 3:  # Color image
        return np.stack([
            wiener(img[..., c], psf=psf, balance=noise, clip=False)
            for c in range(img.shape[2])
        ], axis=-1)
    else:  # Grayscale
        return wiener(img, psf=psf, balance=noise, clip=False)


def richardson_lucy_deconv(img, psf=None, num_iter=10):
    """
    Apply Richardson-Lucy deconvolution to a single- or multi-channel image.
    Args:
        img (np.ndarray): Image as numpy array in range [0, 1]
        psf (np.ndarray): Point spread function
        num_iter (int): Number of iterations
    Returns:
        np.ndarray: Deconvolved image
    """
    if psf is None:
        psf = np.zeros((13, 13), dtype=np.float32)
        psf[6, 6] = 1
        psf = gaussian_filter(psf, sigma=2)

    if img.ndim == 3:
        return np.stack([
            richardson_lucy(img[..., c], psf, num_iter=num_iter, clip=False)
            for c in range(img.shape[2])
        ], axis=-1)
    else:
        return richardson_lucy(img, psf, num_iter=num_iter, clip=False)


def tv_denoise(img, weight=0.1):
    from skimage.restoration import denoise_tv_chambolle
    return denoise_tv_chambolle(img, weight=weight, multichannel=True)


def wavelet_denoise(img, wavelet='db1', mode='soft', level=1):
    from skimage.restoration import denoise_wavelet
    return denoise_wavelet(img, wavelet=wavelet, mode=mode, multichannel=True, method='BayesShrink', rescale_sigma=True)


def apply_bm3d(img_np, sigma=0.1):
    """
    Apply BM3D denoising to grayscale or RGB image.
    Input and output in [0, 1] range.
    """
    if img_np.ndim == 3:
        return np.stack([bm3d.bm3d(img_np[..., c], sigma_psd=sigma, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
                         for c in range(img_np.shape[2])], axis=-1)
    else:
        return bm3d.bm3d(img_np, sigma_psd=sigma, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)


def apply_nlm(img_np, h=0.1, fast_mode=True):
    """
    Apply NLM denoising to grayscale or RGB image.
    Input and output in [0, 1] range.
    """
    patch_kw = dict(patch_size=5, patch_distance=6, multichannel=True)
    return denoise_nl_means(img_np, h=h, fast_mode=fast_mode, **patch_kw)
