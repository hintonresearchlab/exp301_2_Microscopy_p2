import torch
import torch.nn.functional as F
import numpy as np
from math import log10
from skimage.metrics import structural_similarity as compare_ssim


def psnr(sr, hr, max_val=1.0):
    """Compute PSNR between super-resolved and high-res images."""
    mse = F.mse_loss(sr, hr, reduction='mean')
    if mse == 0:
        return float('inf')
    return 20 * log10(max_val / torch.sqrt(mse))


def ssim(sr, hr):
    """Compute SSIM between SR and HR images. Convert to numpy first."""
    sr_img = tensor_to_image(sr)
    hr_img = tensor_to_image(hr)

    # If grayscale, skip channel dimension
    if sr_img.ndim == 3 and sr_img.shape[2] == 3:
        multichannel = True
    else:
        multichannel = False

    return compare_ssim(hr_img, sr_img, data_range=1.0, channel_axis=-1 if multichannel else None)


def tensor_to_image(tensor):
    """
    Convert a torch tensor [B, C, H, W] or [C, H, W] to a numpy image [H, W, C].
    Assumes tensor values are in [0, 1].
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image in batch
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = np.clip(img, 0, 1)
    return img



