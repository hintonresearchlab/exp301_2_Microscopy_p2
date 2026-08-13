import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch
from torch import nn, amp
from torchvision.transforms.functional import to_pil_image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import numpy as np


def save_sr_comparison(lr, sr, hr, save_path, title=None):
    """
    Save a side-by-side comparison of LR → SR → HR images.
    Args:
        lr, sr, hr: Tensors of shape (C, H, W), assumed in [0, 1] range
        save_path: Path to save the image
    """
    if isinstance(lr, torch.Tensor):
        lr, sr, hr = [img.cpu().detach().clamp(0, 1) for img in [lr, sr, hr]]

    comparison = torch.stack([lr, sr, hr], dim=0)
    grid = vutils.make_grid(comparison, nrow=3, padding=5)

    plt.figure(figsize=(12, 4))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_psnr(psnr_dict, save_path=None, title=None, show=True):
    """
    Plot PSNR values over epochs.
    Args:
        psnr_values: List of PSNR values
        save_path: Path to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    for name, values in psnr_dict.items():
        plt.plot(range(1, len(values) + 1), values, label=name)

    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('PSNR over Epochs')
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()
    plt.close()


def plot_metrics(metrics_dict, save_path=None, title=None, show=True):
    """
    Plot multiple training/validation metrics over epochs.

    Args:
        metrics_dict (dict): Dictionary where keys are metric names and values are lists of values per epoch.
                             e.g., {'train_loss': [...], 'val_psnr': [...]}.
        save_path (str, optional): If provided, saves the plot to this path.
        title (str, optional): Title for the plot. Defaults to 'Training Metrics'.
        show (bool): If True, displays the plot using plt.show(). Set to False for headless mode.
    """
    plt.figure(figsize=(10, 6))
    for name, values in metrics_dict.items():
        plt.plot(range(1, len(values) + 1), values, label=name)

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(title or 'Training Metrics')
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()
    plt.close()


def visualize_batch(loader, save_path=None, n=4, title='Sample Batches'):
    """
    Visualizes n LR→HR image pairs from a batch.
    """
    batch = next(iter(loader))

    lr_imgs = batch['lr'][:n].cpu().detach().clamp(0, 1)
    hr_imgs = batch['hr'][:n].cpu().detach().clamp(0, 1)

    fig, axs = plt.subplots(nrows=n, ncols=2, figsize=(6, 3 * n))

    for i in range(n):
        axs[i][0].imshow(to_pil_image(lr_imgs[i]), cmap='gray')
        axs[i][0].set_title('LR')
        axs[i][1].imshow(to_pil_image(hr_imgs[i]), cmap='gray')
        axs[i][1].set_title('HR')
        for j in range(2):
            axs[i][j].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    plt.close()


def visualize_model_predictions(model, val_dataset, device, save_dir='results/visuals', num_samples=5, fixed_indices=None):
    """
    Visualize LR → SR → HR predictions on the validation dataset.

    Args:
        model (torch.nn.Module): Trained model.
        val_dataset (torch.utils.data.Dataset): Validation dataset.
        device (str): 'cuda' or 'cpu'
        save_dir (str): Where to save the visualizations.
        num_samples (int): Number of samples to visualize.
        fixed_indices (list[int], optional): List of fixed indices to visualize.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    if fixed_indices is None:
        fixed_indices = list(range(min(num_samples, len(val_dataset))))
    else:
        fixed_indices = fixed_indices[:num_samples]

    with torch.no_grad():
        for idx in fixed_indices:
            sample = val_dataset[idx]
            lr = sample['lr'].unsqueeze(0).to(device)
            hr = sample['hr'].to('cpu')
            filename = sample.get('filename', f'sample_{idx}')

            # Forward pass
            sr = model(lr).squeeze(0).cpu().clamp(0, 1)

            # Clamp & move lr to CPU for plotting
            lr = lr.squeeze(0).cpu().clamp(0, 1)

            # Plot
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(to_pil_image(lr), cmap='gray')
            axs[0].set_title('Low-Res Input')

            axs[1].imshow(to_pil_image(sr), cmap='gray')
            axs[1].set_title('Super-Resolved Output')

            axs[2].imshow(to_pil_image(hr), cmap='gray')
            axs[2].set_title('High-Res Ground Truth')

            for ax in axs:
                ax.axis('off')

            plt.tight_layout()
            save_path = os.path.join(save_dir, f"{filename}_comparison.png")
            plt.savefig(save_path)
            plt.close()

            print(f"Saved: {save_path}")


def visualize_model_res(model, preload, best_model_epoch, config, val_dataset, device, psnr_model, ssim_model, model_type, save_dir='results/visuals',
                        num_samples=5, fixed_indices=None, suffix='', logger=None):
    """
    Visualize LR → SR → HR predictions on the validation dataset with PSNR & SSIM.

    Args:
        model (torch.nn.Module): Trained SR model.
        config (Namespace or dict): Config with exp_name, device, etc.
        val_dataset (Dataset): Validation dataset with 'lr', 'hr', and optionally 'filename'.
        device (str): 'cuda' or 'cpu'.
        save_dir (str): Directory to save visualizations.
        num_samples (int): Number of samples to show.
        fixed_indices (list[int]): Indices to visualize. Random if None.
        suffix (str): Optional suffix for image filename.
        logger (logging.Logger, optional): Logger to log progress.
    """
    os.makedirs(save_dir, exist_ok=True)
    if preload:
        ckpt_path = os.path.join(
            'logs', config.exp_name, 'best_model_epoch_' + str(best_model_epoch) + '.pth')
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {ckpt_path}")
        print(f"Loading model from {ckpt_path}")

        state_dict = torch.load(ckpt_path, map_location=device)
        # state_dict = rename_state_dict_keys(state_dict)
        model.load_state_dict(state_dict)
        model.to(device=device, memory_format=torch.channels_last)
        model.eval()
    else:
        model.eval()

    if fixed_indices is None:
        fixed_indices = list(range(min(num_samples, len(val_dataset))))
    else:
        fixed_indices = fixed_indices[:num_samples]

    with torch.no_grad():
        for idx in fixed_indices:
            sample = val_dataset[idx]
            lr = sample['lr'].to(device=device)
            lr = lr.unsqueeze(0) if lr.dim() == 3 else lr
            hr = sample['hr'].to(device=device)
            hr = hr.unsqueeze(0) if hr.dim() == 3 else hr

            filename = sample.get('filename', f'sample_{idx}')

            with amp.autocast(device_type=str(device)):
                if model_type == 'pre':
                    lr = nn.functional.interpolate(
                        lr, size=(hr.shape[2], hr.shape[3]), mode='bicubic', align_corners=False)

                sr = model(lr)

            # Detach & convert to numpy
            lr_img = lr[0].cpu().clamp(0, 1)
            sr_img = sr[0].cpu().clamp(0, 1)
            hr_img = hr[0].cpu().clamp(0, 1)

            # Compute metrics
            psnr_val = psnr_model(sr, hr).mean().item()
            ssim_val = ssim_model(sr, hr).mean().item()

            # Plot
            fig, axs = plt.subplots(1, 3, figsize=(14, 5))
            axs[0].imshow(to_pil_image(lr_img), cmap='gray')
            axs[0].set_title('Low-Res Input')
            axs[1].imshow(to_pil_image(sr_img), cmap='gray')
            axs[1].set_title(
                f'SR Output\nPSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}')
            axs[2].imshow(to_pil_image(hr_img), cmap='gray')
            axs[2].set_title('Ground Truth (HR)')

            for ax in axs:
                ax.axis('off')

            plt.tight_layout()
            save_path = os.path.join(save_dir, f"{filename}{suffix}_viz.png")
            plt.savefig(save_path)
            plt.close()

            if logger:
                logger.info(f"Saved visualization: {save_path}")
            else:
                print(f"Saved: {save_path}")


def rename_state_dict_keys(state_dict):
    renamed_dict = {}
    for k, v in state_dict.items():
        if k.startswith("upsampling"):
            new_key = k.replace("upsampling", "upsample")
        else:
            new_key = k
        renamed_dict[new_key] = v
    return renamed_dict
