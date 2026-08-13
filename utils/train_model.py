# train.py
import os
import sys
import time
import torch
from torch import nn, optim, amp
from tqdm import tqdm

# from logger import get_logger
# from image_assessment import psnr, ssim
# logger = get_logger("train")


def train(model, dataloader, optimizer, pixel_criterion, psnr_model, scaler, epoch, logger, model_type, device='cuda'):

    model.train()
    total_loss = 0.0
    total_psnr = 0.0

    num_batches = len(dataloader)

    start = time.time()

    loop = tqdm(enumerate(dataloader), total=num_batches,
                desc=f"Epoch {epoch+1} [Train]", leave=False)

    for batch_idx, batch in loop:
        lr = batch['lr'].to(
            device, memory_format=torch.channels_last, non_blocking=True)
        hr = batch['hr'].to(
            device, memory_format=torch.channels_last, non_blocking=True)

        model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast(device_type=device):
            if model_type == 'pre':
                lr = nn.functional.interpolate(
                    lr, size=(hr.shape[2], hr.shape[3]), mode='bicubic', align_corners=False)
                sr = model(lr)
                sr = torch.clamp(sr, 0.0, 1.0)
            elif model_type == 'post':
                sr = model(lr)

            # Compute loss
            loss = pixel_criterion(sr, hr)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        psnr = psnr_model(sr, hr).mean().item()

        total_loss += loss.item()
        total_psnr += psnr

        loop.set_postfix(loss=loss.item(), psnr=psnr)

        # if batch_idx % 50 == 0:
        #     print(
        #         f"Epoch {epoch+1} | Batch {batch_idx}/{num_batches} | PSNR: {psnr:.4f} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    elapsed = time.time() - start

    logger.info(
        f"[Epoch {epoch+1}] Training completed in {elapsed:.2f}s | Avg Trainning Loss: {avg_loss:.4f} | Avg PSNR: {avg_psnr:.4f}")
    # print(
    # f"[Epoch {epoch+1}] Avg Training Loss: {avg_loss:.4f} | Avg PSNR: {avg_psnr:.4f}")
    # print(f"[Epoch {epoch+1}] Training Time: {elapsed:.2f}s")

    return avg_loss, avg_psnr


def validate(model, dataloader, psnr_model, ssim_model, logger, model_type, device='cuda'):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = len(dataloader)

    loop = tqdm(dataloader, total=num_batches, desc="[Validate]", leave=False)

    with torch.no_grad():
        for batch in loop:
            lr = batch['lr'].to(
                device, memory_format=torch.channels_last, non_blocking=True)
            hr = batch['hr'].to(
                device, memory_format=torch.channels_last, non_blocking=True)

            with amp.autocast(device_type=device):
                if model_type == 'pre':
                    lr = nn.functional.interpolate(
                        lr, size=(hr.shape[2], hr.shape[3]), mode='bicubic', align_corners=False)
                    sr = model(lr)
                    sr = torch.clamp(sr, 0.0, 1.0)
                elif model_type == 'post':
                    sr = model(lr)

            psnr = psnr_model(sr, hr).mean().item()
            ssim = ssim_model(sr, hr).mean().item()

            total_psnr += psnr
            total_ssim += ssim

            loop.set_postfix(psnr=psnr, ssim=ssim)

    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches

    # print(f"[Validation] PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    logger.info(f"[Validation] PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim


def test(model, dataloader, psnr_model, ssim_model, logger, model_type, device='cuda'):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = len(dataloader)

    logger.info("[Testing] Starting evaluation on test set...")

    loop = tqdm(dataloader, total=num_batches, desc="[Test]", leave=False)

    with torch.no_grad():
        for batch in loop:
            lr = batch['lr'].to(
                device, memory_format=torch.channels_last, non_blocking=True)
            hr = batch['hr'].to(
                device, memory_format=torch.channels_last, non_blocking=True)

            with amp.autocast(device_type=device):
                if model_type == 'pre':
                    lr = nn.functional.interpolate(
                        lr, size=(hr.shape[2], hr.shape[3]), mode='bicubic', align_corners=False)
                    sr = model(lr)
                    sr = torch.clamp(sr, 0.0, 1.0)
                elif model_type == 'post':
                    sr = model(lr)

            psnr = psnr_model(sr, hr).mean().item()
            ssim = ssim_model(sr, hr).mean().item()

            total_psnr += psnr
            total_ssim += ssim

            loop.set_postfix(psnr=psnr, ssim=ssim)

    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches

    logger.info(f"[Test] PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim
