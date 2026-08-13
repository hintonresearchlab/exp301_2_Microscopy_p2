# train.py
import os
import sys

import torch
from torch import nn
from torch import optim
from tqdm import tqdm

from utils.logger import get_logger
from utils.image_assessment import psnr, ssim


def define_optimizer(model, config) -> optim.Adam:
    optimizer = optim.Adam(
        model.parameters(), config.model_lr, config.model_betas)

    return optimizer


# def train_model(model, train_dataset, val_dataset, config):

#     # Initialize the number of training epochs
#     start_epoch = 0

#     # Initialize training to generate network evaluation indicators
#     best_psnr = 0.0
#     best_ssim = 0.0


#     logger = get_logger(config.log_dir, config.exp_name)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     logger.info(f"Using device: {device}")
#     logger.info(f"Training on {len(train_dataset)} samples")
#     logger.info(f"Validation on {len(val_dataset)} samples")
#     logger.info(f"Batch size: {config.batch_size}")
#     logger.info(f"Learning rate: {config.lr}")
#     logger.info(f"Epochs: {config.epochs}")
#     logger.info(f"Logs directory: {config.log_dir}")
#     logger.info(f"Experiment name: {config.exp_name}")
#     logger.info(f"Model: {model}")
#     logger.info(f"Training configuration: {config}")

#     # Loss functions
#     pixel_criterion = nn.L1Loss()
#     adversarial_criterion = nn.BCEWithLogitsLoss()

#     # Optimizers
#     optimizer = define_optimizer(model, config)
#     print("Define all optimizer functions successfully.")

#     # Create an IQA evaluation model
#     psnr_model = psnr(config.upscale_factor, config.only_test_y_channel)
#     psnr_model = psnr_model.to( device=config.device, memory_format=torch.channels_last, non_blocking=True)
#     ssim_model = ssim(config.upscale_factor, config.only_test_y_channel)
#     ssim_model = ssim_model.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)


#     # train_loader = DataLoader(
#     #     train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
#     # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

#     for epoch in range(start_epoch, config.epochs):
#         train(model, train_dataset, val_dataset, epoch, config, optimizer)


#         g_loss_total, d_loss_total = 0.0, 0.0
#         best_epoch = -1

#         pbar = tqdm(train_dataset, desc=f"Epoch {epoch+1}/{config.epochs}")

#         for batch in pbar:
#             lr = batch["lr"].to(device)
#             hr = batch["hr"].to(device)

#             # ---------------------
#             # Train Discriminator
#             # ---------------------
#             with torch.no_grad():
#                 sr = generator(lr).detach()
#             real_output = discriminator(hr)
#             fake_output = discriminator(sr)

#             real_labels = torch.ones_like(real_output).to(device)
#             fake_labels = torch.zeros_like(fake_output).to(device)

#             d_loss_real = adversarial_criterion(real_output, real_labels)
#             d_loss_fake = adversarial_criterion(fake_output, fake_labels)
#             d_loss = d_loss_real + d_loss_fake

#             disc_optimizer.zero_grad()
#             d_loss.backward()
#             disc_optimizer.step()

#             # ---------------------
#             # Train Generator
#             # ---------------------
#             sr = generator(lr)
#             fake_output = discriminator(sr)
#             adversarial_loss = adversarial_criterion(fake_output, real_labels)
#             pixel_loss = pixel_criterion(sr, hr)

#             g_loss = pixel_loss + config.gan_weight * adversarial_loss

#             gen_optimizer.zero_grad()
#             g_loss.backward()
#             gen_optimizer.step()

#             g_loss_total += g_loss.item()
#             d_loss_total += d_loss.item()

#             pbar.set_postfix(g_loss=g_loss.item(), d_loss=d_loss.item())

#         avg_g_loss = g_loss_total / len(train_dataset)
#         avg_d_loss = d_loss_total / len(train_dataset)
#         logger.info(
#             f"[Epoch {epoch+1}] Generator Loss: {avg_g_loss:.6f}, Discriminator Loss: {avg_d_loss:.6f}")

#         # ---------------------
#         # Validation
#         # ---------------------
#         generator.eval()
#         val_loss, total_psnr, total_ssim = 0.0, 0.0, 0.0
#         with torch.no_grad():
#             for batch in val_dataset:
#                 lr = batch["lr"].to(device)
#                 hr = batch["hr"].to(device)
#                 sr = generator(lr)

#                 loss = pixel_criterion(sr, hr)
#                 val_loss += loss.item()

#                 total_psnr += psnr(sr, hr)
#                 total_ssim += ssim(sr, hr)

#         avg_val_loss = val_loss / len(val_dataset)
#         avg_psnr = total_psnr / len(val_dataset)
#         avg_ssim = total_ssim / len(val_dataset)

#         logger.info(
#             f"[Epoch {epoch+1}] Validation Loss: {avg_val_loss:.6f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")

#         # Save checkpoint
#         save_dir = os.path.join(config.log_dir, "checkpoints", config.exp_name)
#         os.makedirs(save_dir, exist_ok=True)

#         if (epoch + 1) % 5 == 0:

#             torch.save(generator.state_dict(), os.path.join(
#                 save_dir, f"G_epoch{epoch+1}.pth"))
#             torch.save(discriminator.state_dict(), os.path.join(
#                 save_dir, f"D_epoch{epoch+1}.pth"))
#             logger.info(f"Checkpoint saved at epoch {epoch+1}")
#         if (epoch + 1) == config.epochs:
#             torch.save(generator.state_dict(), os.path.join(
#                 save_dir, f"G_final.pth"))
#             torch.save(discriminator.state_dict(), os.path.join(
#                 save_dir, f"D_final.pth"))
#             logger.info(f"Final checkpoint saved")

#         # Save best model based on SSIM
#         if avg_ssim > best_ssim:
#             best_ssim = avg_ssim
#             best_epoch = epoch + 1
#             torch.save(generator.state_dict(), os.path.join(
#                 save_dir, f"G_best_ssim_epoch_{epoch+1}.pth"))
#             torch.save(discriminator.state_dict(),
#                        os.path.join(save_dir, f"D_best_ssim_epoch_{epoch+1}.pth"))
#             logger.info(
#                 f"[Epoch {epoch+1}] 🔥 New Best SSIM: {best_ssim:.4f} (saved)")

#         # Save best model based on PSNR
#         if avg_psnr > best_psnr:
#             best_psnr = avg_psnr
#             best_epoch = epoch + 1
#             torch.save(generator.state_dict(), os.path.join(
#                 save_dir, f"G_best_psnr_epoch_{epoch+1}.pth"))
#             torch.save(discriminator.state_dict(),
#                        os.path.join(save_dir, f"D_best_psnr_epoch_{epoch+1}.pth"))
#             logger.info(
#                 f"[Epoch {epoch+1}] 🔥 New Best PSNR: {best_psnr:.2f} (saved)")


def validate(model, val_loader, criterion, device, logger):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            sr = model(lr)
            loss = criterion(sr, hr)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    logger.info(f"Validation Loss: {avg_val_loss:.6f}")
