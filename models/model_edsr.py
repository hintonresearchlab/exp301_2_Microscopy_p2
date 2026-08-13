import math

import torch
from torch import nn
import torch.nn.functional as F

import math


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rcb(x)

        out = torch.mul(out, 0.1)
        out = torch.add(out, identity)

        return out


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor *
                      upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsample_block(x)

        return out


class WienerFilterLayer(nn.Module):
    def __init__(self, kernel_size=15, sigma=3, K=0.01):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.K = K
        self.psf = self._create_psf(kernel_size, sigma)

    def _create_psf(self, size, sigma):
        ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        kernel /= kernel.sum()
        return kernel.view(1, 1, size, size)

    def forward(self, x):
        B, C, H, W = x.shape
        psf = self.psf.to(x.device).expand(C, -1, -1, -1)
        pad = (0, W - self.kernel_size, 0, H - self.kernel_size)
        psf = F.pad(psf, pad)

        x_fft = torch.fft.fft2(x)
        psf_fft = torch.fft.fft2(psf)
        psf_conj = torch.conj(psf_fft)

        wiener_filter = psf_conj / (psf_fft * psf_conj + self.K)
        result_fft = x_fft * wiener_filter
        result = torch.fft.ifft2(result_fft).real
        return result


class RLDeconvLayer(nn.Module):
    def __init__(self, kernel_size=15, sigma=3, iterations=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.iterations = iterations
        self.psf = self._create_psf(kernel_size, sigma)

    def _create_psf(self, size, sigma):
        ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        kernel /= kernel.sum()
        return kernel.view(1, 1, size, size)

    def forward(self, x):
        psf = self.psf.to(x.device)
        psf_flip = torch.flip(psf, [2, 3])
        eps = 1e-8
        estimate = x.clone()

        for _ in range(self.iterations):
            conv_est = F.conv2d(estimate, psf, padding='same', groups=1)
            ratio = x / (conv_est + eps)
            estimate *= F.conv2d(ratio, psf_flip, padding='same', groups=1)

        return estimate


class EDSRBase(nn.Module):
    def __init__(self, upscale_factor, in_chans=1):
        super().__init__()
        self.mean = torch.Tensor([0.5]).view(1, in_chans, 1, 1)
        self.conv1 = nn.Conv2d(in_chans, 64, 3, 1, 1)
        self.trunk = nn.Sequential(*[ResidualConvBlock(64) for _ in range(16)])
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.upsample = nn.Sequential(*[
            UpsampleBlock(64, 2) for _ in range(int(math.log2(upscale_factor)))
        ])
        self.conv3 = nn.Conv2d(64, in_chans, 3, 1, 1)

    def forward_backbone(self, x):
        x = x.sub_(self.mean.to(x.device)).mul_(255.)
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out = self.conv2(out) + out1
        return out


# class EDSR(EDSRBase):
#     def forward(self, x):
#         out = self.forward_backbone(x)
#         out = self.upsample(out)
#         out = self.conv3(out)
#         return out.div_(255.).add_(self.mean.to(out.device))


class EDSRwithRL(EDSRBase):
    def __init__(self, upscale_factor, in_chans=1, rl_iters=5):
        super().__init__(upscale_factor, in_chans)
        self.rl = RLDeconvLayer(iterations=rl_iters)

    def forward(self, x):
        out = self.forward_backbone(x)
        out = self.rl(out)
        out = self.upsample(out)
        out = self.conv3(out)
        return out.div_(255.).add_(self.mean.to(out.device))


class EDSRwithWiener(EDSRBase):
    def __init__(self, upscale_factor, in_chans=1):
        super().__init__(upscale_factor, in_chans)
        self.wiener = WienerFilterLayer()

    def forward(self, x):
        out = self.forward_backbone(x)
        out = self.wiener(out)
        out = self.upsample(out)
        out = self.conv3(out)
        return out.div_(255.).add_(self.mean.to(out.device))


class EDSR(nn.Module):
    def __init__(self, upscale_factor: int, in_chans: int = 1) -> None:
        super(EDSR, self).__init__()
        # First layer
        self.conv1 = nn.Conv2d(in_chans, 64, (3, 3), (1, 1), (1, 1))

        # Residual blocks
        trunk = []
        for _ in range(16):
            trunk.append(ResidualConvBlock(64))
        self.trunk = nn.Sequential(*trunk)

        # Second layer
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        # Upsampling layers
        upsampling = []
        for _ in range(int(math.log(upscale_factor, 2))):
            upsampling.append(UpsampleBlock(64, 2))

        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Conv2d(64, in_chans, (3, 3), (1, 1), (1, 1))

        self.register_buffer("mean", torch.Tensor(
            [0.5]).view(1, in_chans, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # The images by subtracting the mean RGB value of the DIV2K dataset.
        out = x.sub_(self.mean).mul_(255.)

        out1 = self.conv1(out)
        out = self.trunk(out1)
        out = self.conv2(out)
        out = torch.add(out, out1)
        out = self.upsampling(out)
        out = self.conv3(out)

        out = out.div_(255.).add_(self.mean)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
