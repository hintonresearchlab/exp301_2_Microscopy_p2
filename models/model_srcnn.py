# models/model_srcnn.py
from scipy.signal.windows import gaussian
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self, in_chans: int):
        super(SRCNN, self).__init__()

        assert isinstance(in_chans, int), type(in_chans)
        assert in_chans > 0, in_chans
        self.in_chans = in_chans

        # First convolutional layer -> feature extraction layer
        self.conv1 = nn.Conv2d(
            in_channels=in_chans, out_channels=1024,
            kernel_size=(5, 5), stride=1, padding=(2, 2)
        )
        self.relu1 = nn.ReLU(inplace=False)

        # Second convolutional layer -> non-linear mapping layer
        self.conv2 = nn.Conv2d(
            1024, 128,
            kernel_size=(1, 1), stride=1
        )
        self.relu2 = nn.ReLU(False)

        # Third convolutional layer  -> reconstruction layer
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=in_chans,
            kernel_size=1, stride=1
        )

        # Initialize weights similar to Keras's glorot_uniform
        self._initialize_weights()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             # Glorot uniform is same as xavier_uniform in PyTorch
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0)
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0,
                                math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)

        nn.init.normal_(self.conv3.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.conv3.bias.data)


class SRCNNwithPSF(nn.Module):
    def __init__(self, in_chans: int, psf_size: int = 9):
        super(SRCNNwithPSF, self).__init__()

        assert isinstance(in_chans, int), type(in_chans)
        assert in_chans > 0, in_chans
        self.in_chans = in_chans
        self.psf_size = psf_size

        # Learnable PSF layer (acts as blind deconvolution kernel)
        self.psf_conv = nn.Conv2d(
            in_channels=in_chans,
            out_channels=in_chans,
            kernel_size=psf_size,
            stride=1,
            padding=psf_size // 2,
            bias=False,
            groups=in_chans  # depthwise blur
        )

        # First convolutional layer -> feature extraction layer
        self.conv1 = nn.Conv2d(
            in_channels=in_chans, out_channels=1024,
            kernel_size=(5, 5), stride=1, padding=(2, 2)
        )
        self.relu1 = nn.ReLU(inplace=False)

        # Second convolutional layer -> non-linear mapping layer
        self.conv2 = nn.Conv2d(
            1024, 128,
            kernel_size=(1, 1), stride=1
        )
        self.relu2 = nn.ReLU(False)

        # Third convolutional layer  -> reconstruction layer
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=in_chans,
            kernel_size=1, stride=1
        )

        # Initialize weights similar to Keras's glorot_uniform
        self._initialize_weights()

    def forward(self, x):
        x = self.psf_conv(x)              # Learnable PSF
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             # Glorot uniform is same as xavier_uniform in PyTorch
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0)
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0,
                                math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                if module.bias is not None:
                    nn.init.zeros_(module.bias.data)

        # PSF Conv - Initialize to approximate Gaussian
        with torch.no_grad():
            center = self.psf_size // 2
            sigma = 1.0
            y, x = torch.meshgrid(torch.arange(
                self.psf_size), torch.arange(self.psf_size), indexing='ij')
            gaussian = torch.exp(-((x - center)**2 +
                                 (y - center)**2) / (2 * sigma**2))
            gaussian /= gaussian.sum()

            for c in range(self.in_chans):
                self.psf_conv.weight.data[c, 0] = gaussian

        nn.init.normal_(self.conv3.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.conv3.bias.data)


def init_gaussian_psf(size: int, std: float = 1.0):
    g = gaussian(size, std)
    g2d = np.outer(g, g)
    g2d /= g2d.sum()
    return torch.tensor(g2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


class RLDeconvolutionLayer(nn.Module):
    def __init__(self, psf_size=9, in_chans=1, num_iters=5):
        super(RLDeconvolutionLayer, self).__init__()
        init_psf = init_gaussian_psf(psf_size)

        self.in_chans = in_chans
        self.psf_size = psf_size
        self.num_iters = num_iters

        # Learnable PSF
        self.psf = nn.Parameter(init_psf.clone(), requires_grad=True)

    def forward(self, x):
        eps = 1e-6
        psf = self.psf
        psf = psf / (psf.sum() + eps)  # Normalize

        y = x
        for i in range(self.num_iters):
            conv_psf = F.conv2d(y, psf, padding=self.psf_size // 2, groups=1)
            conv_psf = conv_psf.clamp(min=eps)

            safe_conv_psf = torch.clamp(conv_psf, min=1e-2)
            ratio = x / safe_conv_psf

            correction = F.conv2d(ratio, torch.flip(
                psf, [2, 3]), padding=self.psf_size // 2, groups=1)
            correction = torch.clamp(ratio, min=0.1, max=10)
            y = y * correction
            y = y / (y.max(dim=2, keepdim=True)
                     [0].max(dim=3, keepdim=True)[0] + eps)
            y = torch.clamp(y * correction, min=0.0, max=1.0)

            # if i % (self.num_iters // 2) == 0:
            #     with torch.no_grad():
            #         print(f"[RL Iter {i+1}]")
            #         print(f"  PSF sum: {psf.sum().item():.6f}")
            #         print(f"  Max conv_psf: {conv_psf.max().item():.6f}")
            #         print(f"  Min conv_psf: {conv_psf.min().item():.6f}")
            #         print(f"  Max ratio: {ratio.max().item():.6f}")
            #         print(f"  Max correction: {correction.max().item():.6f}")
            #         print(
            #             f"  Max y: {y.max().item():.6f} | Min y: {y.min().item():.6f}")
            #         print("-" * 40)

        return y


class SRCNNwithRL(nn.Module):
    def __init__(self, in_chans: int, psf_size: int = 9, num_rl_iters: int = 5):
        super(SRCNNwithRL, self).__init__()
        self.rl_layer = RLDeconvolutionLayer(
            psf_size=psf_size, in_chans=in_chans, num_iters=num_rl_iters)

        self.conv1 = nn.Conv2d(in_chans, 1024, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(1024, 128, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(128, in_chans, kernel_size=1)

        self._initialize_weights()

    def forward(self, x):
        x = self.rl_layer(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, math.sqrt(
                    2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.conv3.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.conv3.bias.data)
