# models/model_srgan.py
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, upscale_factor=4):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 3 * (upscale_factor ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


def build_srgan(config):
    gen = Generator(upscale_factor)
    disc = Discriminator()
    return gen, disc
