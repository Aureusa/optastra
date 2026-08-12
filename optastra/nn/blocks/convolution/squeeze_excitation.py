from __future__ import annotations
import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """
    Channel-attention block (Hu et al., 2018, arXiv:1709.01507).
    Global-pools each channel to a scalar, learns a per-channel gate via a
    small bottleneck MLP, rescales the input by that gate. Used inside
    MBConv"""

    def __init__(self, channels: int, reduction_ratio: float = 0.25, reduced_from: int | None = None):
        super().__init__()
        # EfficientNet reduces relative to the BLOCK's input channels, not
        # the expanded channels -- reduced_from lets MBConv pass that in
        # explicitly rather than this class guessing from `channels` alone.
        reduced_channels = max(1, int((reduced_from or channels) * reduction_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.act(self.fc1(scale))
        scale = self.gate(self.fc2(scale))
        return x * scale
    