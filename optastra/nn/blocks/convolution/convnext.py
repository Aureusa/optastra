import torch
import torch.nn as nn

from .layernorm2d import LayerNorm2d
from .conv_norm_act import ConvNormAct
from ..transformer.stochastic_depth import StochasticDepth


class ConvNeXtBlock(nn.Module):
    """
    Depthwise 7x7 conv -> LayerNorm (channels-last) -> pointwise MLP
    (4x expansion, GELU) -> pointwise back down. Inverted-bottleneck,
    pre-norm.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)   # depthwise
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = StochasticDepth(drop_path)   # reuse your existing transformer block's DropPath

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)   # (B,C,H,W) -> (B,H,W,C) for channels-last LayerNorm/Linear
        x = self.norm(x)
        x = self.act(self.pwconv1(x))
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (B,H,W,C) -> (B,C,H,W) back to channels-first
        x = self.drop_path(x)
        return residual + x
    

class ConvNeXtDownsample(nn.Module):
    """LayerNorm -> 2x2 conv, stride 2. Sits between stages (not inside the
    first block of a stage, unlike ResNet) -- ConvNeXt keeps blocks
    stride-1 throughout and puts all downsampling in these explicit layers."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = LayerNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))
    