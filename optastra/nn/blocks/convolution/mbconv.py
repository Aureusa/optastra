from __future__ import annotations
import torch
import torch.nn as nn

from .conv_norm_act import ConvNormAct
from .squeeze_excitation import SqueezeExcitation
from ..transformer.stochastic_depth import StochasticDepth


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Conv (Sandler et al. 2018 / Tan & Le 2019).
    1x1 expand -> depthwise KxK -> Squeeze-Excitation -> 1x1 project,
    residual connection only when stride=1 and in_channels==out_channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        se_ratio: float = 0.25,
        drop_path: float = 0.0,
    ):
        super().__init__()
        expanded_channels = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        self.expand = expand_ratio != 1

        if self.expand:
            self.expand_conv = ConvNormAct(
                in_channels=in_channels, out_channels=expanded_channels, kernel_size=1,
                norm="batchnorm", activation="silu",
            )

        self.depthwise = ConvNormAct(
            in_channels=expanded_channels, out_channels=expanded_channels, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2, groups=expanded_channels,
            norm="batchnorm", activation="silu",
        )

        self.se = SqueezeExcitation(expanded_channels, reduction_ratio=se_ratio, reduced_from=in_channels)

        self.project = ConvNormAct(
            in_channels=expanded_channels, out_channels=out_channels, kernel_size=1,
            norm="batchnorm", activation=None,
        )

        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.expand_conv(x) if self.expand else x
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = residual + self.drop_path(out)
        return out