"""
This module implements the backbone of the AlexNet architecture,
a classic convolutional neural network
(CNN) model introduced by Alex Krizhevsky et al. in 2012.
The architecture consists of multiple convolutional layers followed by 
fully connected layers, and it was designed for image classification tasks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn as nn

from .base import Backbone

from ..nn.blocks.convolution.lrn import LocalResponseNorm
from ..nn.blocks.convolution.conv_norm_act import ConvNormAct
from ..nn.features import FeatureSpec, FeatureMaps


__all__ = ["AlexNetBackbone", "AlexNetConfig"]


@dataclass
class AlexNetConfig:
    """Config for the AlexNet backbone.

    Defaults are faithful to the paper: conv + ReLU with bias, LRN after the
    first two convs, no BatchNorm. `norm="batchnorm"` gives a modernized variant
    (usually together with `lrn=False`)."""

    in_channels: int = 3
    channels: list[int] = field(default_factory=lambda: [96, 256, 384, 256, 256])
    norm: str | None = None
    lrn: bool = True


class AlexNetBackbone(Backbone):
    """AlexNet feature extractor with a single output map "out" at stride 32.

    The pools use no padding, so the output grid loses a border: a 224x224
    input gives 6x6 (not 7x7). The *stride* between output cells is still
    4 * 2 * 2 * 2 = 32 input pixels, which is what out_spec declares -- e.g.
    growing the input by 64 px grows the output by exactly 2 cells.
    """

    def __init__(self, cfg: AlexNetConfig):
        """
        Implements the backbone of the AlexNet architecture.
        The architecture consists of the following layers:
            - Conv -> LRN -> MaxPool
            - Conv -> LRN -> MaxPool
            - Conv
            - Conv
            - Conv -> MaxPool

        :param cfg: AlexNet configuration.
        """
        super(AlexNetBackbone, self).__init__()
        self.cfg = cfg
        in_channels = cfg.in_channels
        channels = cfg.channels

        if len(channels) != 5:
            raise ValueError("AlexNetConfig.channels must contain exactly 5 values")

        def conv(c_in: int, c_out: int, **kwargs) -> ConvNormAct:
            return ConvNormAct(in_channels=c_in, out_channels=c_out, norm=cfg.norm, activation="relu", **kwargs)

        def lrn() -> nn.Module:
            return LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0) if cfg.lrn else nn.Identity()

        self.features = nn.Sequential(
            conv(in_channels, channels[0], kernel_size=11, stride=4, padding=2),
            lrn(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            conv(channels[0], channels[1], kernel_size=5, padding=2),
            lrn(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            conv(channels[1], channels[2], kernel_size=3, padding=1),
            conv(channels[2], channels[3], kernel_size=3, padding=1),
            conv(channels[3], channels[4], kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.out_spec = FeatureSpec(
            channels={"out": channels[4]},
            strides={"out": 32},
        )

    def forward(self, x: torch.Tensor) -> FeatureMaps:
        x = self.features(x)
        return FeatureMaps(
            feature_maps={"out": x},
        )


alexnet_configs = {
    "alexnet": AlexNetConfig(),
}


@Backbone.register(config=alexnet_configs["alexnet"])
def alexnet(cfg: AlexNetConfig) -> AlexNetBackbone:
    """
    Factory function to create an AlexNet backbone.

    :param cfg: AlexNet configuration.
    :return: An instance of AlexNetBackbone.
    """
    return AlexNetBackbone(cfg)
