"""
This module implements the backbone of the AlexNet architecture,
a classic convolutional neural network
(CNN) model introduced by Alex Krizhevsky et al. in 2012.
The architecture consists of multiple convolutional layers followed by 
fully connected layers, and it was designed for image classification tasks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import torch.nn as nn

from .base import Backbone, BackboneFeatures, FeatureSpec

from ..nn.blocks.convolution.lrn import LocalResponseNorm
from ..nn.blocks.convolution.conv_norm_act import ConvNormAct

from ._registry import register_backbone


__all__ = ["AlexNetBackbone"]


@dataclass
class AlexNetConfig:
    """Config for the AlexNet backbone."""

    in_channels: int = 3
    channels: list[int] = field(default_factory=lambda: [96, 256, 384, 256, 256])


class AlexNetBackbone(Backbone):
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

        self.features = nn.Sequential(
            ConvNormAct(in_channels=in_channels, out_channels=channels[0], kernel_size=11, stride=4, padding=2),
            LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvNormAct(in_channels=channels[0], out_channels=channels[1], kernel_size=5, padding=2),
            LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvNormAct(in_channels=channels[1], out_channels=channels[2], kernel_size=3, padding=1),
            ConvNormAct(in_channels=channels[2], out_channels=channels[3], kernel_size=3, padding=1),
            ConvNormAct(in_channels=channels[3], out_channels=channels[4], kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.out_spec = FeatureSpec(
            channels={"out": channels[4]},
            strides={"out": 32},
        )

    def forward(self, x):
        x = self.features(x)
        return BackboneFeatures(
            feature_maps={"out": x},
        )


alexnet_configs = {
    "alexnet": AlexNetConfig(),
}


@register_backbone(config=alexnet_configs["alexnet"])
def alexnet(cfg: AlexNetConfig) -> AlexNetBackbone:
    """
    Factory function to create an AlexNet backbone.

    :param cfg: AlexNet configuration.
    :return: An instance of AlexNetBackbone.
    """
    return AlexNetBackbone(cfg)