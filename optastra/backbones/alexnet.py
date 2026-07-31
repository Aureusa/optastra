"""
This module implements the backbone of the AlexNet architecture,
a classic convolutional neural network
(CNN) model introduced by Alex Krizhevsky et al. in 2012.
The architecture consists of multiple convolutional layers followed by 
fully connected layers, and it was designed for image classification tasks.
"""
import torch
from torch import nn

from .base import Backbone, BackboneFeatures

from ..nn.blocks.convolutions.lrn import LocalResponseNorm
from ..nn.blocks.convolutions.conv_norm_act import ConvNormAct

from ._registry import register_backbone


__all__ = ["AlexNetBackbone"]


class AlexNetBackbone(Backbone):
    def __init__(
            self,
            in_channels: int = 3,
            channels: list[int] = [96, 256, 384, 256, 256],
        ):
        """
        Implements the backbone of the AlexNet architecture.
        The architecture consists of the following layers:
            - Conv -> LRN -> MaxPool
            - Conv -> LRN -> MaxPool
            - Conv
            - Conv
            - Conv -> MaxPool

        :param in_channels: Number of input channels. Default is 3 for RGB images.
        :param channels: List of output channels for each convolutional layer.
        """
        super(AlexNetBackbone, self).__init__()
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

        self.out_channels = {
            "out": 256
        }
        self.out_strides = {
            "out": 32
        }

    def forward(self, x):
        x = self.features(x)
        return BackboneFeatures(
            feature_maps={"out": x},
        )


@register_backbone
def alexnet(**kwargs) -> AlexNetBackbone:
    """
    Factory function to create an AlexNet backbone.

    :param pretrained: If True, loads pretrained weights. Default is False.
    :param kwargs: Additional keyword arguments for the AlexNetBackbone constructor.
    :return: An instance of AlexNetBackbone.
    """
    model = AlexNetBackbone(in_channels=3, channels=[96, 256, 384, 256, 256], **kwargs)
    return model