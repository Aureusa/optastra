"""
VGG backbone implementation. It adopts the design of VGG from the original
paper "Very Deep Convolutional Networks for Large-Scale Image Recognition" by
Karen Simonyan and Andrew Zisserman (2014).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Backbone, BackboneFeatures
from ..nn.blocks.convolution.conv_norm_act import ConvNormAct

from ._registry import register_backbone


__all__ = ["VGG"]

class VGGStem(nn.Module):
    """3x3 conv stride 1 -> BN -> ReLU. Output stride 1 (this is C1)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 64, preact: bool = False):
        super().__init__()
        self.conv = ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            norm="batchnorm",
            activation="relu",
            preact=preact,
        )

    def forward(self, x):
        return self.conv(x)
    

class VGG(Backbone):
    """Generic VGG. Returns multi-stage features C1-C5 for use in necks like FPN.

    Stage strides relative to input: C1 = 1, C2 = 2, C3 = 4, C4 = 8, C5 = 16.
    """

    def __init__(
        self,
        layers: list[int],
        in_channels: int = 3,
        stem_channels: int = 64,
        preact: bool = False
    ):
        super().__init__()
        """
        Initializes the VGG backbone:
        Conv -> Conv -> MaxPool (C1)
        Conv -> Conv -> MaxPool (C2)
        Conv -> Conv -> Conv -> MaxPool (C3)
        Conv -> Conv -> Conv -> MaxPool (C4)
        Conv -> Conv -> Conv -> MaxPool (C5)

        :param layers: A list containing the number of blocks in each stage.
        :param in_channels: Number of input channels. Default is 3.
        :param stem_channels: Number of output channels for the stem. Default is 64.
        :param preact: Whether to use pre-activation in the stem. Default is False.
        """
        stage_channels = [stem_channels * (2 ** i) for i in range(len(layers))]
        VGG_stages = []

        for i, num_blocks in enumerate(layers):
            stage = []
            current_in_channels = in_channels
            if i == 0:
                current_in_channels = in_channels
            else:
                current_in_channels = stage_channels[i - 1]

            for j in range(num_blocks):
                if i == 0 and j == 0:
                    stage.append(VGGStem(in_channels, stage_channels[i], preact=preact))
                    current_in_channels = stage_channels[i]
                else:
                    out_channels = stage_channels[i]
                    stage.append(ConvNormAct(
                        in_channels=current_in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        norm="batchnorm",
                        activation="relu",
                        preact=preact,
                    ))
                    current_in_channels = out_channels
            VGG_stages.append(nn.Sequential(*stage))

            # Add a max pooling layer after each stage except the last one
            if i < len(layers) - 1:
                VGG_stages.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.stages = nn.Sequential(*VGG_stages)

        self.out_channels = {
            f"C{i + 1}": stage_channels[i] for i in range(len(layers))
        }
        self.out_strides = {f"C{i + 1}": 2 ** i for i in range(len(layers))}

    def forward(self, x):
        feature_maps = {}
        stage_outputs = []
        for stage in self.stages:
            x = stage(x)
            if isinstance(stage, nn.Sequential):
                stage_outputs.append(x)

        for i, stage_out in enumerate(stage_outputs):
            feature_maps[f"C{i + 1}"] = stage_out

        return BackboneFeatures(feature_maps=feature_maps)

@register_backbone
def vgg16(**kwargs) -> VGG:
    """Factory function to create a VGG16 backbone.

    :param in_channels: Number of input channels. Default is 3.
    :param stem_channels: Number of output channels for the stem. Default is 64.
    :param preact: Whether to use pre-activation in the stem. Default is False.
    """
    return VGG(layers=[2, 2, 3, 3, 3], **kwargs)

@register_backbone
def vgg19(**kwargs) -> VGG:
    """Factory function to create a VGG19 backbone.

    :param in_channels: Number of input channels. Default is 3.
    :param stem_channels: Number of output channels for the stem. Default is 64.
    :param preact: Whether to use pre-activation in the stem. Default is False.
    """
    return VGG(layers=[2, 2, 4, 4, 4], **kwargs)

@register_backbone
def vgg11(**kwargs) -> VGG:
    """Factory function to create a VGG11 backbone.

    :param in_channels: Number of input channels. Default is 3.
    :param stem_channels: Number of output channels for the stem. Default is 64.
    :param preact: Whether to use pre-activation in the stem. Default is False.
    """
    return VGG(layers=[1, 1, 2, 2, 2], **kwargs)

@register_backbone
def vgg13(**kwargs) -> VGG:
    """Factory function to create a VGG13 backbone.

    :param in_channels: Number of input channels. Default is 3.
    :param stem_channels: Number of output channels for the stem. Default is 64.
    :param preact: Whether to use pre-activation in the stem. Default is False.
    """
    return VGG(layers=[2, 2, 2, 2, 2], **kwargs)
