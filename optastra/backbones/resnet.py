"""
ResNet backbone implementation. It adopts the design of ResNet from the original
paper "Deep Residual Learning for Image Recognition" by Kaiming He et al. (2015).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Backbone, BackboneFeatures
from ..nn.blocks.convolution.residual import ResidualBlock, BottleneckResidualBlock
from ..nn.blocks.convolution.conv_norm_act import ConvNormAct

from ._registry import register_backbone


__all__ = ["ResNet"]

class ResNetStem(nn.Module):
    """7x7 conv stride 2 -> BN -> ReLU -> 3x3 maxpool stride 2. Output stride 4 (this is C1)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 64, preact: bool = False):
        super().__init__()
        self.conv = ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=2,
            norm="batchnorm",
            activation="relu",
            preact=preact,
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(self.conv(x))
    

class ResNet(Backbone):
    """Generic ResNet. Returns multi-stage features C2-C5 for use in necks like FPN.

    Stage strides relative to input: C1 (stem) = 4, C2 = 4, C3 = 8, C4 = 16, C5 = 32.
    """

    def __init__(
        self,
        block: type[ResidualBlock | BottleneckResidualBlock],
        layers: list[int],
        in_channels: int = 3,
        stem_channels: int = 64,
        preact: bool = False
    ):
        """
        Initializes the ResNet backbone.

        :param block: The type of residual block to use (ResidualBlock or BottleneckResidualBlock).
        :param layers: A list containing the number of blocks in each stage.
        :param in_channels: Number of input channels. Default is 3.
        :param stem_channels: Number of output channels for the stem. Default is 64.
        """
        super().__init__()
        self.stem = ResNetStem(in_channels, stem_channels, preact=preact)

        stage_channels = [64, 128, 256, 512]
        stage_strides = [1, 2, 2, 2]  # stage1 keeps stride (pool already halved it)

        self.stages = nn.ModuleList()
        in_ch = stem_channels
        for width, depth, stride in zip(stage_channels, layers, stage_strides):
            self.stages.append(self._make_stage(block, in_ch, width, depth, stride, preact=preact))
            in_ch = width * block.expansion

        self.out_channels = {
            f"C{i + 2}": stage_channels[i] * block.expansion for i in range(4)
        }
        self.out_strides = {"C2": 4, "C3": 8, "C4": 16, "C5": 32}

    @staticmethod
    def _make_stage(
            block: type[ResidualBlock | BottleneckResidualBlock],
            in_channels: int,
            out_channels: int,
            depth: int,
            stride: int,
            preact: bool = False
        ) -> nn.Sequential:
        """
        Creates a stage of the ResNet architecture consisting of multiple residual blocks.
        
        :param block: The type of residual block to use (ResidualBlock or BottleneckResidualBlock).
        :param in_channels: Number of input channels to the stage.
        :param out_channels: Number of output channels for the blocks in the stage.
        :param depth: Number of residual blocks in the stage.
        :param stride: Stride for the first block in the stage. Subsequent blocks will have a stride of 1.
        :return: A sequential container of residual blocks forming the stage
        """
        layers = [block(in_channels, out_channels, stride=stride, preact=preact)]
        for _ in range(depth - 1):
            layers.append(block(out_channels * block.expansion, out_channels, preact=preact))
        return nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> BackboneFeatures:
        """
        Forward pass through the ResNet backbone.
        
        :param images: Input tensor of shape (B, C, H, W).
        :return: BackboneFeatures containing feature maps from each stage.
        """
        x = self.stem(images)
        feature_maps = {}
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feature_maps[f"C{i + 2}"] = x
        return BackboneFeatures(feature_maps=feature_maps)


@register_backbone
def resnet18(**kwargs) -> ResNet:
    return ResNet(ResidualBlock, [2, 2, 2, 2], **kwargs)

@register_backbone
def resnet34(**kwargs) -> ResNet:
    return ResNet(ResidualBlock, [3, 4, 6, 3], **kwargs)

@register_backbone
def resnet50(**kwargs) -> ResNet:
    return ResNet(BottleneckResidualBlock, [3, 4, 6, 3], **kwargs)

@register_backbone
def resnet101(**kwargs) -> ResNet:
    return ResNet(BottleneckResidualBlock, [3, 4, 23, 3], **kwargs)

@register_backbone
def resnet152(**kwargs) -> ResNet:
    return ResNet(BottleneckResidualBlock, [3, 8, 36, 3], **kwargs)
