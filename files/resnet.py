from __future__ import annotations

import torch
import torch.nn as nn

from vision.backbones.base import Backbone, BackboneFeatures
from vision.nn.blocks.shortcuts import ShortcutProjection
from vision.nn.convolutions.conv_norm_act import ConvNormAct


class ResNetStem(nn.Module):
    """7x7 conv stride 2 -> BN -> ReLU -> 3x3 maxpool stride 2. Output stride 4 (this is C1)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        self.conv = ConvNormAct(in_channels, out_channels, kernel_size=7, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(self.conv(x))


class BasicBlock(nn.Module):
    """Two 3x3 convs, used in ResNet-18/34."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 3, stride=stride)
        self.conv2 = ConvNormAct(out_channels, out_channels, 3, activation=None)

        needs_projection = stride != 1 or in_channels != out_channels * self.expansion
        self.shortcut = (
            ShortcutProjection(in_channels, out_channels * self.expansion, stride=stride)
            if needs_projection
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv2(self.conv1(x))
        return self.act(out + identity)


class BottleneckBlock(nn.Module):
    """1x1 reduce -> 3x3 -> 1x1 expand (x4), used in ResNet-50/101/152."""

    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 1)
        self.conv2 = ConvNormAct(out_channels, out_channels, 3, stride=stride)
        self.conv3 = ConvNormAct(
            out_channels, out_channels * self.expansion, 1, activation=None
        )

        needs_projection = stride != 1 or in_channels != out_channels * self.expansion
        self.shortcut = (
            ShortcutProjection(in_channels, out_channels * self.expansion, stride=stride)
            if needs_projection
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv3(self.conv2(self.conv1(x)))
        return self.act(out + identity)


class ResNet(Backbone):
    """Generic ResNet. Returns multi-stage features C2-C5 for use in necks like FPN.

    Stage strides relative to input: C1 (stem) = 4, C2 = 4, C3 = 8, C4 = 16, C5 = 32.
    """

    def __init__(
        self,
        block: type[BasicBlock | BottleneckBlock],
        layers: list[int],
        in_channels: int = 3,
        stem_channels: int = 64,
    ):
        super().__init__()
        self.stem = ResNetStem(in_channels, stem_channels)

        stage_channels = [64, 128, 256, 512]
        stage_strides = [1, 2, 2, 2]  # stage1 keeps stride (pool already halved it)

        self.stages = nn.ModuleList()
        in_ch = stem_channels
        for width, depth, stride in zip(stage_channels, layers, stage_strides):
            self.stages.append(self._make_stage(block, in_ch, width, depth, stride))
            in_ch = width * block.expansion

        self.out_channels = {
            f"C{i + 2}": stage_channels[i] * block.expansion for i in range(4)
        }
        self.out_strides = {"C2": 4, "C3": 8, "C4": 16, "C5": 32}

    @staticmethod
    def _make_stage(block, in_channels, out_channels, depth, stride):
        layers = [block(in_channels, out_channels, stride=stride)]
        for _ in range(depth - 1):
            layers.append(block(out_channels * block.expansion, out_channels))
        return nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> BackboneFeatures:
        x = self.stem(images)
        feature_maps = {}
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feature_maps[f"C{i + 2}"] = x
        return BackboneFeatures(feature_maps=feature_maps)


def resnet18(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs) -> ResNet:
    return ResNet(BottleneckBlock, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs) -> ResNet:
    return ResNet(BottleneckBlock, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs) -> ResNet:
    return ResNet(BottleneckBlock, [3, 8, 36, 3], **kwargs)
