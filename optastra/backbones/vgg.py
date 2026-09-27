"""
VGG backbone implementation. It adopts the design of VGG from the original
paper "Very Deep Convolutional Networks for Large-Scale Image Recognition" by
Karen Simonyan and Andrew Zisserman (2014), with BatchNorm after every conv
(the common "VGG-BN" variant).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn as nn

from .base import Backbone
from ..nn.blocks.convolution.conv_norm_act import ConvNormAct
from ..nn.features import FeatureSpec, FeatureMaps


__all__ = ["VGG", "VGGConfig"]


@dataclass
class VGGConfig:
    """Config for the VGG family.

    `layers[i]` is the number of 3x3 convs in stage i. Stage widths start at
    `stem_channels` and double every stage, capped at `max_channels`
    (64-128-256-512-512 for the standard VGG)."""

    layers: list[int] = field(default_factory=lambda: [2, 2, 3, 3, 3])
    in_channels: int = 3
    stem_channels: int = 64
    max_channels: int = 512
    preact: bool = False


class VGG(Backbone):
    """Generic VGG. Every stage is `layers[i]` x (3x3 Conv -> BN -> ReLU) followed
    by a 2x2 max-pool, and C{i+1} is the (pooled) output of stage i, so it follows
    the same "C_k has stride 2**k" convention as the other backbones:

        C1 = 2, C2 = 4, C3 = 8, C4 = 16, C5 = 32   (C5 == torchvision's vgg.features output)
    """

    def __init__(self, cfg: VGGConfig):
        """
        Initializes the VGG backbone (VGG-16 shown):
        Conv -> Conv -> MaxPool (C1)
        Conv -> Conv -> MaxPool (C2)
        Conv -> Conv -> Conv -> MaxPool (C3)
        Conv -> Conv -> Conv -> MaxPool (C4)
        Conv -> Conv -> Conv -> MaxPool (C5)

        :param cfg: VGG configuration.
        """
        super().__init__()
        self.cfg = cfg

        stage_channels = [min(cfg.stem_channels * (2 ** i), cfg.max_channels) for i in range(len(cfg.layers))]

        self.stages = nn.ModuleList()
        in_ch = cfg.in_channels
        for num_convs, out_ch in zip(cfg.layers, stage_channels):
            convs = []
            for _ in range(num_convs):
                convs.append(ConvNormAct(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    stride=1,
                    norm="batchnorm",
                    activation="relu",
                    preact=cfg.preact,
                ))
                in_ch = out_ch
            self.stages.append(nn.Sequential(*convs, nn.MaxPool2d(kernel_size=2, stride=2)))

        self.out_spec = FeatureSpec(
            channels={f"C{i + 1}": ch for i, ch in enumerate(stage_channels)},
            strides={f"C{i + 1}": 2 ** (i + 1) for i in range(len(stage_channels))},
        )

    def forward(self, x: torch.Tensor) -> FeatureMaps:
        feature_maps = {}
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feature_maps[f"C{i + 1}"] = x
        return FeatureMaps(feature_maps=feature_maps)


vgg_configs = {
    "vgg11": VGGConfig(layers=[1, 1, 2, 2, 2]),
    "vgg13": VGGConfig(layers=[2, 2, 2, 2, 2]),
    "vgg16": VGGConfig(layers=[2, 2, 3, 3, 3]),
    "vgg19": VGGConfig(layers=[2, 2, 4, 4, 4]),
}


@Backbone.register(config=vgg_configs["vgg16"])
def vgg16(cfg: VGGConfig) -> VGG:
    """Factory function to create a VGG16 backbone.

    :param cfg: VGG configuration.
    """
    return VGG(cfg)

@Backbone.register(config=vgg_configs["vgg19"])
def vgg19(cfg: VGGConfig) -> VGG:
    """Factory function to create a VGG19 backbone.

    :param cfg: VGG configuration.
    """
    return VGG(cfg)

@Backbone.register(config=vgg_configs["vgg11"])
def vgg11(cfg: VGGConfig) -> VGG:
    """Factory function to create a VGG11 backbone.

    :param cfg: VGG configuration.
    """
    return VGG(cfg)

@Backbone.register(config=vgg_configs["vgg13"])
def vgg13(cfg: VGGConfig) -> VGG:
    """Factory function to create a VGG13 backbone.

    :param cfg: VGG configuration.
    """
    return VGG(cfg)
