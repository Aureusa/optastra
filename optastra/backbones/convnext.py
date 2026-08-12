"""
ConvNeXt backbone implementation, following "A ConvNet for the 2020s"
(Liu et al., 2022, arXiv:2201.03545).

Structurally a ResNet: patchify stem, 4 stages of blocks producing
multi-scale features C2-C5. The "twist" is entirely inside the block
(depthwise conv, inverted bottleneck, LayerNorm, GELU, pre-norm) and in
using explicit strided-conv downsampling layers between stages instead of
stride-2 inside the first block of each stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn as nn

from .base import Backbone
from ..nn.features import FeatureMaps, FeatureSpec
from ..nn.blocks.convolution.convnext import ConvNeXtBlock, ConvNeXtDownsample
from ..nn.blocks.convolution.layernorm2d import LayerNorm2d

from ._registry import register_backbone


__all__ = ["ConvNeXt"]


@dataclass
class ConvNeXtConfig:
    """Config for the ConvNeXt family."""
    depths: list[int] = field(default_factory=lambda: [3, 3, 9, 3])
    dims: list[int] = field(default_factory=lambda: [96, 192, 384, 768])
    in_channels: int = 3
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0
    layer_scale_init: float = 1e-6


class ConvNeXtStem(nn.Module):
    """4x4 conv, stride 4 (non-overlapping 'patchify' stem, unlike ResNet's
    7x7 stride-2 + maxpool). Output stride 4 -- this is C1, matching ResNet's
    stem stride exactly so downstream stage strides line up (C2=4, C3=8, ...)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 96):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=4)
        self.norm = LayerNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))
    

class ConvNeXt(Backbone):
    """Generic ConvNeXt. Returns multi-stage features C2-C5 for use in necks
    like FPN -- same out_spec shape as ResNet, so FPN/detection architectures
    work with either backbone family unmodified.

    Stage strides relative to input: C1 (stem) = 4, C2 = 4, C3 = 8, C4 = 16, C5 = 32.
    """

    def __init__(self, cfg: ConvNeXtConfig):
        """
        Initializes the ConvNeXt backbone.

        :param cfg: ConvNeXtConfig specifying depths, dims, and block hyperparameters.
        """
        super().__init__()
        self.cfg = cfg

        self.stem = ConvNeXtStem(cfg.in_channels, cfg.dims[0])

        # linearly scale drop_path rate across all blocks, deepest block gets
        # the highest rate -- same convention as ViT's drop_path_rate
        total_depth = sum(cfg.depths)
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, total_depth)]

        self.downsamples = nn.ModuleList()
        self.stages = nn.ModuleList()
        block_idx = 0
        for i, (depth, dim) in enumerate(zip(cfg.depths, cfg.dims)):
            if i == 0:
                self.downsamples.append(nn.Identity())   # stem already produced dims[0] at stride 4
            else:
                self.downsamples.append(ConvNeXtDownsample(cfg.dims[i - 1], dim))

            stage = nn.Sequential(*[
                ConvNeXtBlock(
                    dim=dim,
                    mlp_ratio=cfg.mlp_ratio,
                    drop_path=dpr[block_idx + j],
                    layer_scale_init=cfg.layer_scale_init,
                )
                for j in range(depth)
            ])
            self.stages.append(stage)
            block_idx += depth

        self.out_spec = FeatureSpec(
            channels={f"C{i + 2}": cfg.dims[i] for i in range(4)},
            strides={"C2": 4, "C3": 8, "C4": 16, "C5": 32},
        )

    def forward(self, images: torch.Tensor) -> FeatureMaps:
        """
        Forward pass through the ConvNeXt backbone.

        :param images: Input tensor of shape (B, C, H, W).
        :return: FeatureMaps containing feature maps from each stage.
        """
        x = self.stem(images)
        feature_maps = {}
        for i, (downsample, stage) in enumerate(zip(self.downsamples, self.stages)):
            x = downsample(x)
            x = stage(x)
            feature_maps[f"C{i + 2}"] = x
        return FeatureMaps(feature_maps=feature_maps)


convnext_configs = {
    "convnext_tiny": ConvNeXtConfig(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]),
    "convnext_small": ConvNeXtConfig(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768]),
    "convnext_base": ConvNeXtConfig(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]),
    "convnext_large": ConvNeXtConfig(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]),\
    "convnext_xlarge": ConvNeXtConfig(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048]),
}


@register_backbone(config=convnext_configs["convnext_tiny"])
def convnext_tiny(cfg: ConvNeXtConfig) -> ConvNeXt:
    return ConvNeXt(cfg)

@register_backbone(config=convnext_configs["convnext_small"])
def convnext_small(cfg: ConvNeXtConfig) -> ConvNeXt:
    return ConvNeXt(cfg)

@register_backbone(config=convnext_configs["convnext_base"])
def convnext_base(cfg: ConvNeXtConfig) -> ConvNeXt:
    return ConvNeXt(cfg)

@register_backbone(config=convnext_configs["convnext_large"])
def convnext_large(cfg: ConvNeXtConfig) -> ConvNeXt:
    return ConvNeXt(cfg)

@register_backbone(config=convnext_configs["convnext_xlarge"])
def convnext_xlarge(cfg: ConvNeXtConfig) -> ConvNeXt:
    return ConvNeXt(cfg)
