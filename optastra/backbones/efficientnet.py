"""
EfficientNet backbone, following "EfficientNet: Rethinking Model Scaling
for Convolutional Neural Networks" (Tan & Le, 2019, arXiv:1905.11946).

B0 defines a base architecture (stage widths/depths/kernel sizes/strides).
B1-B7 are NOT separately designed -- they're B0 scaled by a single compound
coefficient phi via:
    depth_multiplier  = alpha ** phi
    width_multiplier  = beta ** phi
    resolution        = base_resolution * (gamma ** phi)
with alpha=1.2, beta=1.1, gamma=1.15 (searched by the paper to satisfy
alpha * beta^2 * gamma^2 ~= 2, so doubling phi ~doubles FLOPs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import torch
import torch.nn as nn

from .base import Backbone
from ..nn.features import FeatureMaps, FeatureSpec
from ..nn.blocks.convolution.mbconv import MBConvBlock
from ..nn.blocks.convolution.conv_norm_act import ConvNormAct

from ._registry import register_backbone


__all__ = ["EfficientNet"]


# B0's base stage definitions: (expand_ratio, channels, num_blocks, stride, kernel_size)
# stage index -> which C-stage it belongs to for FeatureSpec (C2-C5)
_BASE_STAGES = [
    # expand, channels, depth, stride, kernel
    (1,  16, 1, 1, 3),   # stage 1 -- stride 1, stays at stem's stride (part of C2)
    (6,  24, 2, 2, 3),   # stage 2 -- C2 -> C3
    (6,  40, 2, 2, 5),   # stage 3 -- C3 -> C4
    (6,  80, 3, 2, 3),   # stage 4 -- C4 -> C5
    (6, 112, 3, 1, 5),   # stage 5 -- stays at C5's stride
    (6, 192, 4, 2, 5),   # stage 6 -- C5 -> C6 (EfficientNet has one more downsample than ResNet/ConvNeXt)
    (6, 320, 1, 1, 3),   # stage 7 -- stays at C6's stride
]
_BASE_STEM_CHANNELS = 32
_BASE_RESOLUTION = 224


def _round_channels(channels: float, width_mult: float, divisor: int = 8) -> int:
    """EfficientNet's channel-rounding rule: scale, then round to nearest
    multiple of `divisor`, never dropping more than 10% from the scaled value."""
    channels *= width_mult
    new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return int(new_channels)


def _round_depth(depth: int, depth_mult: float) -> int:
    return int(math.ceil(depth * depth_mult))


@dataclass
class EfficientNetConfig:
    width_mult: float = 1.0
    depth_mult: float = 1.0
    resolution: int = 224     # informational only -- backbone itself is resolution-agnostic
    in_channels: int = 3
    se_ratio: float = 0.25
    drop_path_rate: float = 0.2
    dropout: float = 0.2      # note: applies at the classification-head level, not inside the backbone


class EfficientNet(Backbone):
    """Generic EfficientNet, parameterized by (width_mult, depth_mult) --
    every Bn variant is the SAME class with different multipliers, not a
    separate architecture. Produces C2-C5 for FPN compatibility, same as
    ResNet/ConvNeXt (the paper's stage 6 -> C6 downsample is folded into
    C5's feature map so out_spec stays a 4-stage C2-C5 contract, matching
    every other backbone in this framework)."""

    def __init__(self, cfg: EfficientNetConfig):
        super().__init__()
        self.cfg = cfg

        stem_channels = _round_channels(_BASE_STEM_CHANNELS, cfg.width_mult)
        self.stem = ConvNormAct(
            in_channels=cfg.in_channels, out_channels=stem_channels, kernel_size=3, stride=2, padding=1,
            norm="batchnorm", activation="silu",
        )

        total_blocks = sum(_round_depth(d, cfg.depth_mult) for _, _, d, _, _ in _BASE_STAGES)
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        # which base-stage index starts each C-level (after stem's stride-2):
        # stem=stride2 (C1). stage1=stride1 (still C2 territory... but ResNet
        # convention wants C2=stride4) -- stage2 is the first stride-2 -> C2 boundary.
        # We treat: stem+stage1 => C2 (stride 4 total), stage2 => C3, stage3 => C4,
        # stages4+5 => C5 (stage5 stride1, stays at C5), stage6+7 collapse into C5
        # as well since this framework's FeatureSpec is a 4-stage contract.
        c_stage_boundaries = {1: "C2", 2: "C3", 3: "C4", 4: "C5"}  # stage index -> new C-level starts here

        in_ch = stem_channels
        block_idx = 0
        feature_map_channels = {}
        self._stage_to_clevel = []

        current_clevel = "C2"
        for stage_i, (expand, base_ch, base_depth, stride, kernel) in enumerate(_BASE_STAGES):
            out_ch = _round_channels(base_ch, cfg.width_mult)
            depth = _round_depth(base_depth, cfg.depth_mult)

            if stage_i in c_stage_boundaries:
                current_clevel = c_stage_boundaries[stage_i]

            blocks = []
            for j in range(depth):
                blocks.append(MBConvBlock(
                    in_channels=in_ch if j == 0 else out_ch,
                    out_channels=out_ch,
                    kernel_size=kernel,
                    stride=stride if j == 0 else 1,
                    expand_ratio=expand,
                    se_ratio=cfg.se_ratio,
                    drop_path=dpr[block_idx],
                ))
                block_idx += 1
            self.stages.append(nn.Sequential(*blocks))
            self._stage_to_clevel.append(current_clevel)
            feature_map_channels[current_clevel] = out_ch   # last stage writing to this C-level wins
            in_ch = out_ch

        self.out_spec = FeatureSpec(
            channels=feature_map_channels,
            strides={"C2": 4, "C3": 8, "C4": 16, "C5": 32},
        )

    def forward(self, images: torch.Tensor) -> FeatureMaps:
        x = self.stem(images)
        feature_maps = {}
        for stage, clevel in zip(self.stages, self._stage_to_clevel):
            x = stage(x)
            feature_maps[clevel] = x   # later stages at the same clevel overwrite -- correct, we want the LAST one
        return FeatureMaps(feature_maps=feature_maps)


# compound scaling coefficients from the paper (alpha, beta, gamma; alpha*beta^2*gamma^2 ~= 2)
_ALPHA, _BETA, _GAMMA = 1.2, 1.1, 1.15

def _scaled_config(phi: float, resolution: int, dropout: float) -> EfficientNetConfig:
    return EfficientNetConfig(
        depth_mult=_ALPHA ** phi,
        width_mult=_BETA ** phi,
        resolution=resolution,
        dropout=dropout,
    )

efficientnet_configs = {
    "efficientnet_b0": _scaled_config(phi=0, resolution=224, dropout=0.2),
    "efficientnet_b1": _scaled_config(phi=0.5, resolution=240, dropout=0.2),
    "efficientnet_b2": _scaled_config(phi=1, resolution=260, dropout=0.3),
    "efficientnet_b3": _scaled_config(phi=2, resolution=300, dropout=0.3),
    "efficientnet_b4": _scaled_config(phi=3, resolution=380, dropout=0.4),
    "efficientnet_b5": _scaled_config(phi=4, resolution=456, dropout=0.4),
    "efficientnet_b6": _scaled_config(phi=5, resolution=528, dropout=0.5),
    "efficientnet_b7": _scaled_config(phi=6, resolution=600, dropout=0.5),
}


@register_backbone(config=efficientnet_configs["efficientnet_b0"])
def efficientnet_b0(cfg: EfficientNetConfig) -> EfficientNet:
    return EfficientNet(cfg)

@register_backbone(config=efficientnet_configs["efficientnet_b1"])
def efficientnet_b1(cfg: EfficientNetConfig) -> EfficientNet:
    return EfficientNet(cfg)

@register_backbone(config=efficientnet_configs["efficientnet_b2"])
def efficientnet_b2(cfg: EfficientNetConfig) -> EfficientNet:
    return EfficientNet(cfg)

@register_backbone(config=efficientnet_configs["efficientnet_b3"])
def efficientnet_b3(cfg: EfficientNetConfig) -> EfficientNet:
    return EfficientNet(cfg)

@register_backbone(config=efficientnet_configs["efficientnet_b4"])
def efficientnet_b4(cfg: EfficientNetConfig) -> EfficientNet:
    return EfficientNet(cfg)

@register_backbone(config=efficientnet_configs["efficientnet_b5"])
def efficientnet_b5(cfg: EfficientNetConfig) -> EfficientNet:
    return EfficientNet(cfg)

@register_backbone(config=efficientnet_configs["efficientnet_b6"])
def efficientnet_b6(cfg: EfficientNetConfig) -> EfficientNet:
    return EfficientNet(cfg)

@register_backbone(config=efficientnet_configs["efficientnet_b7"])
def efficientnet_b7(cfg: EfficientNetConfig) -> EfficientNet:
    return EfficientNet(cfg)
