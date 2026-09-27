"""
EfficientNet backbone, following "EfficientNet: Rethinking Model Scaling
for Convolutional Neural Networks" (Tan & Le, 2019, arXiv:1905.11946).

B0 defines a base architecture (stage widths/depths/kernel sizes/strides).
B1-B7 are NOT separately designed -- they're B0 scaled by a compound
coefficient phi:
    depth_multiplier  = alpha ** phi
    width_multiplier  = beta ** phi
    resolution        = base_resolution * (gamma ** phi)
with alpha=1.2, beta=1.1, gamma=1.15 (searched by the paper to satisfy
alpha * beta^2 * gamma^2 ~= 2, so doubling phi ~doubles FLOPs).

The released B1-B7 models use rounded, hand-picked multipliers rather than
the exact powers, so the variant table below copies the official values
(from the reference TF implementation / timm) instead of recomputing them.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn

from .base import Backbone
from ..nn.features import FeatureMaps, FeatureSpec
from ..nn.layers import drop_path_rates
from ..nn.blocks.convolution.mbconv import MBConvBlock
from ..nn.blocks.convolution.conv_norm_act import ConvNormAct


__all__ = ["EfficientNet", "EfficientNetConfig", "EFFICIENTNET_VARIANTS"]


# B0's base stage definitions. The stem already has stride 2, so the running
# stride after each stage is: 2, 4, 8, 16, 16, 32, 32.
_BASE_STAGES = [
    # expand, channels, depth, stride, kernel
    (1,  16, 1, 1, 3),   # stage 1 -- stride 2  (C1)
    (6,  24, 2, 2, 3),   # stage 2 -- stride 4  (C2)
    (6,  40, 2, 2, 5),   # stage 3 -- stride 8  (C3)
    (6,  80, 3, 2, 3),   # stage 4 -- stride 16
    (6, 112, 3, 1, 5),   # stage 5 -- stride 16 (C4 = last stage at stride 16)
    (6, 192, 4, 2, 5),   # stage 6 -- stride 32
    (6, 320, 1, 1, 3),   # stage 7 -- stride 32 (C5 = last stage at stride 32)
]
_BASE_STEM_CHANNELS = 32


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
    in_channels: int = 3
    se_ratio: float = 0.25
    drop_path_rate: float = 0.2


class EfficientNet(Backbone):
    """Generic EfficientNet, parameterized by (width_mult, depth_mult) --
    every Bn variant is the SAME class with different multipliers, not a
    separate architecture.

    Produces C2-C5 for FPN compatibility, same as ResNet/ConvNeXt: each
    C-level is the output of the LAST stage running at stride 2**k (the same
    choice as timm's `features_only`), e.g. for B0: C2=24, C3=40, C4=112, C5=320
    channels. The resolution-agnostic backbone works at any input size; the
    paper's per-variant training resolution is listed in EFFICIENTNET_VARIANTS.
    """

    def __init__(self, cfg: EfficientNetConfig):
        super().__init__()
        self.cfg = cfg

        stem_channels = _round_channels(_BASE_STEM_CHANNELS, cfg.width_mult)
        self.stem = ConvNormAct(
            in_channels=cfg.in_channels, out_channels=stem_channels, kernel_size=3, stride=2, padding=1,
            norm="batchnorm", activation="silu",
        )

        total_blocks = sum(_round_depth(d, cfg.depth_mult) for _, _, d, _, _ in _BASE_STAGES)
        dpr = drop_path_rates(cfg.drop_path_rate, total_blocks)

        self.stages = nn.ModuleList()
        self._stage_to_clevel = []     # C-level name of each stage's output, by its running stride
        feature_map_channels = {}
        feature_map_strides = {}

        in_ch = stem_channels
        block_idx = 0
        total_stride = 2               # the stem is stride 2
        for expand, base_ch, base_depth, stride, kernel in _BASE_STAGES:
            out_ch = _round_channels(base_ch, cfg.width_mult)
            depth = _round_depth(base_depth, cfg.depth_mult)

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
            in_ch = out_ch

            total_stride *= stride
            clevel = f"C{int(math.log2(total_stride))}"
            self._stage_to_clevel.append(clevel)
            if clevel != "C1":         # out_spec is the usual C2-C5 contract
                feature_map_channels[clevel] = out_ch   # last stage at this stride wins
                feature_map_strides[clevel] = total_stride

        self.out_spec = FeatureSpec(
            channels=feature_map_channels,
            strides=feature_map_strides,
        )

    def forward(self, images: torch.Tensor) -> FeatureMaps:
        x = self.stem(images)
        feature_maps = {}
        for stage, clevel in zip(self.stages, self._stage_to_clevel):
            x = stage(x)
            if clevel in self.out_spec.channels:
                feature_maps[clevel] = x   # later stages at the same stride overwrite -- we want the LAST one
        return FeatureMaps(feature_maps=feature_maps)


# Official per-variant values: (width_mult, depth_mult, train resolution, head dropout).
# Only the multipliers configure the backbone; resolution and dropout are listed as
# the paper's recommended input size and classifier-head dropout for that variant.
EFFICIENTNET_VARIANTS: dict[str, tuple[float, float, int, float]] = {
    "efficientnet_b0": (1.0, 1.0, 224, 0.2),
    "efficientnet_b1": (1.0, 1.1, 240, 0.2),
    "efficientnet_b2": (1.1, 1.2, 260, 0.3),
    "efficientnet_b3": (1.2, 1.4, 300, 0.3),
    "efficientnet_b4": (1.4, 1.8, 380, 0.4),
    "efficientnet_b5": (1.6, 2.2, 456, 0.4),
    "efficientnet_b6": (1.8, 2.6, 528, 0.5),
    "efficientnet_b7": (2.0, 3.1, 600, 0.5),
}

efficientnet_configs = {
    name: EfficientNetConfig(width_mult=width, depth_mult=depth)
    for name, (width, depth, _resolution, _dropout) in EFFICIENTNET_VARIANTS.items()
}

# Every variant is the same class with a different config, so instead of one
# 3-line factory function per variant we register the class itself under each name.
for _name, _cfg in efficientnet_configs.items():
    Backbone.register(EfficientNet, config=_cfg, name=_name)
