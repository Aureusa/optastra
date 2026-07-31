from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from ..backbones.base import BackboneFeatures
from .base import NeckFeatures, Neck
from ._registry import register_neck
from ..nn.blocks.convolution.conv_norm_act import ConvNormAct


__all__ = ["FPN"]


class FPN(Neck):
    """Feature Pyramid Network (Lin et al., CVPR 2017, arXiv:1612.03144).

    Consumes multi-stage backbone features (e.g. C2-C5) and produces a pyramid
    of feature maps (P2-P5) at a common channel width, each carrying both the
    fine spatial detail of shallow stages and the strong semantics of deep ones.
    """

    def __init__(
        self,
        in_channels: dict[str, int],  # e.g. {"C2": 256, "C3": 512, "C4": 1024, "C5": 2048}
        out_channels: int = 256,
        preact: bool = False
    ):
        super().__init__()
        self.stage_names = sorted(in_channels.keys())  # ["C2", "C3", "C4", "C5"]

        self.laterals = nn.ModuleDict(
            {
                name: ConvNormAct(
                    in_channels=in_channels[name],
                    out_channels=out_channels,
                    kernel_size=1,
                    norm=None,
                    activation=None,
                    preact=preact
                )
                for name in self.stage_names
            }
        )
        self.outputs = nn.ModuleDict(
            {
                name: ConvNormAct(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    norm=None,
                    activation=None,
                    preact=preact
                )
                for name in self.stage_names
            }
        )

    def forward(self, features: BackboneFeatures) -> NeckFeatures:
        laterals = {
            name: self.laterals[name](features.feature_maps[name])
            for name in self.stage_names
        }

        # top-down pathway: start from the deepest stage, upsample + add into shallower ones
        merged = {self.stage_names[-1]: laterals[self.stage_names[-1]]}
        for name in reversed(self.stage_names[:-1]):
            deeper_name = self.stage_names[self.stage_names.index(name) + 1]
            upsampled = F.interpolate(
                merged[deeper_name], scale_factor=2, mode="nearest"
            )
            merged[name] = laterals[name] + upsampled

        # 3x3 smoothing conv per level to reduce aliasing from the upsample-add
        outputs = {
            name.replace("C", "P"): self.outputs[name](merged[name])
            for name in self.stage_names
        }
        return NeckFeatures(feature_maps=outputs)


@register_neck
def fpn(**kwargs) -> FPN:
    """Factory function to create an FPN neck.

    :param in_channels: dict of stage_name -> channel count, e.g. {"C2": 256, "C3": 512, "C4": 1024, "C5": 2048}
    :param out_channels: int, number of channels in the output feature maps (default: 256)
    :param preact: bool, whether to use pre-activation in the convolution
    :return: FPN instance
    """
    return FPN(**kwargs)
