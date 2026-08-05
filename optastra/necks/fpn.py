from __future__ import annotations

from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F

from .base import Neck
from ._registry import register_neck
from ..nn.blocks.convolution.conv_norm_act import ConvNormAct
from ..nn.features import FeatureSpec, FeatureMaps


__all__ = ["FPN"]


@dataclass
class FPNConfig:
    """Config for the FPN neck."""
    out_channels: int = 256
    preact: bool = False


class FPN(Neck):
    """Feature Pyramid Network (Lin et al., CVPR 2017, arXiv:1612.03144).

    Consumes multi-stage backbone features (e.g. C2-C5) and produces a pyramid
    of feature maps (P2-P5) at a common channel width, each carrying both the
    fine spatial detail of shallow stages and the strong semantics of deep ones.
    """

    def __init__(
        self,
        in_spec: FeatureSpec,
        cfg: FPNConfig,
    ):
        super().__init__()
        self.cfg = cfg
        # Unpack the cfg into local variables for convenience
        in_spec.require("channels", "strides") # Ensure that the in_spec has both channels and strides defined
        in_channels = in_spec.channels
        out_channels = cfg.out_channels
        preact = cfg.preact

        self.stage_names = sorted(in_channels.keys())  # e.g. ["C2", "C3", "C4", "C5"]

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

        self.out_spec = FeatureSpec(
            channels={name.replace("C", "P"): out_channels for name in self.stage_names},
            strides={name.replace("C", "P"): 2 ** (self.stage_names.index(name) + 2) for name in self.stage_names},
        )

    def forward(self, features: FeatureMaps) -> FeatureMaps:
        laterals = {
            name: self.laterals[name](features.feature_maps[name])
            for name in self.stage_names
        }

        # top-down pathway: start from the deepest stage, upsample + add into shallower ones
        merged = {self.stage_names[-1]: laterals[self.stage_names[-1]]}
        for name in reversed(self.stage_names[:-1]):
            deeper_name = self.stage_names[self.stage_names.index(name) + 1]
            upsampled = F.interpolate(
                merged[deeper_name], size=laterals[name].shape[-2:], mode="nearest"
            )
            merged[name] = laterals[name] + upsampled

        # 3x3 smoothing conv per level to reduce aliasing from the upsample-add
        outputs = {
            name.replace("C", "P"): self.outputs[name](merged[name])
            for name in self.stage_names
        }
        return FeatureMaps(feature_maps=outputs)


fpn_configs = {
    "fpn": FPNConfig(
        out_channels=256,
        preact=False,
    )
}


@register_neck(config=fpn_configs["fpn"])
def fpn(in_spec: FeatureSpec, cfg: FPNConfig) -> FPN:
    """Factory function to create an FPN neck.

    :param in_spec: FeatureSpec instance describing the output of a preceeding feature extractor
    :param cfg: FPNConfig instance containing the configuration for the FPN
    :return: FPN instance
    """
    return FPN(in_spec, cfg)
