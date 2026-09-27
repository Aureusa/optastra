from __future__ import annotations

from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F

from .base import Neck
from ..nn.blocks.convolution.conv_norm_act import ConvNormAct
from ..nn.features import FeatureSpec, FeatureMaps


__all__ = ["FPN", "FPNConfig"]


@dataclass
class FPNConfig:
    """Config for the FPN neck."""
    out_channels: int = 256


def _pyramid_name(stage: str) -> str:
    """Backbone stage name -> pyramid level name: "C3" -> "P3" (other names are kept)."""
    return "P" + stage[1:] if stage.startswith("C") else stage


class FPN(Neck):
    """Feature Pyramid Network (Lin et al., CVPR 2017, arXiv:1612.03144).

    Consumes multi-stage backbone features (e.g. C2-C5) and produces a pyramid
    of feature maps (P2-P5) at a common channel width, each carrying both the
    fine spatial detail of shallow stages and the strong semantics of deep ones.

    Stages are ordered by their stride in `in_spec` (finest first), and each
    output level keeps the stride of the stage it came from. A single-stage
    input (e.g. a ViT's stride-16 map) gives a one-level "pyramid".
    """

    def __init__(
        self,
        in_spec: FeatureSpec,
        cfg: FPNConfig,
    ):
        super().__init__()
        self.cfg = cfg
        in_spec.require("channels", "strides") # Ensure that the in_spec has both channels and strides defined
        in_channels = in_spec.channels
        out_channels = cfg.out_channels

        missing = set(in_channels) - set(in_spec.strides)
        if missing:
            raise ValueError(f"FPN needs a stride for every stage; missing strides for {sorted(missing)}.")
        # finest -> coarsest, e.g. ["C2", "C3", "C4", "C5"] (sorting names would put "C10" before "C2")
        self.stage_names = sorted(in_channels, key=lambda name: in_spec.strides[name])

        self.laterals = nn.ModuleDict(
            {
                name: ConvNormAct(
                    in_channels=in_channels[name],
                    out_channels=out_channels,
                    kernel_size=1,
                    norm=None,
                    activation=None,
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
                )
                for name in self.stage_names
            }
        )

        self.out_spec = FeatureSpec(
            channels={_pyramid_name(name): out_channels for name in self.stage_names},
            strides={_pyramid_name(name): in_spec.strides[name] for name in self.stage_names},
        )

    def forward(self, features: FeatureMaps) -> FeatureMaps:
        laterals = {
            name: self.laterals[name](features.feature_maps[name])
            for name in self.stage_names
        }

        # top-down pathway: start from the deepest stage, upsample + add into shallower ones
        merged = {self.stage_names[-1]: laterals[self.stage_names[-1]]}
        for shallow, deeper in zip(reversed(self.stage_names[:-1]), reversed(self.stage_names[1:])):
            upsampled = F.interpolate(
                merged[deeper], size=laterals[shallow].shape[-2:], mode="nearest"
            )
            merged[shallow] = laterals[shallow] + upsampled

        # 3x3 smoothing conv per level to reduce aliasing from the upsample-add
        outputs = {
            _pyramid_name(name): self.outputs[name](merged[name])
            for name in self.stage_names
        }
        return FeatureMaps(feature_maps=outputs)


fpn_configs = {
    "fpn": FPNConfig(
        out_channels=256,
    )
}


@Neck.register(config=fpn_configs["fpn"])
def fpn(in_spec: FeatureSpec, cfg: FPNConfig) -> FPN:
    """Factory function to create an FPN neck.

    :param in_spec: FeatureSpec instance describing the output of a preceeding feature extractor
    :param cfg: FPNConfig instance containing the configuration for the FPN
    :return: FPN instance
    """
    return FPN(in_spec, cfg)
