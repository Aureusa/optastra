import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Sequence

from ..nn.blocks.convolution.conv_norm_act import ConvNormAct
from ..nn.features import FeatureMaps, FeatureSpec
from .base import ProposalGenerator
from ._registry import register_proposal_generator


@dataclass
class RPNConfig:
    num_anchors: int = 3
    box_dim: int = 4
    conv_dims: Sequence[int] = (-1,)
    in_features: tuple[str, ...] = ()


class RPN(ProposalGenerator):
    """
    Simplified Detectron2-style RPN head.

    The module consumes one feature map or a list of feature maps and predicts:
    - objectness logits per anchor location
    - box deltas per anchor location
    """

    def __init__(
        self,
        in_spec: FeatureSpec,
        cfg: RPNConfig,
    ):
        super().__init__()
        in_spec.require("channels", "strides")

        stage_names = tuple(sorted(in_spec.channels.keys())) if not cfg.in_features else cfg.in_features
        for name in stage_names:
            if name not in in_spec.channels:
                raise ValueError(f"Requested feature '{name}' is missing from in_spec.channels")

        in_channels_per_stage = [in_spec.channels[name] for name in stage_names]
        if len(set(in_channels_per_stage)) != 1:
            raise ValueError("RPN expects selected input feature maps to have the same channel count")

        in_channels = in_channels_per_stage[0]
        num_anchors = cfg.num_anchors
        box_dim = cfg.box_dim
        conv_dims = cfg.conv_dims

        self.in_features = stage_names
        cur_channels = in_channels

        if len(conv_dims) == 1:
            out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]
            if out_channels <= 0:
                raise ValueError(f"Conv output channels must be > 0, got {out_channels}")
            self.conv = self._make_conv(cur_channels, out_channels)
            cur_channels = out_channels
        else:
            convs: list[nn.Module] = []
            for conv_dim in conv_dims:
                out_channels = cur_channels if conv_dim == -1 else conv_dim
                if out_channels <= 0:
                    raise ValueError(f"Conv output channels must be > 0, got {out_channels}")
                convs.append(self._make_conv(cur_channels, out_channels))
                cur_channels = out_channels
            self.conv = nn.Sequential(*convs)

        self.cls_logits = nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(cur_channels, num_anchors * box_dim, kernel_size=1, stride=1)

        self.out_spec = FeatureSpec(
            channels={
                **{f"{name}_objectness": num_anchors for name in self.in_features},
                **{f"{name}_deltas": num_anchors * box_dim for name in self.in_features},
            },
            strides={
                **{f"{name}_objectness": in_spec.strides[name] for name in self.in_features},
                **{f"{name}_deltas": in_spec.strides[name] for name in self.in_features},
            },
        )

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    @staticmethod
    def _make_conv(in_channels: int, out_channels: int) -> nn.Module:
        return ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=None,
            activation="relu",
        )

    def forward(self, features: FeatureMaps) -> FeatureMaps:
        objectness_maps: dict[str, torch.Tensor] = {}
        delta_maps: dict[str, torch.Tensor] = {}

        for name in self.in_features:
            feat = features.feature_maps[name]
            hidden = self.conv(feat)
            objectness_maps[f"{name}_objectness"] = self.cls_logits(hidden)
            delta_maps[f"{name}_deltas"] = self.bbox_pred(hidden)

        return FeatureMaps(feature_maps={**objectness_maps, **delta_maps})


rpn_configs = {
    "rpn": RPNConfig(),
}


@register_proposal_generator(config=rpn_configs["rpn"])
def rpn(in_spec: FeatureSpec, cfg: RPNConfig) -> RPN:
    return RPN(in_spec, cfg)
    