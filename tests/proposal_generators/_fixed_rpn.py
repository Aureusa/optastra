"""Helpers for hand-built RPN cases: an RPN whose head outputs are fixed tensors."""
import math

import torch
import torch.nn as nn

from optastra.nn.features import FeatureSpec
from optastra.proposal_generators.rpn import RPN, RPNConfig


class FixedOutput(nn.Module):
    """Ignores its input and returns a fixed tensor (kept in the autograd graph via a scalar param)."""

    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.register_buffer("value", value)
        self.gain = nn.Parameter(torch.ones(()))

    def forward(self, x):
        return self.value * self.gain


def make_fixed_rpn(objectness: dict, deltas: dict, strides: dict, **cfg_overrides) -> RPN:
    """RPN over len(strides) levels whose per-level outputs are the given (N, A*K, H, W) tensors."""
    levels = tuple(strides)
    cfg = RPNConfig(anchor_scales=(1.0,), aspect_ratios=(0.5, 1.0, 2.0), in_features=levels, **cfg_overrides)
    rpn = RPN(FeatureSpec(channels={name: 4 for name in levels}, strides=dict(strides)), cfg)

    class PerLevel(nn.Module):
        """The RPN applies one shared conv to every level; dispatch on the feature map's size."""

        def __init__(self, outputs):
            super().__init__()
            self.outputs = nn.ModuleList(FixedOutput(v) for v in outputs.values())

        def forward(self, x):
            for module in self.outputs:
                if module.value.shape[-2:] == x.shape[-2:]:
                    return module(x)
            raise AssertionError("no fixed output for this level")

    rpn.conv = nn.Identity()
    rpn.cls_logits = PerLevel(objectness)
    rpn.bbox_pred = PerLevel(deltas)
    return rpn


def anchor_box(cx: float, cy: float, stride: float, ratio: float) -> list[float]:
    """Independent re-derivation of an anchor (scale 1.0): area stride^2, h / w = ratio."""
    w = math.sqrt(stride * stride / ratio)
    h = ratio * w
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
