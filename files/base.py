from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class BackboneFeatures:
    """Structured backbone output so CNNs and transformers share one interface.

    feature_maps: dict of stage_name -> tensor, e.g. {"C2": ..., "C3": ..., ...}
        Only CNNs populate this in general; used by necks like FPN.
    pooled: optional (B, C) global embedding, e.g. after global average pool.
    patch_tokens / cls_token: optional, populated by transformer backbones.
    """

    feature_maps: dict[str, torch.Tensor] = field(default_factory=dict)
    pooled: torch.Tensor | None = None
    patch_tokens: torch.Tensor | None = None
    cls_token: torch.Tensor | None = None


class Backbone(nn.Module):
    """A backbone only produces features -- it knows nothing about tasks."""

    #: maps stage name -> output channel count, e.g. {"C2": 256, "C3": 512, ...}
    out_channels: dict[str, int]
    #: maps stage name -> stride relative to input image, e.g. {"C2": 4, "C3": 8, ...}
    out_strides: dict[str, int]

    def forward(self, images: torch.Tensor) -> BackboneFeatures:
        raise NotImplementedError
