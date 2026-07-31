from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..backbones.base import BackboneFeatures

from ._registry import get_neck_entrypoint


__all__ = ["Neck", "NeckFeatures"]

@dataclass
class NeckFeatures:
    """Structured neck output so CNNs and transformers share one interface.

    feature_maps: dict of stage_name -> tensor, e.g. {"P2": ..., "P3": ..., ...}
        Only CNNs populate this in general; used by heads like RetinaNet.
    pooled: optional (B, C) global embedding, e.g. after global average pool.
    patch_tokens / cls_token: optional, populated by transformer necks.
    """

    feature_maps: dict[str, torch.Tensor] = field(default_factory=dict)
    pooled: torch.Tensor | None = None
    patch_tokens: torch.Tensor | None = None
    cls_token: torch.Tensor | None = None


class Neck(nn.Module, ABC):
    """A neck only produces features -- it knows nothing about tasks."""

    #: maps stage name -> output channel count, e.g. {"P2": 256, "P3": 256, ...}
    out_channels: dict[str, int]
    #: maps stage name -> stride relative to input image, e.g. {"P2": 4, "P3": 8, ...}
    out_strides: dict[str, int]

    @classmethod
    def create(cls, name: str) -> Neck:
        """Create a neck by name.

        Args:
            name: Name of the neck to create.
            **kwargs: Additional keyword arguments passed to the neck constructor.

        Returns:
            An instance of the requested neck.
        """
        entrypoint = get_neck_entrypoint(name)
        return entrypoint()

    def forward(self, features: BackboneFeatures) -> NeckFeatures:
        raise NotImplementedError