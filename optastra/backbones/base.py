from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, fields, replace
from typing import Any
import torch
import torch.nn as nn

from ._registry import get_backbone_entrypoint, get_backbone_default_config


__all__ = ["Backbone", "BackboneFeatures", "FeatureSpec"]

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


@dataclass
class FeatureSpec:
    """What a backbone/neck produces, so the next component can configure itself."""
    channels: dict[str, int] = field(default_factory=dict)   # stage -> channels
    strides: dict[str, int] = field(default_factory=dict)    # stage -> stride
    embed_dim: int | None = None     # for transformer/global embeddings
    num_tokens: int | None = None


class Backbone(nn.Module, ABC):
    """A backbone only produces features -- it knows nothing about tasks."""

    @classmethod
    def create(cls, name: str, **overrides) -> Backbone: # Factory method to create a backbone by name
        """Create a backbone by name, optionally loading pretrained weights.

        :param name: Name of the backbone to create.
        :param overwrites: Optional keyword arguments to overwrite the default configuration.
        :return: An instance of the backbone.
        """
        entrypoint = get_backbone_entrypoint(name)
        default_cfg = get_backbone_default_config(name)
        cfg = replace(default_cfg, **overrides)  # raises on unknown fields
        return entrypoint(cfg)

    @classmethod
    def describe(cls, name: str) -> dict[str, int]: # Factory method to describe a backbone by name
        """Describe a backbone by name, returning its out_channels and out_strides.

        :param name: Name of the backbone to describe.
        :return: A dictionary containing the out_channels and out_strides of the backbone.
        """
        cfg = get_backbone_default_config(name)
        print(f"{name}:")
        for f in fields(cfg):
            current = getattr(cfg, f.name)
            print(f"  {f.name}: {f.type}  = {current!r}")

    @classmethod
    def config(cls, name: str) -> Any: # Factory method to get the default config of a backbone by name
        """Get the default configuration for a backbone by name.

        :param name: Name of the backbone to get the configuration for.
        :return: The default configuration of the backbone.
        """
        return get_backbone_default_config(name)

    def forward(self, images: torch.Tensor) -> BackboneFeatures:
        raise NotImplementedError
