from __future__ import annotations

from abc import ABC
import os
from typing import Any, Mapping
import torch
import torch.nn as nn

from ..core.factory import Factory
from ..core.registry import FamilyRegistry
from ..nn.features import FeatureMaps, FeatureSpec
from .weights import load_backbone_weights, export_backbone


__all__ = ["Backbone", "load_backbone_weights", "export_backbone"]


class Backbone(nn.Module, Factory["Backbone"], ABC):
    """A backbone only produces features -- it knows nothing about tasks."""

    out_spec: FeatureSpec  # type: ignore
    _registry = FamilyRegistry("backbone")

    @classmethod
    def create(
            cls,
            name: str,
            *,
            weights: str | os.PathLike | Mapping[str, Any] | None = None,
            weights_prefix: str | None = None,
            **overrides,
        ) -> "Backbone":
        """
        Build a registered backbone, optionally loading pretrained weights:

            Backbone.create("resnet50", weights="runs/byol/ckpt_1000.pt")

        :param weights: a state_dict, or a path to one / to an optastra
            checkpoint. The backbone's key prefix inside it
            ("online_backbone.", "backbone.", "0.", ...) is auto-detected;
            loading is strict. See `optastra.backbones.weights`.
        :param weights_prefix: set the key prefix explicitly instead.
        """
        backbone = super().create(name, **overrides)
        if weights is not None:
            load_backbone_weights(backbone, weights, prefix=weights_prefix)
        return backbone

    @classmethod
    def _post_create(cls, backbone: "Backbone") -> "Backbone":
        """Ensure every created backbone exposes a valid FeatureSpec."""
        if not isinstance(backbone.out_spec, FeatureSpec):
            raise ValueError(
                f"{backbone.__class__.__name__} must define an 'out_spec' attribute of type FeatureSpec. Check docs for details."
            )
        return backbone

    def forward(self, images: torch.Tensor) -> FeatureMaps:
        raise NotImplementedError
