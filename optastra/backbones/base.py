from __future__ import annotations

from abc import ABC
import torch
import torch.nn as nn

from ..core.factory import Factory
from ..core.registry import FamilyRegistry
from ..nn.features import FeatureMaps, FeatureSpec


__all__ = ["Backbone"]


class Backbone(nn.Module, Factory["Backbone"], ABC):
    """A backbone only produces features -- it knows nothing about tasks."""

    out_spec: FeatureSpec  # type: ignore
    _registry = FamilyRegistry("backbone")

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
