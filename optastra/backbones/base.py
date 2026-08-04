from __future__ import annotations

from abc import ABC
from dataclasses import fields, replace
from typing import Any
import torch
import torch.nn as nn

from ._registry import _registry
from ..core.factory import Factory
from ..nn.features import FeatureMaps, FeatureSpec


__all__ = ["Backbone"]


class Backbone(nn.Module, Factory["Backbone"], ABC):
    """A backbone only produces features -- it knows nothing about tasks."""

    out_spec: FeatureSpec  # type: ignore
    _registry = _registry

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
