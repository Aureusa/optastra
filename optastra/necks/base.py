from __future__ import annotations

from abc import ABC
import torch.nn as nn

from ..core.factory import SpecFactory
from ..core.registry import FamilyRegistry
from ..nn.features import FeatureMaps, FeatureSpec


__all__ = ["Neck"]


class Neck(nn.Module, SpecFactory["Neck"], ABC):
    """A neck only produces features -- it knows nothing about tasks."""

    _registry = FamilyRegistry("neck")
    out_spec: FeatureSpec

    @classmethod
    def _post_create(cls, neck: "Neck") -> "Neck":
        """Ensure every created neck exposes a valid FeatureSpec."""
        if not isinstance(neck.out_spec, FeatureSpec):
            raise ValueError(
                f"{neck.__class__.__name__} must define an 'out_spec' attribute of type FeatureSpec. Check docs for details."
            )
        return neck

    def forward(self, features: FeatureMaps) -> FeatureMaps:
        raise NotImplementedError
    