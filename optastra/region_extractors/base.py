from __future__ import annotations

from abc import ABC
import torch.nn as nn

from ._registry import _registry
from ..nn.features import FeatureMaps, FeatureSpec
from ..core.factory import SpecFactory


__all__ = ["RegionExtractor"]


class RegionExtractor(nn.Module, SpecFactory["RegionExtractor"], ABC):
    out_spec: FeatureSpec
    _registry = _registry

    @classmethod
    def _post_create(cls, module: "RegionExtractor") -> "RegionExtractor":
        if not isinstance(module.out_spec, FeatureSpec):
            raise ValueError(
                f"{module.__class__.__name__} must define 'out_spec' as FeatureSpec."
            )
        return module

    def forward(self, features: FeatureMaps, rois):
        raise NotImplementedError
