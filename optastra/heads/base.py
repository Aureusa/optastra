from __future__ import annotations
from abc import ABC
import torch
import torch.nn as nn

from ..core.factory import SpecFactory
from ..core.registry import FamilyRegistry
from ..nn.features import FeatureMaps, HeadOutput


__all__ = ["Head"]


class Head(nn.Module, SpecFactory["Head"], ABC):
    """
    A head turns features into predictions: it consumes the FeatureMaps of a
    backbone/neck (validated against their FeatureSpec at construction time)
    and returns a HeadOutput (logits, regression values, boxes, masks, ...).
    It computes no losses and knows nothing about targets -- that is the Task's job.
    """

    _registry = FamilyRegistry("head")

    def forward(self, features: FeatureMaps) -> HeadOutput:
        raise NotImplementedError

    def _pooled(self, features: FeatureMaps) -> torch.Tensor:
        """features.pooled, with a clear error when the upstream stage didn't produce one."""
        if features.pooled is None:
            raise ValueError(
                f"{self.__class__.__name__} reads FeatureMaps.pooled, but it is None. Put a pooling "
                f"neck (e.g. 'global_avg_pool' or 'token_pool') between a spatial backbone and this head."
            )
        return features.pooled
