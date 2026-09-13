from __future__ import annotations
from abc import ABC
import torch.nn as nn

from ..core.factory import SpecFactory
from ..core.registry import FamilyRegistry
from ..nn.features import FeatureMaps, HeadOutput


__all__ = ["Head"]


class Head(nn.Module, SpecFactory["Head"], ABC):
    """A head only produces features -- it knows nothing about tasks."""

    _registry = FamilyRegistry("head")
    
    def forward(self, features: FeatureMaps) -> HeadOutput:
        raise NotImplementedError
    