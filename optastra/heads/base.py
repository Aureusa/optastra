from __future__ import annotations
from abc import ABC
import torch.nn as nn

from ._registry import _registry
from ..core.factory import SpecFactory
from ..nn.features import FeatureMaps, HeadOutput


__all__ = ["Head"]


class Head(nn.Module, SpecFactory["Head"], ABC):
    """A head only produces features -- it knows nothing about tasks."""

    _registry = _registry
    
    def forward(self, features: FeatureMaps) -> HeadOutput:
        raise NotImplementedError
    