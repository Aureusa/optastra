from __future__ import annotations
from abc import ABC, abstractmethod

from ..data.sample import Sample
from ._registry import _registry, _batch_registry
from ..core.factory import Factory


__all__ = ["Transform", "BatchTransform"]


class Transform(ABC, Factory["Transform"]):
    """Operates on one Sample -- must move image and target together
    (a flip that moves pixels but not boxes silently corrupts detection).
    """

    _registry = _registry

    @abstractmethod
    def __call__(self, sample: Sample) -> Sample:
        raise NotImplementedError

class BatchTransform(ABC, Factory["BatchTransform"]):
    """Operates on an already-collated batch dict, not a single Sample --
    for augmentations that mix multiple samples together."""
    _registry = _batch_registry  # separate registry: batch_transforms/_registry.py

    @abstractmethod
    def __call__(self, batch: dict) -> dict:
        raise NotImplementedError
    