from __future__ import annotations

import torch

from ..core.factory import Factory
from ._registry import sampler_registry


class Sampler(Factory["Sampler"]):
    """Base class for balanced positive/negative samplers used by RPN and RCNN heads."""
    _registry = sampler_registry

    def sample(self, labels: torch.Tensor, *, positive_value: int = 1, negative_value: int = 0) -> torch.Tensor:
        raise NotImplementedError("BalancedSampler subclasses must implement the sample method.")