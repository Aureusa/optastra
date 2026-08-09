from __future__ import annotations
import torch
import torch.nn as nn


class StochasticDepth(nn.Module):
    """Per-sample residual-branch dropout (DropPath). Identity at eval."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob) # Bernoulli distribution to create a mask
        return x * mask / keep_prob
    