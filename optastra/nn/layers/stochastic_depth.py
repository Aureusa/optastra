from __future__ import annotations
import torch
import torch.nn as nn


__all__ = ["StochasticDepth", "drop_path_rates"]


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


def drop_path_rates(max_rate: float, num_blocks: int) -> list[float]:
    """
    Per-block drop-path probabilities, increasing linearly from 0 (first
    block) to `max_rate` (last block) -- the standard schedule used by
    ViT, ConvNeXt and EfficientNet.
    """
    return [x.item() for x in torch.linspace(0, max_rate, num_blocks)]
