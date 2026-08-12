from __future__ import annotations
import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """
    LayerNorm over the channel dimension of a (B, C, H, W) tensor.
    nn.LayerNorm only normalizes the last dim, so this permutes to
    channels-last, normalizes, permutes back to channels-first.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)
    