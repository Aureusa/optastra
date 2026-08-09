from __future__ import annotations
import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .stochastic_depth import StochasticDepth
from ..readout.mlp import MLP


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block: LN -> MHSA -> residual, LN -> MLP -> residual.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            dropout: float = 0.0,
            attn_dropout: float = 0.0,
            drop_path: float = 0.0
        ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, qkv_bias, attn_dropout, dropout)
        self.drop_path1 = StochasticDepth(drop_path)

        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            num_layers=2,
            dropout=dropout
        )
        self.drop_path2 = StochasticDepth(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.ln1(x)))
        x = x + self.drop_path2(self.mlp(self.ln2(x)))
        return x
    