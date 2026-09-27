from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention:
        softmax(q @ k^T / sqrt(head_dim)) @ v, per head, then an output projection.

    The attention itself is computed by F.scaled_dot_product_attention, which is
    numerically the same formula but dispatches to fused (flash / memory-efficient)
    kernels when available.
    """

    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True,
                 attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_dropout = attn_dropout   # dropout prob on the attention weights (training only)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # each (B, num_heads, N, head_dim)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            scale=self.scale,
        )                                                   # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)          # (B, N, C)
        return self.proj_dropout(self.proj(out))
