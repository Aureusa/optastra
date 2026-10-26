"""Attention with masks, cross-attention and additive biases.

``MultiHeadSelfAttention`` (used by the ViT) takes no mask. Set models --
DETR-style decoders, tokenised tables, relational reasoning over objects --
need padding masks (variable token counts), cross-attention (queries ->
image features) and additive attention biases (e.g. relative geometry), so
these blocks add an ``attn_mask`` argument.

``attn_mask`` follows ``F.scaled_dot_product_attention``: a bool tensor
(True = may attend) or a float tensor added to the logits, broadcastable to
(B, num_heads, L_q, L_k). Every query row must allow at least one key,
otherwise its softmax is undefined (NaN).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..readout.mlp import MLP


__all__ = ["MultiHeadAttention", "MaskedTransformerBlock", "CrossAttentionBlock", "padding_to_attn_mask"]


class MultiHeadAttention(nn.Module):
    """softmax(q k^T / sqrt(head_dim) + mask) v with separate q / k / v projections.

    :param kdim: width of the key / value inputs if it differs from ``dim``.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, kdim: int | None = None):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        kdim = kdim or dim
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kdim, dim)
        self.v = nn.Linear(kdim, dim)
        self.proj = nn.Linear(dim, dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v = self._heads(self.q(query)), self._heads(self.k(key)), self._heads(self.v(value))
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0,
        )
        B, _, L, _ = out.shape
        return self.proj(out.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim))   # explicit: B may be 0


def padding_to_attn_mask(key_mask: torch.Tensor) -> torch.Tensor:
    """(B, L_k) bool, True = real key -> (B, 1, 1, L_k) attn_mask."""
    return key_mask[:, None, None, :]


class MaskedTransformerBlock(nn.Module):
    """Pre-norm self-attention block (as TransformerBlock) with an attn_mask and an
    optional positional term added to queries and keys only."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, num_layers=2, dropout=dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, pos: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        qk = h if pos is None else h + pos
        x = x + self.attn(qk, qk, h, attn_mask)
        return x + self.mlp(self.ln2(x))


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + MLP: queries attend to a memory sequence."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, kdim: int | None = None):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(kdim or dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout, kdim=kdim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, num_layers=2, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.ln_q(x)
        kv = self.ln_kv(memory)
        q_in = q if query_pos is None else q + query_pos
        k_in = kv if memory_pos is None else kv + memory_pos
        x = x + self.attn(q_in, k_in, kv, attn_mask)
        return x + self.mlp(self.ln2(x))
