"""DETR transformer layers (Carion et al., ECCV 2020, arXiv:2005.12872).

Unlike the ViT's pre-norm ``TransformerBlock``, DETR's layers are post-norm
by default (``normalize_before=False``) and add positional encodings to the
queries and keys of every attention -- never to the values -- so position
steers *where* attention looks without being mixed into the content.

The decoder layer's cross-attention has two flavours:

* ``"standard"`` (DETR): q = tgt + query_pos, k = memory + pos.
* ``"conditional"`` (Conditional DETR, Meng et al., ICCV 2021, arXiv:2108.06152):
  content and spatial parts are kept apart and concatenated per head,
  q_h = [q_content_h, q_spatial_h], k_h = [k_content_h, k_pos_h], where the
  spatial query is the sine embedding of the query's reference point
  (``query_sine_embed``). Used by both Conditional and DAB-DETR.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masked_attention import MultiHeadAttention


__all__ = ["DETREncoderLayer", "DETRDecoderLayer", "ConditionalCrossAttention"]


class _FFN(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


def _with_pos(x: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
    return x if pos is None else x + pos


class DETREncoderLayer(nn.Module):
    """Self-attention over the flattened feature map + FFN."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 normalize_before: bool = False):
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = _FFN(d_model, dim_feedforward, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, pos: torch.Tensor | None = None, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.normalize_before:
            h = self.norm1(src)
            src = src + self.dropout1(self.self_attn(_with_pos(h, pos), _with_pos(h, pos), h, attn_mask))
            return src + self.dropout2(self.ffn(self.norm2(src)))
        qk = _with_pos(src, pos)
        src = self.norm1(src + self.dropout1(self.self_attn(qk, qk, src, attn_mask)))
        return self.norm2(src + self.dropout2(self.ffn(src)))


class ConditionalCrossAttention(nn.Module):
    """Cross-attention with per-head concatenated content and spatial parts
    (Conditional DETR). q / k have 2 * head_dim per head, v has head_dim."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, first_layer: bool = False):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.num_heads, self.head_dim, self.dropout = num_heads, d_model // num_heads, dropout
        self.first_layer = first_layer
        self.q_content = nn.Linear(d_model, d_model)
        self.q_pos = nn.Linear(d_model, d_model) if first_layer else None   # only the first layer adds query_pos
        self.q_sine = nn.Linear(d_model, d_model)
        self.k_content = nn.Linear(d_model, d_model)
        self.k_pos = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, L, d)

    def forward(self, tgt, memory, query_pos, pos, query_sine_embed, attn_mask=None):
        q = self.q_content(tgt)
        k = self.k_content(memory)
        k_pos = self.k_pos(pos) if pos is not None else torch.zeros_like(k)
        if self.first_layer:
            # The first layer's content query has no positional information yet,
            # so (as in Conditional DETR) it also gets the query / memory positions.
            if query_pos is not None:
                q = q + self.q_pos(query_pos)
            k = k + k_pos
        q = torch.cat([self._heads(q), self._heads(self.q_sine(query_sine_embed))], dim=-1)
        k = torch.cat([self._heads(k), self._heads(k_pos)], dim=-1)
        v = self._heads(self.v(memory))
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        B, _, L, _ = out.shape
        return self.proj(out.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim))


class DETRDecoderLayer(nn.Module):
    """Self-attention among queries, cross-attention to the memory, FFN."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 normalize_before: bool = False, cross_attention: str = "standard",
                 first_layer: bool = False):
        super().__init__()
        if cross_attention not in ("standard", "conditional"):
            raise ValueError(f"cross_attention must be 'standard' or 'conditional', got {cross_attention!r}.")
        self.normalize_before = normalize_before
        self.cross_attention = cross_attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = (MultiHeadAttention(d_model, num_heads, dropout) if cross_attention == "standard"
                           else ConditionalCrossAttention(d_model, num_heads, dropout, first_layer=first_layer))
        self.ffn = _FFN(d_model, dim_feedforward, dropout)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout1, self.dropout2, self.dropout3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)

    def _cross(self, tgt, memory, query_pos, pos, query_sine_embed, attn_mask):
        if self.cross_attention == "standard":
            return self.cross_attn(_with_pos(tgt, query_pos), _with_pos(memory, pos), memory, attn_mask)
        if query_sine_embed is None:
            raise ValueError("Conditional cross-attention needs query_sine_embed (the queries' reference points).")
        return self.cross_attn(tgt, memory, query_pos, pos, query_sine_embed, attn_mask)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        self_attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
        query_sine_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        :param tgt: (B, M, D) query content.
        :param memory: (B, L, D) flattened features; ``pos`` (B, L, D) their positions.
        :param query_pos: (B, M, D) query positions (added to self-attention q / k).
        :param query_sine_embed: (B, M, D) spatial query for conditional cross-attention.
        """
        if self.normalize_before:
            h = self.norm1(tgt)
            qk = _with_pos(h, query_pos)
            tgt = tgt + self.dropout1(self.self_attn(qk, qk, h, self_attn_mask))
            tgt = tgt + self.dropout2(self._cross(self.norm2(tgt), memory, query_pos, pos, query_sine_embed,
                                                  cross_attn_mask))
            return tgt + self.dropout3(self.ffn(self.norm3(tgt)))
        qk = _with_pos(tgt, query_pos)
        tgt = self.norm1(tgt + self.dropout1(self.self_attn(qk, qk, tgt, self_attn_mask)))
        tgt = self.norm2(tgt + self.dropout2(self._cross(tgt, memory, query_pos, pos, query_sine_embed,
                                                         cross_attn_mask)))
        return self.norm3(tgt + self.dropout3(self.ffn(tgt)))
