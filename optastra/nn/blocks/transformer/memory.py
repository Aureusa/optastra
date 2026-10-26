"""Flatten (multi-level) feature maps into a transformer memory sequence.

Shared by the DETR encoder (a Neck) and query decoders: both must agree on
which cells are real (not batch padding) and where each cell is.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

from .pos_embed import sine_embed_2d


__all__ = ["FlatMemory", "flatten_feature_levels", "unflatten_feature_levels"]


class FlatMemory:
    """memory (B, L, D), pos (B, L, D), mask (B, L) bool (True = real cell),
    centres (B, L, 2) cell centres as fractions of each image's size, and
    shapes [(h, w) per level] to undo the flattening."""

    def __init__(self, memory, pos, mask, centres, shapes):
        self.memory, self.pos, self.mask, self.centres, self.shapes = memory, pos, mask, centres, shapes


def flatten_feature_levels(
    maps: Sequence[torch.Tensor],
    strides: Sequence[int],
    image_sizes: Sequence[tuple[int, int]] | None = None,
    level_embed: torch.Tensor | None = None,
) -> FlatMemory:
    """
    :param maps: per level (B, D, h, w) maps of the padded batch.
    :param strides: per level stride in pixels.
    :param image_sizes: per-image (h, w) before padding; None = every image fills the batch.
    :param level_embed: optional (num_levels, D) added to each level's cells (multi-scale memory).

    A cell (i, j) of a stride-s level is real for an image of size (h, w) if
    i < ceil(h / s) and j < ceil(w / s) -- it sees at least one real pixel, the
    same rule as DETR's down-sampled padding mask. Its centre is
    ((j + 0.5) s / w, (i + 0.5) s / h), which also feeds the sine position
    encoding, so positions are image-relative and padding-independent.
    """
    B = maps[0].shape[0]
    device = maps[0].device
    if image_sizes is None:
        image_sizes = [(maps[0].shape[-2] * strides[0], maps[0].shape[-1] * strides[0])] * B
    sizes = torch.tensor([[w, h] for h, w in image_sizes], device=device, dtype=torch.float32)   # (B, 2) as (w, h)
    mems, poss, masks, centres, shapes = [], [], [], [], []
    for lvl, (fmap, s) in enumerate(zip(maps, strides)):
        _, D, h, w = fmap.shape
        ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * s
        xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * s
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")
        px = torch.stack([cx, cy], dim=-1).reshape(1, h * w, 2)               # pixel centres
        norm = px / sizes[:, None, :]                                          # (B, hw, 2)
        cols = torch.tensor([math.ceil(iw / s) for _, iw in image_sizes], device=device)
        rows = torch.tensor([math.ceil(ih / s) for ih, _ in image_sizes], device=device)
        jj = torch.arange(w, device=device).repeat(h)
        ii = torch.arange(h, device=device).repeat_interleave(w)
        mask = (jj[None, :] < cols[:, None]) & (ii[None, :] < rows[:, None])
        mem = fmap.flatten(2).transpose(1, 2)
        if level_embed is not None:
            mem = mem + level_embed[lvl].to(mem.dtype)
        mems.append(mem)
        poss.append(sine_embed_2d(norm, D).to(fmap.dtype))
        masks.append(mask)
        centres.append(norm)
        shapes.append((h, w))
    return FlatMemory(torch.cat(mems, 1), torch.cat(poss, 1), torch.cat(masks, 1), torch.cat(centres, 1), shapes)


def unflatten_feature_levels(memory: torch.Tensor, shapes: Sequence[tuple[int, int]]) -> list[torch.Tensor]:
    """(B, L, D) -> per level (B, D, h, w)."""
    out, start = [], 0
    for h, w in shapes:
        out.append(memory[:, start:start + h * w].transpose(1, 2).reshape(memory.shape[0], -1, h, w))
        start += h * w
    return out
