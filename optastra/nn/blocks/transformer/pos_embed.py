from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "LearnedPosEmbed",
    "SinusoidalPosEmbed",
    "RotaryPosEmbed2D",
    "interpolate_pos_embed",
    "get_2d_sincos_pos_embed",
]


def _as_hw(grid_size: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize a grid size given as an int (square) or (H, W) to an (H, W) tuple."""
    if isinstance(grid_size, int):
        return grid_size, grid_size
    h, w = grid_size
    return int(h), int(w)


def interpolate_pos_embed(
        pos_embed: torch.Tensor,
        old_grid_size: int | tuple[int, int],
        new_grid_size: int | tuple[int, int],
        num_prefix_tokens: int = 1,
    ) -> torch.Tensor:
    """
    Resize a learned positional embedding grid via bicubic interpolation --
    lets a ViT pretrained at one img_size be fine-tuned (or run) at another
    without retraining pos_embed from scratch.

    :param pos_embed: (1, num_prefix_tokens + H_old * W_old, dim).
    :param old_grid_size: grid the embedding was trained on, int (square) or (H, W).
    :param new_grid_size: grid to resize to, int (square) or (H, W).
    :param num_prefix_tokens: leading tokens (e.g. the cls token) that are copied unchanged.
    :return: (1, num_prefix_tokens + H_new * W_new, dim).
    """
    (old_h, old_w), (new_h, new_w) = _as_hw(old_grid_size), _as_hw(new_grid_size)
    if (old_h, old_w) == (new_h, new_w):
        return pos_embed
    prefix_pos, patch_pos = pos_embed[:, :num_prefix_tokens], pos_embed[:, num_prefix_tokens:]
    dim = pos_embed.shape[-1]
    patch_pos = patch_pos.reshape(1, old_h, old_w, dim).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(new_h, new_w), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_h * new_w, dim)
    return torch.cat([prefix_pos, patch_pos], dim=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int | tuple[int, int], cls_token: bool = True) -> torch.Tensor:
    """Returns (1, H*W [+1], embed_dim) fixed sinusoidal positions.
    embed_dim must be divisible by 4 (split in half for each axis, then
    half again for sin/cos). The first half of the channels encodes the row,
    the second half the column; the optional cls position is all zeros."""
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by 4 for 2D sincos.")

    grid_h, grid_w = _as_hw(grid_size)
    rows, cols = torch.meshgrid(
        torch.arange(grid_h, dtype=torch.float32),
        torch.arange(grid_w, dtype=torch.float32),
        indexing="ij",
    )   # each (H, W); flattened row-major -> same order as the patch tokens

    pos_embed_h = _sincos_1d(embed_dim // 2, rows.reshape(-1))
    pos_embed_w = _sincos_1d(embed_dim // 2, cols.reshape(-1))
    pos_embed = torch.cat([pos_embed_h, pos_embed_w], dim=1)   # (H*W, embed_dim)

    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    return pos_embed.unsqueeze(0)   # (1, N [+1], embed_dim)


def _sincos_1d(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    omega = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    out = positions.unsqueeze(1) * omega.unsqueeze(0)   # (N, embed_dim//2)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)   # (N, embed_dim)


class SinusoidalPosEmbed(nn.Module):
    """Fixed 2D sin-cos positional embedding (no learnable parameters).

    The table is a deterministic function of (embed_dim, grid), so it is stored
    as a NON-persistent buffer: it moves with .to(device) but is not written to
    (or expected in) state_dicts -- every model rebuilds the identical table at
    construction. When forward() sees a different grid (another input size), the
    table for that grid is computed on the fly."""

    def __init__(self, embed_dim: int, grid_size: int | tuple[int, int], cls_token: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = _as_hw(grid_size)
        self.cls_token = cls_token
        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.grid_size, cls_token)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def forward(self, x: torch.Tensor, grid_size: int | tuple[int, int] | None = None) -> torch.Tensor:
        grid_size = self.grid_size if grid_size is None else _as_hw(grid_size)
        if grid_size == self.grid_size:
            pos_embed = self.pos_embed
        else:
            pos_embed = get_2d_sincos_pos_embed(self.embed_dim, grid_size, self.cls_token).to(x)
        return x + pos_embed


class LearnedPosEmbed(nn.Module):
    """Learned positional embedding for a reference grid of patches (+ optional cls).

    Inputs on a different grid use a bicubically interpolated copy (forward), and
    checkpoints saved at another grid size are interpolated on load
    (_load_from_state_dict) -- so `load_state_dict` just works across img_size."""

    def __init__(self, embed_dim: int, grid_size: int | tuple[int, int], cls_token: bool = True):
        super().__init__()
        self.grid_size = _as_hw(grid_size)
        self.num_prefix_tokens = 1 if cls_token else 0
        num_tokens = self.grid_size[0] * self.grid_size[1] + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02) # Truncated normal initialization, std=0.02 is standard for ViT

    def forward(self, x: torch.Tensor, grid_size: int | tuple[int, int] | None = None) -> torch.Tensor:
        grid_size = self.grid_size if grid_size is None else _as_hw(grid_size)
        pos_embed = interpolate_pos_embed(self.pos_embed, self.grid_size, grid_size, self.num_prefix_tokens)
        return x + pos_embed

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # A checkpoint trained at another img_size has a different number of
        # positions: resize it to this module's grid before the normal copy.
        # (Only square checkpoint grids can be inferred from the token count.)
        key = prefix + "pos_embed"
        loaded = state_dict.get(key)
        if loaded is not None and loaded.shape[1] != self.pos_embed.shape[1]:
            old_num_patches = loaded.shape[1] - self.num_prefix_tokens
            old_side = math.isqrt(old_num_patches)
            if old_side * old_side == old_num_patches:
                state_dict[key] = interpolate_pos_embed(
                    loaded, old_side, self.grid_size, self.num_prefix_tokens
                )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


# ---------------------------------------------------------------------------
# 3. Rotary (2D RoPE) -- Experimental, not used by the built-in ViT yet
# ---------------------------------------------------------------------------

class RotaryPosEmbed2D(nn.Module):
    """Precomputes rotation frequencies for 2D RoPE. Call `.rotate(q_or_k)`
    inside attention, on q and k separately, AFTER the qkv projection and
    BEFORE the attention matmul -- this class does not touch token
    embeddings or the residual stream directly.

    Half of the rotated frequency pairs encode the patch row, the other half
    the column, so q_i . k_j depends only on the (row, col) offset between i and j."""

    def __init__(self, head_dim: int, grid_size: int):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by 4 for 2D RoPE.")
        freqs = 1.0 / (10000 ** (torch.arange(0, head_dim // 4, dtype=torch.float32) / (head_dim // 4)))
        grid = torch.arange(grid_size, dtype=torch.float32)
        freqs_grid = torch.outer(grid, freqs)   # (grid_size, head_dim//4)
        # combine h and w frequencies, one half of head_dim per axis
        freqs_h = freqs_grid.unsqueeze(1).expand(-1, grid_size, -1).reshape(-1, head_dim // 4)
        freqs_w = freqs_grid.unsqueeze(0).expand(grid_size, -1, -1).reshape(-1, head_dim // 4)
        freqs_full = torch.cat([freqs_h, freqs_w], dim=-1)   # (grid_size**2, head_dim//2)
        cos, sin = freqs_full.cos(), freqs_full.sin()
        self.register_buffer("cos", torch.cat([cos, cos], dim=-1), persistent=False)  # (N, head_dim)
        self.register_buffer("sin", torch.cat([sin, sin], dim=-1), persistent=False)

    def rotate(self, x: torch.Tensor, num_prefix_tokens: int = 1) -> torch.Tensor:
        """x: (B, num_heads, N, head_dim). num_prefix_tokens (e.g. cls token)
        are passed through unrotated -- RoPE only applies to patch tokens."""
        prefix, patches = x[:, :, :num_prefix_tokens], x[:, :, num_prefix_tokens:]
        x1, x2 = patches.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        patches = patches * self.cos + rotated * self.sin
        return torch.cat([prefix, patches], dim=2)   # re-attach prefix along the token dim
