from __future__ import annotations
import torch
import torch.nn as nn


__all__ = ["LearnedPosEmbed", "SinusoidalPosEmbed", "RotaryPosEmbed2D", "interpolate_pos_embed"]


def interpolate_pos_embed(pos_embed: torch.Tensor, old_grid_size: int, new_grid_size: int) -> torch.Tensor:
    """
    Resize a learned positional embedding grid via bicubic interpolation --
    lets a ViT pretrained at one img_size be fine-tuned at another without
    retraining pos_embed from scratch.
    """
    cls_pos, patch_pos = pos_embed[:, :1], pos_embed[:, 1:]
    dim = pos_embed.shape[-1]
    patch_pos = patch_pos.reshape(1, old_grid_size, old_grid_size, dim).permute(0, 3, 1, 2)
    patch_pos = torch.nn.functional.interpolate(
        patch_pos, size=(new_grid_size, new_grid_size), mode="bicubic", align_corners=False
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid_size * new_grid_size, dim)
    return torch.cat([cls_pos, patch_pos], dim=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = True) -> torch.Tensor:
    """Returns (1, grid_size**2 [+1], embed_dim) fixed sinusoidal positions.
    embed_dim must be divisible by 4 (split in half for each axis, then
    half again for sin/cos)."""
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by 4 for 2D sincos.")

    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="ij")   # (2, grid_size, grid_size)
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)

    pos_embed_h = _sincos_1d(embed_dim // 2, grid[0].reshape(-1))
    pos_embed_w = _sincos_1d(embed_dim // 2, grid[1].reshape(-1))
    pos_embed = torch.cat([pos_embed_h, pos_embed_w], dim=1)   # (grid_size**2, embed_dim)

    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    return pos_embed.unsqueeze(0)   # (1, N [+1], embed_dim)


def _sincos_1d(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    omega = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    out = positions.unsqueeze(1) * omega.unsqueeze(0)   # (N, embed_dim//2)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)   # (N, embed_dim)


class SinusoidalPosEmbed(nn.Module):
    """Fixed positional embedding -- computed once at construction, stored
    as a buffer (moves with .to(device), never appears in state_dict
    optimizer updates, and by default IS saved/loaded via state_dict
    unless persistent=False)."""

    def __init__(self, embed_dim: int, grid_size: int, cls_token: bool = True):
        super().__init__()
        pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed
    

class LearnedPosEmbed(nn.Module):
    def __init__(self, embed_dim: int, grid_size: int, cls_token: bool = True):
        super().__init__()
        num_tokens = grid_size ** 2 + (1 if cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02) # Truncated normal initialization, std=0.02 is standard for ViT

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed


# ---------------------------------------------------------------------------
# 3. Rotary (2D RoPE) -- Experimental
# ---------------------------------------------------------------------------

class RotaryPosEmbed2D(nn.Module):
    """Precomputes rotation frequencies for 2D RoPE. Call `.rotate(q_or_k)`
    inside attention, on q and k separately, AFTER the qkv projection and
    BEFORE the attention matmul -- this class does not touch token
    embeddings or the residual stream directly."""

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
        return torch.cat([prefix, patches], dim=1) if num_prefix_tokens else patches
    