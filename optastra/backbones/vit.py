"""
Vision Transformer backbone, following "An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import torch
import torch.nn as nn

from .base import Backbone
from ..nn.features import FeatureMaps, FeatureSpec
from ..nn.layers import drop_path_rates
from ..nn.blocks.transformer.patch_embed import PatchEmbedding
from ..nn.blocks.transformer.transformer_block import TransformerBlock
from ..nn.blocks.transformer.pos_embed import LearnedPosEmbed, SinusoidalPosEmbed


__all__ = ["ViT", "ViTConfig"]


@dataclass
class ViTConfig:
    img_size: int = 224           # reference size: sizes the learned pos-embed; other sizes are interpolated
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    cls_token: bool = True
    dropout: float = 0.0
    attn_dropout: float = 0.0
    drop_path_rate: float = 0.0   # linearly scaled across depth, standard practice
    pos_embed_dropout: float = 0.0
    pos_embed_type: str = field(default="learned", metadata={"choices": ["learned", "sinusoidal"]})


class ViT(Backbone):
    """
    Vision Transformer. The forward pass returns a FeatureMaps with:

    - ``patch_tokens``: (B, N, D) final-normed patch tokens,
    - ``cls_token``: (B, D) final-normed cls token (None if ``cfg.cls_token=False``),
    - ``pooled``: the cls token, or the mean patch token when there is no cls token,
    - ``feature_maps[level]``: the patch tokens reshaped to a (B, D, H/P, W/P)
      spatial map -- a single-scale dense feature at stride P (``"C4"`` for P=16),
      so pooling necks and FPN can consume a ViT like a one-stage CNN.

    ``out_spec`` therefore carries embed_dim/num_tokens (token view) *and*
    channels/strides for that one level (spatial view). ``num_tokens`` is the
    patch count at the reference ``img_size``; forward() accepts any input size
    (learned positions are interpolated, sin-cos positions are recomputed).
    """
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg

        pos_embed_cls = {"learned": LearnedPosEmbed, "sinusoidal": SinusoidalPosEmbed}[cfg.pos_embed_type]

        self.patch_embed = PatchEmbedding(cfg.img_size, cfg.patch_size, cfg.in_channels, cfg.embed_dim)
        num_patches = self.patch_embed.num_patches

        self.num_prefix_tokens = 1 if cfg.cls_token else 0
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim)) if cfg.cls_token else None
        self.pos_embed = pos_embed_cls(cfg.embed_dim, self.patch_embed.grid_size, cls_token=cfg.cls_token)
        self.pos_dropout = nn.Dropout(cfg.pos_embed_dropout)

        dpr = drop_path_rates(cfg.drop_path_rate, cfg.depth)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=cfg.embed_dim, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias, dropout=cfg.dropout, attn_dropout=cfg.attn_dropout,
                drop_path=dpr[i],
            )
            for i in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)

        # Name the dense map after the pyramid level closest to its stride (P=16 -> "C4", P=8 -> "C3").
        self.feature_level = f"C{round(math.log2(cfg.patch_size))}"
        self.out_spec = FeatureSpec(
            channels={self.feature_level: cfg.embed_dim},
            strides={self.feature_level: cfg.patch_size},
            embed_dim=cfg.embed_dim,
            num_tokens=num_patches,
        )

        self._init_weights()

    def _init_weights(self):
        # The positional embedding initializes itself (trunc-normal if learned, sin-cos if fixed).
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, images: torch.Tensor) -> FeatureMaps:
        B, _, H, W = images.shape
        grid_size = (H // self.cfg.patch_size, W // self.cfg.patch_size)

        # Create patch embeddings, prepend the cls token and add positional embeddings
        x = self.patch_embed(images)                           # (B, num_patches, embed_dim)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)   # (B, 1 + num_patches, embed_dim)
        x = self.pos_dropout(self.pos_embed(x, grid_size))

        # Run it through the transformer blocks and layer norm
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Split the output into cls_token and patch_tokens
        patch_out = x[:, self.num_prefix_tokens:]              # (B, num_patches, embed_dim)
        cls_out = x[:, 0] if self.cls_token is not None else None
        pooled = cls_out if cls_out is not None else patch_out.mean(dim=1)

        # Tokens are row-major over the patch grid -> reshape back to a (B, D, H/P, W/P) map
        spatial = patch_out.transpose(1, 2).reshape(B, -1, *grid_size)
        return FeatureMaps(
            feature_maps={self.feature_level: spatial},
            cls_token=cls_out,
            patch_tokens=patch_out,
            pooled=pooled,
        )


vit_configs = {
    "vit_tiny": ViTConfig(embed_dim=192, depth=12, num_heads=3),
    "vit_small": ViTConfig(embed_dim=384, depth=12, num_heads=6),
    "vit_base": ViTConfig(embed_dim=768, depth=12, num_heads=12),
    "vit_large": ViTConfig(embed_dim=1024, depth=24, num_heads=16),
}


@Backbone.register(config=vit_configs["vit_tiny"])
def vit_tiny(cfg: ViTConfig) -> ViT:
    return ViT(cfg)

@Backbone.register(config=vit_configs["vit_small"])
def vit_small(cfg: ViTConfig) -> ViT:
    return ViT(cfg)

@Backbone.register(config=vit_configs["vit_base"])
def vit_base(cfg: ViTConfig) -> ViT:
    return ViT(cfg)

@Backbone.register(config=vit_configs["vit_large"])
def vit_large(cfg: ViTConfig) -> ViT:
    return ViT(cfg)
