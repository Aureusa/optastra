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
from ..nn.blocks.transformer.patch_embed import PatchEmbedding
from ..nn.blocks.transformer.transformer_block import TransformerBlock
from ..nn.blocks.transformer.pos_embed import LearnedPosEmbed, SinusoidalPosEmbed, interpolate_pos_embed
from ._registry import register_backbone


__all__ = ["ViT"]


@dataclass
class ViTConfig:
    img_size: int = 224
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
    Vision Transformer. Populates embed_dim/patch_tokens/cls_token on
    FeatureSpec/FeatureMaps -- never channels/strides, since there's
    no multi-scale spatial pyramid here. Necks/heads that need channels/strides
    (e.g. FPN) structurally cannot consume this backbone's output -- that's
    intentional, per FeatureSpec.require().
    """
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg

        pos_embed_cls = {"learned": LearnedPosEmbed, "sinusoidal": SinusoidalPosEmbed}[cfg.pos_embed_type]

        self.patch_embed = PatchEmbedding(cfg.img_size, cfg.patch_size, cfg.in_channels, cfg.embed_dim)
        num_patches = self.patch_embed.num_patches

        if cfg.cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = pos_embed_cls(cfg.embed_dim, self.patch_embed.grid_size, cls_token=cfg.cls_token)
        self.pos_dropout = nn.Dropout(cfg.pos_embed_dropout)

        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, cfg.depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=cfg.embed_dim, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias, dropout=cfg.dropout, attn_dropout=cfg.attn_dropout,
                drop_path=dpr[i],
            )
            for i in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)

        self.out_spec = FeatureSpec(embed_dim=cfg.embed_dim, num_tokens=num_patches)

        self._init_weights()

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        super().load_state_dict(state_dict, strict=strict)

        # If loading pretrained weights, call `interpolate_pos_embed` to resize
        # the positional embedding to match the current grid size (num_patches).
        if self.cfg.pos_embed_type == "learned" and "pos_embed" in state_dict:
            old_pos_embed = state_dict["pos_embed"]
            new_pos_embed = interpolate_pos_embed(
                old_pos_embed,
                old_grid_size=int(math.sqrt(old_pos_embed.shape[1] - 1)),
                new_grid_size=int(math.sqrt(self.patch_embed.num_patches))
            )
            self.pos_embed.pos_embed.data.copy_(new_pos_embed)

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed.pos_embed, std=0.02)
        if hasattr(self, "cls_token"):
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
        B = images.shape[0]

        # Create patch embeddings and add positional embeddings
        x = self.patch_embed(images)                          # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)                  # (B, num_patches+1, embed_dim)
        x = self.pos_dropout(self.pos_embed(x))                                 # (B, num_patches+1, embed_dim)

        # Run it through the transformer blocks and layer norm
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Split the output into cls_token and patch_tokens, and return as FeatureMaps
        cls_out = x[:, 0]
        patch_out = x[:, 1:]
        return FeatureMaps(cls_token=cls_out, patch_tokens=patch_out, pooled=cls_out)


vit_configs = {
    "vit_tiny": ViTConfig(embed_dim=192, depth=12, num_heads=3),
    "vit_small": ViTConfig(embed_dim=384, depth=12, num_heads=6),
    "vit_base": ViTConfig(embed_dim=768, depth=12, num_heads=12),
    "vit_large": ViTConfig(embed_dim=1024, depth=24, num_heads=16),
}


@register_backbone(config=vit_configs["vit_tiny"])
def vit_tiny(cfg: ViTConfig) -> ViT:
    return ViT(cfg)

@register_backbone(config=vit_configs["vit_small"])
def vit_small(cfg: ViTConfig) -> ViT:
    return ViT(cfg)

@register_backbone(config=vit_configs["vit_base"])
def vit_base(cfg: ViTConfig) -> ViT:
    return ViT(cfg)

@register_backbone(config=vit_configs["vit_large"])
def vit_large(cfg: ViTConfig) -> ViT:
    return ViT(cfg)
