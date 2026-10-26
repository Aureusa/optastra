from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .base import Neck
from ..nn.blocks.transformer.detr_layers import DETREncoderLayer
from ..nn.blocks.transformer.memory import flatten_feature_levels, unflatten_feature_levels
from ..nn.features import FeatureMaps, FeatureSpec


__all__ = ["DETREncoder", "DETREncoderConfig", "detr_encoder"]

IMAGE_SIZES = "image_sizes"   # FeatureMaps.extra key, same string as optastra.detection.keys.IMAGE_SIZES


@dataclass
class DETREncoderConfig:
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    normalize_before: bool = False
    in_features: tuple[str, ...] = ()   # () = only the coarsest map (DETR: C5); several = one joint sequence


class DETREncoder(Neck):
    """DETR's transformer encoder as a neck (Carion et al., 2020).

    Each selected map is projected to ``d_model`` (1x1 conv), flattened, and
    all of them pass as one sequence through ``num_layers`` self-attention
    layers with sine position encodings (and a level embedding when several
    maps are used); the result is reshaped back to maps. Padded cells are
    masked using ``FeatureMaps.extra["image_sizes"]`` when present (the
    architecture puts it there), so batch padding never leaks into real cells.

    Self-attention is dense: cost grows with the square of the number of
    cells, so prefer a stride-16/32 map, or tile large inputs. DETR uses only
    C5 (stride 32), the default here; small objects want a finer level.

    Output: the encoded maps under their input names, at ``d_model`` channels
    and the same strides; ``extra`` is passed through.
    """

    def __init__(self, in_spec: FeatureSpec, cfg: DETREncoderConfig):
        super().__init__()
        in_spec.require("channels", "strides")
        self.cfg = cfg
        names = list(cfg.in_features) or [max(in_spec.channels, key=lambda n: in_spec.strides[n])]
        missing = [n for n in names if n not in in_spec.channels or n not in in_spec.strides]
        if missing:
            raise ValueError(f"DETREncoder: features {missing} not in the in_spec ({list(in_spec.channels)}).")
        D = cfg.d_model
        if D % 4 != 0:
            raise ValueError(f"d_model ({D}) must be divisible by 4 (2D sine position encodings).")
        self.feature_names = names
        self.strides = [in_spec.strides[n] for n in names]
        self.input_proj = nn.ModuleDict({n: nn.Conv2d(in_spec.channels[n], D, 1) for n in names})
        self.level_embed = nn.Parameter(torch.randn(len(names), D) * 0.02) if len(names) > 1 else None
        self.layers = nn.ModuleList(
            DETREncoderLayer(D, cfg.num_heads, cfg.dim_feedforward, cfg.dropout, cfg.normalize_before)
            for _ in range(cfg.num_layers)
        )
        self.norm = nn.LayerNorm(D) if cfg.normalize_before else nn.Identity()   # pre-norm needs a final norm
        self.out_spec = FeatureSpec(channels={n: D for n in names}, strides={n: in_spec.strides[n] for n in names})

    def forward(self, features: FeatureMaps) -> FeatureMaps:
        maps = [self.input_proj[n](features.feature_maps[n]) for n in self.feature_names]
        flat = flatten_feature_levels(maps, self.strides, features.extra.get(IMAGE_SIZES), self.level_embed)
        x = flat.memory
        attn_mask = flat.mask[:, None, None, :]
        for layer in self.layers:
            x = layer(x, flat.pos, attn_mask)
        x = self.norm(x)
        out = unflatten_feature_levels(x, flat.shapes)
        return FeatureMaps(feature_maps=dict(zip(self.feature_names, out)), extra=dict(features.extra))


@Neck.register(config=DETREncoderConfig())
def detr_encoder(in_spec: FeatureSpec, cfg: DETREncoderConfig) -> DETREncoder:
    return DETREncoder(in_spec, cfg)
