# optastra/nn/features.py
from __future__ import annotations
from dataclasses import dataclass, field
import torch

__all__ = ["FeatureMaps", "FeatureSpec"]

@dataclass
class FeatureMaps:
    """Runtime output of any feature-producing stage."""
    feature_maps: dict[str, torch.Tensor] = field(default_factory=dict)
    pooled: torch.Tensor | None = None
    patch_tokens: torch.Tensor | None = None
    cls_token: torch.Tensor | None = None

@dataclass
class FeatureSpec:
    """
    Construction-time twin of FeatureMaps -- shapes/metadata, no tensors.

    Populate only the fields that apply to your architecture. Downstream
    components declare which fields they need and fail loudly
    at construction time if a required field is missing -- they never guess.
    """
    channels: dict[str, int] = field(default_factory=dict)
    strides: dict[str, int] = field(default_factory=dict)
    embed_dim: int | None = None
    num_tokens: int | None = None

    def require(self, *fields: str) -> None:
        """Raise a clear error if this spec doesn't carry what a consumer needs."""
        missing = [f for f in fields if not getattr(self, f)]
        if missing:
            raise ValueError(
                f"This output spec is missing {missing}, "
                f"which the next component requires. "
                f"Available: channels={list(self.channels)}, "
                f"strides={list(self.strides)}, embed_dim={self.embed_dim}, "
                f"num_tokens={self.num_tokens}."
            )
        