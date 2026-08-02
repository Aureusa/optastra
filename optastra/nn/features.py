# optastra/nn/features.py
from __future__ import annotations
from dataclasses import dataclass, field
import torch

__all__ = ["FeatureMaps", "FeatureSpec", "HeadOutput"]


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


@dataclass
class HeadOutput:
    """
    Structured, task-agnostic prediction output. Every head returns this;
    a Task reads only the fields relevant to it, ignoring the rest.

    logits: (B, num_classes) -- classification, or per-box class logits for detection
    values: (B, ...) -- regression outputs
    boxes: (B, N, 4) -- detection/localization
    scores: (B, N) -- detection confidence, paired with boxes
    masks: (B, N, H, W) or (B, C, H, W) -- segmentation
    embedding: (B, D) -- representation learning / contrastive heads
    extra: dict[str, torch.Tensor] -- escape hatch for any other outputs, e.g. keypoints, flow, etc.
    """
    logits: torch.Tensor | None = None
    values: torch.Tensor | None = None
    boxes: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    masks: torch.Tensor | None = None
    embedding: torch.Tensor | None = None
    extra: dict[str, torch.Tensor] = field(default_factory=dict)  # escape hatch, see below
        