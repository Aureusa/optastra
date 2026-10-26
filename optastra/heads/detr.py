"""Set-prediction heads for decoder tokens: points (keypoint_head) and boxes (detr_box_head).

Both read ``FeatureMaps.patch_tokens`` (B, M, D) from a Decoder. When the
decoder's queries carry reference points (``extra["reference_points"]``,
Conditional / DAB-DETR), positions are predicted as offsets from them in
inverse-sigmoid space; otherwise (vanilla DETR) absolutely. Coordinates are
fractions of each image's size (see ``optastra.detection.keys``). An
architecture applies the same head to every decoder layer for auxiliary losses.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from .base import Head
from ..decoders import keys as decoder_keys
from ..decoders.query_decoder import inverse_sigmoid
from ..detection import keys
from ..nn.blocks.readout.mlp import MLP
from ..nn.features import FeatureMaps, FeatureSpec, HeadOutput


__all__ = [
    "KeypointHead", "KeypointHeadConfig", "keypoint_head",
    "DETRBoxHead", "DETRBoxHeadConfig", "detr_box_head",
]


def _tokens(head: Head, features: FeatureMaps) -> torch.Tensor:
    if features.patch_tokens is None or features.patch_tokens.dim() != 3:
        raise ValueError(f"{type(head).__name__} reads decoder tokens FeatureMaps.patch_tokens (B, M, D); "
                         f"put it after a Decoder.")
    return features.patch_tokens


def _locate(raw: torch.Tensor, reference: torch.Tensor | None) -> torch.Tensor:
    """sigmoid(raw), or sigmoid(inverse_sigmoid(reference) + raw) with a reference point."""
    if reference is None:
        return raw.float().sigmoid()
    return (inverse_sigmoid(reference.float()) + raw.float()).sigmoid()


@dataclass
class KeypointHeadConfig:
    hidden_features: int = 256
    num_layers: int = 3
    predict_uncertainty: bool = False   # also predict a log scale per coordinate (Laplace NLL)
    prior_prob: float = 0.1             # initial objectness probability (focal-loss style bias init)


class KeypointHead(Head):
    """Decoder tokens -> one point + one objectness logit per query.

    ``HeadOutput.scores`` (B, M) are the objectness logits; ``extra["points"]``
    (B, M, 2) the points and, with ``predict_uncertainty``,
    ``extra["point_log_scales"]`` (B, M, 2) the log of the predicted scale
    (both in fractions of the image size). The offset layer is zero-initialised:
    a query with a reference point starts exactly on it.
    """

    def __init__(self, in_spec: FeatureSpec, cfg: KeypointHeadConfig):
        super().__init__()
        in_spec.require("embed_dim")
        self.cfg = cfg
        D = in_spec.embed_dim
        self.location = MLP(D, cfg.hidden_features, 4 if cfg.predict_uncertainty else 2, cfg.num_layers)
        nn.init.zeros_(self.location.mlp[-1].weight)
        nn.init.zeros_(self.location.mlp[-1].bias)
        self.objectness = nn.Linear(D, 1)
        nn.init.constant_(self.objectness.bias, -math.log((1 - cfg.prior_prob) / cfg.prior_prob))
        self.out_spec = FeatureSpec(embed_dim=2)

    def forward(self, features: FeatureMaps) -> HeadOutput:
        z = _tokens(self, features)
        out = self.location(z)
        extra = {keys.POINTS: _locate(out[..., :2], features.extra.get(decoder_keys.REFERENCE_POINTS))}
        if self.cfg.predict_uncertainty:
            extra[keys.POINT_LOG_SCALES] = out[..., 2:].float()
        return HeadOutput(scores=self.objectness(z).squeeze(-1), extra=extra)


@dataclass
class DETRBoxHeadConfig:
    num_classes: int = 80        # foreground classes; one extra "no object" logit is added
    hidden_features: int = 256
    num_layers: int = 3


class DETRBoxHead(Head):
    """DETR's prediction FFNs: class logits (B, M, num_classes + 1), the last
    one meaning "no object", and boxes (B, M, 4) as normalised (cx, cy, w, h).
    With reference points the box centre is an offset from them."""

    def __init__(self, in_spec: FeatureSpec, cfg: DETRBoxHeadConfig):
        super().__init__()
        in_spec.require("embed_dim")
        self.cfg = cfg
        D = in_spec.embed_dim
        self.class_embed = nn.Linear(D, cfg.num_classes + 1)
        self.bbox_embed = MLP(D, cfg.hidden_features, 4, cfg.num_layers, activation="relu")
        self.out_spec = FeatureSpec(embed_dim=cfg.num_classes + 1)

    def forward(self, features: FeatureMaps) -> HeadOutput:
        z = _tokens(self, features)
        raw = self.bbox_embed(z)
        centre = _locate(raw[..., :2], features.extra.get(decoder_keys.REFERENCE_POINTS))
        boxes = torch.cat([centre, raw[..., 2:].float().sigmoid()], dim=-1)
        return HeadOutput(logits=self.class_embed(z), boxes=boxes)


@Head.register(config=KeypointHeadConfig())
def keypoint_head(in_spec: FeatureSpec, cfg: KeypointHeadConfig) -> KeypointHead:
    return KeypointHead(in_spec, cfg)


@Head.register(config=DETRBoxHeadConfig())
def detr_box_head(in_spec: FeatureSpec, cfg: DETRBoxHeadConfig) -> DETRBoxHead:
    return DETRBoxHead(in_spec, cfg)
