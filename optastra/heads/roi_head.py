from __future__ import annotations

from dataclasses import dataclass

from ._registry import register_head
from .base import Head
from ..nn.blocks.readout.mlp import MLP
from ..nn.features import FeatureMaps, FeatureSpec, HeadOutput


@dataclass
class ROIBoxHeadConfig:
    fc_hidden_features: int = 256
    fc_num_layers: int = 2
    activation: str = "relu"
    dropout: float = 0.0
    num_classes: int = 80
    class_agnostic_box_regression: bool = True


class ROIBoxHead(Head):
    """Shared MLP trunk followed by class logits and box deltas branches."""

    def __init__(self, in_spec: FeatureSpec, cfg: ROIBoxHeadConfig):
        super().__init__()
        self.cfg = cfg
        if in_spec.embed_dim is None:
            raise ValueError("ROIBoxHead requires in_spec.embed_dim to be defined.")

        self.num_classes = cfg.num_classes
        self.class_agnostic_box_regression = cfg.class_agnostic_box_regression

        self.trunk = MLP(
            in_features=in_spec.embed_dim,
            hidden_features=cfg.fc_hidden_features,
            out_features=cfg.fc_hidden_features,
            num_layers=cfg.fc_num_layers,
            activation=cfg.activation,
            norm=None,
            dropout=cfg.dropout,
        )

        # +1 for background class, matching standard Fast/Faster R-CNN classification.
        self.cls_score = MLP(
            in_features=cfg.fc_hidden_features,
            hidden_features=cfg.fc_hidden_features,
            out_features=cfg.num_classes + 1,
            num_layers=2,
            activation=cfg.activation,
            norm=None,
            dropout=cfg.dropout,
        )

        box_out_dim = 4 if cfg.class_agnostic_box_regression else 4 * cfg.num_classes
        self.bbox_pred = MLP(
            in_features=cfg.fc_hidden_features,
            hidden_features=cfg.fc_hidden_features,
            out_features=box_out_dim,
            num_layers=2,
            activation=cfg.activation,
            norm=None,
            dropout=cfg.dropout,
        )

    def forward(self, features: FeatureMaps) -> HeadOutput:
        if features.pooled is None:
            raise ValueError("ROIBoxHead requires features.pooled to be present.")

        shared = self.trunk(features.pooled)
        logits = self.cls_score(shared)
        deltas = self.bbox_pred(shared)
        return HeadOutput(logits=logits, values=deltas)


roi_box_head_configs = {
    "roi_box_head": ROIBoxHeadConfig(),
}


@register_head(config=roi_box_head_configs["roi_box_head"])
def roi_box_head(in_spec: FeatureSpec, cfg: ROIBoxHeadConfig) -> ROIBoxHead:
    return ROIBoxHead(in_spec, cfg)