from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from .base import Head
from ..nn.blocks._pytorch_primitives import get_activation, get_dropout
from ..nn.features import FeatureMaps, FeatureSpec, HeadOutput


__all__ = ["ROIBoxHead", "ROIBoxHeadConfig"]


@dataclass
class ROIBoxHeadConfig:
    fc_hidden_features: int = 256
    fc_num_layers: int = 2
    activation: str = "relu"
    dropout: float = 0.0
    num_classes: int = 80
    class_agnostic_box_regression: bool = True
    # "flatten": the whole (C, S, S) ROI feature feeds the fc trunk (the standard
    # Fast/Faster R-CNN "2fc" head, keeps the spatial layout inside the box).
    # "pooled": the ROI feature is mean-pooled to (C,) first -- far fewer
    # parameters, but position inside the box is lost.
    roi_features: str = "flatten"


class ROIBoxHead(Head):
    """``fc_num_layers`` x (Linear -> activation) trunk followed by two linear
    predictors: class logits (num_classes + 1, background last) and box deltas
    (4, or 4 * num_classes for class-specific regression)."""

    def __init__(self, in_spec: FeatureSpec, cfg: ROIBoxHeadConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.roi_features == "flatten":
            in_spec.require("channels", "num_tokens")
            if "roi" not in in_spec.channels:
                raise ValueError("ROIBoxHead(roi_features='flatten') requires 'roi' feature maps in the input spec.")
            in_features = in_spec.channels["roi"] * in_spec.num_tokens
        elif cfg.roi_features == "pooled":
            if in_spec.embed_dim is None:
                raise ValueError("ROIBoxHead(roi_features='pooled') requires in_spec.embed_dim to be defined.")
            in_features = in_spec.embed_dim
        else:
            raise ValueError(f"roi_features must be 'flatten' or 'pooled', got {cfg.roi_features!r}.")

        self.num_classes = cfg.num_classes
        self.class_agnostic_box_regression = cfg.class_agnostic_box_regression

        layers: list[nn.Module] = []
        for _ in range(cfg.fc_num_layers):
            layers += [
                nn.Linear(in_features, cfg.fc_hidden_features),
                get_activation(cfg.activation),
                get_dropout("dropout", p=cfg.dropout),
            ]
            in_features = cfg.fc_hidden_features
        self.trunk = nn.Sequential(*layers)

        # +1 for background class, matching standard Fast/Faster R-CNN classification.
        self.cls_score = nn.Linear(in_features, cfg.num_classes + 1)
        box_out_dim = 4 if cfg.class_agnostic_box_regression else 4 * cfg.num_classes
        self.bbox_pred = nn.Linear(in_features, box_out_dim)

        # Small-std predictor init (as in Detectron2): start near uniform class
        # scores and near-zero box refinements.
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.zeros_(self.cls_score.bias)
        nn.init.zeros_(self.bbox_pred.bias)

    def forward(self, features: FeatureMaps) -> HeadOutput:
        if self.cfg.roi_features == "flatten":
            if "roi" not in features.feature_maps:
                raise ValueError("ROIBoxHead requires ROI feature maps under features.feature_maps['roi'].")
            x = features.feature_maps["roi"].flatten(start_dim=1)
        else:
            if features.pooled is None:
                raise ValueError("ROIBoxHead requires features.pooled to be present.")
            x = features.pooled

        shared = self.trunk(x)
        return HeadOutput(logits=self.cls_score(shared), values=self.bbox_pred(shared))


roi_box_head_configs = {
    "roi_box_head": ROIBoxHeadConfig(),
}


@Head.register(config=roi_box_head_configs["roi_box_head"])
def roi_box_head(in_spec: FeatureSpec, cfg: ROIBoxHeadConfig) -> ROIBoxHead:
    return ROIBoxHead(in_spec, cfg)
