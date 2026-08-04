from dataclasses import dataclass
from typing import Union

from .base import Head
from ._registry import register_head
from ..nn.blocks.readout.mlp import MLP
from ..nn.features import FeatureSpec, HeadOutput, FeatureMaps


__all__ = ["BBoxRegressionHead"]


@dataclass
class BBoxRegressionHeadConfig:
    hidden_features: int = 1024
    num_layers: int = 2
    activation: str = "gelu"
    norm: Union[str, None] = None
    dropout: float = 0.0
    box_dim: int = 4
    class_agnostic: bool = True
    num_classes: int = 1000


class BBoxRegressionHead(Head):
    """A simple ROI box regression head that predicts bbox deltas."""

    def __init__(self, in_spec: FeatureSpec, cfg: BBoxRegressionHeadConfig):
        super().__init__()
        self.cfg = cfg
        in_features = in_spec.embed_dim
        out_dim = cfg.box_dim if cfg.class_agnostic else cfg.box_dim * cfg.num_classes

        self.mlp = MLP(
            in_features=in_features,
            hidden_features=cfg.hidden_features,
            out_features=out_dim,
            num_layers=cfg.num_layers,
            activation=cfg.activation,
            norm=cfg.norm,
            dropout=cfg.dropout,
        )

    def forward(self, x: FeatureMaps) -> HeadOutput:
        deltas = self.mlp(x.pooled)
        return HeadOutput(values=deltas)


regression_head_configs = {
    "vanilla_box_regression_head": BBoxRegressionHeadConfig(),
}


@register_head(config=regression_head_configs["vanilla_box_regression_head"])
def vanilla_box_regression_head(
    in_spec: FeatureSpec,
    cfg: BBoxRegressionHeadConfig,
) -> BBoxRegressionHead:
    return BBoxRegressionHead(in_spec, cfg)
