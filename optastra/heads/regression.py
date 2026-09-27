from dataclasses import dataclass
from typing import Union

from .base import Head
from ..nn.blocks.readout.mlp import MLP
from ..nn.features import FeatureSpec, HeadOutput, FeatureMaps


__all__ = ["RegressionHead", "RegressionHeadConfig", "BBoxRegressionHead", "BBoxRegressionHeadConfig"]


@dataclass
class RegressionHeadConfig:
    hidden_features: int = 256
    num_layers: int = 2
    activation: str = "gelu"
    norm: Union[str, None] = None
    dropout: float = 0.0
    num_outputs: int = 1   # continuous values predicted per image


class RegressionHead(Head):
    """MLP regression head: pooled (B, embed_dim) features -> (B, num_outputs)
    continuous values (e.g. redshift, size, mass), returned as HeadOutput.values.
    Pairs with the "regression_task" Task."""

    def __init__(self, in_spec: FeatureSpec, cfg: RegressionHeadConfig):
        super().__init__()
        self.cfg = cfg
        in_spec.require("embed_dim")
        self.mlp = MLP(
            in_features=in_spec.embed_dim,
            hidden_features=cfg.hidden_features,
            out_features=cfg.num_outputs,
            num_layers=cfg.num_layers,
            activation=cfg.activation,
            norm=cfg.norm,
            dropout=cfg.dropout,
        )

    def forward(self, x: FeatureMaps) -> HeadOutput:
        return HeadOutput(values=self.mlp(self._pooled(x)))


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
        in_spec.require("embed_dim")
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
        deltas = self.mlp(self._pooled(x))
        return HeadOutput(values=deltas)


regression_head_configs = {
    "vanilla_regression_head": RegressionHeadConfig(),
    "vanilla_box_regression_head": BBoxRegressionHeadConfig(),
}


@Head.register(config=regression_head_configs["vanilla_regression_head"])
def vanilla_regression_head(in_spec: FeatureSpec, cfg: RegressionHeadConfig) -> RegressionHead:
    return RegressionHead(in_spec, cfg)


@Head.register(config=regression_head_configs["vanilla_box_regression_head"])
def vanilla_box_regression_head(
    in_spec: FeatureSpec,
    cfg: BBoxRegressionHeadConfig,
) -> BBoxRegressionHead:
    return BBoxRegressionHead(in_spec, cfg)
