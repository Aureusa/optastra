from dataclasses import dataclass
from typing import Union

from .base import Head

from ..nn.blocks.readout.mlp import MLP
from ..nn.features import FeatureSpec, HeadOutput, FeatureMaps


__all__ = ["ClassificationHead", "ClassificationHeadConfig"]


@dataclass
class ClassificationHeadConfig:
    """Configuration for a classification head."""
    hidden_features: int = 1024
    num_layers: int = 2
    activation: str = "gelu"
    norm: Union[str, None] = None
    dropout: float = 0.0
    num_classes: int = 1000


class ClassificationHead(Head):
    """MLP classification head: pooled (B, embed_dim) features -> (B, num_classes) logits."""

    def __init__(
            self,
            in_spec: FeatureSpec,
            cfg: ClassificationHeadConfig
        ):
        super().__init__()
        self.cfg = cfg
        in_spec.require("embed_dim")
        in_features = in_spec.embed_dim

        self.num_classes = cfg.num_classes

        self.mlp = MLP(
            in_features=in_features,
            hidden_features=cfg.hidden_features,
            out_features=cfg.num_classes,
            num_layers=cfg.num_layers,
            activation=cfg.activation,
            norm=cfg.norm,
            dropout=cfg.dropout
        )

    def forward(self, x: FeatureMaps) -> HeadOutput:
        """Forward pass through the classification head.

        :param x: Input features from the backbone or neck.
        :return: HeadOutput containing the logits.
        """
        logits = self.mlp(self._pooled(x))
        return HeadOutput(logits=logits)


classification_head_configs = {
    "vanilla_classification_head": ClassificationHeadConfig(),
}


@Head.register(config=classification_head_configs["vanilla_classification_head"])
def vanilla_classification_head(in_spec: FeatureSpec, cfg: ClassificationHeadConfig) -> ClassificationHead:
    """
    Factory function to create a vanilla classification head.

    :param in_spec: FeatureSpec instance describing the output of a preceeding feature extractor
    :param cfg: Configuration object for the classification head.
    :return: An instance of ClassificationHead.
    """
    return ClassificationHead(in_spec, cfg)
