import torch
import torch.nn as nn
from typing import Union

from .base import Head, HeadFeatures
from ._registry import register_head

from ..nn.blocks.readout.mlp import MLP
from ..nn.blocks.readout.pooling import GlobalAvgPool2d, GlobalMaxPool2d

from ..backbones import BackboneFeatures
from ..necks import NeckFeatures


class ClassificationHead(Head):
    """A simple classification head that produces logits and predictions."""

    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            stage: str = "C5",  # Stage from which to take features, e.g., "C5" for ResNet
            pooling: str = "avg",  # "avg" for GlobalAvgPool2d, "max" for GlobalMaxPool2d
            num_layers: int = 2,
            activation: str = "gelu",
            norm: Union[str, None] = None,
            dropout: float = 0.0,
            num_classes: int = 1000
        ):
        super().__init__()
        assert pooling in ["avg", "max"], "Pooling must be either 'avg' or 'max'."
        self.pooling = GlobalAvgPool2d() if pooling == "avg" else GlobalMaxPool2d()
        self.num_classes = num_classes

        self.stage = stage

        self.mlp = MLP(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=num_classes,
            num_layers=num_layers,
            activation=activation,
            norm=norm,
            dropout=dropout
        )

    def forward(self, x: Union[BackboneFeatures, NeckFeatures]) -> HeadFeatures:
        """Forward pass through the classification head.

        :param x: Input features from the backbone or neck.
        :return: HeadFeatures containing logits and predictions.
        """
        assert isinstance(x, (BackboneFeatures, NeckFeatures)), "Input must be BackboneFeatures or NeckFeatures."
        assert self.stage in x.feature_maps, f"Stage '{self.stage}' not found in input features. Check the available stages: {list(x.feature_maps.keys())}"
        features = x.feature_maps[self.stage]  # Example: using the last stage feature map

        features = self.pooling(features)  # Apply global pooling

        logits = self.mlp(features)
        predictions = torch.softmax(logits, dim=-1)
        return HeadFeatures(logits=logits, predictions=predictions)


@register_head
def vanilla_classification_head(
        in_features: int,
        hidden_features: int,
        stage: str = "C5",
        pooling: str = "avg",
        num_layers: int = 2,
        activation: str = "gelu",
        norm: Union[str, None] = None,
        dropout: float = 0.0,
        num_classes: int = 1000
    ) -> ClassificationHead:
    """
    Factory function to create a vanilla classification head.

    :param in_features: Number of input features from the backbone.
    :param hidden_features: Number of hidden features in the MLP.
    :param stage: Stage from which to take features (e.g., "C5").
    :param pooling: Type of pooling to use ("avg" or "max").
    :param num_layers: Number of layers in the MLP.
    :param activation: Activation function to use in the MLP.
    :param norm: Normalization method to use in the MLP.
    :param dropout: Dropout rate to use in the MLP.
    :param num_classes: Number of output classes for classification.
    :return: An instance of ClassificationHead.
    """
    return ClassificationHead(
        in_features=in_features,
        hidden_features=hidden_features,
        stage=stage,
        pooling=pooling,
        num_layers=num_layers,
        activation=activation,
        norm=norm,
        dropout=dropout,
        num_classes=num_classes
    )