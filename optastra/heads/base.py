from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

import torch
import torch.nn as nn
from typing import Union

from ..backbones import BackboneFeatures
from ..necks import NeckFeatures

from ._registry import get_head_entrypoint


__all__ = ["Head", "HeadFeatures"]


@dataclass
class HeadFeatures:
    """A dataclass for holding the features produced by a head.

    Attributes:
        logits: The raw output logits from the head.
        predictions: The processed predictions from the head (e.g., after applying softmax).
        loss: The computed loss for the head, if applicable.
    """
    logits: torch.Tensor
    predictions: torch.Tensor


class Head(nn.Module, ABC):
    """Base class for all heads in the Optastra framework.

    A head is a component that takes features from a backbone and produces outputs
    such as logits and predictions.
    """

    @classmethod
    def create(cls, name: str) -> Head: # Factory method to create a backbone by name
        """Create a head by name.

        :param name: Name of the head to create.
        :return: An instance of the requested head.
        """
        entrypoint = get_head_entrypoint(name)
        return entrypoint()

    def forward(self, x: Union[BackboneFeatures, NeckFeatures]) -> HeadFeatures:
        """Forward pass through the head.

        Args:
            x: Input tensor from the backbone.

        Returns:
            HeadFeatures containing logits and predictions.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    