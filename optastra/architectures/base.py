from __future__ import annotations
from abc import ABC
from typing import Any
import torch.nn as nn

from ._registry import _registry
from ..core.factory import Factory


__all__ = ["Architecture"]


class Architecture(nn.Module, Factory["Architecture"], ABC):

    _registry = _registry

    def forward(self, x: Any) -> Any:
        raise NotImplementedError("Subclasses must implement the forward method.")
    