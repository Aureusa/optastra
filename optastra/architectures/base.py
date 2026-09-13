from __future__ import annotations
from abc import ABC
from typing import Any
import torch.nn as nn

from ..core.factory import Factory
from ..core.registry import FamilyRegistry


__all__ = ["Architecture"]


class Architecture(nn.Module, Factory["Architecture"], ABC):

    _registry = FamilyRegistry("architecture")

    def forward(self, x: Any) -> Any:
        raise NotImplementedError("Subclasses must implement the forward method.")
    