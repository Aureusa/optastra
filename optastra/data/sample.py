from dataclasses import dataclass, field
from typing import Any
import torch


@dataclass
class Sample:
    """
    What one dataset item looks like. Populate only what your task family needs.
    """
    image: torch.Tensor | None = None
    views: list[torch.Tensor] | None = None   # populated instead of `image` for SSL algorithms
    target: dict[str, Any] = field(default_factory=dict)  # "label", "boxes", "labels", "mask", ...
    meta: dict[str, Any] = field(default_factory=dict)     # e.g. id, path, orig size
