from dataclasses import dataclass
import torch

from .base import Transform
from ._registry import register_transform


__all__ = ["ToFloat"]


@dataclass
class ToFloatConfig:
    scale: bool = True   # divide by 255 if input is uint8; no-op if already float


class ToFloat(Transform):
    """Converts sample.image to float32, scaled to [0,1] if it arrived as
    uint8. Always the first transform in any pipeline touching raw dataset
    output -- every photometric/geometric op downstream assumes float."""

    def __init__(self, cfg: ToFloatConfig = ToFloatConfig()):
        self.cfg = cfg

    def __call__(self, sample):
        img = sample.image
        if not img.is_floating_point():
            img = img.to(torch.float32)
            if self.cfg.scale:
                img = img / 255.0
        sample.image = img
        return sample


@register_transform(config=ToFloatConfig())
def to_float(cfg): return ToFloat(cfg)
