from dataclasses import dataclass
import torch

from .base import Transform
from .functional import to_float_image


__all__ = ["ToFloat"]


@dataclass
class ToFloatConfig:
    scale: bool = True          # unsigned ints -> [0, 1] (uint8 /255, uint16 /65535); False -> plain cast
    signed_scale: float = 1.0   # int16/int32/int64 are divided by this when scale=True (1.0 keeps raw counts)


class ToFloat(Transform):
    """Converts sample.image to float32. Dtype-aware:

    * uint8 / uint16 / uint32 -> divided by the dtype max, i.e. [0, 1];
    * int16 / int32 / int64   -> divided by `signed_scale` (these usually hold
      raw or background-subtracted counts with no natural full scale);
    * floating point          -> unchanged (HDR values are kept as they are).

    With `scale=False` integers are only cast. Put it first in any pipeline
    touching raw dataset output."""

    def __init__(self, cfg: ToFloatConfig = ToFloatConfig()):
        self.cfg = cfg

    def __call__(self, sample):
        img = sample.image
        if not img.is_floating_point():
            img = to_float_image(img, self.cfg.signed_scale) if self.cfg.scale else img.to(torch.float32)
        sample.image = img
        return sample


@Transform.register(config=ToFloatConfig())
def to_float(cfg): return ToFloat(cfg)
