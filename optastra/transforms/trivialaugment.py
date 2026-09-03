"""
TrivialAugment (Müller & Hutter, 2021, arXiv:2103.10158). Deliberately
minimal: pick ONE op uniformly at random, sample its magnitude uniformly
at random from the full range (not a fixed schedule like RandAugment) --
no tuning of num_ops or a fixed magnitude required.
"""
from dataclasses import dataclass, field
import random

from .base import Transform
from .ops import ALL_OPS, PHOTOMETRIC_OPS
from ._registry import register_transform


__all__ = ["TrivialAugment"]


@dataclass
class TrivialAugmentConfig:
    ops: list[str] = field(default_factory=lambda: list(ALL_OPS.keys()))
    magnitude_min: float = 0.0
    magnitude_max: float = 10.0


class TrivialAugment(Transform):
    """
    TrivialAugment: pick one op uniformly at random, sample its magnitude
    uniformly at random. The default op set is the full set of photometric+geometric ops,
    but for target-aware tasks (detection/segmentation) you may want to restrict to photometric-only ops.
    """
    def __init__(self, cfg: TrivialAugmentConfig = TrivialAugmentConfig()):
        self.cfg = cfg

    def __call__(self, sample):
        op_name = random.choice(self.cfg.ops)
        magnitude = random.uniform(self.cfg.magnitude_min, self.cfg.magnitude_max)   # uniform, not fixed -- the whole point
        sample.image = ALL_OPS[op_name](sample.image, magnitude)
        return sample


@register_transform(config=TrivialAugmentConfig())
def trivial_augment(cfg): return TrivialAugment(cfg)


@register_transform(config=TrivialAugmentConfig())
def trivial_augment_weak(cfg):
    cfg.magnitude_min = 0.0
    cfg.magnitude_max = 4.0
    return TrivialAugment(cfg)
 
 
@register_transform(config=TrivialAugmentConfig())
def trivial_augment_strong(cfg):
    cfg.magnitude_min = 6.0
    cfg.magnitude_max = 10.0
    return TrivialAugment(cfg)
