"""
TrivialAugment (Müller & Hutter, 2021, arXiv:2103.10158). Deliberately
minimal: pick ONE op uniformly at random, sample its magnitude uniformly
at random from the full range (not a fixed schedule like RandAugment) --
no tuning of num_ops or a fixed magnitude required.
"""
from __future__ import annotations
from dataclasses import dataclass

from . import functional as FN
from . import rng
from .base import Transform
from .ops import ALL_OP_NAMES, apply_op, check_op_names


__all__ = ["TrivialAugment"]


@dataclass
class TrivialAugmentConfig:
    ops: tuple[str, ...] = ALL_OP_NAMES
    magnitude_min: float = 0.0
    magnitude_max: float = 10.0
    value_range: tuple[float, float] | None = None   # None -> inferred per image, see functional.py


class TrivialAugment(Transform):
    """
    TrivialAugment: pick one op uniformly at random, sample its magnitude
    uniformly at random. The default op set is the full set of
    photometric+geometric ops; geometric ops move boxes/masks with the image.
    Works on float images with any channel count and value range; integer
    images are converted with `functional.to_float_image` first.
    """
    def __init__(self, cfg: TrivialAugmentConfig = TrivialAugmentConfig()):
        check_op_names(cfg.ops)
        self.cfg = cfg

    def __call__(self, sample):
        sample.image = FN.to_float_image(sample.image)
        value_range = FN.infer_value_range(sample.image, self.cfg.value_range)
        op_name = rng.choice(self.cfg.ops)
        magnitude = rng.uniform(self.cfg.magnitude_min, self.cfg.magnitude_max)   # uniform, not fixed -- the whole point
        return apply_op(sample, op_name, magnitude, value_range)


@Transform.register(config=TrivialAugmentConfig())
def trivial_augment(cfg): return TrivialAugment(cfg)


@Transform.register(config=TrivialAugmentConfig(magnitude_min=0.0, magnitude_max=4.0))
def trivial_augment_weak(cfg): return TrivialAugment(cfg)


@Transform.register(config=TrivialAugmentConfig(magnitude_min=6.0, magnitude_max=10.0))
def trivial_augment_strong(cfg): return TrivialAugment(cfg)
