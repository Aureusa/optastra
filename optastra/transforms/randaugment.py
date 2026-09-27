"""
RandAugment implementation based on the original paper:
https://arxiv.org/abs/1909.13719

The original implementation is available at:
https://www.github.com/tensorflow/tpu/tree/master/models/official/efficientnet

Op implementations live in `ops.py`. The default op set is photometric-only;
`rand_augment_all_ops` adds the geometric ops, which move boxes/masks along
with the image.
"""
from __future__ import annotations
from dataclasses import dataclass

from . import functional as FN
from . import rng
from .base import Transform
from .ops import ALL_OP_NAMES, PHOTOMETRIC_OP_NAMES, apply_op, check_op_names


__all__ = ["RandAugment"]


@dataclass
class RandAugmentConfig:
    num_ops: int = 2
    magnitude: int = 9   # standard RandAugment N,M notation
    ops: tuple[str, ...] = PHOTOMETRIC_OP_NAMES
    value_range: tuple[float, float] | None = None   # None -> inferred per image, see functional.py


class RandAugment(Transform):
    """Apply `num_ops` distinct ops, chosen uniformly from `ops`, at a fixed `magnitude`.
    Works on float images with any channel count and value range; integer
    images are converted with `functional.to_float_image` first."""

    def __init__(self, cfg: RandAugmentConfig = RandAugmentConfig()):
        check_op_names(cfg.ops)
        self.cfg = cfg

    def __call__(self, sample):
        sample.image = FN.to_float_image(sample.image)
        value_range = FN.infer_value_range(sample.image, self.cfg.value_range)
        for op_name in rng.sample(self.cfg.ops, k=self.cfg.num_ops):
            sample = apply_op(sample, op_name, self.cfg.magnitude, value_range)
        return sample


@Transform.register(config=RandAugmentConfig())
def rand_augment(cfg): return RandAugment(cfg)


@Transform.register(config=RandAugmentConfig(ops=ALL_OP_NAMES))
def rand_augment_all_ops(cfg): return RandAugment(cfg)
