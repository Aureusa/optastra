"""
AutoAugment: learned augmentation policies (Cubuk et al., 2019, arXiv:1805.09501).
Uses the canonical ImageNet policy discovered via RL search in the original
paper -- a fixed list of 25 sub-policies, each a pair of (op, prob, magnitude).
One sub-policy is sampled uniformly per call; each op within it applies
independently at its own probability.
"""
from __future__ import annotations
from dataclasses import dataclass

from . import functional as FN
from . import rng
from .base import Transform
from .ops import apply_op, check_op_names


__all__ = ["AutoAugment"]

# (op_name, prob, magnitude) pairs, two per sub-policy -- the published
# ImageNet policy from the AutoAugment paper, Table 12.
_IMAGENET_POLICY = (
    (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
    (("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)),
    (("Equalize", 0.8, 8), ("Equalize", 0.6, 3)),
    (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
    (("Equalize", 0.4, 7), ("Solarize", 0.2, 4)),
    (("Equalize", 0.4, 4), ("Rotate", 0.8, 8)),
    (("Solarize", 0.6, 3), ("Equalize", 0.6, 7)),
    (("Posterize", 0.8, 5), ("Equalize", 1.0, 2)),
    (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
    (("Equalize", 0.6, 8), ("Posterize", 0.4, 6)),
    (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
    (("Rotate", 0.4, 9), ("Equalize", 0.6, 2)),
    (("Equalize", 0.0, 7), ("Equalize", 0.8, 8)),
    (("Equalize", 0.6, 4), ("Sharpness", 0.3, 3)),
    (("Solarize", 0.4, 5), ("AutoContrast", 0.9, 3)),
    (("Sharpness", 0.4, 7), ("Color", 0.7, 4)),
    (("Equalize", 0.3, 5), ("AutoContrast", 0.4, 2)),
    (("Color", 0.6, 3), ("Equalize", 1.0, 8)),
    (("AutoContrast", 0.4, 6), ("Solarize", 0.6, 5)),
    (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
    (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
    (("Sharpness", 0.4, 7), ("Solarize", 0.4, 4)),
    (("Contrast", 0.9, 8), ("Sharpness", 0.5, 8)),
    (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),
    (("Equalize", 0.3, 7), ("AutoContrast", 0.4, 8)),
)


def _scale_policy(policy, factor: float, cap: float = 10.0):
    return [
        [(op, prob, max(0.0, min(mag * factor, cap))) for (op, prob, mag) in sub_policy]
        for sub_policy in policy
    ]


@dataclass
class AutoAugmentConfig:
    policy: tuple = _IMAGENET_POLICY
    magnitude_scale: float = 1.0
    value_range: tuple[float, float] | None = None   # None -> inferred per image, see functional.py


class AutoAugment(Transform):
    """The published policy includes geometric ops (Rotate, TranslateX), which
    move boxes/masks together with the image (see ops.py). Works on float
    images with any channel count and value range; integer images are
    converted with `functional.to_float_image` first."""

    def __init__(self, cfg: AutoAugmentConfig = AutoAugmentConfig()):
        check_op_names([op for sub_policy in cfg.policy for op, _, _ in sub_policy])
        self.cfg = cfg
        self._policy = _scale_policy(cfg.policy, cfg.magnitude_scale)

    def __call__(self, sample):
        sample.image = FN.to_float_image(sample.image)
        value_range = FN.infer_value_range(sample.image, self.cfg.value_range)
        for op_name, prob, magnitude in rng.choice(self._policy):
            if rng.rand() < prob:
                sample = apply_op(sample, op_name, magnitude, value_range)
        return sample


@Transform.register(config=AutoAugmentConfig())
def auto_augment(cfg): return AutoAugment(cfg)


@Transform.register(config=AutoAugmentConfig(magnitude_scale=0.5))
def auto_augment_weak(cfg): return AutoAugment(cfg)


@Transform.register(config=AutoAugmentConfig(magnitude_scale=1.3))
def auto_augment_strong(cfg): return AutoAugment(cfg)
