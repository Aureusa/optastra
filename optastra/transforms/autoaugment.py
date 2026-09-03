"""
AutoAugment: learned augmentation policies (Cubuk et al., 2019, arXiv:1805.09501).
Uses the canonical ImageNet policy discovered via RL search in the original
paper -- a fixed list of 25 sub-policies, each a pair of (op, prob, magnitude).
One sub-policy is sampled uniformly per call; each op within it applies
independently at its own probability.
"""
from dataclasses import dataclass, field
import random

from .base import Transform
from .ops import ALL_OPS
from ._registry import register_transform


__all__ = ["AutoAugment"]

# (op_name, prob, magnitude) pairs, two per sub-policy -- the published
# ImageNet policy from the AutoAugment paper, Table 12.
_IMAGENET_POLICY = [
    [("Posterize", 0.4, 8), ("Rotate", 0.6, 9)],
    [("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)],
    [("Equalize", 0.8, 8), ("Equalize", 0.6, 3)],
    [("Posterize", 0.6, 7), ("Posterize", 0.6, 6)],
    [("Equalize", 0.4, 7), ("Solarize", 0.2, 4)],
    [("Equalize", 0.4, 4), ("Rotate", 0.8, 8)],
    [("Solarize", 0.6, 3), ("Equalize", 0.6, 7)],
    [("Posterize", 0.8, 5), ("Equalize", 1.0, 2)],
    [("Rotate", 0.2, 3), ("Solarize", 0.6, 8)],
    [("Equalize", 0.6, 8), ("Posterize", 0.4, 6)],
    [("Rotate", 0.8, 8), ("Color", 0.4, 0)],
    [("Rotate", 0.4, 9), ("Equalize", 0.6, 2)],
    [("Equalize", 0.0, 7), ("Equalize", 0.8, 8)],
    [("Equalize", 0.6, 4), ("Sharpness", 0.3, 3)],
    [("Solarize", 0.4, 5), ("AutoContrast", 0.9, 3)],
    [("Sharpness", 0.4, 7), ("Color", 0.7, 4)],
    [("Equalize", 0.3, 5), ("AutoContrast", 0.4, 2)],
    [("Color", 0.6, 3), ("Equalize", 1.0, 8)],
    [("AutoContrast", 0.4, 6), ("Solarize", 0.6, 5)],
    [("Rotate", 0.8, 8), ("Color", 1.0, 2)],
    [("Color", 0.8, 8), ("Solarize", 0.8, 7)],
    [("Sharpness", 0.4, 7), ("Solarize", 0.4, 4)],
    [("Contrast", 0.9, 8), ("Sharpness", 0.5, 8)],
    [("Color", 0.7, 7), ("TranslateX", 0.5, 8)],
    [("Equalize", 0.3, 7), ("AutoContrast", 0.4, 8)],
]


def _scale_policy(policy, factor: float, cap: float = 10.0):
    return [
        [(op, prob, max(0.0, min(mag * factor, cap))) for (op, prob, mag) in sub_policy]
        for sub_policy in policy
    ]


@dataclass
class AutoAugmentConfig:
    policy: list = field(default_factory=lambda: _IMAGENET_POLICY)
    magnitude_scale: float = 1.0


class AutoAugment(Transform):
    """Note: the published policy includes geometric ops (Rotate, ShearX/Y,
    TranslateX/Y) which move pixels -- like RandAugment's default op set,
    this is safe for classification but NOT target-aware. Restrict `policy`
    to photometric-only sub-policies before using with detection/segmentation."""

    def __init__(self, cfg: AutoAugmentConfig = AutoAugmentConfig()):
        self.cfg = cfg
        self._policy = _scale_policy(cfg.policy, cfg.magnitude_scale)

    def __call__(self, sample):
        sub_policy = random.choice(self._policy)
        for op_name, prob, magnitude in sub_policy:
            if random.random() < prob:
                sample.image = ALL_OPS[op_name](sample.image, magnitude)
        return sample


@register_transform(config=AutoAugmentConfig())
def auto_augment(cfg): return AutoAugment(cfg)


@register_transform(config=AutoAugmentConfig())
def auto_augment_weak(cfg):
    cfg.magnitude_scale = 0.5
    return AutoAugment(cfg)

 
@register_transform(config=AutoAugmentConfig())
def auto_augment_strong(cfg):
    cfg.magnitude_scale = 1.3
    return AutoAugment(cfg)
