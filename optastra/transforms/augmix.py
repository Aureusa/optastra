"""
AugMix (Hendrycks et al., 2020, arXiv:1912.02781). Builds several
independent augmentation chains from the same image, mixes them via a
Dirichlet-weighted sum, then interpolates with the original image via Beta
weighting -- produces diverse but structurally coherent augmentations,
aimed at corruption robustness. Photometric ops only by design (the paper
explicitly excludes geometric ops that could push augmented images out of
plausible distribution)."""
from __future__ import annotations
from dataclasses import dataclass
import torch

from . import functional as FN
from . import rng
from .base import Transform
from .ops import PHOTOMETRIC_OP_NAMES, PHOTOMETRIC_OPS, check_op_names


__all__ = ["AugMix"]


@dataclass
class AugMixConfig:
    num_chains: int = 3
    chain_depth: int = -1   # -1 -> random depth 1-3 per chain, as in the paper
    magnitude: float = 3.0
    alpha: float = 1.0      # Dirichlet/Beta concentration
    ops: tuple[str, ...] = PHOTOMETRIC_OP_NAMES
    value_range: tuple[float, float] | None = None   # None -> inferred per image, see functional.py


class AugMix(Transform):
    """Photometric only, so targets are never touched. Works on float images
    with any channel count and value range; the result is clamped to the
    value range, not to [0, 1]."""

    def __init__(self, cfg: AugMixConfig = AugMixConfig()):
        check_op_names(cfg.ops, allowed=PHOTOMETRIC_OPS)
        self.cfg = cfg

    def _augment_chain(self, img: torch.Tensor, value_range: tuple[float, float]) -> torch.Tensor:
        depth = self.cfg.chain_depth if self.cfg.chain_depth > 0 else rng.randint(1, 3)
        out = img
        for _ in range(depth):
            op = PHOTOMETRIC_OPS[rng.choice(self.cfg.ops)]
            out = op(out, rng.uniform(0.1, self.cfg.magnitude), value_range)
        return out

    def __call__(self, sample):
        img = FN.to_float_image(sample.image)
        value_range = FN.infer_value_range(img, self.cfg.value_range)
        weights = rng.dirichlet(self.cfg.alpha, self.cfg.num_chains)
        mix = torch.zeros_like(img)
        for i in range(self.cfg.num_chains):
            mix += weights[i] * self._augment_chain(img, value_range)

        m = rng.beta(self.cfg.alpha, self.cfg.alpha)
        sample.image = (m * img + (1 - m) * mix).clamp(*value_range)
        return sample


@Transform.register(config=AugMixConfig())
def augmix(cfg): return AugMix(cfg)


@Transform.register(config=AugMixConfig(num_chains=2, chain_depth=1, magnitude=1.5))
def augmix_weak(cfg): return AugMix(cfg)


@Transform.register(config=AugMixConfig(num_chains=3, chain_depth=3, magnitude=7.0))
def augmix_strong(cfg): return AugMix(cfg)
