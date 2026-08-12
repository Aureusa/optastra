"""
AugMix (Hendrycks et al., 2020, arXiv:1912.02781). Builds several
independent augmentation chains from the same image, mixes them via a
Dirichlet-weighted sum, then interpolates with the original image via Beta
weighting -- produces diverse but structurally coherent augmentations,
aimed at corruption robustness. Photometric ops only by design (the paper
explicitly excludes geometric ops that could push augmented images out of
plausible distribution)."""
from dataclasses import dataclass, field
import random
import torch

from .base import Transform
from .ops import PHOTOMETRIC_OPS
from ._registry import register_transform


__all__ = ["AugMix"]


@dataclass
class AugMixConfig:
    num_chains: int = 3
    chain_depth: int = -1   # -1 -> random depth 1-3 per chain, as in the paper
    magnitude: float = 3.0
    alpha: float = 1.0      # Dirichlet/Beta concentration
    ops: list[str] = field(default_factory=lambda: list(PHOTOMETRIC_OPS.keys()))


class AugMix(Transform):
    def __init__(self, cfg: AugMixConfig = AugMixConfig()):
        self.cfg = cfg

    def _augment_chain(self, img: torch.Tensor) -> torch.Tensor:
        depth = self.cfg.chain_depth if self.cfg.chain_depth > 0 else random.randint(1, 3)
        out = img
        for _ in range(depth):
            op_name = random.choice(self.cfg.ops)
            out = PHOTOMETRIC_OPS[op_name](out, random.uniform(0.1, self.cfg.magnitude))
        return out

    def __call__(self, sample):
        img = sample.image
        weights = torch.distributions.Dirichlet(
            torch.full((self.cfg.num_chains,), self.cfg.alpha)
        ).sample()
        mix = torch.zeros_like(img)
        for i in range(self.cfg.num_chains):
            mix += weights[i] * self._augment_chain(img)

        m = torch.distributions.Beta(self.cfg.alpha, self.cfg.alpha).sample()
        sample.image = (m * img + (1 - m) * mix).clamp(0, 1)
        return sample


@register_transform(config=AugMixConfig())
def augmix(cfg): return AugMix(cfg)
