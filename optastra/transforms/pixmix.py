"""
PixMix (Hendrycks et al., 2022, arXiv:2112.05135). Mixes the augmented
image with a "fractal/structurally complex" image from an auxiliary
mixing set, via random additive or multiplicative blending -- goes further
than AugMix by introducing genuinely different visual structure, not just
photometric perturbation of the same image.

Requires a `mixing_set`: a Dataset or list of PIL/tensor images unrelated
to the training data. The original paper uses fractal images and public
art datasets; this implementation accepts any indexable image source, so
users can supply domain-appropriate mixing images rather than a fixed
hardcoded set.
"""
from dataclasses import dataclass, field
import random
import torch

from .base import Transform
from .ops import PHOTOMETRIC_OPS
from ._registry import register_transform


__all__ = ["PixMix"]


def _generate_fractal(size: int, device=None) -> torch.Tensor:
    """
    Fallback mixing image when no mixing_set is provided: a simple
    midpoint-displacement-style noise fractal, cheap and dependency-free.
    Not a substitute for the paper's curated fractal set, but keeps PixMix
    usable out of the box without requiring an external asset download.
    """
    noise = torch.rand(1, size, size, device=device)
    for _ in range(3):
        noise = torch.nn.functional.avg_pool2d(noise, kernel_size=3, stride=1, padding=1)
        noise = noise + 0.3 * torch.rand_like(noise)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise.expand(3, size, size).clone()


@dataclass
class PixMixConfig:
    num_mixing_rounds: int = 3
    beta: float = 3.0   # controls blend strength
    ops: list[str] = field(default_factory=lambda: list(PHOTOMETRIC_OPS.keys()))
    preaugment_magnitude: float = 3.0


class PixMix(Transform):
    def __init__(self, cfg: PixMixConfig = PixMixConfig(), mixing_set=None):
        self.cfg = cfg
        self.mixing_set = mixing_set   # optional indexable image source

    def _get_mixing_image(self, size: int, device) -> torch.Tensor:
        if self.mixing_set is not None:
            img = self.mixing_set[random.randrange(len(self.mixing_set))]
            return img.to(device)
        return _generate_fractal(size, device=device)

    def _mix(self, img: torch.Tensor, mixer: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            # additive
            w = torch.distributions.Beta(self.cfg.beta, self.cfg.beta).sample().item()
            out = w * img + (1 - w) * mixer
        else:
            # multiplicative (geometric-mean-like blend)
            out = img * (mixer ** torch.distributions.Beta(self.cfg.beta, self.cfg.beta).sample().item())
        return out.clamp(0, 1)

    def __call__(self, sample):
        img = sample.image
        op_name = random.choice(self.cfg.ops)
        img = PHOTOMETRIC_OPS[op_name](img, random.uniform(0.1, self.cfg.preaugment_magnitude))

        size = img.shape[-1]
        for _ in range(self.cfg.num_mixing_rounds):
            mixer = self._get_mixing_image(size, img.device)
            if random.random() < 0.5:
                img = self._mix(img, mixer)
            else:
                op_name = random.choice(self.cfg.ops)
                img = PHOTOMETRIC_OPS[op_name](img, random.uniform(0.1, self.cfg.preaugment_magnitude))

        sample.image = img
        return sample


@register_transform(config=PixMixConfig())
def pixmix(cfg): return PixMix(cfg)

