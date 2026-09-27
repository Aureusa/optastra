"""
PixMix (Hendrycks et al., 2022, arXiv:2112.05135). Mixes the augmented
image with a "fractal/structurally complex" image from an auxiliary
mixing set, via random additive or multiplicative blending -- goes further
than AugMix by introducing genuinely different visual structure, not just
photometric perturbation of the same image.

Takes an optional `mixing_set`: a Dataset or list of tensor images (or
Samples) unrelated to the training data. The original paper uses fractal
images and public art datasets; this implementation accepts any indexable
image source, so users can supply domain-appropriate mixing images rather
than a fixed hardcoded set. Without one, a cheap procedural fractal is used.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torchvision.transforms.functional as F

from . import functional as FN
from . import rng
from .base import Transform
from .ops import PHOTOMETRIC_OP_NAMES, PHOTOMETRIC_OPS, check_op_names
from ..data.sample import Sample


__all__ = ["PixMix"]


def _generate_fractal(channels: int, height: int, width: int) -> torch.Tensor:
    """
    Fallback mixing image when no mixing_set is provided: a simple
    midpoint-displacement-style noise fractal, cheap and dependency-free.
    Not a substitute for the paper's curated fractal set, but keeps PixMix
    usable out of the box without requiring an external asset download.
    Returns a (channels, height, width) tensor in [0, 1], same pattern in every channel.
    """
    g = rng.get_generator()
    noise = torch.rand(1, height, width, generator=g)
    for _ in range(3):
        noise = torch.nn.functional.avg_pool2d(noise, kernel_size=3, stride=1, padding=1)
        noise = noise + 0.3 * torch.rand(noise.shape, generator=g)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise.expand(channels, height, width).clone()


@dataclass
class PixMixConfig:
    num_mixing_rounds: int = 3
    beta: float = 3.0   # controls blend strength
    ops: tuple[str, ...] = PHOTOMETRIC_OP_NAMES
    preaugment_magnitude: float = 3.0
    value_range: tuple[float, float] | None = None   # None -> inferred per image, see functional.py


class PixMix(Transform):
    """Photometric only, so targets are never touched. Works on float images
    with any channel count, non-square sizes and any value range: blending
    happens in [0, 1] "unit" space (image mapped through its value range),
    and the result is mapped back -- clamped to the value range, not to [0, 1].

    Mixing images must have the input's channel count (or 1 channel, which is
    broadcast); they are resized to the input's HxW and normalized by their
    own value range."""

    def __init__(self, cfg: PixMixConfig = PixMixConfig(), mixing_set=None):
        check_op_names(cfg.ops, allowed=PHOTOMETRIC_OPS)
        self.cfg = cfg
        self.mixing_set = mixing_set   # optional indexable image source

    def _get_mixing_image(self, img: torch.Tensor) -> torch.Tensor:
        """A mixer shaped like `img`, with values in [0, 1]."""
        c, h, w = img.shape[-3:]
        if self.mixing_set is None:
            return _generate_fractal(c, h, w).to(img.device)

        mixer = self.mixing_set[rng.randint(0, len(self.mixing_set) - 1)]
        if isinstance(mixer, Sample):
            mixer = mixer.image
        mixer = FN.to_float_image(mixer).to(img.device, torch.float32)
        if mixer.shape[-3] == 1 and c != 1:
            mixer = mixer.expand(c, *mixer.shape[-2:])
        elif mixer.shape[-3] != c:
            raise ValueError(f"PixMix mixing image has {mixer.shape[-3]} channels, input has {c}.")
        if tuple(mixer.shape[-2:]) != (h, w):
            mixer = F.resize(mixer, [h, w])
        lo, hi = FN.infer_value_range(mixer)
        return ((mixer - lo) / max(hi - lo, 1e-12)).clamp(0.0, 1.0)

    def _mix(self, img: torch.Tensor, mixer: torch.Tensor, value_range: tuple[float, float]) -> torch.Tensor:
        lo, hi = value_range
        scale = max(hi - lo, 1e-12)
        unit = ((img - lo) / scale).clamp(0.0, 1.0)
        w = rng.beta(self.cfg.beta, self.cfg.beta)
        if rng.rand() < 0.5:
            out = w * unit + (1 - w) * mixer      # additive
        else:
            out = unit * (mixer ** w)             # multiplicative (geometric-mean-like blend)
        return out.clamp(0.0, 1.0) * scale + lo

    def _augment(self, img: torch.Tensor, value_range: tuple[float, float]) -> torch.Tensor:
        op = PHOTOMETRIC_OPS[rng.choice(self.cfg.ops)]
        return op(img, rng.uniform(0.1, self.cfg.preaugment_magnitude), value_range)

    def __call__(self, sample):
        img = FN.to_float_image(sample.image)
        value_range = FN.infer_value_range(img, self.cfg.value_range)
        img = self._augment(img, value_range)
        for _ in range(self.cfg.num_mixing_rounds):
            if rng.rand() < 0.5:
                img = self._mix(img, self._get_mixing_image(img), value_range)
            else:
                img = self._augment(img, value_range)
        sample.image = img
        return sample


@Transform.register(config=PixMixConfig())
def pixmix(cfg): return PixMix(cfg)
