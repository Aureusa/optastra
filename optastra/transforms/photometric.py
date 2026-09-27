"""
Photometric transforms -- they never touch the target, so they are safe for
every task family. All of them first convert integer images to float
(`functional.to_float_image`) and then work on any channel count and value
range; see functional.py for how the value range is inferred.
"""
from __future__ import annotations
from dataclasses import dataclass
import torchvision.transforms.functional as F

from . import functional as FN
from . import rng
from .base import Transform


__all__ = ["ColorJitter", "RandomGrayscale", "GaussianBlur", "Solarize"]


@dataclass
class ColorJitterConfig:
    strength: float = 0.5   # scales all four sub-jitters together, matches SimCLR/BYOL convention
    p: float = 0.8
    value_range: tuple[float, float] | None = None   # None -> inferred per image, see functional.py


class ColorJitter(Transform):
    """Random brightness, contrast, saturation, then hue (in that order).

    Channels: brightness/contrast/saturation work for any C (saturation
    blends each band towards the per-pixel band mean for C != 3, and is a
    no-op for C == 1). Hue is only defined for RGB, so it is skipped when
    C != 3."""

    def __init__(self, cfg: ColorJitterConfig = ColorJitterConfig()):
        self.cfg = cfg
        s = cfg.strength
        self.brightness, self.contrast, self.saturation, self.hue = 0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s

    def __call__(self, sample):
        img = FN.to_float_image(sample.image)
        sample.image = img
        if rng.rand() >= self.cfg.p:
            return sample
        value_range = FN.infer_value_range(img, self.cfg.value_range)
        for fn, factor_range in [
            (FN.adjust_brightness, self.brightness),
            (FN.adjust_contrast, self.contrast),
            (FN.adjust_saturation, self.saturation),
        ]:
            factor = rng.uniform(max(0, 1 - factor_range), 1 + factor_range)
            img = fn(img, factor, value_range)
        if img.shape[-3] == 3:
            img = FN.adjust_hue(img, rng.uniform(-self.hue, self.hue), value_range)
        sample.image = img
        return sample


@dataclass
class RandomGrayscaleConfig:
    p: float = 0.2


class RandomGrayscale(Transform):
    """Replace every channel by the image's luminance, keeping the channel
    count. For RGB this is the usual ITU-R 601 grayscale; for any other C
    every band becomes the band mean (a "panchromatic" image)."""

    def __init__(self, cfg: RandomGrayscaleConfig = RandomGrayscaleConfig()):
        self.cfg = cfg

    def __call__(self, sample):
        sample.image = FN.to_float_image(sample.image)
        if rng.rand() < self.cfg.p:
            sample.image = FN.grayscale(sample.image)
        return sample


@dataclass
class GaussianBlurConfig:
    p: float = 0.5
    sigma_range: tuple[float, float] = (0.1, 2.0)
    kernel_size: int | None = None   # None -> derived from image size (odd, ~10% of shorter side)


class GaussianBlur(Transform):
    """Gaussian blur of every channel independently; any C and value range."""

    def __init__(self, cfg: GaussianBlurConfig = GaussianBlurConfig()):
        self.cfg = cfg

    def __call__(self, sample):
        img = FN.to_float_image(sample.image)
        sample.image = img
        if rng.rand() >= self.cfg.p:
            return sample
        k = self.cfg.kernel_size
        if k is None:
            shorter = min(img.shape[-2], img.shape[-1])
            k = max(3, int(0.1 * shorter) | 1)   # force odd
        sigma = rng.uniform(*self.cfg.sigma_range)
        sample.image = F.gaussian_blur(img, kernel_size=[k, k], sigma=[sigma, sigma])
        return sample


@dataclass
class SolarizeConfig:
    p: float = 0.2
    threshold: float = 0.5   # fraction of the value range: pixels >= lo + threshold * (hi - lo) are inverted
    value_range: tuple[float, float] | None = None   # None -> inferred per image, see functional.py


class Solarize(Transform):
    """Invert every value above a threshold. `threshold` is relative to the
    value range, so for standard [0, 1] images it is simply the pixel value."""

    def __init__(self, cfg: SolarizeConfig = SolarizeConfig()):
        self.cfg = cfg

    def __call__(self, sample):
        img = FN.to_float_image(sample.image)
        sample.image = img
        if rng.rand() < self.cfg.p:
            lo, hi = FN.infer_value_range(img, self.cfg.value_range)
            sample.image = FN.solarize(img, lo + self.cfg.threshold * (hi - lo), (lo, hi))
        return sample


@Transform.register(config=GaussianBlurConfig())
def gaussian_blur(cfg): return GaussianBlur(cfg)

@Transform.register(config=SolarizeConfig())
def solarize(cfg): return Solarize(cfg)

@Transform.register(config=RandomGrayscaleConfig())
def random_grayscale(cfg): return RandomGrayscale(cfg)

@Transform.register(config=ColorJitterConfig())
def color_jitter(cfg): return ColorJitter(cfg)
