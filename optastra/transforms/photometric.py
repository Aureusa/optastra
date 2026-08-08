from __future__ import annotations
from dataclasses import dataclass
import random
import torchvision.transforms.functional as F

from .base import Transform
from .functional import safe_solarize
from ._registry import register_transform


__all__ = ["ColorJitter", "RandomGrayscale", "GaussianBlur", "Solarize"]


@dataclass
class ColorJitterConfig:
    strength: float = 0.5   # scales all four sub-jitters together, matches SimCLR/BYOL convention
    p: float = 0.8


class ColorJitter(Transform):
    """Photometric only -- never touches target, safe on any task family."""

    def __init__(self, cfg: ColorJitterConfig = ColorJitterConfig()):
        self.cfg = cfg
        s = cfg.strength
        self.brightness, self.contrast, self.saturation, self.hue = 0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s

    def __call__(self, sample):
        if random.random() >= self.cfg.p:
            return sample
        img = sample.image
        for fn, factor_range in [
            (F.adjust_brightness, self.brightness),
            (F.adjust_contrast, self.contrast),
            (F.adjust_saturation, self.saturation),
        ]:
            factor = random.uniform(max(0, 1 - factor_range), 1 + factor_range)
            img = fn(img, factor)
        hue_factor = random.uniform(-self.hue, self.hue)
        img = F.adjust_hue(img, hue_factor)
        sample.image = img
        return sample


@dataclass
class RandomGrayscaleConfig:
    p: float = 0.2


class RandomGrayscale(Transform):
    def __init__(self, cfg: RandomGrayscaleConfig = RandomGrayscaleConfig()):
        self.cfg = cfg

    def __call__(self, sample):
        if random.random() < self.cfg.p:
            sample.image = F.rgb_to_grayscale(sample.image, num_output_channels=3)
        return sample
    

@dataclass
class GaussianBlurConfig:
    p: float = 0.5
    sigma_range: tuple[float, float] = (0.1, 2.0)
    kernel_size: int | None = None   # None -> derived from image size (odd, ~10% of shorter side)


class GaussianBlur(Transform):
    def __init__(self, cfg: GaussianBlurConfig = GaussianBlurConfig()):
        self.cfg = cfg

    def __call__(self, sample):
        if random.random() >= self.cfg.p:
            return sample
        img = sample.image
        k = self.cfg.kernel_size
        if k is None:
            shorter = min(img.shape[-2], img.shape[-1])
            k = max(3, int(0.1 * shorter) | 1)   # force odd
        sigma = random.uniform(*self.cfg.sigma_range)
        sample.image = F.gaussian_blur(img, kernel_size=[k, k], sigma=[sigma, sigma])
        return sample
    

@dataclass
class SolarizeConfig:
    p: float = 0.2
    threshold: float = 0.5   # image assumed in [0,1] range; adjust if your pipeline uses [0,255]


class Solarize(Transform):
    def __init__(self, cfg: SolarizeConfig = SolarizeConfig()):
        self.cfg = cfg

    def __call__(self, sample):
        if random.random() < self.cfg.p:
            sample.image = safe_solarize(sample.image, self.cfg.threshold)
        return sample


@register_transform(config=GaussianBlurConfig())
def gaussian_blur(cfg): return GaussianBlur(cfg)

@register_transform(config=SolarizeConfig())
def solarize(cfg): return Solarize(cfg)

@register_transform(config=RandomGrayscaleConfig())
def random_grayscale(cfg): return RandomGrayscale(cfg)

@register_transform(config=ColorJitterConfig())
def color_jitter(cfg): return ColorJitter(cfg)
