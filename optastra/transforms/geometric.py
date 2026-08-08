from dataclasses import dataclass
import random
import torchvision.transforms.functional as F

from .base import Transform
from ._registry import register_transform
from ..nn.blocks.geometry.boxes import flip_boxes


__all__ = ["RandomHFlip", "RandomVFlip"]


@dataclass
class RandomFlipConfig:
    p: float = 0.5

class RandomHFlip(Transform):
    def __init__(self, cfg: RandomFlipConfig = RandomFlipConfig()):
        self.cfg = cfg

    def __call__(self, sample):
        if random.random() >= self.cfg.p:
            return sample
        sample.image = F.hflip(sample.image)
        if "boxes" in sample.target:
            sample.target["boxes"] = flip_boxes(sample.target["boxes"], sample.image.shape[-1])
        if "masks" in sample.target:
            sample.target["masks"] = F.hflip(sample.target["masks"])
        return sample

class RandomVFlip(Transform):
    def __init__(self, cfg: RandomFlipConfig = RandomFlipConfig()):
        self.cfg = cfg

    def __call__(self, sample):
        if random.random() >= self.cfg.p:
            return sample
        sample.image = F.vflip(sample.image)
        if "boxes" in sample.target:
            sample.target["boxes"] = flip_boxes(
                sample.target["boxes"],
                sample.image.shape[-1],
                sample.image.shape[-2],
                f_type="v"
            )
        if "masks" in sample.target:
            sample.target["masks"] = F.vflip(sample.target["masks"])
        return sample


@register_transform(config=RandomFlipConfig())
def random_hflip(cfg: RandomFlipConfig): return RandomHFlip(cfg)

@register_transform(config=RandomFlipConfig())
def random_vflip(cfg: RandomFlipConfig): return RandomVFlip(cfg)
