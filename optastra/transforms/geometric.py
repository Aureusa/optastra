from dataclasses import dataclass
import math
import random
import torchvision.transforms.functional as F

from .base import Transform
from ._registry import register_transform
from ..nn.blocks.geometry.boxes import flip_boxes


__all__ = ["RandomHFlip", "RandomVFlip", "RandomResizedCrop"]

@dataclass
class ResizeConfig:
    size: int = 224


class Resize(Transform):
    """Resize image and target to the specified size."""

    def __init__(self, cfg: ResizeConfig = ResizeConfig()):
        self.cfg = cfg

    def _resize_boxes(self, sample):
        if "boxes" in sample.target:
            boxes = sample.target["boxes"].clone()
            boxes[:, [0, 2]] *= (self.cfg.size / sample.image.shape[-1])
            boxes[:, [1, 3]] *= (self.cfg.size / sample.image.shape[-2])
            sample.target["boxes"] = boxes
        return sample

    def _resize_masks(self, sample):
        if "masks" in sample.target:
            masks = sample.target["masks"]
            sample.target["masks"] = F.resize(masks, [self.cfg.size, self.cfg.size])
        return sample

    def __call__(self, sample):
        sample.image = F.resize(sample.image, [self.cfg.size, self.cfg.size])
        sample = self._resize_boxes(sample)
        sample = self._resize_masks(sample)
        return sample


@dataclass
class RandomResizedCropConfig:
    size: int = 224
    scale: tuple[float, float] = (0.08, 1.0)
    ratio: tuple[float, float] = (3 / 4, 4 / 3)


class RandomResizedCrop(Transform):
    """Target-aware: crop box is sampled once, then applied consistently
    to image, boxes, and masks -- same discipline as RandomHFlip."""

    def __init__(self, cfg: RandomResizedCropConfig = RandomResizedCropConfig()):
        self.cfg = cfg

    def _sample_crop_box(self, height: int, width: int) -> tuple[int, int, int, int]:
        area = height * width
        log_ratio = (math.log(self.cfg.ratio[0]), math.log(self.cfg.ratio[1]))
        for _ in range(10):
            target_area = area * random.uniform(*self.cfg.scale)
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < w <= width and 0 < h <= height:
                top = random.randint(0, height - h)
                left = random.randint(0, width - w)
                return top, left, h, w
        # fallback: center crop at the largest square that fits
        s = min(height, width)
        return (height - s) // 2, (width - s) // 2, s, s

    def __call__(self, sample):
        _, height, width = sample.image.shape
        top, left, h, w = self._sample_crop_box(height, width)
        sample.image = F.resized_crop(sample.image, top, left, h, w, [self.cfg.size, self.cfg.size])

        if "boxes" in sample.target:
            boxes = sample.target["boxes"].clone()
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left).clamp(0, w) * (self.cfg.size / w)
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top).clamp(0, h) * (self.cfg.size / h)
            # drop boxes that the crop reduced to zero area
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            sample.target["boxes"] = boxes[keep]
            if "labels" in sample.target:
                sample.target["labels"] = sample.target["labels"][keep]

        if "masks" in sample.target:
            masks = sample.target["masks"][:, top:top + h, left:left + w]
            sample.target["masks"] = F.resize(masks, [self.cfg.size, self.cfg.size])

        return sample


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


@register_transform(config=RandomResizedCropConfig())
def random_resized_crop(cfg): return RandomResizedCrop(cfg)

@register_transform(config=RandomFlipConfig())
def random_hflip(cfg: RandomFlipConfig): return RandomHFlip(cfg)

@register_transform(config=RandomFlipConfig())
def random_vflip(cfg: RandomFlipConfig): return RandomVFlip(cfg)
