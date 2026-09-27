"""
Geometric transforms. Every one of them moves the image AND its targets
together -- a transform that moves pixels but not boxes silently corrupts
detection/segmentation training.

Target conventions (per Sample):

* `target["boxes"]`  -- (N, 4) float, XYXY in absolute pixel coordinates
  (pixel i covers [i, i+1)). Boxes are clipped to the image; boxes left with
  zero width or height are dropped together with their `labels` and `masks`.
* `target["masks"]`  -- (N, H, W) instance masks; `target["mask"]` -- (H, W)
  semantic map. Warped with nearest-neighbour interpolation so binary / integer
  masks stay binary / integer; dtype is preserved. Areas that come from outside
  the image are filled with 0.
* anything else in `target` (e.g. a class "label") is left untouched.

The `*_sample` helpers below are the building blocks; use them to write your
own target-aware transforms.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from . import rng
from .base import Transform
from ..data.sample import Sample
from ..nn.blocks.geometry.boxes import flip_boxes


__all__ = [
    "Crop", "Resize", "RandomCrop", "RandomRotation", "RandomHFlip", "RandomVFlip", "RandomResizedCrop",
    "crop_sample", "resize_sample", "affine_sample",
]

DENSE_KEYS = ("masks", "mask")              # per-pixel targets, warped with nearest-neighbour
INSTANCE_KEYS = ("boxes", "labels", "masks")  # per-instance targets, filtered together


# ---- target helpers ---------------------------------------------------------

def _map_dense_targets(target: dict, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
    """Apply a spatial op to every dense target. `fn` gets a float (N, H, W)
    tensor; interpolating ops must use nearest-neighbour."""
    for key in DENSE_KEYS:
        if key not in target:
            continue
        masks = target[key]
        as_3d = masks if masks.ndim >= 3 else masks.unsqueeze(0)
        if as_3d.numel() == 0:
            # torchvision can't interpolate an empty stack; only the output size matters
            out_hw = fn(torch.zeros(1, *as_3d.shape[-2:])).shape[-2:]
            out = masks.new_zeros(*as_3d.shape[:-2], *out_hw)
        else:
            out = fn(as_3d.to(torch.float32)).to(masks.dtype)
        target[key] = out if masks.ndim >= 3 else out.squeeze(0)


def _clip_and_filter_boxes(target: dict, width: int, height: int) -> None:
    """Clip boxes to the image and drop degenerate ones (with their labels/masks)."""
    boxes = target["boxes"].clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
    target["boxes"] = boxes
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not bool(keep.all()):
        for key in INSTANCE_KEYS:
            if key in target:
                target[key] = target[key][keep]


def crop_sample(sample: Sample, top: int, left: int, height: int, width: int) -> Sample:
    """Crop the window [top, top+height) x [left, left+width). Parts of the
    window outside the image are zero-padded (like F.crop)."""
    sample.image = F.crop(sample.image, top, left, height, width)
    _map_dense_targets(sample.target, lambda m: F.crop(m, top, left, height, width))
    if "boxes" in sample.target:
        boxes = sample.target["boxes"].clone()
        boxes[:, 0::2] -= left
        boxes[:, 1::2] -= top
        sample.target["boxes"] = boxes
        _clip_and_filter_boxes(sample.target, width, height)
    return sample


def resize_sample(
    sample: Sample,
    height: int,
    width: int,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> Sample:
    """Resize to (height, width); boxes are scaled by new/old size."""
    old_h, old_w = sample.image.shape[-2:]
    sample.image = F.resize(sample.image, [height, width], interpolation=interpolation)
    # NEAREST_EXACT samples at pixel centres; torch's legacy NEAREST is shifted by up to a pixel
    _map_dense_targets(sample.target, lambda m: F.resize(m, [height, width], interpolation=InterpolationMode.NEAREST_EXACT))
    if "boxes" in sample.target:
        boxes = sample.target["boxes"].clone()
        boxes[:, 0::2] *= width / old_w
        boxes[:, 1::2] *= height / old_h
        sample.target["boxes"] = boxes
    return sample


def _affine_forward_matrix(angle: float, translate, scale: float, shear) -> torch.Tensor:
    """2x3 forward map of F.affine, in pixel coordinates relative to the image
    centre (y points down, so positive `angle` is clockwise on screen).
    torchvision builds the *inverse* of exactly this matrix
    (`_get_inverse_affine_matrix`); boxes need the forward direction."""
    rot = math.radians(angle)
    sx, sy = math.radians(shear[0]), math.radians(shear[1])
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)
    return torch.tensor([
        [scale * a, scale * b, float(translate[0])],
        [scale * c, scale * d, float(translate[1])],
    ], dtype=torch.float64)


def _affine_boxes(boxes: torch.Tensor, matrix: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """New box = axis-aligned bounding box of the 4 transformed corners."""
    x1, y1, x2, y2 = boxes.to(torch.float64).unbind(-1)
    corners = torch.stack([
        torch.stack([x1, y1], -1), torch.stack([x2, y1], -1),
        torch.stack([x1, y2], -1), torch.stack([x2, y2], -1),
    ], dim=1)                                                    # (N, 4, 2)
    centre = torch.tensor([width / 2, height / 2], dtype=torch.float64)
    moved = (corners - centre) @ matrix[:, :2].T + matrix[:, 2] + centre
    return torch.cat([moved.amin(dim=1), moved.amax(dim=1)], dim=-1).to(boxes.dtype)


def affine_sample(
    sample: Sample,
    angle: float = 0.0,
    translate: tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    shear: tuple[float, float] = (0.0, 0.0),
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    image_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Sample:
    """F.affine applied to image, masks (nearest) and boxes (bounding box of the
    moved corners, clipped). Arguments follow F.affine: `angle` in degrees,
    clockwise. `image_fn` optionally replaces the image warp (e.g. a
    reflect-padded rotation) -- it must implement the same geometry."""
    height, width = sample.image.shape[-2:]
    translate, shear = list(translate), list(shear)
    if image_fn is None:
        sample.image = F.affine(sample.image, angle=angle, translate=translate, scale=scale,
                                shear=shear, interpolation=interpolation)
    else:
        sample.image = image_fn(sample.image)
    _map_dense_targets(sample.target, lambda m: F.affine(
        m, angle=angle, translate=translate, scale=scale, shear=shear,
        interpolation=InterpolationMode.NEAREST,
    ))
    if "boxes" in sample.target:
        matrix = _affine_forward_matrix(angle, translate, scale, shear)
        sample.target["boxes"] = _affine_boxes(sample.target["boxes"], matrix, width, height)
        _clip_and_filter_boxes(sample.target, width, height)
    return sample


def _hflip(img: torch.Tensor) -> torch.Tensor:
    """F.hflip, but also for uint16 (torch.flip has no CPU uint16 kernel)."""
    return img.index_select(-1, torch.arange(img.shape[-1] - 1, -1, -1, device=img.device))


def _center_offset(size: int, crop: int) -> int:
    """Top/left offset F.center_crop uses (negative -> the crop zero-pads)."""
    return int(round((size - crop) / 2.0)) if size >= crop else -((crop - size) // 2)


# ---- transforms -------------------------------------------------------------

@dataclass
class CropConfig:
    size: int = 224


class Crop(Transform):
    """Center crop image and targets to `size` x `size`."""

    def __init__(self, cfg: CropConfig = CropConfig()):
        self.cfg = cfg

    def __call__(self, sample):
        h, w = sample.image.shape[-2:]
        s = self.cfg.size
        return crop_sample(sample, _center_offset(h, s), _center_offset(w, s), s, s)


@dataclass
class RandomCropConfig:
    size: int = 224


class RandomCrop(Transform):
    """Crop a fixed-size window at a uniformly random position inside the image."""

    def __init__(self, cfg: RandomCropConfig = RandomCropConfig()):
        self.cfg = cfg

    def __call__(self, sample):
        h, w = sample.image.shape[-2:]
        size = self.cfg.size
        if h < size or w < size:
            raise ValueError(f"Image ({h}x{w}) is smaller than crop size {size}")
        top = rng.randint(0, h - size)
        left = rng.randint(0, w - size)
        return crop_sample(sample, top, left, size, size)


@dataclass
class RandomRotationConfig:
    degrees: tuple[float, float] = (0.0, 180.0)
    interpolation: str = "bilinear"


class RandomRotation(Transform):
    """
    Random rotation about the image centre, counter-clockwise by an angle
    drawn uniformly from `degrees`. The image is reflection-padded first, so
    the corners are filled with mirrored content instead of black.

    Targets: masks are rotated with nearest-neighbour (the corners are 0 --
    mirrored content is not annotated), boxes become the bounding box of
    their rotated corners, clipped to the image. Note that a rotated box's
    bounding box is looser than the object, and mirrored copies of objects
    in the corners carry no box.

    Interpolation modes (image only):
    ``nearest``, ``nearest-exact``, ``bilinear``, ``bicubic``, ``box``, ``hamming``, ``lanczos``.
    """
    interpolation_modes = {
        "nearest": InterpolationMode.NEAREST,
        "nearest-exact": InterpolationMode.NEAREST_EXACT,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "box": InterpolationMode.BOX,
        "hamming": InterpolationMode.HAMMING,
        "lanczos": InterpolationMode.LANCZOS,
    }

    def __init__(self, cfg: RandomRotationConfig = RandomRotationConfig()):
        self.cfg = cfg
        self.interpolation = self.interpolation_modes[self.cfg.interpolation]

    def _rotate_reflect(self, image, angle):
        h, w = image.shape[-2:]
        # pad enough that the rotated corners never expose the border
        pad = int(math.ceil((math.sqrt(h ** 2 + w ** 2) - min(h, w)) / 2)) + 1
        padded = F.pad(image, pad, padding_mode="reflect")
        rotated = F.rotate(padded, angle, interpolation=self.interpolation, expand=False)
        return F.center_crop(rotated, [h, w])

    def __call__(self, sample):
        angle = rng.uniform(*self.cfg.degrees)
        # F.rotate is counter-clockwise, F.affine clockwise -> same geometry with -angle
        return affine_sample(sample, angle=-angle, image_fn=lambda img: self._rotate_reflect(img, angle))


@dataclass
class ResizeConfig:
    size: int = 224


class Resize(Transform):
    """Resize image and targets to `size` x `size` (aspect ratio not kept)."""

    def __init__(self, cfg: ResizeConfig = ResizeConfig()):
        self.cfg = cfg

    def __call__(self, sample):
        return resize_sample(sample, self.cfg.size, self.cfg.size)


@dataclass
class RandomResizedCropConfig:
    size: int = 224
    scale: tuple[float, float] = (0.08, 1.0)
    ratio: tuple[float, float] = (3 / 4, 4 / 3)


class RandomResizedCrop(Transform):
    """Crop a random area/aspect-ratio window, then resize it to `size` x `size`.
    The window is sampled once and applied to image, boxes and masks."""

    def __init__(self, cfg: RandomResizedCropConfig = RandomResizedCropConfig()):
        self.cfg = cfg

    def _sample_crop_box(self, height: int, width: int) -> tuple[int, int, int, int]:
        area = height * width
        log_ratio = (math.log(self.cfg.ratio[0]), math.log(self.cfg.ratio[1]))
        for _ in range(10):
            target_area = area * rng.uniform(*self.cfg.scale)
            aspect_ratio = math.exp(rng.uniform(*log_ratio))
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < w <= width and 0 < h <= height:
                top = rng.randint(0, height - h)
                left = rng.randint(0, width - w)
                return top, left, h, w
        # fallback: center crop at the largest square that fits
        s = min(height, width)
        return (height - s) // 2, (width - s) // 2, s, s

    def __call__(self, sample):
        height, width = sample.image.shape[-2:]
        top, left, h, w = self._sample_crop_box(height, width)
        sample = crop_sample(sample, top, left, h, w)
        return resize_sample(sample, self.cfg.size, self.cfg.size)


@dataclass
class RandomFlipConfig:
    p: float = 0.5


class RandomHFlip(Transform):
    """Horizontal flip with probability `p`."""

    def __init__(self, cfg: RandomFlipConfig = RandomFlipConfig()):
        self.cfg = cfg

    def __call__(self, sample):
        if rng.rand() >= self.cfg.p:
            return sample
        sample.image = _hflip(sample.image)
        _map_dense_targets(sample.target, _hflip)
        if "boxes" in sample.target:
            sample.target["boxes"] = flip_boxes(sample.target["boxes"], sample.image.shape[-1])
        return sample


class RandomVFlip(Transform):
    """Vertical flip with probability `p`."""

    def __init__(self, cfg: RandomFlipConfig = RandomFlipConfig()):
        self.cfg = cfg

    def __call__(self, sample):
        if rng.rand() >= self.cfg.p:
            return sample
        sample.image = F.vflip(sample.image)
        _map_dense_targets(sample.target, F.vflip)
        if "boxes" in sample.target:
            sample.target["boxes"] = flip_boxes(
                sample.target["boxes"], sample.image.shape[-1], sample.image.shape[-2], f_type="v"
            )
        return sample


@Transform.register(config=CropConfig())
def crop(cfg: CropConfig): return Crop(cfg)


@Transform.register(config=RandomCropConfig())
def random_crop(cfg: RandomCropConfig): return RandomCrop(cfg)


@Transform.register(config=RandomRotationConfig())
def random_rotation(cfg: RandomRotationConfig): return RandomRotation(cfg)


@Transform.register(config=ResizeConfig())
def resize(cfg: ResizeConfig): return Resize(cfg)


@Transform.register(config=RandomResizedCropConfig())
def random_resized_crop(cfg): return RandomResizedCrop(cfg)


@Transform.register(config=RandomFlipConfig())
def random_hflip(cfg: RandomFlipConfig): return RandomHFlip(cfg)


@Transform.register(config=RandomFlipConfig())
def random_vflip(cfg: RandomFlipConfig): return RandomVFlip(cfg)
