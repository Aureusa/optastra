"""
RandAugment implementation based on the original paper:
https://arxiv.org/abs/1909.13719

The original implementation is available at:
https://www.github.com/tensorflow/tpu/tree/master/models/official/efficientnet


TODO: Split _OPS into categories: _PHOTOMETRIC_OPS and _GEOMETRIC_OPS,
and allow users to select which categories to use. This is usefull as if you
rotate the image and do not transform the boxes, the boxes will be misaligned with the image.
Photometric ops do not have this problem, so they are safe to use with detection tasks.
"""
from dataclasses import dataclass, field
import math
import random

import torch
import torchvision.transforms.functional as F

from .base import Transform
from ._registry import register_transform


__all__ = ["RandAugment"]


def _to_uint8(img):
    return (img * 255.0).clamp(0, 255).to(torch.uint8)


def _from_uint8(img):
    return img.to(torch.float32) / 255.0


def _identity(img, m):
    return img


def _rotate(img, m):
    degrees = random.choice([-1, 1]) * (m / 10.0) * 30
    return F.rotate(img, degrees)


def _posterize(img, m):
    bits = int(round(8 - (m / 10.0) * 4))
    bits = max(1, min(8, bits))

    # F.posterize does not support floating point images, so we convert to uint8 and back
    if img.is_floating_point():
        img_uint8 = (img * 255.0).clamp(0, 255).to(torch.uint8)
        out = F.posterize(img_uint8, bits)
        return out.to(torch.float32) / 255.0

    return F.posterize(img, bits)


def _autocontrast(img, m):
    if img.is_floating_point():
        img = _to_uint8(img)
        img = F.autocontrast(img)
        return _from_uint8(img)

    return F.autocontrast(img)


def _solarize(img, m):
    threshold = 1.0 - (m / 10.0)

    # If the image is floating point, we need to ensure that the threshold is less
    # than the maximum pixel value in the image. Otherwise, torchvision will raise an error.
    if img.is_floating_point():
        # torchvision requires threshold < max(img)
        threshold = min(threshold, float(img.max()) - 1e-6)
        threshold = max(threshold, 0.0)

    else:
        threshold = min(threshold * 255, 255)

    return F.solarize(img, threshold)


def _equalize(img, m):
    if img.is_floating_point():
        img = _to_uint8(img)
        img = F.equalize(img)
        return _from_uint8(img)

    return F.equalize(img)


def _color(img, m):
    factor = 1 + random.choice([-1, 1]) * (m / 10.0) * 0.9
    return F.adjust_saturation(img, factor)


def _contrast(img, m):
    factor = 1 + random.choice([-1, 1]) * (m / 10.0) * 0.9
    return F.adjust_contrast(img, factor)


def _brightness(img, m):
    factor = 1 + random.choice([-1, 1]) * (m / 10.0) * 0.9
    return F.adjust_brightness(img, factor)


def _sharpness(img, m):
    factor = 1 + random.choice([-1, 1]) * (m / 10.0) * 0.9
    return F.adjust_sharpness(img, factor)


def _shear_x(img, m):
    # RandAugment shear max ~= 0.3
    # torchvision expects degrees
    degrees = math.degrees(math.atan(0.3 * (m / 10.0)))
    degrees *= random.choice([-1, 1])

    return F.affine(
        img,
        angle=0,
        translate=[0, 0],
        scale=1.0,
        shear=[degrees, 0],
    )


def _shear_y(img, m):
    degrees = math.degrees(math.atan(0.3 * (m / 10.0)))
    degrees *= random.choice([-1, 1])

    return F.affine(
        img,
        angle=0,
        translate=[0, 0],
        scale=1.0,
        shear=[0, degrees],
    )


def _translate_x(img, m):
    max_shift = img.shape[-1] * 0.3
    shift = int(random.choice([-1, 1]) * (m / 10.0) * max_shift)

    return F.affine(
        img,
        angle=0,
        translate=[shift, 0],
        scale=1.0,
        shear=[0, 0],
    )


def _translate_y(img, m):
    max_shift = img.shape[-2] * 0.3
    shift = int(random.choice([-1, 1]) * (m / 10.0) * max_shift)

    return F.affine(
        img,
        angle=0,
        translate=[0, shift],
        scale=1.0,
        shear=[0, 0],
    )


_OPS = {
    "Identity": _identity,
    "AutoContrast": _autocontrast,
    "Equalize": _equalize,

    "Rotate": _rotate,
    "Posterize": _posterize,
    "Solarize": _solarize,

    "Color": _color,
    "Contrast": _contrast,
    "Brightness": _brightness,
    "Sharpness": _sharpness,

    "ShearX": _shear_x,
    "ShearY": _shear_y,
    "TranslateX": _translate_x,
    "TranslateY": _translate_y,
}


@dataclass
class RandAugmentConfig:
    num_ops: int = 2
    magnitude: int = 9   # standard RandAugment N,M notation

    ops: list[str] = field(
        default_factory=lambda: list(_OPS.keys())
    )


class RandAugment(Transform):

    def __init__(self, cfg=None):
        self.cfg = cfg or RandAugmentConfig()

    def __call__(self, sample):

        chosen = random.sample(
            self.cfg.ops,
            k=self.cfg.num_ops,
        )

        for op_name in chosen:
            op = _OPS[op_name]
            sample.image = op(
                sample.image,
                self.cfg.magnitude,
            )

        return sample


@register_transform(config=RandAugmentConfig())
def rand_augment(cfg):
    return RandAugment(cfg)
