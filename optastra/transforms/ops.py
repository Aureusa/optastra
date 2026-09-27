"""Shared photometric+geometric op set, used by RandAugment, AutoAugment,
TrivialAugment, AugMix and PixMix. Single source of truth for op
implementations so magnitude semantics stay consistent across policies.

Magnitudes follow the RandAugment convention: 0 (weakest) .. 10 (strongest).

* Photometric ops: `op(img, magnitude, value_range) -> img`. `img` is float
  with any channel count; `value_range` is the (lo, hi) already resolved by
  the calling policy (see `functional.infer_value_range`).
* Geometric ops: `op(sample, magnitude) -> sample`. They move boxes and
  masks together with the image (see `geometric.affine_sample`), so they
  are safe for detection/segmentation too.

Use `apply_op` to run either kind by name.
"""
import math

from . import functional as FN
from . import rng
from .geometric import affine_sample
from ..data.sample import Sample


__all__ = ["PHOTOMETRIC_OPS", "GEOMETRIC_OPS", "ALL_OPS", "apply_op", "check_op_names"]


def _factor(m: float) -> float:
    """1 +/- up to 0.9, the enhance-factor used by Color/Contrast/Brightness/Sharpness."""
    return 1 + rng.sign() * (m / 10.0) * 0.9


# ---- photometric: (img, magnitude, value_range) -> img ----------------------

def identity(img, m, vr): return img
def autocontrast(img, m, vr): return FN.autocontrast(img, vr)
def equalize(img, m, vr): return FN.equalize(img, vr)
def posterize(img, m, vr): return FN.posterize(img, int(round(8 - (m / 10.0) * 4)), vr)
def solarize(img, m, vr): return FN.solarize(img, vr[1] - (m / 10.0) * (vr[1] - vr[0]), vr)
def color(img, m, vr): return FN.adjust_saturation(img, _factor(m), vr)
def contrast(img, m, vr): return FN.adjust_contrast(img, _factor(m), vr)
def brightness(img, m, vr): return FN.adjust_brightness(img, _factor(m), vr)
def sharpness(img, m, vr): return FN.adjust_sharpness(img, _factor(m), vr)


# ---- geometric: (sample, magnitude) -> sample -------------------------------

def rotate(sample, m):
    return affine_sample(sample, angle=rng.sign() * (m / 10.0) * 30)

def shear_x(sample, m):
    deg = math.degrees(math.atan(0.3 * (m / 10.0))) * rng.sign()
    return affine_sample(sample, shear=(deg, 0.0))

def shear_y(sample, m):
    deg = math.degrees(math.atan(0.3 * (m / 10.0))) * rng.sign()
    return affine_sample(sample, shear=(0.0, deg))

def translate_x(sample, m):
    shift = int(rng.sign() * (m / 10.0) * sample.image.shape[-1] * 0.3)
    return affine_sample(sample, translate=(shift, 0))

def translate_y(sample, m):
    shift = int(rng.sign() * (m / 10.0) * sample.image.shape[-2] * 0.3)
    return affine_sample(sample, translate=(0, shift))


# Ordered dicts: the key order is the canonical op order (and the default op
# tuples below), so op selection never depends on set/hash ordering.
PHOTOMETRIC_OPS = {
    "Identity": identity, "AutoContrast": autocontrast, "Equalize": equalize,
    "Posterize": posterize, "Solarize": solarize, "Color": color,
    "Contrast": contrast, "Brightness": brightness, "Sharpness": sharpness,
}
GEOMETRIC_OPS = {
    "Rotate": rotate, "ShearX": shear_x, "ShearY": shear_y,
    "TranslateX": translate_x, "TranslateY": translate_y,
}
ALL_OPS = {**PHOTOMETRIC_OPS, **GEOMETRIC_OPS}

PHOTOMETRIC_OP_NAMES = tuple(PHOTOMETRIC_OPS)
ALL_OP_NAMES = tuple(ALL_OPS)


def check_op_names(names, allowed: dict = ALL_OPS) -> None:
    """Fail at construction time, not mid-training, on a typo'd op name."""
    unknown = [n for n in names if n not in allowed]
    if unknown:
        raise ValueError(f"Unknown op(s) {unknown}; choose from {list(allowed)}.")


def apply_op(sample: Sample, name: str, magnitude: float, value_range: tuple[float, float]) -> Sample:
    """Run op `name` on the sample: geometric ops move image + targets,
    photometric ops only touch the (float) image."""
    if name in GEOMETRIC_OPS:
        return GEOMETRIC_OPS[name](sample, magnitude)
    sample.image = PHOTOMETRIC_OPS[name](sample.image, magnitude, value_range)
    return sample
