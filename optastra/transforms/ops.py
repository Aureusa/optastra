"""Shared photometric+geometric op set, used by RandAugment, AutoAugment,
TrivialAugment, and AugMix. Single source of truth for op implementations
so magnitude semantics stay consistent across policies."""
import math
import random
import torch
import torchvision.transforms.functional as F

from .functional import safe_posterize, safe_autocontrast, safe_equalize, safe_solarize


def identity(img, m): return img
def autocontrast(img, m): return safe_autocontrast(img)
def equalize(img, m): return safe_equalize(img)
def rotate(img, m): return F.rotate(img, random.choice([-1, 1]) * (m / 10.0) * 30)
def posterize(img, m): return safe_posterize(img, int(round(8 - (m / 10.0) * 4)))
def solarize(img, m): return safe_solarize(img, 1.0 - (m / 10.0))
def color(img, m): return F.adjust_saturation(img, 1 + random.choice([-1, 1]) * (m / 10.0) * 0.9)
def contrast(img, m): return F.adjust_contrast(img, 1 + random.choice([-1, 1]) * (m / 10.0) * 0.9)
def brightness(img, m): return F.adjust_brightness(img, 1 + random.choice([-1, 1]) * (m / 10.0) * 0.9)
def sharpness(img, m): return F.adjust_sharpness(img, 1 + random.choice([-1, 1]) * (m / 10.0) * 0.9)

def shear_x(img, m):
    deg = math.degrees(math.atan(0.3 * (m / 10.0))) * random.choice([-1, 1])
    return F.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[deg, 0])

def shear_y(img, m):
    deg = math.degrees(math.atan(0.3 * (m / 10.0))) * random.choice([-1, 1])
    return F.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[0, deg])

def translate_x(img, m):
    shift = int(random.choice([-1, 1]) * (m / 10.0) * img.shape[-1] * 0.3)
    return F.affine(img, angle=0, translate=[shift, 0], scale=1.0, shear=[0, 0])

def translate_y(img, m):
    shift = int(random.choice([-1, 1]) * (m / 10.0) * img.shape[-2] * 0.3)
    return F.affine(img, angle=0, translate=[0, shift], scale=1.0, shear=[0, 0])


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