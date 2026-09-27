"""
Dtype-, channel- and range-agnostic photometric ops.

torchvision's functional ops assume uint8 or float RGB in [0, 1]: they clamp
float results to [0, 1], only accept 1 or 3 channels, and posterize/equalize
only accept uint8. That is wrong for e.g. 5-band float32 astronomy images
with values in [-10, 1000]. The functions here work on float tensors of
shape (..., C, H, W) with any C and any value range.

Value range
-----------
Ops that need to know what "black" and "white" are (solarize, posterize,
equalize, autocontrast, and the final clamp of every blend) take
`value_range=(lo, hi)`. When it is None, `infer_value_range` decides:

* images whose values all lie in [0, 1] -> (0, 1), the standard float convention;
* anything else -> (img.min(), img.max()) of that image.

Pass an explicit `value_range` when you know your data's physical range --
per-image inference means an image's brightest pixel is its "white".
Augmentation policies infer the range once per sample and pass it to every
op in the chain, so op N+1 isn't fooled by what op N did.

Channels
--------
"Grayscale" (luminance) is the ITU-R 601 weighted sum for C == 3 -- same as
torchvision -- and the plain mean over channels for any other C. Contrast,
saturation and grayscale are defined in terms of it, so they work for any
C. Hue is inherently RGB and raises for C != 3.
"""
from __future__ import annotations

import torch
import torchvision.transforms.functional as F


__all__ = [
    "to_float_image", "infer_value_range", "luminance",
    "adjust_brightness", "adjust_contrast", "adjust_saturation", "adjust_sharpness", "adjust_hue",
    "grayscale", "solarize", "posterize", "autocontrast", "equalize",
]

ValueRange = tuple[float, float]

# Unsigned integer images span their full dtype range; divide by the max to land in [0, 1].
UNSIGNED_FULL_SCALE = {torch.uint8: 255.0, torch.uint16: 65535.0, torch.uint32: 4294967295.0}


def to_float_image(img: torch.Tensor, signed_scale: float = 1.0) -> torch.Tensor:
    """float32 copy of an integer image; floating-point images pass through.

    uint8/uint16/uint32 are divided by their dtype max (-> [0, 1]). Signed
    integers (int16/int32/int64) have no natural full scale -- they usually
    hold raw or background-subtracted counts -- so they are divided by
    `signed_scale` (default 1: keep the raw values).
    """
    if img.is_floating_point():
        return img
    divisor = UNSIGNED_FULL_SCALE.get(img.dtype, signed_scale)
    return img.to(torch.float32) / divisor


def infer_value_range(img: torch.Tensor, value_range: ValueRange | None = None) -> ValueRange:
    """`value_range` if given, else (0, 1) for images inside [0, 1], else the image's (min, max)."""
    if value_range is not None:
        return float(value_range[0]), float(value_range[1])
    lo, hi = float(img.min()), float(img.max())
    if lo >= 0.0 and hi <= 1.0:
        return 0.0, 1.0
    return lo, hi


def luminance(img: torch.Tensor) -> torch.Tensor:
    """(..., 1, H, W) grayscale: ITU-R 601 weights for RGB, channel mean otherwise."""
    if img.shape[-3] == 3:
        r, g, b = img.unbind(dim=-3)
        return (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(-3)
    return img.mean(dim=-3, keepdim=True)


def _blend(img1: torch.Tensor, img2: torch.Tensor, ratio: float, value_range: ValueRange | None) -> torch.Tensor:
    lo, hi = infer_value_range(img1, value_range)
    return (ratio * img1 + (1.0 - ratio) * img2).clamp(lo, hi)


def adjust_brightness(img: torch.Tensor, factor: float, value_range: ValueRange | None = None) -> torch.Tensor:
    """Scale all values by `factor` (blend towards 0), clamped to the value range."""
    return _blend(img, torch.zeros_like(img), factor, value_range)


def adjust_contrast(img: torch.Tensor, factor: float, value_range: ValueRange | None = None) -> torch.Tensor:
    """Blend towards the image's mean luminance. Any channel count."""
    mean = luminance(img).mean(dim=(-3, -2, -1), keepdim=True)
    return _blend(img, mean, factor, value_range)


def adjust_saturation(img: torch.Tensor, factor: float, value_range: ValueRange | None = None) -> torch.Tensor:
    """Blend each pixel towards its own luminance. For C != 3 this pushes every
    band towards the per-pixel band mean ("spectral saturation"); for C == 1
    it is a no-op."""
    return _blend(img, luminance(img), factor, value_range)


def adjust_sharpness(img: torch.Tensor, factor: float, value_range: ValueRange | None = None) -> torch.Tensor:
    """Blend with a 3x3-smoothed copy (PIL/torchvision kernel; border pixels kept)."""
    if img.shape[-1] <= 2 or img.shape[-2] <= 2:
        return img
    kernel = torch.ones(3, 3, dtype=img.dtype, device=img.device)
    kernel[1, 1] = 5.0
    kernel /= kernel.sum()
    c = img.shape[-3]
    flat = img.reshape(-1, c, *img.shape[-2:])
    smoothed = torch.nn.functional.conv2d(flat, kernel.expand(c, 1, 3, 3), groups=c)
    degenerate = flat.clone()
    degenerate[..., 1:-1, 1:-1] = smoothed
    return _blend(img, degenerate.reshape(img.shape), factor, value_range)


def adjust_hue(img: torch.Tensor, hue_factor: float, value_range: ValueRange | None = None) -> torch.Tensor:
    """Hue rotation. Inherently RGB: raises for C != 3. The image is mapped to
    [0, 1] with the value range, rotated by torchvision, and mapped back."""
    if img.shape[-3] != 3:
        raise ValueError(f"adjust_hue needs a 3-channel RGB image, got {img.shape[-3]} channels.")
    lo, hi = infer_value_range(img, value_range)
    scale = max(hi - lo, 1e-12)
    unit = ((img - lo) / scale).clamp(0.0, 1.0)
    return F.adjust_hue(unit, hue_factor) * scale + lo


def grayscale(img: torch.Tensor) -> torch.Tensor:
    """Replace every channel by the luminance -- keeps the channel count
    (for C == 3 identical to torchvision's rgb_to_grayscale(..., 3))."""
    return luminance(img).expand_as(img).clone()


def solarize(img: torch.Tensor, threshold: float, value_range: ValueRange | None = None) -> torch.Tensor:
    """Invert (lo + hi - x) every value >= `threshold`, in the image's own units.

    No clamping of `threshold`: a threshold above the image max leaves the
    image unchanged, as it should.
    """
    lo, hi = infer_value_range(img, value_range)
    return torch.where(img >= threshold, lo + hi - img, img)


def _to_levels(img: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Quantize into 256 equal-width bins over [lo, hi] -> levels 0..255 (as float).
    Same binning as torchvision's float->uint8 conversion, so for [0, 1]
    images the value k/255 lands exactly on level k."""
    unit = ((img - lo) / max(hi - lo, 1e-12)).clamp(0.0, 1.0)
    return torch.floor(unit * (256.0 - 1e-3))


def _from_levels(levels: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return levels / 255.0 * (hi - lo) + lo


def posterize(img: torch.Tensor, bits: int, value_range: ValueRange | None = None) -> torch.Tensor:
    """Keep only the top `bits` (1..8) bits of each value on a 256-level grid
    spanning the value range -- identical to torchvision on uint8/[0, 1] images."""
    bits = max(1, min(8, int(bits)))
    lo, hi = infer_value_range(img, value_range)
    step = 2 ** (8 - bits)
    levels = torch.floor(_to_levels(img, lo, hi) / step) * step
    return _from_levels(levels, lo, hi)


def autocontrast(img: torch.Tensor, value_range: ValueRange | None = None) -> torch.Tensor:
    """Stretch each channel linearly so its min -> lo and its max -> hi.
    Constant channels are left unchanged."""
    lo, hi = infer_value_range(img, value_range)
    cmin = img.amin(dim=(-2, -1), keepdim=True)
    cmax = img.amax(dim=(-2, -1), keepdim=True)
    constant = cmax <= cmin
    scale = (hi - lo) / torch.where(constant, torch.ones_like(cmax), cmax - cmin)
    stretched = ((img - cmin) * scale + lo).clamp(lo, hi)
    return torch.where(constant, img, stretched)


def equalize(img: torch.Tensor, value_range: ValueRange | None = None) -> torch.Tensor:
    """Per-channel histogram equalization over 256 bins spanning the value
    range -- same lookup-table construction as torchvision's uint8 equalize,
    so results are quantized to 256 levels."""
    lo, hi = infer_value_range(img, value_range)
    levels = _to_levels(img, lo, hi).long()
    flat = levels.reshape(-1, levels.shape[-2] * levels.shape[-1])   # one row per channel
    out = torch.empty_like(flat)
    for i, channel in enumerate(flat):
        hist = torch.bincount(channel, minlength=256)
        nonzero = hist[hist != 0]
        step = torch.div(nonzero[:-1].sum(), 255, rounding_mode="floor")
        if step == 0:
            out[i] = channel
            continue
        lut = torch.div(torch.cumsum(hist, 0) + torch.div(step, 2, rounding_mode="floor"), step, rounding_mode="floor")
        lut = torch.nn.functional.pad(lut, [1, 0])[:-1].clamp(0, 255)
        out[i] = lut[channel]
    return _from_levels(out.reshape(img.shape).to(img.dtype), lo, hi)
