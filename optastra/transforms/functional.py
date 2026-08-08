"""
Dtype-safe wrappers around torchvision ops that only support uint8
(posterize/equalize/autocontrast) or have threshold semantics sensitive to
the image's actual value range (solarize). Shared by every Transform that
needs these ops, regardless of whether it's RandAugment-driven or standalone.
"""
import torch
import torchvision.transforms.functional as F


def to_uint8(img: torch.Tensor) -> torch.Tensor:
    return (img * 255.0).clamp(0, 255).to(torch.uint8)


def from_uint8(img: torch.Tensor) -> torch.Tensor:
    return img.to(torch.float32) / 255.0


def safe_posterize(img: torch.Tensor, bits: int) -> torch.Tensor:
    bits = max(1, min(8, bits))
    if img.is_floating_point():
        return from_uint8(F.posterize(to_uint8(img), bits))
    return F.posterize(img, bits)


def safe_autocontrast(img: torch.Tensor) -> torch.Tensor:
    if img.is_floating_point():
        return from_uint8(F.autocontrast(to_uint8(img)))
    return F.autocontrast(img)


def safe_equalize(img: torch.Tensor) -> torch.Tensor:
    if img.is_floating_point():
        return from_uint8(F.equalize(to_uint8(img)))
    return F.equalize(img)


def safe_solarize(img: torch.Tensor, threshold: float) -> torch.Tensor:
    """threshold given in [0,1] terms; clamped against the image's actual
    max so torchvision's threshold < max(img) requirement never raises,
    and correctly rescaled for non-float images."""
    if img.is_floating_point():
        threshold = min(threshold, float(img.max()) - 1e-6)
        threshold = max(threshold, 0.0)
    else:
        threshold = min(threshold * 255, 255)
    return F.solarize(img, threshold)
