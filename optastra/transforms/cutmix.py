"""
CutMix (Yun et al., 2019, arXiv:1905.04899). Randomly cuts a patch from one
image and pastes it onto another image, and mixes the labels accordingly.
This implementation is designed to work with batches of images and labels.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from .base import BatchTransform
from ._registry import register_batch_transform


__all__ = ["CutMix"]


@dataclass
class CutMixConfig:
    alpha: float = 1.0
    p: float = 0.5


def _rand_bbox(h: int, w: int, lam: float) -> tuple[int, int, int, int]:
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
    cy, cx = torch.randint(h, (1,)).item(), torch.randint(w, (1,)).item()
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    return y1, y2, x1, x2


class CutMix(BatchTransform):
    def __init__(self, cfg: CutMixConfig = CutMixConfig()):
        self.cfg = cfg

    def __call__(self, batch):
        if torch.rand(()) >= self.cfg.p:
            return batch

        images, labels = batch["inputs"], batch["targets"]["labels"]
        B, _, H, W = images.shape
        perm = torch.randperm(B, device=images.device)

        lam = torch.distributions.Beta(self.cfg.alpha, self.cfg.alpha).sample().item()
        y1, y2, x1, x2 = _rand_bbox(H, W, lam)
        images[:, :, y1:y2, x1:x2] = images[perm][:, :, y1:y2, x1:x2]

        # recompute lam from actual patch area, since rounding can shift it
        lam_actual = 1.0 - ((y2 - y1) * (x2 - x1) / (H * W))
        batch["inputs"] = images
        batch["targets"] = {"labels": labels, "labels_b": labels[perm], "lam": lam_actual}
        return batch


@register_batch_transform(config=CutMixConfig())
def cutmix(cfg): return CutMix(cfg)


@register_batch_transform(config=CutMixConfig())
def cutmix_weak(cfg):
    cfg.p = 0.25
    cfg.alpha = 0.5
    return CutMix(cfg)
 
 
@register_batch_transform(config=CutMixConfig())
def cutmix_strong(cfg):
    cfg.p = 0.8
    cfg.alpha = 2.0  # concentrates lam near 0.5 -> a consistently sized patch, not occasional huge swaps
    return CutMix(cfg)
