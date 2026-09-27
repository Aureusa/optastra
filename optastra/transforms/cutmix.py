"""
CutMix (Yun et al., 2019, arXiv:1905.04899). Randomly cuts a patch from one
image and pastes it onto another image, and mixes the labels accordingly.
This implementation is designed to work with batches of images and labels.
"""
from __future__ import annotations
from dataclasses import dataclass

from . import rng
from .base import BatchTransform


__all__ = ["CutMix"]


@dataclass
class CutMixConfig:
    alpha: float = 1.0
    p: float = 0.5
    inputs_key: str = "inputs"     # batch[inputs_key]: (B, C, H, W) images
    targets_key: str = "targets"   # batch[targets_key]: dict of per-batch targets
    labels_key: str = "labels"     # batch[targets_key][labels_key]: (B,) class labels


def _rand_bbox(h: int, w: int, lam: float) -> tuple[int, int, int, int]:
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
    cy, cx = rng.randint(0, h - 1), rng.randint(0, w - 1)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    return y1, y2, x1, x2


class CutMix(BatchTransform):
    """Paste a random patch of each image's partner (x[perm]) into it.

    The targets dict is kept (other entries preserved) and gains
    `<labels_key>_b` (the partners' labels) and `lam` (fraction of the image
    that is still the original, recomputed from the actual patch area)."""

    def __init__(self, cfg: CutMixConfig = CutMixConfig()):
        self.cfg = cfg

    def __call__(self, batch):
        if rng.rand() >= self.cfg.p:
            return batch

        images, targets = batch[self.cfg.inputs_key], batch[self.cfg.targets_key]
        if not isinstance(targets, dict):
            raise TypeError(f"CutMix needs batch['{self.cfg.targets_key}'] to be a dict, got {type(targets).__name__}.")
        labels = targets[self.cfg.labels_key]
        B, _, H, W = images.shape
        perm = rng.randperm(B)

        lam = rng.beta(self.cfg.alpha, self.cfg.alpha)
        y1, y2, x1, x2 = _rand_bbox(H, W, lam)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[perm.to(images.device)][:, :, y1:y2, x1:x2]

        # recompute lam from actual patch area, since rounding can shift it
        lam_actual = 1.0 - ((y2 - y1) * (x2 - x1) / (H * W))
        batch[self.cfg.inputs_key] = images
        batch[self.cfg.targets_key] = {
            **targets,
            f"{self.cfg.labels_key}_b": labels[perm.to(labels.device)],
            "lam": lam_actual,
        }
        return batch


@BatchTransform.register(config=CutMixConfig())
def cutmix(cfg): return CutMix(cfg)


@BatchTransform.register(config=CutMixConfig(alpha=0.5, p=0.25))
def cutmix_weak(cfg): return CutMix(cfg)


# alpha=2 concentrates lam near 0.5 -> a consistently sized patch, not occasional huge swaps
@BatchTransform.register(config=CutMixConfig(alpha=2.0, p=0.8))
def cutmix_strong(cfg): return CutMix(cfg)
