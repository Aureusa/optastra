from __future__ import annotations
from dataclasses import dataclass
from . import rng
from .base import BatchTransform


__all__ = ["MixUp"]


@dataclass
class MixUpConfig:
    alpha: float = 0.2
    p: float = 0.5
    inputs_key: str = "inputs"     # batch[inputs_key]: (B, C, H, W) images
    targets_key: str = "targets"   # batch[targets_key]: dict of per-batch targets
    labels_key: str = "labels"     # batch[targets_key][labels_key]: (B,) class labels


class MixUp(BatchTransform):
    """Blend each image with a random partner from the batch:
    x = lam * x + (1 - lam) * x[perm], lam ~ Beta(alpha, alpha).

    The targets dict is kept (other entries preserved) and gains
    `<labels_key>_b` (the partners' labels) and `lam`; the classification loss
    interpolates the two label losses by `lam`."""

    def __init__(self, cfg: MixUpConfig = MixUpConfig()):
        self.cfg = cfg

    def __call__(self, batch):
        if rng.rand() >= self.cfg.p:
            return batch
        images, targets = batch[self.cfg.inputs_key], batch[self.cfg.targets_key]
        if not isinstance(targets, dict):
            raise TypeError(f"MixUp needs batch['{self.cfg.targets_key}'] to be a dict, got {type(targets).__name__}.")
        labels = targets[self.cfg.labels_key]
        lam = rng.beta(self.cfg.alpha, self.cfg.alpha)
        perm = rng.randperm(images.size(0))
        batch[self.cfg.inputs_key] = lam * images + (1 - lam) * images[perm.to(images.device)]
        batch[self.cfg.targets_key] = {
            **targets,
            f"{self.cfg.labels_key}_b": labels[perm.to(labels.device)],
            "lam": lam,
        }
        return batch


@BatchTransform.register(config=MixUpConfig())
def mixup(cfg): return MixUp(cfg)


@BatchTransform.register(config=MixUpConfig(alpha=0.05, p=0.25))
def mixup_weak(cfg): return MixUp(cfg)


@BatchTransform.register(config=MixUpConfig(alpha=1.0, p=0.8))
def mixup_strong(cfg): return MixUp(cfg)
