from __future__ import annotations
from dataclasses import dataclass
import torch
from .base import BatchTransform
from ._registry import register_batch_transform


__all__ = ["MixUp"]


@dataclass
class MixUpConfig:
    alpha: float = 0.2
    p: float = 0.5


class MixUp(BatchTransform):
    def __init__(self, cfg: MixUpConfig = MixUpConfig()):
        self.cfg = cfg

    def __call__(self, batch):
        if torch.rand(()) >= self.cfg.p:
            return batch
        images, labels = batch["inputs"], batch["targets"]["labels"]
        lam = torch.distributions.Beta(self.cfg.alpha, self.cfg.alpha).sample()
        perm = torch.randperm(images.size(0), device=images.device)
        batch["inputs"] = lam * images + (1 - lam) * images[perm]
        # soft-label mixing: classification loss needs to accept this shape --
        # store both label sets + lambda, let the Task's loss handle it
        batch["targets"] = {"labels": labels, "labels_b": labels[perm], "lam": lam}
        return batch


@register_batch_transform(config=MixUpConfig())
def mixup(cfg): return MixUp(cfg)