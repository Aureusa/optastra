"""Loss functions shared by several criteria (plain functions, not a registry family)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


__all__ = ["sigmoid_focal_loss"]


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Element-wise focal loss (Lin et al., ICCV 2017), no reduction.

    FL = alpha_t * (1 - p_t)^gamma * BCE(logits, targets); ``alpha < 0``
    disables the class balancing. Same values as ``torchvision.ops.sigmoid_focal_loss``.
    """
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t) ** gamma
    if alpha >= 0:
        loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))
    return loss
