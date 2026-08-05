from __future__ import annotations

from dataclasses import dataclass

import torch

from .._registry import register_sampler


@dataclass
class BalancedSamplerConfig:
    batch_size: int = 512
    positive_fraction: float = 0.25


class BalancedSampler:
    """Balanced positive/negative sampler used by RPN and RCNN heads."""

    def __init__(self, cfg: BalancedSamplerConfig):
        self.cfg = cfg

    def sample(self, labels: torch.Tensor, *, positive_value: int = 1, negative_value: int = 0) -> torch.Tensor:
        pos_idx = torch.where(labels == positive_value)[0]
        neg_idx = torch.where(labels == negative_value)[0]

        num_pos = min(int(self.cfg.batch_size * self.cfg.positive_fraction), int(pos_idx.numel()))
        num_neg = min(self.cfg.batch_size - num_pos, int(neg_idx.numel()))

        if num_pos > 0:
            pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=labels.device)[:num_pos]]
        else:
            pos_idx = labels.new_zeros((0,), dtype=torch.long)

        if num_neg > 0:
            neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=labels.device)[:num_neg]]
        else:
            neg_idx = labels.new_zeros((0,), dtype=torch.long)

        return torch.cat((pos_idx, neg_idx), dim=0)


sampler_configs = {
    "rcnn_balanced_sampler": BalancedSamplerConfig(batch_size=512, positive_fraction=0.25),
    "rpn_balanced_sampler": BalancedSamplerConfig(batch_size=256, positive_fraction=0.5),
}


@register_sampler(config=sampler_configs["rcnn_balanced_sampler"])
def rcnn_balanced_sampler(cfg: BalancedSamplerConfig) -> BalancedSampler:
    return BalancedSampler(cfg)


@register_sampler(config=sampler_configs["rpn_balanced_sampler"])
def rpn_balanced_sampler(cfg: BalancedSamplerConfig) -> BalancedSampler:
    return BalancedSampler(cfg)
