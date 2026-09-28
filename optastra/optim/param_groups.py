from __future__ import annotations
from dataclasses import dataclass, field
import torch.nn as nn


__all__ = ["ParamGroupConfig", "build_param_groups", "DEFAULT_NO_DECAY_NAMES"]

NORM_MODULES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.LayerNorm, nn.GroupNorm)

# Parameters that are conventionally never weight-decayed, matched against
# any dotted component of the parameter name (so both `cls_token` and
# `backbone.pos_embed.pos_embed` match). Decaying a positional embedding or
# a class token pulls it towards zero, which only hurts.
DEFAULT_NO_DECAY_NAMES = ("pos_embed", "cls_token", "dist_token", "mask_token", "relative_position_bias_table")


@dataclass
class ParamGroupConfig:
    no_decay_norm_and_bias: bool = True
    # Parameter names (dotted components) excluded from weight decay,
    # independently of `no_decay_norm_and_bias`. Pass () to disable.
    no_decay_names: tuple[str, ...] = DEFAULT_NO_DECAY_NAMES
    # module-path prefix -> LR multiplier, e.g. {"backbone": 0.1}. Longest
    # matching prefix wins, so {"backbone": 0.1, "backbone.stem": 0.01} is
    # unambiguous rather than depending on dict iteration order.
    lr_multipliers: dict[str, float] = field(default_factory=dict)


def _multiplier_for(name: str, lr_multipliers: dict[str, float]) -> float:
    matches = [prefix for prefix in lr_multipliers if name.startswith(prefix)]
    if not matches:
        return 1.0
    longest = max(matches, key=len)
    return lr_multipliers[longest]


def _is_no_decay(name: str, param: nn.Parameter, cfg: ParamGroupConfig, norm_param_ids: set[int]) -> bool:
    if set(name.split(".")) & set(cfg.no_decay_names):
        return True
    if cfg.no_decay_norm_and_bias:
        return name.endswith(".bias") or name == "bias" or id(param) in norm_param_ids
    return False


def build_param_groups(
    model: nn.Module,
    cfg: ParamGroupConfig,
    base_lr: float,
    base_weight_decay: float = 0.0,
) -> list[dict]:
    """Splits params into groups by (LR multiplier x decay eligibility), so
    'no decay on norm/bias' and 'backbone at 0.1x LR' compose automatically.

    base_weight_decay is threaded in explicitly (rather than omitted and left
    to the optimizer's default) so every group's weight_decay is always a
    real float -- PyTorch param groups don't support a "use optimizer
    default" sentinel; an explicit key always wins over the optimizer-level
    default, so omitting it silently reintroduces decay on the no-decay
    bucket if the optimizer's own default is nonzero.
    """
    norm_param_ids = {
        id(p) for m in model.modules() if isinstance(m, NORM_MODULES) for p in m.parameters()
    }

    buckets: dict[tuple[float, bool], list[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = _is_no_decay(name, param, cfg, norm_param_ids)
        mult = _multiplier_for(name, cfg.lr_multipliers)
        buckets.setdefault((mult, no_decay), []).append(param)

    groups = []
    for (mult, no_decay), params in buckets.items():
        groups.append({
            "params": params,
            "lr": base_lr * mult,
            "weight_decay": 0.0 if no_decay else base_weight_decay,
        })
    return groups
