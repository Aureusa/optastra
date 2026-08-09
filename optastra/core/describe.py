from __future__ import annotations
import yaml
import torch
from .component_ref import _serialize_config
from .experiment import ExperimentConfig


def resolve_experiment(cfg: ExperimentConfig) -> dict:
    return _serialize_config(cfg)


def print_experiment(cfg: ExperimentConfig) -> None:
    print(yaml.safe_dump(resolve_experiment(cfg), sort_keys=False))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
