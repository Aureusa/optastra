from __future__ import annotations
from dataclasses import dataclass
import yaml

from .component_ref import (
    ComponentRef,
    ComponentRefConfigMixin,
    component_field,
    config_from_plain,
    config_to_plain,
)
from ..architectures.base import Architecture
from ..tasks.base import Task
from ..optim.base import Optimizer
from ..optim.scheduler_base import Scheduler


@dataclass
class ExperimentConfig(ComponentRefConfigMixin):
    architecture: ComponentRef = component_field(Architecture)
    task: ComponentRef = component_field(Task)
    optimizer: ComponentRef = component_field(Optimizer, default_name="adamw")
    scheduler: ComponentRef | None = component_field(Scheduler, optional=True)

    seed: int = 0
    max_iter: int = 10_000
    batch_size: int = 32
    output_dir: str = "runs/exp"

    def to_dict(self) -> dict:
        """Plain, YAML-safe dict. ComponentRefs (including ones nested inside
        overrides) are written as {"name": ..., "overrides": {...}} so
        from_dict() rebuilds exactly the same config."""
        return config_to_plain(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "ExperimentConfig":
        return cls(**config_from_plain(raw))

    def to_yaml(self, path: str | None = None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)
