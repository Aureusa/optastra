from __future__ import annotations
from dataclasses import dataclass, asdict
import yaml

from .component_ref import ComponentRef, component_field
from ..architectures.base import Architecture
from ..tasks.base import Task
from ..optim.base import Optimizer
from ..optim.scheduler_base import Scheduler


@dataclass
class ExperimentConfig:
    architecture: ComponentRef = component_field(Architecture)
    task: ComponentRef = component_field(Task)
    optimizer: ComponentRef = component_field(Optimizer, default_name="adamw")
    scheduler: ComponentRef | None = component_field(Scheduler, optional=True)

    seed: int = 0
    max_iter: int = 10_000
    batch_size: int = 32
    output_dir: str = "runs/exp"

    def to_yaml(self, path: str | None = None) -> str:
        payload = {
            "architecture": asdict(self.architecture),
            "task": asdict(self.task),
            "optimizer": asdict(self.optimizer),
            "scheduler": asdict(self.scheduler) if self.scheduler else None,
            "seed": self.seed, "max_iter": self.max_iter,
            "batch_size": self.batch_size, "output_dir": self.output_dir,
        }
        text = yaml.safe_dump(payload, sort_keys=False)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        mk = lambda d: ComponentRef(**d) if d else None
        return cls(
            architecture=mk(raw["architecture"]), task=mk(raw["task"]),
            optimizer=mk(raw["optimizer"]), scheduler=mk(raw.get("scheduler")),
            seed=raw["seed"], max_iter=raw["max_iter"],
            batch_size=raw["batch_size"], output_dir=raw["output_dir"],
        )
    