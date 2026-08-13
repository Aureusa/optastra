from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace, fields
from typing import Any, Mapping, Literal
import torch

from ._registry import _registry
from ..nn.features import HeadOutput
from ..core.factory import Factory


__all__ = ["Task", "TaskStepOutput", "Stage"]


Stage = Literal["train", "val", "test", "predict"]


@dataclass
class TaskStepOutput:
    loss: torch.Tensor | None
    losses: dict[str, torch.Tensor] = field(default_factory=dict)
    metrics: dict[str, torch.Tensor | float] = field(default_factory=dict)
    predictions: Any = None         # decoded, user-facing
    raw_predictions: Any = None     # model output, optional for debugging
    targets: Any = None             # preprocessed targets, optional


class Task(ABC, Factory["Task"]):
    required_fields: tuple[str, ...] = ()
    collate: str = "default_collate"
    _registry = _registry
    
    def run_step(self, model, batch: Mapping[str, Any], stage: Stage = "train") -> TaskStepOutput:
        self.validate_batch(batch, stage)
        inputs, raw_targets = self.split_inputs_targets(batch, stage)
        targets = self.preprocess_targets(raw_targets) if raw_targets is not None else None

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            raw_preds = self.forward_model(model, inputs)
        self.validate_predictions(raw_preds)

        losses, total_loss = {}, None
        if stage in ("train", "val") and targets is not None:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                losses = self.compute_losses(raw_preds, targets)
            total_loss = self.reduce_losses(losses)

        metrics = {}
        if stage in ("val", "test") and targets is not None:
            metrics = self.compute_metrics(raw_preds, targets)

        decoded = None
        if stage in ("val", "test", "predict"):
            decoded = self.decode_predictions(raw_preds)

        return TaskStepOutput(loss=total_loss, losses=losses, metrics=metrics,
                               predictions=decoded, raw_predictions=raw_preds, targets=targets)

    def validate_predictions(self, raw_preds: Any) -> None:
        if not isinstance(raw_preds, HeadOutput):
            raise TypeError(f"Model output must be a HeadOutput, got {type(raw_preds)}.")
        missing = [f for f in self.required_fields if getattr(raw_preds, f, None) is None]
        if missing:
            raise ValueError(f"{type(self).__name__} requires {missing}, got {raw_preds}.")

    @abstractmethod
    def validate_batch(self, batch: Mapping[str, Any], stage: Stage = "train"):
        """Validate the batch structure and contents for the given stage."""
        raise NotImplementedError

    @abstractmethod
    def split_inputs_targets(self, batch: Mapping[str, Any], stage: Stage = "train") -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Split the batch into inputs and raw targets."""
        raise NotImplementedError

    @abstractmethod
    def preprocess_targets(self, raw_targets: Mapping[str, Any]) -> Mapping[str, Any]:
        """Preprocess raw targets into a format suitable for loss computation."""
        raise NotImplementedError

    @abstractmethod
    def forward_model(self, model, inputs: Mapping[str, Any]) -> Any:
        """Forward pass through the model."""
        raise NotImplementedError

    @abstractmethod
    def compute_losses(self, raw_preds: Any, targets: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Compute losses based on raw predictions and targets."""
        raise NotImplementedError

    @abstractmethod
    def reduce_losses(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Reduce multiple loss components into a single scalar loss."""
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self, raw_preds: Any, targets: Mapping[str, Any]) -> dict[str, torch.Tensor | float]:
        """Compute metrics based on raw predictions and targets."""
        raise NotImplementedError

    @abstractmethod
    def decode_predictions(self, raw_preds: Any) -> Any:
        """Decode raw predictions into a user-facing format."""
        raise NotImplementedError


# TODO: Implement a MultiTask class that combines several tasks
# sharing one model but reading different HeadOutput keys.
# This is escpecially useful for multi-task learning scenarios where a
# single model outputs multiple types of predictions (e.g., object detection and segmentation)
# and each task has its own loss and metrics.
# In Masked R-CNN for example we have a detection task and a segmentation task
# that share the same backbone and neck but have different heads and loss functions. In such cases,
# a MultiTask class can orchestrate the training and evaluation of these tasks together,
# ensuring that the model learns to perform well on all tasks simultaneously (each head has its own task).
# class MultiTask(Task):
#     """Combines several tasks that share one model but read different HeadOutput keys."""
#     def __init__(self, tasks: dict[str, Task]):
#         self.tasks = tasks  # e.g. {"boxes": DetectionTask(...), "masks": MaskTask(...)}

#     def run_step(self, model, batch, stage="train"):
#         raw_preds = model(batch["inputs"])          # dict[str, HeadOutput]
#         outputs = {k: t.run_step_from_preds(raw_preds[k], batch, stage) for k, t in self.tasks.items()}
#         total_loss = sum(o.loss for o in outputs.values() if o.loss is not None)
#         return TaskStepOutput(loss=total_loss, losses={...}, metrics={...}, predictions=outputs)
