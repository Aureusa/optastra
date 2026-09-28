from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Literal, Protocol
import torch

from ..nn.features import HeadOutput
from ..core.distributed import sum_across_processes
from ..core.factory import Factory
from ..core.registry import FamilyRegistry


__all__ = ["Task", "TaskStepOutput", "Stage", "Evaluator", "MeanMetricEvaluator", "infer_batch_size"]


Stage = Literal["train", "val", "test", "predict"]


@dataclass
class TaskStepOutput:
    loss: torch.Tensor | None
    losses: dict[str, torch.Tensor] = field(default_factory=dict)
    metrics: dict[str, torch.Tensor | float] = field(default_factory=dict)
    predictions: Any = None         # decoded, user-facing
    raw_predictions: Any = None     # model output, optional for debugging
    targets: Any = None             # preprocessed targets, optional


class Evaluator(Protocol):
    """Accumulates per-batch TaskStepOutputs over a whole eval pass and
    reduces them to dataset-level scalars. Built by `Task.build_evaluator()`
    and driven by `Trainer.evaluate()`:

        evaluator.reset()
        for batch in loader:
            evaluator.process(task.run_step(model, batch, "val"), batch)
        results = evaluator.summarize()   # e.g. {"accuracy": 0.91, "total_loss": 0.32}

    Keeping this separate from `compute_metrics` matters for any metric that
    isn't a per-batch mean (accuracy over unequal batches, mAP, ...).

    Multi-process (DDP) evaluation: each process evaluates its own shard, then
    `Trainer.evaluate()` calls `sync()` before `summarize()`. `sync()` must
    combine the accumulators of all processes (in place), typically by
    summing counts and running sums with
    `optastra.core.distributed.sum_across_processes` -- which is why
    evaluators accumulate sums rather than per-batch results. Without
    `sync()`, an evaluator only works in single-process runs.
    """

    def reset(self) -> None: ...
    def process(self, output: TaskStepOutput, batch: Mapping[str, Any]) -> None: ...
    def sync(self) -> None: ...
    def summarize(self) -> dict[str, float]: ...


def infer_batch_size(batch: Mapping[str, Any]) -> int:
    """Number of samples in a collated batch: the leading dim of
    `batch["inputs"]` (or its length if it is a list), or of the first view
    for multi-view batches. Falls back to 1 (i.e. unweighted) if unknown."""
    inputs = batch.get("inputs") if isinstance(batch, Mapping) else None
    if inputs is None and isinstance(batch, Mapping) and batch.get("views"):
        inputs = batch["views"][0]
    if torch.is_tensor(inputs):
        return int(inputs.shape[0])
    if isinstance(inputs, (list, tuple)):
        return len(inputs)
    return 1


class MeanMetricEvaluator:
    """Default Evaluator: sample-weighted mean of `output.loss` (reported as
    "total_loss") and every entry of `output.metrics`. Each batch is weighted
    by its size, so the result equals the per-sample mean over the whole
    dataset even when the last batch is smaller -- assuming each per-batch
    value is itself a mean over that batch."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    def process(self, output: TaskStepOutput, batch: Mapping[str, Any]) -> None:
        n = infer_batch_size(batch)
        values = dict(output.metrics)
        if output.loss is not None:
            values["total_loss"] = output.loss
        for k, v in values.items():
            self._sums[k] += float(v) * n
            self._counts[k] += n

    def sync(self) -> None:
        """Add up every process's sums and counts (multi-process evaluation)."""
        total = sum_across_processes(
            {f"sum/{k}": v for k, v in self._sums.items()} | {f"count/{k}": v for k, v in self._counts.items()}
        )
        self._sums = defaultdict(float, {k[4:]: v for k, v in total.items() if k.startswith("sum/")})
        self._counts = defaultdict(int, {k[6:]: v for k, v in total.items() if k.startswith("count/")})

    def summarize(self) -> dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums if self._counts[k] > 0}


class Task(ABC, Factory["Task"]):
    """Owns what is computed for a batch: splitting inputs/targets, the
    forward pass, losses, metrics and decoding. It is deliberately free of
    runtime policy -- mixed precision (autocast), gradient accumulation,
    clipping and devices are the Trainer's job, so `run_step` behaves the
    same on CPU, on GPU, and under any precision the Trainer wraps it in."""

    required_fields: tuple[str, ...] = ()
    collate: str = "default_collate"
    _registry = FamilyRegistry("task")

    def run_step(self, model, batch: Mapping[str, Any], stage: Stage = "train") -> TaskStepOutput:
        self.validate_batch(batch, stage)
        inputs, raw_targets = self.split_inputs_targets(batch, stage)
        targets = self.preprocess_targets(raw_targets) if raw_targets is not None else None

        raw_preds = self.forward_model(model, inputs)
        self.validate_predictions(raw_preds)

        losses, total_loss = {}, None
        if stage in ("train", "val") and targets is not None:
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

    def build_evaluator(self) -> Evaluator:
        """Fresh Evaluator for one eval pass. The default averages loss and
        `compute_metrics` output weighted by batch size; override for metrics
        that need dataset-level accumulation (counts, mAP, ...)."""
        return MeanMetricEvaluator()

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
