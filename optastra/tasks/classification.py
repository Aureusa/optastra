import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Mapping

from .base import Task, Stage, TaskStepOutput
from ..nn.features import HeadOutput


__all__ = ["ClassificationTask", "ClassificationEvaluator"]


@dataclass
class ClassificationTaskConfig:
    label_smoothing: float = 0.0
    reduction: str = "mean"  # Options: 'mean', 'sum', 'none'


class ClassificationEvaluator:
    """Dataset-level accuracy from accumulated counts (correct / total) plus
    the sample-weighted mean loss -- exact regardless of batch sizes."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.correct = 0
        self.total = 0
        self.loss_sum = 0.0
        self.loss_count = 0

    def process(self, output: TaskStepOutput, batch: Mapping[str, Any]) -> None:
        labels = output.targets["labels"]
        n = labels.shape[0]
        self.correct += int((output.predictions == labels).sum().item())
        self.total += n
        if output.loss is not None:
            # output.loss is a mean over the batch -> weight it by the batch size.
            self.loss_sum += float(output.loss) * n
            self.loss_count += n

    def summarize(self) -> dict[str, float]:
        results = {}
        if self.total > 0:
            results["accuracy"] = self.correct / self.total
        if self.loss_count > 0:
            results["total_loss"] = self.loss_sum / self.loss_count
        return results


class ClassificationTask(Task):
    required_fields = ("logits",) # Ensure that the model output contains 'logits' for classification tasks
    collate = "dense"
    def __init__(self, cfg: ClassificationTaskConfig = ClassificationTaskConfig()):
        self.cfg = cfg
        self.reduction = cfg.reduction

    def compute_losses(self, raw_preds, targets):
        if "labels_b" in targets:
            # CutMix/MixUp path: interpolate the two label losses by lam
            loss_a = F.cross_entropy(raw_preds.logits, targets["labels"],
                                      label_smoothing=self.cfg.label_smoothing, reduction=self.reduction)
            loss_b = F.cross_entropy(raw_preds.logits, targets["labels_b"],
                                      label_smoothing=self.cfg.label_smoothing, reduction=self.reduction)
            lam = targets["lam"]
            loss = lam * loss_a + (1 - lam) * loss_b
        else:
            # Standard classification path
            loss = F.cross_entropy(raw_preds.logits, targets["labels"],
                                    label_smoothing=self.cfg.label_smoothing,
                                    reduction=self.reduction)
        return {"ce_loss": loss}

    def validate_batch(self, batch: Mapping[str, Any], stage: Stage = "train"):
        if stage in ("train", "val", "test") and ("inputs" not in batch or "targets" not in batch):
            raise ValueError(f"Batch must contain 'inputs' and 'targets' keys for stage '{stage}'.")
        elif stage == "predict" and "inputs" not in batch:
            raise ValueError(f"Batch must contain 'inputs' key for prediction stage.")
    
    def split_inputs_targets(self, batch: Mapping[str, Any], stage: Stage = "train") -> tuple[Mapping[str, Any], Any]:
        if stage in ("train", "val", "test"):
            return batch["inputs"], batch["targets"]
        elif stage == "predict":
            return batch["inputs"], None

    def preprocess_targets(self, raw_targets: Mapping[str, Any] | torch.Tensor) -> Mapping[str, Any]:
        if isinstance(raw_targets, Mapping):
            # pass through mixed-label fields untouched if CutMix/MixUp already ran
            out = {"labels": raw_targets["labels"].long()}
            if "labels_b" in raw_targets:
                out["labels_b"] = raw_targets["labels_b"].long()
                out["lam"] = raw_targets["lam"]
            return out
        return {"labels": raw_targets.long()}

    def forward_model(self, model, inputs: Mapping[str, Any]) -> Any:
        return model(inputs)

    def reduce_losses(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(losses.values())

    # TODO: Implement a more comprehensive metric computation for classification tasks
    def compute_metrics(self, raw_preds: HeadOutput, targets: Mapping[str, Any]) -> dict[str, float]:
        with torch.no_grad():
            preds = torch.argmax(raw_preds.logits, dim=1)
            correct = (preds == targets["labels"]).sum().item()
            total = targets["labels"].size(0)
            accuracy = correct / total
        return {"accuracy": accuracy}

    def decode_predictions(self, raw_preds: HeadOutput) -> Any:
        return torch.argmax(raw_preds.logits, dim=1)

    def build_evaluator(self) -> ClassificationEvaluator:
        return ClassificationEvaluator()


classification_task_config = {
    # Same as the dataclass defaults (no label smoothing) -- pass
    # label_smoothing=0.1 explicitly if you want it.
    "classification_task": ClassificationTaskConfig(label_smoothing=0.0, reduction="mean")
}

@Task.register(config=classification_task_config["classification_task"])
def classification_task(cfg: ClassificationTaskConfig) -> ClassificationTask:
    return ClassificationTask(cfg)
