import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Mapping

from ._registry import register_task
from .base import Task, Stage
from ..nn.features import HeadOutput


__all__ = ["ClassificationTask"]


@dataclass
class ClassificationTaskConfig:
    label_smoothing: float = 0.0
    reduction: str = "mean"  # Options: 'mean', 'sum', 'none'


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


classification_task_config = {
    "classification_task": ClassificationTaskConfig(label_smoothing=0.1, reduction="mean")
}

@register_task(config=classification_task_config["classification_task"])
def classification_task(cfg: ClassificationTaskConfig) -> ClassificationTask:
    return ClassificationTask(cfg)
