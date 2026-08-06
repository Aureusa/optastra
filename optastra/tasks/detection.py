from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from ..core.component_ref import ComponentRef, resolve_component, component_field
from ._registry import register_task
from .base import Stage, Task
from ..detection import DetectionCriterion, Postprocessor
from ..nn.features import FeatureMaps, HeadOutput


@dataclass
class DetectionTaskConfig:
    num_classes: int = 80
    criterion: ComponentRef = component_field(DetectionCriterion, default_name="rcnn_criterion")
    postprocessor: ComponentRef = component_field(Postprocessor, default_name="rcnn_postprocessor")


class DetectionTask(Task):
    required_fields = ()
    collate = "ragged"

    def __init__(self, cfg: DetectionTaskConfig = DetectionTaskConfig()):
        self.cfg = cfg

        self.criterion = resolve_component(cfg, "criterion", num_classes=cfg.num_classes)
        self.postprocessor = resolve_component(cfg, "postprocessor")

    def validate_predictions(self, raw_preds: Any) -> None:
        if not isinstance(raw_preds, HeadOutput):
            raise TypeError(f"Model output must be a HeadOutput, got {type(raw_preds)}.")
        self.criterion.validate_predictions(raw_preds)

    def validate_batch(self, batch: Mapping[str, Any], stage: Stage = "train"):
        if "inputs" not in batch:
            raise ValueError("DetectionTask expects 'inputs' in batch.")
        if stage in ("train", "val", "test") and "targets" not in batch:
            raise ValueError("DetectionTask expects 'targets' for train/val/test.")

    def split_inputs_targets(self, batch: Mapping[str, Any], stage: Stage = "train"):
        if stage == "predict":
            return batch["inputs"], None
        return batch["inputs"], batch["targets"]

    def preprocess_targets(self, raw_targets: list[Mapping[str, Any]]) -> list[dict[str, torch.Tensor]]:
        processed: list[dict[str, torch.Tensor]] = []
        for target in raw_targets:
            if "boxes" not in target or "labels" not in target:
                raise ValueError("Each detection target must have 'boxes' and 'labels'.")
            out: dict[str, torch.Tensor] = {
                "boxes": target["boxes"].float(),
                "labels": target["labels"].long(),
            }
            if "masks" in target:
                out["masks"] = target["masks"].float()
            processed.append(out)
        return processed

    def forward_model(self, model, inputs):
        if isinstance(inputs, Mapping) and "rois" in inputs:
            return model(inputs["images"], inputs["rois"])
        return model(inputs)

    def compute_losses(self, raw_preds: HeadOutput, targets: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return self.criterion.compute_losses(raw_preds, targets)

    def reduce_losses(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(losses.values())

    def compute_metrics(self, raw_preds: HeadOutput, targets: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        return self.criterion.compute_metrics(raw_preds, targets)

    def decode_predictions(self, raw_preds: HeadOutput):
        image_size = None
        rpn_output = raw_preds.extra.get("rpn")
        if isinstance(rpn_output, FeatureMaps):
            image_size = rpn_output.extra.get("image_size")
        return self.postprocessor.process(raw_preds, num_classes=self.cfg.num_classes, image_size=image_size)


detection_task_configs = {
    "detection_task": DetectionTaskConfig(),
}


@register_task(config=detection_task_configs["detection_task"])
def detection_task(cfg: DetectionTaskConfig) -> DetectionTask:
    return DetectionTask(cfg)
