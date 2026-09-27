from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ..core.component_ref import ComponentRef, component_field, ComponentRefConfigMixin
from .base import Stage, Task
from .criterion_based import CriterionBasedTask
from ..detection import DetectionCriterion, Postprocessor


__all__ = ["DetectionTask", "DetectionTaskConfig"]


@dataclass
class DetectionTaskConfig(ComponentRefConfigMixin):
    # Foreground classes (background excluded). Must equal the architecture's
    # num_classes -- the criterion checks the logits and raises on a mismatch.
    num_classes: int = 80
    criterion: ComponentRef = component_field(DetectionCriterion, default_name="rcnn_criterion")
    postprocessor: ComponentRef = component_field(Postprocessor, default_name="rcnn_postprocessor")


class DetectionTask(CriterionBasedTask):
    """Box / instance detection with a pluggable criterion and postprocessor.

    Batch contract (produced by the ``ragged`` collate):
        inputs       (N, C, H, W) images, padded to a common size
        targets      list of per-image dicts: boxes (G, 4) XYXY, labels (G,) in [0, num_classes), optional masks
        image_sizes  optional list of per-image (h, w) before padding
        rois         optional (R, 5) precomputed proposals (Fast R-CNN)

    The model is called as ``model(images, rois=..., image_sizes=..., gt_boxes=...)``
    with only the keys present in the batch; ``gt_boxes`` is passed during
    training so the architecture can add them to its proposals.
    """

    required_fields = ()
    collate = "ragged"

    def __init__(self, cfg: DetectionTaskConfig = DetectionTaskConfig()):
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.criterion = cfg.criterion.resolve(DetectionCriterion, num_classes=cfg.num_classes)
        self.postprocessor = cfg.postprocessor.resolve(Postprocessor)

    def validate_batch(self, batch: Mapping[str, Any], stage: Stage = "train"):
        if "inputs" not in batch:
            raise ValueError("DetectionTask expects 'inputs' in batch.")
        if stage in ("train", "val", "test") and "targets" not in batch:
            raise ValueError("DetectionTask expects 'targets' for train/val/test.")

    def split_inputs_targets(self, batch: Mapping[str, Any], stage: Stage = "train"):
        inputs: dict[str, Any] = {"images": batch["inputs"]}
        for key in ("image_sizes", "rois"):
            if batch.get(key) is not None:
                inputs[key] = batch[key]
        if stage == "predict":
            return inputs, None
        if stage == "train":
            inputs["gt_boxes"] = [target["boxes"] for target in batch["targets"]]
        return inputs, batch["targets"]

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
        if not isinstance(inputs, Mapping):
            return model(inputs)
        kwargs = {key: inputs[key] for key in ("rois", "image_sizes", "gt_boxes") if key in inputs}
        return model(inputs["images"], **kwargs)


detection_task_configs = {
    "detection_task": DetectionTaskConfig(),
}


@Task.register(config=detection_task_configs["detection_task"])
def detection_task(cfg: DetectionTaskConfig) -> DetectionTask:
    return DetectionTask(cfg)
