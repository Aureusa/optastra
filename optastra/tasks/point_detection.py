"""Point detection: find every object's position (x, y), e.g. sources in an
astronomical field or heads in a crowd. Pairs with a point head
(``keypoint_head``, e.g. in ``point_detr_r18_fpn``)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ..core.component_ref import ComponentRef, ComponentRefConfigMixin, component_field
from ..detection import DetectionCriterion, Postprocessor
from .base import Stage, Task
from .criterion_based import CriterionBasedTask


__all__ = ["PointDetectionTask", "PointDetectionTaskConfig", "point_detection_task"]


@dataclass
class PointDetectionTaskConfig(ComponentRefConfigMixin):
    criterion: ComponentRef = component_field(DetectionCriterion, default_name="point_set_criterion")
    postprocessor: ComponentRef = component_field(Postprocessor, default_name="point_postprocessor")


class PointDetectionTask(CriterionBasedTask):
    """Batch contract (the ``ragged`` collate):
        inputs       (N, C, H, W) images, padded to a common size
        targets      list of per-image dicts: points (G, 2) pixel (x, y)
        image_sizes  optional list of per-image (h, w) before padding

    The model is called as ``model(images, image_sizes=...)``.
    """

    required_fields = ()
    collate = "ragged"

    def __init__(self, cfg: PointDetectionTaskConfig = PointDetectionTaskConfig()):
        self.cfg = cfg
        self.num_classes = 1
        self.criterion = cfg.criterion.resolve(DetectionCriterion)
        self.postprocessor = cfg.postprocessor.resolve(Postprocessor)

    def validate_batch(self, batch: Mapping[str, Any], stage: Stage = "train"):
        if "inputs" not in batch:
            raise ValueError("PointDetectionTask expects 'inputs' in batch.")
        if stage != "predict" and "targets" not in batch:
            raise ValueError("PointDetectionTask expects 'targets' for train/val/test.")

    def split_inputs_targets(self, batch: Mapping[str, Any], stage: Stage = "train"):
        inputs = {"images": batch["inputs"], "image_sizes": batch.get("image_sizes")}
        return inputs, (None if stage == "predict" else batch["targets"])

    def preprocess_targets(self, raw_targets: list[Mapping[str, Any]]) -> list[dict[str, torch.Tensor]]:
        out = []
        for target in raw_targets:
            if "points" not in target:
                raise ValueError("Each point-detection target must have 'points' (G, 2).")
            out.append({"points": torch.as_tensor(target["points"]).float().reshape(-1, 2)})
        return out

    def forward_model(self, model, inputs):
        return model(inputs["images"], image_sizes=inputs["image_sizes"])


@Task.register(config=PointDetectionTaskConfig())
def point_detection_task(cfg: PointDetectionTaskConfig) -> PointDetectionTask:
    return PointDetectionTask(cfg)
