import torch

from optastra.core.component_ref import ComponentRef
from optastra.architectures import Architecture
from optastra.tasks import Task


def test_detection_task_computes_roi_and_rpn_losses_with_faster_rcnn():
    model = Architecture.create(
        "faster_rcnn_r18_fpn",
        num_classes=3,
        proposal_generator=ComponentRef("rpn"),
        region_extractor=ComponentRef("roi_align", {"stage": "P2", "output_size": 7}),
        roi_box_head=ComponentRef("roi_box_head", {"fc_hidden_features": 64}),
    )
    task = Task.create(
        "detection_task",
        num_classes=3,
        criterion=ComponentRef("rcnn_criterion", {
            "roi_sampler": ComponentRef("rcnn_balanced_sampler", {"batch_size": 64, "positive_fraction": 0.25}),
            "rpn_sampler": ComponentRef("rpn_balanced_sampler", {"batch_size": 128, "positive_fraction": 0.5}),
        }),
    )

    batch = {
        "inputs": torch.randn(2, 3, 128, 128),
        "targets": [
            {
                "boxes": torch.tensor([[10.0, 12.0, 70.0, 80.0], [20.0, 25.0, 100.0, 100.0]]),
                "labels": torch.tensor([0, 2], dtype=torch.long),
            },
            {
                "boxes": torch.tensor([[15.0, 15.0, 90.0, 95.0]]),
                "labels": torch.tensor([1], dtype=torch.long),
            },
        ],
    }

    out = task.run_step(model, batch, stage="train")

    assert out.loss is not None
    assert "roi_cls_loss" in out.losses
    assert "roi_box_loss" in out.losses
    assert "rpn_objectness_loss" in out.losses
    assert "rpn_box_loss" in out.losses