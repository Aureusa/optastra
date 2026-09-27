import torch

from optastra.data import build_dataloader
from optastra.data.sample import Sample
from optastra.tasks import Task

from optastra.architectures import Architecture
from optastra.architectures.fast_rcnn import FastRCNN
from optastra.core.component_ref import ComponentRef


def test_fast_rcnn_forward_requires_rois_and_returns_box_outputs():
    model = Architecture.create(
        "fast_rcnn_r18_fpn",
        num_classes=5,
        roi_box_head=ComponentRef("roi_box_head", {"fc_hidden_features": 64}),
        region_extractor=ComponentRef("roi_align", {"stage": "P2", "output_size": 7}),
    )

    images = torch.randn(2, 3, 128, 128)
    rois = torch.tensor(
        [
            [0, 0, 0, 64, 64],
            [1, 16, 16, 96, 96],
        ],
        dtype=torch.float32,
    )

    out = model(images, rois)
    assert out.logits.shape == (2, 6)
    assert out.values.shape == (2, 4)
    assert "roi_boxes" in out.extra


def test_fast_rcnn_variant_builds_from_registry():
    model = Architecture.create("fast_rcnn_r50_fpn", num_classes=3)
    assert isinstance(model, FastRCNN)

def test_fast_rcnn_trains_from_the_ragged_collate_with_precomputed_proposals():
    torch.manual_seed(0)
    model = Architecture.create(
        "fast_rcnn_r18_fpn",
        num_classes=2,
        roi_box_head=ComponentRef("roi_box_head", {"fc_hidden_features": 16}),
    )
    task = Task.create("detection_task", num_classes=2)
    dataset = [
        Sample(image=torch.rand(3, 64, 48), target={
            "boxes": torch.tensor([[4.0, 4.0, 30.0, 40.0]]), "labels": torch.tensor([1]),
            "proposals": torch.tensor([[5.0, 5.0, 28.0, 38.0], [0.0, 0.0, 10.0, 10.0]]),
        }),
        Sample(image=torch.rand(3, 40, 64), target={
            "boxes": torch.tensor([[10.0, 8.0, 50.0, 30.0]]), "labels": torch.tensor([0]),
            "proposals": torch.tensor([[12.0, 6.0, 48.0, 32.0]]),
        }),
    ]
    batch = next(iter(build_dataloader(dataset, task=task, batch_size=2, collate_kwargs={"size_divisibility": 32})))

    out = task.run_step(model, batch, stage="train")

    assert batch["inputs"].shape == (2, 3, 64, 64)
    assert batch["rois"][:, 0].tolist() == [0.0, 0.0, 1.0]
    assert torch.isfinite(out.loss)
    # 3 proposals + 2 appended GT boxes went through the ROI head
    assert out.raw_predictions.logits.shape == (5, 3)
