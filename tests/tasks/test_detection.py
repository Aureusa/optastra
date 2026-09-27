import pytest
import torch

from optastra.core.component_ref import ComponentRef
from optastra.architectures import Architecture
from optastra.detection import Postprocessor, keys
from optastra.nn.features import HeadOutput
from optastra.tasks import CriterionBasedTask, Task
from optastra.tasks.detection import DetectionTask


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

# ---------------------------------------------------------------------------
# Shared small model / batch
# ---------------------------------------------------------------------------

def _small_faster_rcnn(num_classes=3, **overrides):
    torch.manual_seed(0)
    return Architecture.create(
        "faster_rcnn_r18_fpn",
        num_classes=num_classes,
        roi_box_head=ComponentRef("roi_box_head", {"fc_hidden_features": 32}),
        **overrides,
    )


def _small_task(num_classes=3):
    return Task.create(
        "detection_task",
        num_classes=num_classes,
        criterion=ComponentRef("rcnn_criterion", {
            "roi_sampler": ComponentRef("rcnn_balanced_sampler", {"batch_size": 64, "positive_fraction": 0.25}),
            "rpn_sampler": ComponentRef("rpn_balanced_sampler", {"batch_size": 64, "positive_fraction": 0.5}),
        }),
    )


def _batch():
    return {
        "inputs": torch.randn(2, 3, 64, 64),
        "targets": [
            {"boxes": torch.tensor([[4.0, 6.0, 40.0, 44.0]]), "labels": torch.tensor([2])},
            {"boxes": torch.tensor([[10.0, 10.0, 50.0, 60.0]]), "labels": torch.tensor([0])},
        ],
        "image_sizes": [(64, 64), (64, 60)],
    }


# ---------------------------------------------------------------------------
# Task wiring
# ---------------------------------------------------------------------------

def test_detection_task_is_a_criterion_based_task():
    assert issubclass(DetectionTask, CriterionBasedTask)


def test_default_architecture_and_task_num_classes_agree():
    task_classes = Task.get_default_config("detection_task").num_classes
    for name in ("faster_rcnn_r50_fpn", "faster_rcnn_r18_c5", "mask_rcnn_r50_fpn", "mask_rcnn_r18_c5"):
        assert Architecture.get_default_config(name).num_classes == task_classes


def test_num_classes_mismatch_between_model_and_task_raises_clear_error():
    model = _small_faster_rcnn(num_classes=5)
    with pytest.raises(ValueError, match="num_classes"):
        _small_task(num_classes=3).run_step(model, _batch(), stage="train")


def test_out_of_range_labels_raise_instead_of_being_clamped():
    batch = _batch()
    batch["targets"][1]["labels"] = torch.tensor([3])  # valid labels are 0..2
    with pytest.raises(ValueError, match=r"\[0, 3\)"):
        _small_task().run_step(_small_faster_rcnn(), batch, stage="train")


def test_train_step_passes_gt_boxes_and_image_sizes_to_the_model():
    model = _small_faster_rcnn()
    out = _small_task().run_step(model, _batch(), stage="train")

    roi_boxes = out.raw_predictions.extra[keys.ROI_BOXES]
    # GT boxes are appended after the proposals, one per image here.
    assert torch.equal(roi_boxes[-2:], torch.tensor([[0.0, 4.0, 6.0, 40.0, 44.0], [1.0, 10.0, 10.0, 50.0, 60.0]]))
    assert out.raw_predictions.extra[keys.IMAGE_SIZES] == [(64, 64), (64, 60)]
    assert torch.isfinite(out.loss)


def test_val_step_decodes_one_sample_per_image_without_gt_boxes():
    model = _small_faster_rcnn().eval()
    batch = _batch()
    out = _small_task().run_step(model, batch, stage="val")

    roi_boxes = out.raw_predictions.extra[keys.ROI_BOXES]
    assert not any(torch.equal(row[1:], batch["targets"][0]["boxes"][0]) for row in roi_boxes)
    assert len(out.predictions) == 2
    for sample, (h, w) in zip(out.predictions, batch["image_sizes"]):
        boxes = sample.target["boxes"]
        assert boxes.shape[1] == 4
        assert (boxes[:, [0, 2]] <= w).all() and (boxes[:, [1, 3]] <= h).all()


# ---------------------------------------------------------------------------
# Gradient isolation: proposals are detached from the RPN
# ---------------------------------------------------------------------------

def test_roi_losses_do_not_backpropagate_into_the_rpn():
    model = _small_faster_rcnn()
    task = _small_task()
    batch = _batch()
    inputs, raw_targets = task.split_inputs_targets(batch, "train")
    raw_preds = task.forward_model(model, inputs)
    losses = task.compute_losses(raw_preds, task.preprocess_targets(raw_targets))

    (losses["roi_cls_loss"] + losses["roi_box_loss"]).backward(retain_graph=True)
    rpn_grads = [p.grad for p in model.proposal_generator.parameters()]
    assert all(g is None or torch.count_nonzero(g) == 0 for g in rpn_grads)
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.roi_head.parameters())

    # Sanity check: the RPN does learn from its own losses.
    (losses["rpn_objectness_loss"] + losses["rpn_box_loss"]).backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.proposal_generator.parameters())


# ---------------------------------------------------------------------------
# Postprocessor (hand-built outputs)
# ---------------------------------------------------------------------------

def _postprocess(raw, num_classes, **overrides):
    return Postprocessor.create("rcnn_postprocessor", **overrides).process(raw, num_classes=num_classes)


def test_postprocessor_decodes_each_class_with_its_own_deltas():
    roi = torch.tensor([[0.0, 10.0, 10.0, 30.0, 30.0]])  # 20x20 proposal
    logits = torch.tensor([[2.0, 2.0, 0.0]])            # both foreground classes pass score_thresh
    deltas = torch.tensor([[0.0, 0.0, 0.0, 0.0,           # class 0: keep the proposal
                            0.5, 0.0, 0.0, 0.0]])         # class 1: shift right by half a width
    raw = HeadOutput(logits=logits, values=deltas, extra={keys.ROI_BOXES: roi, keys.IMAGE_SIZES: [(64, 64)]})

    (sample,) = _postprocess(raw, num_classes=2)

    by_label = {int(l): b.tolist() for l, b in zip(sample.target["labels"], sample.target["boxes"])}
    assert by_label[0] == pytest.approx([10.0, 10.0, 30.0, 30.0])
    assert by_label[1] == pytest.approx([20.0, 10.0, 40.0, 30.0])


def test_postprocessor_returns_one_sample_per_image_and_clips_to_each_image():
    # All ROIs belong to image 0/1; image 2 has none but must still get a (empty) Sample.
    roi = torch.tensor([[0.0, 0.0, 0.0, 50.0, 50.0], [1.0, 0.0, 0.0, 50.0, 50.0]])
    logits = torch.tensor([[5.0, 0.0], [5.0, 0.0]])
    deltas = torch.zeros((2, 4))
    raw = HeadOutput(
        logits=logits, values=deltas,
        extra={keys.ROI_BOXES: roi, keys.IMAGE_SIZES: [(64, 64), (20, 30), (64, 64)]},
    )

    samples = _postprocess(raw, num_classes=1)

    assert len(samples) == 3
    assert samples[0].target["boxes"].tolist() == [[0.0, 0.0, 50.0, 50.0]]
    assert samples[1].target["boxes"].tolist() == [[0.0, 0.0, 30.0, 20.0]]
    assert samples[2].target["boxes"].shape == (0, 4)
    assert samples[1].meta["image_size"] == (20, 30)


def test_postprocessor_pastes_the_predicted_class_mask_into_the_image():
    roi = torch.tensor([[0.0, 2.0, 2.0, 10.0, 6.0]])  # 8 wide, 4 high
    logits = torch.tensor([[0.0, 5.0, 0.0]])        # class 1 only
    masks = torch.full((1, 2, 4, 4), 10.0)          # class 0 channel: everything on
    masks[0, 1, :, 2:] = -10.0                      # class 1 channel: left half only
    raw = HeadOutput(
        logits=logits, values=torch.zeros((1, 4)), masks=masks,
        extra={keys.ROI_BOXES: roi, keys.IMAGE_SIZES: [(12, 12)]},
    )

    (sample,) = _postprocess(raw, num_classes=2)

    assert sample.target["labels"].tolist() == [1]
    pred = sample.target["masks"]
    assert pred.shape == (1, 12, 12) and pred.dtype == torch.bool
    expected = torch.zeros((12, 12), dtype=torch.bool)
    expected[2:6, 2:6] = True  # left half of the 8x4 box
    assert torch.equal(pred[0], expected)


def test_criterion_based_task_decode_passes_num_classes():
    raw = HeadOutput(
        logits=torch.tensor([[5.0, 0.0, 0.0, 0.0]]), values=torch.zeros((1, 4)),
        extra={keys.ROI_BOXES: torch.tensor([[0.0, 1.0, 1.0, 5.0, 5.0]]), keys.IMAGE_SIZES: [(8, 8)]},
    )
    (sample,) = Task.create("detection_task", num_classes=3).decode_predictions(raw)
    assert sample.target["labels"].tolist() == [0]
