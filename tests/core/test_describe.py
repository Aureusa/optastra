from optastra.core.component_ref import ComponentRef
from optastra.core.describe import resolve_experiment
from optastra.core.experiment import ExperimentConfig


def test_resolve_experiment_expands_nested_components():
    cfg = ExperimentConfig(
        architecture=ComponentRef(
            "mask_rcnn_r50_fpn",
            {
                "num_classes": 20,
                "backbone": ComponentRef("resnet50", {"stem_channels": 32}),
            },
        ),
        task=ComponentRef("detection_task", {"num_classes": 2}),
        optimizer=ComponentRef("adamw", {"lr": 1e-4}),
        scheduler=ComponentRef("warmup_cosine", {"warmup_steps": 1000}),
        max_iter=1_115_000,
        batch_size=16,
        output_dir="runs/exp1",
    )

    resolved = resolve_experiment(cfg)

    assert "__default__" not in resolved["architecture"]
    assert "__overrides__" not in resolved["architecture"]

    assert "__default__" not in resolved["task"]
    assert "__overrides__" not in resolved["task"]

    assert resolved["architecture"]["num_classes"] == 20
    assert resolved["task"]["num_classes"] == 2

    assert resolved["architecture"]["backbone"]["name"] == "resnet50"
    assert "layers" in resolved["architecture"]["backbone"]

    assert resolved["architecture"]["neck"]["name"] == "fpn"
    assert resolved["architecture"]["proposal_generator"]["name"] == "rpn"
    assert resolved["architecture"]["roi_box_head"]["name"] == "roi_box_head"
    assert resolved["architecture"]["roi_box_head"]["num_classes"] == 80
    assert resolved["architecture"]["mask_head"]["name"] == "mask_rcnn_head"
    assert resolved["architecture"]["mask_head"]["num_classes"] == 80

    assert resolved["task"]["criterion"]["name"] == "rcnn_criterion"
    assert resolved["task"]["criterion"]["num_classes"] == 80
    assert resolved["task"]["postprocessor"]["name"] == "rcnn_postprocessor"
    assert "bbox_reg_weights" in resolved["task"]["postprocessor"]
