"""Every registered default config must be YAML-safe and round-trip unchanged."""
from dataclasses import is_dataclass

import pytest
import yaml

from optastra import ComponentRef, ExperimentConfig, list_all_registered_components
from optastra.core.component_ref import config_from_plain, config_to_plain
from optastra.core.factory import FACTORIES


def _all_components():
    return [
        (family, name)
        for family, names in list_all_registered_components().items()
        for name in names
    ]


@pytest.mark.parametrize("family, name", _all_components(), ids=lambda v: v if isinstance(v, str) else None)
def test_default_config_round_trips_through_yaml(family, name):
    cfg = FACTORIES[family].get_default_config(name)
    if cfg is None:
        pytest.skip("configless component")
    assert is_dataclass(cfg)

    text = yaml.safe_dump(config_to_plain(cfg), sort_keys=False)   # must not raise
    loaded = type(cfg)(**config_from_plain(yaml.safe_load(text)))
    # YAML has no tuple type, so tuple-valued fields come back as lists; what must
    # survive is the serialized form, i.e. dumping the loaded config is lossless.
    assert config_to_plain(loaded) == config_to_plain(cfg)


def test_config_to_plain_rejects_classes():
    with pytest.raises(TypeError, match="not YAML-safe"):
        config_to_plain({"block": ComponentRef})


def test_experiment_config_yaml_round_trip_with_nested_refs(tmp_path):
    cfg = ExperimentConfig(
        architecture=ComponentRef(
            "faster_rcnn_r50_fpn",
            {
                "num_classes": 5,
                "backbone": ComponentRef("resnet18", {"layers": [1, 1, 1, 1], "block": "basic"}),
                "neck": ComponentRef("fpn", {"out_channels": 64}),
            },
        ),
        task=ComponentRef("detection_task", {"num_classes": 5}),
        optimizer=ComponentRef("adamw", {"lr": 1e-4}),
        scheduler=ComponentRef("warmup_cosine", {"warmup_steps": 10}),
        seed=3,
        output_dir="runs/x",
    )
    path = tmp_path / "exp.yaml"
    text = cfg.to_yaml(str(path))

    assert yaml.safe_load(text)["architecture"]["overrides"]["backbone"] == {
        "name": "resnet18", "overrides": {"layers": [1, 1, 1, 1], "block": "basic"},
    }
    loaded = ExperimentConfig.from_yaml(str(path))
    assert loaded == cfg
    assert isinstance(loaded.architecture.overrides["backbone"], ComponentRef)


def test_experiment_config_yaml_round_trip_without_scheduler(tmp_path):
    cfg = ExperimentConfig(architecture="fast_rcnn_r18_fpn", task="detection_task")
    path = tmp_path / "exp.yaml"
    cfg.to_yaml(str(path))
    assert ExperimentConfig.from_yaml(str(path)) == cfg
