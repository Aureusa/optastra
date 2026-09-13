"""Config-driven model assembly: the "configuration change, not code change"
goal from the README's Research Direction section.

Defines an ExperimentConfig, round-trips it through YAML, then builds the
real model/task/optimizer graph from the loaded config. Building a Fast
R-CNN this way only constructs `nn.Module`s (no forward pass, no dataset),
so it stays fast and CPU-only.

Run with:
    python examples/03_experiment_config_yaml.py
"""
import tempfile
from pathlib import Path

from optastra import ComponentRef, ExperimentConfig, build_experiment_from_config, print_experiment


def main() -> None:
    cfg = ExperimentConfig(
        architecture=ComponentRef("fast_rcnn_r18_fpn", {"num_classes": 5}),
        task=ComponentRef("detection_task", {"num_classes": 5}),
        optimizer=ComponentRef("adamw", {"lr": 1e-4}),
        output_dir="runs/example_detection",
    )

    print("--- resolved config (defaults + overrides merged) ---")
    print_experiment(cfg)

    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = Path(tmp) / "experiment.yaml"
        cfg.to_yaml(str(yaml_path))
        loaded = ExperimentConfig.from_yaml(str(yaml_path))

        built = build_experiment_from_config(loaded)

    print("\n--- built objects ---")
    print(f"model:     {type(built['model']).__name__}")
    print(f"task:      {type(built['task']).__name__}")
    print(f"optimizer: {type(built['optimizer']).__name__}")


if __name__ == "__main__":
    main()
