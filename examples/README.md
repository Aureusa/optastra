# Optastra Examples

Small, self-contained, CPU-runnable scripts demonstrating core Optastra
usage patterns. None of these need a GPU, a dataset, or a cluster --
they use synthetic tensors and construct real modules from the registry.

Install the package first (from the repo root):

```bash
pip install -e .
```

Then run any example directly:

```bash
python examples/01_quickstart_classification.py
python examples/02_custom_transform.py
python examples/03_experiment_config_yaml.py
python examples/04_list_and_describe_components.py
python examples/05_full_training_pipeline.py
```

| Script | Demonstrates |
| --- | --- |
| `01_quickstart_classification.py` | `Backbone -> Neck -> Head` composition via `build_sequential_model`, one train step. |
| `02_custom_transform.py` | Registering a brand-new `Transform` (the same pattern used by every family). |
| `03_experiment_config_yaml.py` | `ExperimentConfig` YAML round-trip and building a full model/task/optimizer graph from config. |
| `04_list_and_describe_components.py` | Discoverability: listing registered families/components and printing a default config. |
| `05_full_training_pipeline.py` | End-to-end: model + `Task` + `Optimizer` + `Scheduler` + augmentations + `Trainer` + hooks (checkpointing, logging, LR scheduling, periodic eval, early stopping) wired together on a toy dataset -- the same shape as a real training script. |

See [`../docs/getting-started.md`](../docs/getting-started.md) for a
narrated walkthrough and [`../docs/extending.md`](../docs/extending.md)
for the full "add a new X" cookbook.
