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
python examples/06_ssl_pretrain_then_finetune.py
python examples/07_object_detection.py
python examples/08_astronomy_multiband.py
python examples/09_custom_backbone.py
python examples/10_regression.py
python examples/11_hooks_checkpoint_resume.py
python examples/12_compare_backbones.py
python examples/13_multi_gpu_ddp.py
torchrun --nproc_per_node=2 examples/13_multi_gpu_ddp.py   # same script, 2 processes / GPUs
```

| Script | Demonstrates |
| --- | --- |
| `01_quickstart_classification.py` | `Backbone -> Neck -> Head` composition via `build_sequential_model`, one train step. |
| `02_custom_transform.py` | Registering a brand-new `Transform` (the same pattern used by every family). |
| `03_experiment_config_yaml.py` | `ExperimentConfig` YAML round-trip and building a full model/task/optimizer graph from config. |
| `04_list_and_describe_components.py` | Discoverability: listing registered families/components and printing a default config. |
| `05_full_training_pipeline.py` | End-to-end: model + `Task` + `Optimizer` + `Scheduler` + augmentations + `Trainer` + hooks (checkpointing, logging, LR scheduling, periodic eval, early stopping) wired together on a toy dataset -- the same shape as a real training script. |
| `06_ssl_pretrain_then_finetune.py` | Self-supervised transfer: BYOL built from config (`Algorithm.create("byol", backbone=...)`), `MultiViewTransform` data, `algo.hooks()` for the EMA target, `export_backbone`, then fine-tuning a classifier from `weights=`. |
| `07_object_detection.py` | Faster R-CNN on variable-size synthetic images: box-aware augmentation, padded ragged batches with `image_sizes`, training, evaluation, and decoded detections (`stage="predict"`). |
| `08_astronomy_multiband.py` | 5-band, background-subtracted, high-dynamic-range cutouts: dtype-aware `to_float`, a custom per-band `asinh_stretch` transform, rotation/flip/blur/RandAugment on float data without clipping, `in_channels=5`. |
| `09_custom_backbone.py` | A new backbone built from `optastra.nn.blocks`, its `FeatureSpec` contract, S/M/L variants registered in a loop (`name=`), plugged into pooling + head and into FPN; wiring errors caught when the model is built. |
| `10_regression.py` | Regression with two continuous outputs per image: `vanilla_regression_head` + `regression_task` (huber loss), dataset-level MAE / RMSE / R², and prediction. |
| `11_hooks_checkpoint_resume.py` | Custom hooks (a new logged metric; a simulated crash), periodic checkpoints, `ResumeHook` resuming to bit-identical weights, and reading `logs/metrics.jsonl`. |
| `12_compare_backbones.py` | ResNet vs EfficientNet vs ConvNeXt vs ViT on the same data, where only the backbone reference changes: params, speed, and validation accuracy side by side. |
| `13_multi_gpu_ddp.py` | Multi-GPU data-parallel training (DDP): the same script under `python` or `torchrun --nproc_per_node=2` -- sharded data, averaged gradients, exact sharded evaluation, rank-0-only checkpoints/logs, a SLURM launch line. Falls back to CPU processes when there are fewer GPUs. |

See [`../docs/tutorials.md`](../docs/tutorials.md) for scenario walkthroughs built on
these scripts, [`../docs/getting-started.md`](../docs/getting-started.md) for a
narrated first run and [`../docs/extending.md`](../docs/extending.md)
for the full "add a new X" cookbook.
