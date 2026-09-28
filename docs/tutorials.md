# Tutorials

Each tutorial below is a scenario you are likely to meet, backed by a
runnable script in [`examples/`](../examples/). Every script runs on CPU
with synthetic data in seconds, so you can read it, run it, and then change
it for your own data. This page explains *why* each script is written the
way it is, and what to change when you move to real data.

| Scenario | Script |
| --- | --- |
| [1. Train an image classifier](#1-train-an-image-classifier) | `01`, `05` |
| [2. Work with multi-band astronomical images](#2-work-with-multi-band-astronomical-images) | `08` |
| [3. Pretrain without labels, then fine-tune](#3-pretrain-without-labels-then-fine-tune) | `06` |
| [4. Detect objects](#4-detect-objects) | `07` |
| [5. Compare backbones and keep configs in YAML](#5-compare-backbones-and-keep-configs-in-yaml) | `12`, `03` |
| [6. Add your own backbone](#6-add-your-own-backbone) | `09` |
| [7. Predict continuous values, and add your own task](#7-predict-continuous-values-and-add-your-own-task) | `10` |
| [8. Customize training: hooks, checkpoints, resuming](#8-customize-training-hooks-checkpoints-resuming) | `11`, `02` |
| [9. Train on several GPUs](#9-train-on-several-gpus) | `13` |

The mental model behind all of them (see [Concepts](concepts.md)):

```text
Dataset -> Sample -> Transform(s) -> collate (chosen by the Task) -> batch
batch -> Task.run_step(model, batch) -> TaskStepOutput(loss, metrics, predictions)
Trainer: loop over batches, backward, optimizer step, and call the Hooks
```

A **model** is a composition (Backbone -> Neck(s) -> Head, or an
`Architecture` preset). A **Task** knows what is being predicted (targets,
loss, metrics, decoding). The **Trainer** knows nothing about either.

---

## 1. Train an image classifier

*Scripts: `01_quickstart_classification.py`, `05_full_training_pipeline.py`*

```python
model = build_sequential_model(
    backbone="resnet18",
    necks=["global_avg_pool"],
    head=("vanilla_classification_head", {"num_classes": 10}),
)
task = Task.create("classification_task")
trainer = Trainer(model, task, Optimizer.create("adamw", model, lr=1e-3), device="cuda", precision="bf16")
trainer.train(build_dataloader(train_set, task=task, batch_size=64, shuffle=True), max_iter=10_000)
```

- Every piece is created by name, and `("name", {...overrides})` changes
  any field of its config. `Backbone.describe("resnet18")` lists the fields.
- `build_dataloader` picks the collate function the Task declares, so
  classification stacks tensors, detection pads images, and SSL collates
  views. You never have to pass a `collate_fn` yourself.
- `05` adds everything a real run needs: augmentation, a warmup-cosine
  schedule (`total_steps` must equal `max_iter`), periodic evaluation,
  best-model checkpoints, and early stopping. All of these are hooks.

**With real data:** write a `Dataset` whose `__getitem__` returns
`Sample(image=<C,H,W tensor>, target={"labels": <int tensor>})` and runs
it through a `Transform` (usually `Compose([...])`). Start the pipeline with
`to_float` if your images are stored as integers.

## 2. Work with multi-band astronomical images

*Script: `08_astronomy_multiband.py`*

Survey cutouts are not 8-bit RGB. They have N bands, fluxes far above 1
and below 0 (after background subtraction), or raw 16-bit counts. Optastra's
transforms are written for that:

- `to_float` is dtype-aware: uint8 and uint16 are scaled to [0, 1], and
  float fluxes pass through unchanged.
- Geometric transforms (rotations by any angle, flips, crops) work for any
  channel count. Photometric ops infer each image's value range, or take
  `value_range=(lo, hi)` explicitly, and never clip to [0, 1].
- Any backbone accepts the band count: `("resnet18", {"in_channels": 5})`.
- Domain-specific preprocessing is one small registered transform. The
  script adds a per-band `asinh_stretch`, the standard way to compress
  astronomical dynamic range, and uses it like any built-in transform.

**With real data:** decide on a normalization first. A per-band asinh
stretch works well for most imaging, and fixed per-band scales are best for
calibrated fluxes. Then choose augmentations that respect the physics:
rotations and flips are almost always safe for extragalactic sources,
colour-changing ops usually are not.

## 3. Pretrain without labels, then fine-tune

*Script: `06_ssl_pretrain_then_finetune.py`*

The typical astronomy situation is millions of unlabelled cutouts and a few
thousand labels. Self-supervised pretraining learns features from the
unlabelled images, and fine-tuning adapts them to the labels.

```python
algo = Algorithm.create("byol", backbone="resnet50")    # or "simclr"; any registered backbone
model = algo.build_model()                              # backbone + projector (+ predictor, EMA target)
views = MultiViewTransform(augment, n_views=2)          # each Sample -> two augmented views
trainer = Trainer(model, algo, optimizer, hooks=algo.hooks() + [CheckpointHook(out_dir, save_every=5000)])
...
export_backbone(Checkpointer(out_dir).latest(), "backbone.pt")       # just the backbone weights
model = build_sequential_model(("resnet50", {"weights": "backbone.pt"}), ["global_avg_pool"], head)
```

- `algo.hooks()` is required: it updates BYOL's EMA target network after
  every optimizer step and saves its schedule in checkpoints.
- `weights=` also accepts a full training checkpoint directly. The
  backbone's key prefix (`online_backbone.`, `backbone.`, `0.`, and
  `torch.compile`'s `_orig_mod.`) is detected automatically.
- To freeze the pretrained backbone for a linear probe, add
  `FreezeBackboneHook()`.

## 4. Detect objects

*Script: `07_object_detection.py`*

```python
model = Architecture.create("faster_rcnn_r18_fpn", num_classes=2)
task = Task.create("detection_task", num_classes=2)
loader = build_dataloader(dataset, task=task, batch_size=4, collate_kwargs={"size_divisibility": 32})
```

- A detection sample is `Sample(image, target={"boxes": (N, 4) XYXY pixels,
  "labels": (N,) in 0..C-1})`. Masks for Mask R-CNN go in
  `target["masks"]` as (N, H, W).
- Geometric augmentations move boxes and masks with the image, and boxes
  that leave the image are dropped along with their labels.
- Images of different sizes are padded into one batch. `image_sizes`
  records each original size, so predictions are clipped per image.
- `task.run_step(model, batch, stage="predict").predictions` is one `Sample`
  per image, with `boxes`, `scores` and `labels` after NMS.
- An `Architecture` is a tree of references: backbone, neck, RPN, ROIAlign
  and box head can each be replaced by name, e.g.
  `Architecture.create("faster_rcnn_r50_fpn", backbone=("convnext_tiny", {}))`.

**With real data:** `CocoDetectionDataset(annotation_json, image_root,
transform, channels="rgb")` reads COCO-format annotations, including RLE
masks. Detection only reports `pred_fg_ratio` during evaluation so far; a
COCO mAP evaluator is planned.

## 5. Compare backbones and keep configs in YAML

*Scripts: `12_compare_backbones.py`, `03_experiment_config_yaml.py`*

Because models are compositions, a comparison study is a loop over
backbone references, with everything else held fixed:

```python
for backbone in ["resnet18", "efficientnet_b0", "convnext_tiny", ("vit_tiny", {"img_size": 32, "patch_size": 4})]:
    model = build_sequential_model(backbone, ["global_avg_pool"], head)
    ...
```

`global_avg_pool` works for CNNs and ViTs alike, because ViTs also expose
their patch tokens as a spatial feature map. `ExperimentConfig` records the
architecture, task, optimizer and scheduler of a run and round-trips them
through YAML, so a study can be described in files instead of code. Data,
trainer and hook settings are not part of `ExperimentConfig` yet (planned).

## 6. Add your own backbone

*Script: `09_custom_backbone.py`*

A backbone takes images, returns `FeatureMaps`, and declares
`self.out_spec`, the channels and stride of each feature level. That spec
is the contract: FPN, pooling necks, heads and detectors are built from it,
and a mismatch fails when the model is built, with a message saying what
is missing.

- Build from `optastra.nn.blocks` (`ConvNormAct`, `ResidualBlock`,
  `SqueezeExcitation`, `TransformerBlock`, ...) rather than writing layers
  from scratch.
- Name spatial levels `C{log2(stride)}` (C2 = stride 4, ...). That is what
  lets detectors and FPN use them unchanged.
- Register a family of variants in a loop with
  `Backbone.register(fn, config=cfg, name="mynet_s")`.

The permanent home for a new backbone is `optastra/backbones/<name>.py`.
[Extending](extending.md#recipe-add-a-new-backbone) has the checklist.

## 7. Predict continuous values, and add your own task

*Script: `10_regression.py`*

```python
model = build_sequential_model("resnet18", ["global_avg_pool"], ("vanilla_regression_head", {"num_outputs": 2}))
task = Task.create("regression_task", loss="huber")        # "mse" (default) | "l1" | "huber"
```

- Targets are `Sample.target["values"]`: a scalar per image, or a
  `(num_outputs,)` tensor when you predict several quantities (e.g.
  redshift and stellar mass).
- `trainer.evaluate()` reports dataset-level `mae`, `rmse` and `r2`. R² is
  computed over the whole validation set, because it can't be averaged from
  per-batch values.

A new *kind of prediction* (multi-label, segmentation, ...) is a new Task.
`optastra/tasks/regression.py` is written as a compact template:

- a **Task** implementing the `run_step` pieces (`split_inputs_targets`,
  `preprocess_targets`, `compute_losses`, `compute_metrics`,
  `decode_predictions`, ...), declaring its collate and the `HeadOutput`
  fields it needs;
- an **Evaluator** (`reset` / `process` / `summarize`) for dataset-level
  metrics. `Trainer.evaluate()` feeds every batch to the Task's evaluator;
- a matching **Head** if no existing one produces the right `HeadOutput`
  field (here `heads/regression.py`).

The model, Trainer and hooks need no changes.

## 8. Customize training: hooks, checkpoints, resuming

*Scripts: `11_hooks_checkpoint_resume.py`, `02_custom_transform.py`*

- A hook is a class with any of `before_train`, `before_step`,
  `after_step`, `after_eval`, ... It reads `state` (model, optimizer,
  current batch, iteration) and writes scalars to `state.storage`. Those
  scalars appear in the console log and in `logs/metrics.jsonl` without any
  other change.
- `state.should_stop = True` ends training cleanly. Early stopping works
  this way, and so does the simulated crash in `11`.
- Checkpoints contain model, optimizer, every hook's state (scheduler,
  early stopping, BYOL schedule, ...) and the RNG state. Resuming with
  `ResumeHook` from an epoch-boundary checkpoint reproduces an
  uninterrupted run exactly, and `11` checks this bit for bit.
- The Trainer owns runtime policy: `precision="bf16" | "fp16"`,
  `grad_accum_steps=N` (one iteration = one optimizer step over N
  batches), and `clip_grad_norm=`.
- `02` shows the transform side of customization: a new augmentation in
  about 15 lines, drawing its randomness from the seedable
  `optastra.transforms.rng` generator.

## 9. Train on several GPUs

*Script: `13_multi_gpu_ddp.py`*

Data-parallel training (PyTorch DDP) needs no code changes: launch the same
script with `torchrun` instead of `python`.

```bash
python   train.py                          # 1 GPU
torchrun --nproc_per_node=2 train.py       # 2 GPUs on this node, one process each
```

On a SLURM cluster, for example one node with 2 A100s:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # one launcher; torchrun starts the 2 workers
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16           # for DataLoader workers (num_workers per process)
srun torchrun --standalone --nproc_per_node=2 train.py
```

What happens under torchrun:

- Each process takes one GPU, trains on a disjoint shard of the data
  (`build_dataloader` sets that up), and gradients are averaged across
  processes every optimizer step.
- `batch_size` is per GPU: with 2 GPUs each optimizer step sees twice the
  data. Either scale the learning rate with the global batch (as `13`
  does), or halve `batch_size` to keep the old global batch. `max_iter`
  still counts optimizer steps, so an epoch takes half as many iterations.
- Evaluation is sharded too, and the results are combined exactly, so
  every process sees the same `val_*` metrics and early stopping / best
  checkpoints agree.
- Checkpoints, `metrics.jsonl` and console logs are written once, by rank 0.
  Custom hooks that only write or print should set `main_process_only = True`.
- Use `optastra.core.distributed` for anything you do yourself:
  `is_main_process()` before printing or saving, `broadcast_object(...)` to
  share a value picked on rank 0 (e.g. a run directory), `barrier()`.

**Getting the most out of A100s:**

- `precision="bf16"`: A100s run bf16 natively, and bf16 needs no loss scaling.
- `torch.set_float32_matmul_precision("high")` at the top of the script
  lets the remaining fp32 matrix multiplies use TF32.
- Feed the GPUs: `build_dataloader(..., num_workers=8, pin_memory=True,
  persistent_workers=True)` per process. If `data_time` in the logs is a
  large part of `iter_time`, data loading is the bottleneck.
- `sync_batchnorm=True` when the per-GPU batch is small (e.g. detection),
  so BatchNorm statistics use the whole global batch.
- Leave `find_unused_parameters` off (it slows every step) unless DDP stops
  with an error about parameters that received no gradient -- then a module
  is being skipped in some forward passes, and the flag handles it.

To test the setup without GPUs: `torchrun --nproc_per_node=2` on a CPU-only
machine runs the same code with CPU processes (example `13` does this
automatically when there are fewer GPUs than processes).
