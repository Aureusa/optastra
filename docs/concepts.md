# Core Concepts

## The building blocks

```text
Backbone            images -> FeatureMaps            (feature extraction only)
Neck                FeatureMaps -> FeatureMaps        (e.g. pooling, FPN)
Head                FeatureMaps -> HeadOutput         (task-specific prediction)
Task                owns losses/metrics/target formatting for a Head's output
Algorithm            a self-supervised objective: builds its model from config, owns per-step
                     state (EMA teacher, schedules); own registry, reuses Task's step logic
Architecture         a full model assembled from Backbone/Neck/Head/etc.
Trainer               orchestrates model + task + optimizer + hooks
```

Each arrow is a real, structured type, not a bare tensor:

- **`FeatureSpec`** (construction-time) describes *what a stage produces*
  -- channels/strides for CNN feature maps, `embed_dim` for a pooled
  embedding, `num_tokens` for patch tokens. Downstream components call
  `in_spec.require("embed_dim")` and fail loudly at construction time if
  what they need isn't there, instead of guessing or crashing deep in a
  forward pass.
- **`FeatureMaps`** (runtime) is the tensor-carrying twin of
  `FeatureSpec`.
- **`HeadOutput`** is what every `Head` returns (`logits`, `values`,
  `boxes`, `scores`, `masks`, `embedding`, plus an `extra` escape hatch).
  A `Task` reads only the fields it needs and validates the rest are
  present via `required_fields`.

This is what makes `Backbone -> Neck -> Head` composition work for both
CNNs and transformers without every combination needing bespoke glue
code: a `GlobalPool` neck turns a ResNet's spatial feature maps into the
same `embed_dim`-carrying `FeatureMaps` that a ViT's CLS token would
also produce, so `ClassificationHead` doesn't care which backbone it's
attached to.

**Architecture** vs. **Algorithm**: a `ViT` is an architecture; `SimCLR`
is a pretraining algorithm that trains one. Keeping them separate means
a new self-supervised method only has to define what a "view" is and how
to compare views -- not a new model class. The backbone is a ComponentRef
in the algorithm's config, and the pretrained weights flow back into any
downstream model through `Backbone.create(name, weights=...)`:

```python
algo = Algorithm.create("byol", backbone="resnet50")   # or ("vit_small", {...})
model = algo.build_model()
trainer = Trainer(model, algo, optimizer, hooks=algo.hooks() + [CheckpointHook("runs/byol")])
# multi-view data: Transform pipeline wrapped in MultiViewTransform(pipeline, n_views=2)

export_backbone("runs/byol/ckpt_1000.pt", "byol_r50.pt")   # optional, strips the prefix
clf = build_sequential_model(("resnet50", {"weights": "byol_r50.pt"}), ["global_avg_pool"], head)
```

`algo.hooks()` must be passed to the Trainer: it runs
`algo.after_optimizer_step(state)` after every step (BYOL's EMA target
update) and stores the algorithm's schedule state in checkpoints.

## Registry -> Factory -> ComponentRef

Every family (backbones, necks, heads, tasks, ...) follows the same
two-piece pattern:

```text
FamilyRegistry      "backbone" -> {name -> (entrypoint, default_config)}
Factory subclass    Backbone.create(name, **overrides), Backbone.register(...)
```

1. **`FamilyRegistry`** (`optastra/core/registry.py`) is a plain
   name -> entrypoint map, one per family, created once as a class
   attribute on that family's base class (`Backbone._registry`,
   `Transform._registry`, ...).
2. **`Factory`** (`optastra/core/factory.py`) is the shared
   `create()` / `register()` / `describe()` / `list_all()` machinery
   every family's base class (`Backbone`, `Task`, `Optimizer`, ...)
   inherits:
   - `create(name, **overrides)` looks up the registered entrypoint,
     merges `**overrides` onto a deep copy of the registered default
     config (a dataclass, via `dataclasses.replace`), and calls the
     entrypoint -- so mutating a built component's `cfg` never changes
     the registered default.
   - `register` is how a new component joins the family -- call it
     directly on the base class, no separate registry module to import:
     ```python
     @Backbone.register(config=MyBackboneConfig())
     def my_backbone(cfg: MyBackboneConfig) -> MyBackbone:
         return MyBackbone(cfg)
     ```
3. **`SpecFactory`** is `Factory` plus one extra required argument:
   `in_spec: FeatureSpec`. Use it for anything that consumes an
   upstream feature spec -- `Neck`, `Head`, `ProposalGenerator`,
   `RegionExtractor`. Everything else (`Backbone`, `Task`, `Algorithm`,
   `Architecture`, `Optimizer`, `Scheduler`, `Transform`) has no
   upstream wiring and uses plain `Factory`.

## ComponentRef: nested, describable component trees

Some configs need to name *other* registered components as fields --
e.g. `FastRCNNConfig.backbone` names a `Backbone`, `DetectionTaskConfig.
criterion` names a `DetectionCriterion`, which itself has matcher/sampler
fields naming further components. `ComponentRef` (`optastra/core/
component_ref.py`) is a plain `(name, overrides)` pair for exactly this:

```python
@dataclass
class FastRCNNConfig(ComponentRefConfigMixin):
    backbone: ComponentRef = component_field(Backbone, default_name="resnet50")
```

Two things happen with a `ComponentRef` field, and they're deliberately
separate:

- **Building** is explicit, in the component's own `__init__` -- no
  indirection, because the call site already knows exactly which
  Factory it needs:
  ```python
  self.backbone = cfg.backbone.resolve(Backbone)
  ```
- **Describing** (`print_experiment()` / `resolve_experiment()`) needs to recurse
  through arbitrarily nested `ComponentRef`s -- e.g. dumping a
  `detection_task` also expands its `criterion`'s `roi_matcher`,
  `roi_sampler`, and so on, several families deep -- without every
  architecture hand-writing its own recursive dump logic. That generic
  walk (`_serialize_config`) needs to know which `Factory` resolves each
  field *without* an `__init__` around to ask, which is what
  `component_field(SomeFactory, ...)`'s metadata is for. This is the one
  place the metadata is read; `resolve_config()` on `ComponentRef`
  expands a ref's overrides against its factory's registered default so
  the resulting YAML is the complete, reproducible, effective config --
  not just the deltas you typed.

Friendly shorthands (`"resnet50"`, `("resnet50", {"stem_channels": 32})`,
`{"name": "resnet50", "overrides": {...}}`) are coerced to a real
`ComponentRef` by `coerce_to_ref()` -- used automatically for dataclass
fields via `ComponentRefConfigMixin.__post_init__`, and called explicitly
wherever a plain function needs the same ergonomics (e.g.
`build_sequential_model`).

## ExperimentConfig

`ExperimentConfig` (`optastra/core/experiment.py`) ties an
`Architecture`, `Task`, `Optimizer`, and optional `Scheduler` together
and round-trips through YAML (`to_yaml` / `from_yaml`). The YAML stores
each `ComponentRef` -- including refs nested inside another ref's
overrides -- as `{name, overrides}` (the deltas you typed, via
`config_to_plain` / `config_from_plain`), so loading it back gives an
equal `ExperimentConfig`; use `print_experiment()` to see the fully
expanded effective config. Config values must be plain data (strings,
numbers, lists, dicts) -- e.g. `ResNetConfig.block` is `"basic"` or
`"bottleneck"`, not a class.
`build_experiment_from_config` resolves every `ComponentRef` and
constructs the real objects.

Not in `ExperimentConfig` yet (planned): the dataset and transforms,
trainer settings (`precision`, `grad_accum_steps`, eval period), hooks,
SSL algorithms, and seeding -- `seed`, `max_iter`, `batch_size` and
`output_dir` are stored but not yet consumed by anything.

## Trainer and hooks

`Trainer.train(dataloader, max_iter)` only ever calls
`task.run_step(model, batch, stage)` and reads the generic
`TaskStepOutput` fields (`loss`, `losses`, `metrics`) -- it contains no
model- or task-specific logic. Behavior is added entirely through
`Hook` subclasses (`optastra/training/hooks/`) with lifecycle methods
(`before_epoch`, `before_step`, `after_step`, `before_eval`, ...);
`default_hooks(...)` returns a sensible starting set (checkpointing,
logging, resume). Hooks run in `Hook.priority` order (lower first), not
registration order.

### Runtime policy lives in the Trainer, not the Task

A `Task` is pure: targets -> forward -> losses / metrics / decoding. How a
step is *executed* is configured on the `Trainer`:

```python
trainer = Trainer(
    model, task, optimizer, device="cuda",
    precision="bf16",       # "fp32" (default) | "bf16" | "fp16" (adds a GradScaler)
    grad_accum_steps=4,     # one iteration = one optimizer step over 4 batches
    clip_grad_norm=1.0,     # pre-clip norm logged to storage as "grad_norm"
)
```

- `precision` wraps `task.run_step` in `torch.autocast` on the model's
  device type (CPU or CUDA), for both training and `evaluate()`.
- With `grad_accum_steps=N`, each micro-batch loss is scaled by `1/N`, so
  the accumulated gradient equals that of one N-times-larger batch.
  `max_iter`, `state.iter`, schedulers and checkpoints all count
  optimizer steps; `before_step` runs once per micro-batch (with
  `state.micro_step`), `after_step` once per optimizer step.
- An epoch ends when the loader is exhausted: `after_epoch` fires just
  before the next epoch's first batch is fetched, followed by
  `before_epoch`.

### Multiple GPUs (DDP)

Launch the same script with `torchrun --nproc_per_node=N` instead of
`python` and it trains data-parallel on N GPUs -- one process per GPU,
nothing to change in the code:

- `Trainer(..., distributed=None)` detects torchrun (`distributed=True` /
  `False` forces it), joins the process group
  (`optastra.core.distributed.init_distributed`), puts process `LOCAL_RANK`
  on GPU `LOCAL_RANK`, and wraps the model in `DistributedDataParallel` when
  `train()` starts -- after the `before_train` hooks, so resuming and
  freezing act on the plain model. `state.model` stays the plain module:
  hooks, checkpoints and `evaluate()` never see the wrapper.
- `build_dataloader` gives each process a disjoint shard through a
  `ShardedSampler` (no padding, no dropped samples; reshuffled every epoch
  with a seed shared by all processes). `batch_size` is per process, so
  the global batch -- and usually the learning rate -- scales with N.
- Gradients are averaged across processes at every optimizer step (once
  per step under gradient accumulation); logged losses are the mean over
  processes.
- `evaluate()` is sharded as well: each process evaluates its shard and the
  task's Evaluator combines them (`Evaluator.sync`), so every process gets
  the same, exact dataset-level metrics -- and hooks that act on them
  (early stopping, best checkpoints) make the same decision everywhere.
- Hooks with `main_process_only = True` -- checkpoint writers, the JSON
  writer, console printers, the visualizer -- run on rank 0 only;
  everything that changes training state (schedulers, EMA, evaluation,
  early stopping, resuming) runs on every process. `setup_logging` lets
  only rank 0 print INFO lines and write the log file.
- `sync_batchnorm=True` computes BatchNorm statistics over the global batch
  (GPU only; useful for small per-GPU batches); `find_unused_parameters=True`
  is needed only if some parameters take no part in some forward passes --
  DDP's error message says so when it happens.

Resuming is exact when the checkpoint was written at the end of an epoch
(a checkpoint from mid-epoch restarts that epoch's data pass).

### Evaluation

`trainer.evaluate(loader)` makes exactly one pass over the loader and
feeds every `TaskStepOutput` to the task's `Evaluator`
(`task.build_evaluator()`: `reset()` / `process(output, batch)` /
`summarize()`). The default evaluator averages `loss` and
`compute_metrics` weighted by batch size; `ClassificationTask` counts
correct / total. The results are written to `EventStorage` once, as
`val_<name>` (e.g. `val_accuracy`, `val_total_loss`), and copied to
`state.eval_results` before the `after_eval` hooks run --
`BestMetricTracker`, `EarlyStoppingHook` and `JSONWriterHook` (an
`"eval_summary"` record in `metrics.jsonl`) read them there. Per-batch
eval scalars (`val_step_*`, `eval_time`) live on the storage's separate
`"eval_iter"` axis, so they never mix into the training smoothing
windows. `EvalHook` only decides *when* `evaluate()` runs.

### Checkpoints

`Checkpointer` saves model (without `torch.compile`'s `_orig_mod.`
prefix, so checkpoints load into compiled and uncompiled models),
optimizer, iteration / epoch, every hook's `state_dict()`, and the
Python / torch RNG states, all of which `ResumeHook` restores.

### Optimizers and schedulers

`Optimizer.create(name, model, param_groups=ParamGroupConfig(...), **cfg)`
builds param groups carrying the config's `lr` and `weight_decay`; biases,
norm parameters and `pos_embed` / `cls_token`-style parameters get
`weight_decay=0`, and `lr_multipliers` scale the LR per module prefix.
`Scheduler.create("warmup_cosine", optimizer, total_steps=max_iter, ...)`
requires `total_steps`; step it with `SchedulerHook`.
