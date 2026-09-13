# Core Concepts

## The building blocks

```text
Backbone            images -> FeatureMaps            (feature extraction only)
Neck                FeatureMaps -> FeatureMaps        (e.g. pooling, FPN)
Head                FeatureMaps -> HeadOutput         (task-specific prediction)
Task                owns losses/metrics/target formatting for a Head's output
Algorithm            a Task specialized for self-supervised pretraining
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
to compare views -- not a new model class.

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
     merges `**overrides` onto the registered default config (a
     dataclass, via `dataclasses.replace`), and calls the entrypoint.
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
- **Describing** (`to_yaml()`, `print_experiment()`) needs to recurse
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
and round-trips through YAML (`to_yaml` / `from_yaml`).
`build_experiment_from_config` resolves every `ComponentRef` and
constructs the real objects.

## Trainer and hooks

`Trainer.train(dataloader, max_iter)` only ever calls
`task.run_step(model, batch, stage)` and reads the generic
`TaskStepOutput` fields (`loss`, `losses`, `metrics`) -- it contains no
model- or task-specific logic. Behavior is added entirely through
`Hook` subclasses (`optastra/training/hooks/`) with lifecycle methods
(`before_step`, `after_step`, `before_eval`, ...); `default_hooks(...)`
returns a sensible starting set (checkpointing, logging, resume).
