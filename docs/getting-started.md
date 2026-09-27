# Getting Started

## Install

```bash
git clone https://github.com/Aureusa/optastra.git
cd optastra
pip install -e .
```

For running the test suite or building docs locally:

```bash
pip install -e ".[dev]"     # pytest
pip install -e ".[docs]"    # mkdocs + mkdocstrings
```

## First import

```python
import optastra
```

That's it -- importing `optastra` registers every built-in backbone,
neck, head, task, algorithm, architecture, optimizer, scheduler,
transform, proposal generator, and region extractor. There is no
separate "plugin registration" step to remember.

## Discover what's available

```python
optastra.Backbone.list_all()               # ["alexnet", "resnet18", "resnet50", "vit_base", ...]
optastra.Transform.list_all(filter="crop") # ["crop", "random_crop", "random_resized_crop"]
optastra.Backbone.describe("resnet50")     # prints the default config field-by-field
```

## Build your first model

A model is a composition of a `Backbone` -> zero or more `Neck`s -> a
`Head`. `build_sequential_model` wires the `FeatureSpec` between stages
for you:

```python
import torch
from optastra import build_sequential_model, Optimizer

model = build_sequential_model(
    backbone="resnet18",
    necks=["global_avg_pool"],
    head=("vanilla_classification_head", {"num_classes": 10}),
)
optimizer = Optimizer.create("adamw", model, lr=1e-3)

images = torch.randn(4, 3, 64, 64)
labels = torch.randint(0, 10, (4,))

output = model(images)  # HeadOutput(logits=...)
loss = torch.nn.functional.cross_entropy(output.logits, labels)
loss.backward()
optimizer.step()
```

Every argument that names a component (`backbone="resnet18"`, an entry
in `necks=[...]`, `head=(...)`) accepts a plain string, a
`(name, overrides)` tuple, a `{"name": ..., "overrides": ...}` dict, or a
`ComponentRef` directly -- see [Concepts](concepts.md#componentref) for
why.

Run it yourself:

```bash
python examples/01_quickstart_classification.py
```

## Train for real: Task + Trainer

A `Task` owns losses, metrics, and how a batch is split into inputs and
targets; the `Trainer` only ever calls `task.run_step(...)` and never
contains model-specific logic. The Trainer owns runtime policy instead:
`precision` (`"fp32"` default, `"bf16"`, `"fp16"`), `grad_accum_steps`
and `clip_grad_norm`. See
[`examples/05_full_training_pipeline.py`](../examples/05_full_training_pipeline.py)
for a full training loop wiring `Trainer`, hooks (`default_hooks`,
`EvalHook`, `SchedulerHook`, ...), and a `DataLoader` built with
`build_dataloader`, and [Concepts](concepts.md#trainer-and-hooks) for how
evaluation results reach storage and `metrics.jsonl`.

## Next steps

- [Tutorials](tutorials.md): scenario walkthroughs (astronomy data, SSL
  pretraining, detection, comparing backbones, custom tasks and hooks),
  each backed by a runnable example.
- [Concepts](concepts.md): how Registry -> Factory -> ComponentRef fit
  together, and the abstractions (`Backbone`, `Neck`, `Head`, `Task`,
  `Algorithm`, `Architecture`, `Trainer`).
- [Extending](extending.md): add your own backbone, transform, task, or
  hook.
- [`examples/`](../examples/): more runnable scripts.
