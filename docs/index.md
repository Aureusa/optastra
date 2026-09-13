# Optastra

**A modular computer vision framework built on PyTorch, focused on
composable architectures, clean implementations, and research
reproducibility.**

## Why Optastra

Most computer vision codebases fall into one of three buckets:
research code that ships with a single paper, deployment-oriented
libraries, or large ecosystems that trade experimentation speed for
stability. Optastra targets the gap in between: a framework where
**implementing a new architecture, pretraining algorithm, or task is a
matter of writing one new file that plugs into everything else**, not
rewriting a training loop.

It grew out of computer vision work in astronomy, where datasets from
different surveys differ enough (wavelength, resolution, PSF, noise,
preprocessing) that pretrained representations often don't transfer
cleanly -- which makes fast, correct experimentation with new backbones,
tasks, and pretraining objectives more valuable than a large model zoo.
The framework itself is domain-independent; astronomy is the motivating
use case, not a limitation.

## The 60-second pitch

Everything in Optastra is a **registered component** built by a
**Factory**:

```python
import optastra

backbone = optastra.Backbone.create("resnet50")
transform = optastra.Transform.create("rand_augment", num_ops=2, magnitude=9)
optimizer = optastra.Optimizer.create("adamw", backbone, lr=1e-4)
```

Models are compositions, not monoliths:

```python
model = optastra.build_sequential_model(
    backbone="resnet18",
    necks=["global_avg_pool"],
    head=("vanilla_classification_head", {"num_classes": 10}),
)
```

And full experiments are configuration, not code:

```python
cfg = optastra.ExperimentConfig(
    architecture=optastra.ComponentRef("faster_rcnn_r50_fpn"),
    task=optastra.ComponentRef("detection_task"),
    optimizer=optastra.ComponentRef("adamw", {"lr": 1e-4}),
)
built = optastra.build_experiment_from_config(cfg)
```

Continue to [Getting Started](getting-started.md) to run this yourself,
or [Concepts](concepts.md) to see how the pieces fit together.
