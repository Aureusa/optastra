# Extending Optastra

Every family follows the same two-piece pattern. Once you've added one
component to a family, adding another is copy-paste-rename.

## Recipe: add a new Backbone

1. Pick (or create) a file under `optastra/backbones/`, e.g.
   `optastra/backbones/my_net.py`.
2. Define a config dataclass, the module class, and a registration
   function:

   ```python
   from dataclasses import dataclass
   import torch.nn as nn

   from .base import Backbone
   from ..nn.features import FeatureMaps, FeatureSpec

   __all__ = ["MyNet"]

   @dataclass
   class MyNetConfig:
       in_channels: int = 3
       width: int = 64

   class MyNet(Backbone):
       def __init__(self, cfg: MyNetConfig):
           super().__init__()
           self.cfg = cfg
           self.stem = nn.Conv2d(cfg.in_channels, cfg.width, 3, padding=1)
           # `out_spec` is required -- Backbone._post_create checks for it.
           self.out_spec = FeatureSpec(channels={"C1": cfg.width}, strides={"C1": 1})

       def forward(self, images) -> FeatureMaps:
           return FeatureMaps(feature_maps={"C1": self.stem(images)})

   @Backbone.register(config=MyNetConfig())
   def my_net(cfg: MyNetConfig) -> MyNet:
       return MyNet(cfg)
   ```

3. Re-export it from `optastra/backbones/__init__.py`:
   `from .my_net import *` (`Backbone.register` already appended
   `"MyNet"` to that module's `__all__`, so the `*` import picks it up).
4. Use it immediately: `Backbone.create("my_net", width=128)`.
5. Add a test under `tests/backbones/` (see `test_resnet.py` for the
   shape).

The same shape applies to `Task`, `Optimizer`, `Scheduler`, `Algorithm`,
`Architecture`, and `Transform` (plain `Factory` subclasses -- no
`in_spec` argument): import the family's base class and call
`<Base>.register(...)` directly, no separate registry module needed.

## Recipe: add a new Neck / Head / ProposalGenerator / RegionExtractor

Identical, except the entrypoint and `create()` take an extra `in_spec:
FeatureSpec` first argument (these are `SpecFactory` subclasses):

```python
@Neck.register(config=MyNeckConfig())
def my_neck(in_spec: FeatureSpec, cfg: MyNeckConfig) -> MyNeck:
    return MyNeck(in_spec, cfg)
```

Call `in_spec.require("channels", "strides")` (or `"embed_dim"`, etc.)
at the top of `__init__` so a mismatched pipeline fails immediately with
a clear message instead of a confusing shape error deep in `forward`.
Set `self.out_spec` before returning -- `Neck`/`Head`/etc. all validate
it via `_post_create`.

## Recipe: add a new Transform

```python
# optastra/transforms/my_transform.py
from dataclasses import dataclass
from .base import Transform

@dataclass
class MyTransformConfig:
    probability: float = 0.5

class MyTransform(Transform):
    def __init__(self, cfg: MyTransformConfig = MyTransformConfig()):
        self.cfg = cfg

    def __call__(self, sample):
        ...  # mutate sample.image (and sample.target if it moves pixels!)
        return sample

@Transform.register(config=MyTransformConfig())
def my_transform(cfg: MyTransformConfig) -> MyTransform:
    return MyTransform(cfg)
```

Add `from .my_transform import *` to `optastra/transforms/__init__.py`.
If your transform mixes multiple samples together (CutMix/MixUp-style),
subclass `BatchTransform` instead and register with
`@BatchTransform.register` -- it operates on a collated batch dict, not
a single `Sample`.

## Recipe: add a new Task

Subclass `Task` and implement the abstract methods
(`validate_batch`, `split_inputs_targets`, `preprocess_targets`,
`forward_model`, `compute_losses`, `reduce_losses`, `compute_metrics`,
`decode_predictions`) -- see `optastra/tasks/classification.py` for the
shortest complete example. Set `required_fields` to the `HeadOutput`
fields your task needs (e.g. `("logits",)`) so a mismatched head fails
loudly via `Task.validate_predictions`.

## Recipe: add a new Hook

```python
# optastra/training/hooks/my_hook.py
from .base import Hook

class MyHook(Hook):
    def after_step(self, state):
        ...  # read state.storage / state.last_output, never state.model internals
```

Hooks are plain classes, not registry-backed -- add yours to
`optastra/training/hooks/__init__.py`'s explicit import list and pass an
instance to `trainer.register_hooks([MyHook(...)])`.

## Conventions checklist

- One file per component (or a small family of closely related ones,
  like `necks/pool.py`).
- Config is a plain `@dataclass`, never a raw dict.
- The registration function's *name* is the string users pass to
  `create()` -- name it what you want to type, not what the class is
  called (e.g. `vanilla_classification_head`, not `ClassificationHead`).
- `Factory` (no upstream wiring) vs `SpecFactory` (needs `in_spec`) --
  see [Concepts](concepts.md#registry---factory---componentref).
- Re-export new public names from the family's `__init__.py`.
- Add a test under the matching `tests/<family>/` directory.
