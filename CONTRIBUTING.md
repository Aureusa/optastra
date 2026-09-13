# Contributing to Optastra

Optastra optimizes for one thing above all: **adding a new idea should
touch as few files as possible.** Every family of components (backbones,
necks, heads, tasks, algorithms, architectures, optimizers, schedulers,
transforms, proposal generators, region extractors) follows the same
convention so that once you've added one component anywhere in the
codebase, you already know how to add the next one anywhere else.

See [`docs/extending.md`](docs/extending.md) for concrete, copy-pasteable
recipes. This document covers the conventions those recipes follow and
the project's baseline expectations.

## The registration convention

Every component family is two things:

1. **A base class** in `<family>/base.py` with a `_registry =
   FamilyRegistry("my_family")` class attribute, inheriting `Factory`
   (no upstream wiring needed -- `Backbone`, `Task`, `Optimizer`, ...) or
   `SpecFactory` (needs an upstream `FeatureSpec` -- `Neck`, `Head`,
   `ProposalGenerator`, `RegionExtractor`):

   ```python
   from optastra.core.factory import Factory
   from optastra.core.registry import FamilyRegistry

   class MyFamilyBase(Factory["MyFamilyBase"]):
       _registry = FamilyRegistry("my_family")
   ```

2. **One dataclass config + one component class + one
   `@MyFamilyBase.register`-decorated factory function per concrete
   component**, usually all in one file:

   ```python
   @dataclass
   class FooConfig:
       ...

   class Foo(MyFamilyBase):
       def __init__(self, cfg: FooConfig): ...

   @MyFamilyBase.register(config=FooConfig())
   def foo(cfg: FooConfig) -> Foo:
       return Foo(cfg)
   ```

There is no separate registry module to import -- `register` is a
classmethod every `Factory`/`SpecFactory` subclass inherits, so a
concrete component file only ever imports the base class it already
subclasses (or, for shared registries like `Algorithm`/`Task`, imports
whichever of the two is more convenient -- they resolve to the same
registry).

The string passed to `@MyFamilyBase.register(...)`'s underlying function
name (`foo` above) is the name users pass to `MyFamilyBase.create("foo")`
-- name it what a user would type, not what the class is called.

## Where things go

- New component -> its family's directory (`optastra/backbones/`,
  `optastra/transforms/`, ...), re-exported via `from .my_file import *`
  in that family's `__init__.py`.
- Tests -> `tests/<family>/test_<component>.py`, mirroring the existing
  layout (see `tests/backbones/test_resnet.py`).
- Runnable, standalone usage demonstrations -> `examples/`, not `tests/`.
  Examples must run on CPU with synthetic data -- no dataset downloads,
  no GPU requirement, no network access.

## Config dataclasses, not raw dicts

Every registered component's tunable parameters live in a `@dataclass`,
never a bare `dict`. This is what lets `Factory.create()` merge
`**overrides` via `dataclasses.replace(default_cfg, **overrides)` and
what lets `ExperimentConfig` serialize an entire experiment to YAML and
back losslessly.

## FeatureSpec discipline

If your component consumes a `FeatureSpec` (any `SpecFactory` subclass),
call `in_spec.require(...)` for whatever fields you need at the top of
`__init__`, and always set `self.out_spec` before returning. Fail at
construction time with a clear message, never at the first forward pass.

## Tests

Run the suite with:

```bash
pip install -e ".[dev]"
pytest
```

New components should have at least: a registration test (it appears in
`list_all()`), a construction test (it builds without error, `out_spec`
is set if applicable), and a forward-pass shape test.

## Docs

If you add or change public API, update the relevant page under `docs/`
(`docs/concepts.md` for new abstractions, `docs/extending.md` for new
recipes) rather than only relying on `docs/api.md`'s auto-generated
docstrings.
