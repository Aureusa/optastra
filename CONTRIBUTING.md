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
subclasses. `Algorithm` subclasses `Task` to reuse its step logic but has
its own registry: SSL methods register with `@Algorithm.register`, tasks
with `@Task.register`.

The registry key is the decorated function's name (`foo` above) -- the
name users pass to `MyFamilyBase.create("foo")` -- so name it what a user
would type, not what the class is called. To register a table of variants
in a loop, pass the key explicitly instead:

```python
for variant, cfg in foo_configs.items():
    MyFamilyBase.register(foo, config=cfg, name=variant)
```

Registration never touches the module's namespace: every component module
declares its own `__all__` (class, config, and factory function(s)), which
is what the family's `from .my_file import *` re-exports.

## Where things go

- New component -> its family's directory (`optastra/backbones/`,
  `optastra/transforms/`, ...), with an explicit `__all__`, re-exported via
  `from .my_file import *` in that family's `__init__.py`.
- Tests -> `tests/<family>/test_<component>.py`, mirroring the existing
  layout (see `tests/backbones/test_resnet.py`).
- Runnable, standalone usage demonstrations -> `examples/`, not `tests/`.
  Examples must run on CPU with synthetic data -- no dataset downloads,
  no GPU requirement, no network access.

## Config dataclasses, not raw dicts

Every registered component's tunable parameters live in a `@dataclass`,
never a bare `dict`. This is what lets `Factory.create()` merge
`**overrides` into a deep copy of the default via `dataclasses.replace`,
and what lets configs be dumped to YAML and loaded back.

Config values must be plain data -- strings, numbers, bools, lists, tuples,
dicts, nested dataclasses or `ComponentRef`s. Never store a class or a
callable in a config (select it by name instead, e.g.
`ResNetConfig.block = "bottleneck"`); `tests/core/test_config_yaml.py`
checks every registered default config for this automatically.

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

Shape tests are not enough on their own: anything that computes a number
(a loss, a box, a learning rate, an EMA update, a transformed coordinate)
also needs a test that checks the *value* against a hand-computed or
reference result. Several past bugs -- misaligned RPN anchors, weight decay
silently set to 0, a BYOL target that never moved -- passed every
shape/key test.

Some checks run automatically for everything registered:
`tests/core/test_config_yaml.py` (every default config is YAML-safe) and
`tests/backbones/test_out_spec_contract.py` (every backbone's measured
feature channels/strides match its `out_spec`, alone and under FPN).

## Docs

If you add or change public API, update the relevant page under `docs/`
(`docs/concepts.md` for new abstractions, `docs/extending.md` for new
recipes) rather than only relying on `docs/api.md`'s auto-generated
docstrings.
