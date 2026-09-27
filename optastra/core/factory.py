from __future__ import annotations
import copy
from dataclasses import replace, fields, asdict
from typing import Any, ClassVar, Generic, TypeVar

from .registry import FamilyRegistry

T = TypeVar("T")


FACTORIES: dict[str, type] = {}


class Factory(Generic[T]):
    """
    Shared create/describe/config/list machinery for any registry-backed
    component family with no upstream wiring (Backbone, Task, Algorithm).
    Subclasses only need to set `_registry` and, optionally, override
    `_post_create` to validate the built instance.

    Components are normally registered with a default config dataclass and
    built as `entrypoint(cfg)`. A component registered without a config is
    "configless": it is built as `entrypoint()` and accepts no overrides.
    """

    _registry: ClassVar[FamilyRegistry]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        registry = cls.__dict__.get("_registry")   # only if THIS class sets it
        if registry is None:
            return   # intermediate/abstract subclasses (e.g. SpecFactory itself) don't register
        if registry.family in FACTORIES and FACTORIES[registry.family] is not cls:
            raise ValueError(
                f"Family '{registry.family}' already registered by "
                f"{FACTORIES[registry.family].__name__}, cannot also register {cls.__name__}."
            )
        FACTORIES[registry.family] = cls

    @classmethod
    def _check_registered(cls, name: str) -> None:
        if not cls._registry.is_registered(name):
            raise ValueError(f"{cls._registry.family} '{name}' is not registered.")

    @classmethod
    def register(cls, fn=None, *, config: Any | None = None, name: str | None = None):
        """
        Register a component under this family's registry. Use directly on
        the family base class -- no need to import a registry module or a
        separate register_x function:

            @Transform.register
            def my_transform(): ...

            @Backbone.register(config=MyBackboneConfig())
            def my_backbone(cfg): ...

        `name` overrides the registry key (default: the function's __name__),
        which lets a table of variants be registered in a loop:

            for variant, cfg in my_configs.items():
                Backbone.register(MyBackbone, config=cfg, name=variant)
        """
        def decorator(inner_fn):
            return cls._registry.register(inner_fn, default_config=config, name=name)
        return decorator(fn) if fn is not None else decorator

    @classmethod
    def _build_cfg(cls, name: str, overrides: dict) -> Any:
        """Registry default + overrides, as a fresh deep copy: mutating a built
        component's cfg (e.g. `model.cfg.layers.append(...)`) can never leak
        back into the registered default."""
        default_cfg = cls._registry.get_default_config(name)
        if default_cfg is None:
            if overrides:
                raise TypeError(
                    f"{cls._registry.family} '{name}' has no config, so it takes no overrides "
                    f"(got {sorted(overrides)})."
                )
            return None
        return replace(copy.deepcopy(default_cfg), **overrides)

    @classmethod
    def create(cls, name: str, **overrides) -> T:
        cls._check_registered(name)
        entrypoint = cls._registry.get_entrypoint(name)
        cfg = cls._build_cfg(name, overrides)
        instance = entrypoint(cfg) if cfg is not None else entrypoint()
        return cls._post_create(instance)

    @classmethod
    def _post_create(cls, instance: T) -> T:
        """
        Optional hook for subclasses to validate the created instance.
        """
        return instance

    @classmethod
    def describe(cls, name: str) -> None:
        cfg = cls._registry.get_default_config(name)
        print(f"{name}:")
        if cfg is None:
            print("  (no config)")
            return
        for f in fields(cfg):
            print(f"  {f.name}: {f.type} = {getattr(cfg, f.name)!r}")

    @classmethod
    def get_default_config(cls, name: str) -> Any:
        """A deep copy of the registered default config (None if configless)."""
        return copy.deepcopy(cls._registry.get_default_config(name))

    @classmethod
    def list_all(cls, module: str | None = None, filter: str | None = None) -> list[str]:
        """
        List all registered components in this family, optionally filtering by module and/or substring.
        """
        return cls._registry.list_component(module=module, filter=filter)


class SpecFactory(Factory[T]):
    """
    Same as Factory, but entrypoint(in_spec, cfg) -- for anything wired
    against an upstream FeatureSpec (Neck, Head, ProposalGenerator,
    RegionExtractor). Accepts either a FeatureSpec or a module exposing
    .out_spec, resolved once here for every subclass.
    """
    @staticmethod
    def _validate_in_spec(in_spec: Any) -> None:
        from ..nn.features import FeatureSpec
        if not isinstance(in_spec, FeatureSpec):
            raise TypeError(f"'in_spec' must be a FeatureSpec, got {type(in_spec)}.")

    @classmethod
    def create(cls, name: str, in_spec, **overrides) -> T:
        cls._validate_in_spec(in_spec)
        cls._check_registered(name)
        entrypoint = cls._registry.get_entrypoint(name)
        cfg = cls._build_cfg(name, overrides)
        instance = entrypoint(in_spec, cfg) if cfg is not None else entrypoint(in_spec)
        return cls._post_create(instance)


def list_all_registered_components(module: str | None = None, filter: str | None = None, family: str | None = None) -> dict[str, list[str]]:
    """
    List all registered components in every family, optionally filtering by module and/or substring.
    """
    if family is not None:
        cls = FACTORIES.get(family)
        if cls is None:
            return {}
        return {family: cls.list_all(module=module, filter=filter)}

    result = {}
    for family, cls in FACTORIES.items():
        r = cls.list_all(module=module, filter=filter)
        if r:
            result[family] = r
    return result


def list_all_registered_families() -> list[str]:
    """
    List all registered component families.
    """
    return list(FACTORIES.keys())


def get_component_parameters(name: str, family: str | None = None) -> dict[str, Any]:
    """
    Get the default config parameters for a registered component
    ({} for a configless component).
    """
    def _params(cls) -> dict[str, Any]:
        cfg = cls.get_default_config(name)
        return asdict(cfg) if cfg is not None else {}

    if family is not None:
        cls = FACTORIES.get(family)
        if cls is None:
            raise ValueError(f"Family '{family}' is not registered.")
        return _params(cls)

    for family, cls in FACTORIES.items():
        if cls._registry.is_registered(name):
            return _params(cls)

    raise ValueError(f"Component '{name}' is not registered in any family.")
