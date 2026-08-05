from __future__ import annotations
from dataclasses import replace, fields
from typing import Any, ClassVar, Generic, TypeVar

from .registry import ComponentRegistry

T = TypeVar("T")


class Factory(Generic[T]):
    """
    Shared create/describe/config/list machinery for any registry-backed
    component family with no upstream wiring (Backbone, Task, Algorithm).
    Subclasses only need to set `_registry` and, optionally, override
    `_post_create` to validate the built instance.
    """

    _registry: ClassVar[ComponentRegistry]

    @classmethod
    def _check_registered(cls, name: str) -> None:
        if not cls._registry.is_registered(name):
            raise ValueError(f"{cls._registry.component_name} '{name}' is not registered.")

    @classmethod
    def _build_cfg(cls, name: str, overrides: dict) -> Any:
        default_cfg = cls._registry.get_default_config(name)
        return replace(default_cfg, **overrides) if default_cfg is not None else None

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
        return cls._registry.get_default_config(name)

    @classmethod
    def list_all(cls, module: str | None = None, filter: str | None = None) -> list[str]:
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
    