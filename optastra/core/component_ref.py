from __future__ import annotations
from dataclasses import dataclass, field, fields as dc_fields, is_dataclass, replace
from typing import Any

from .factory import Factory


def component_field(
        factory: type[Factory],
        *,
        default_name: str | None = None,
        default_overrides: dict | None = None,
        optional: bool = False,
        **kwargs
    ) -> Any:
    """
    Declares a ComponentRef field AND which Factory resolves it -- one
    place, read by resolve_component() at build time and by
    resolve_config() at describe time. They can never disagree because
    there's only one mapping, not two.
    """
    if optional and default_name is None:
        maker = lambda: None
    else:
        maker = lambda: ComponentRef(default_name or "", dict(default_overrides or {}))
    return field(default_factory=maker, metadata={"factory": factory}, **kwargs)


@dataclass
class ComponentRef:
    """Names a registered component + overrides. Carries no factory --
    the factory is context (which field this ref sits in), not data."""
    name: str
    overrides: dict[str, Any] = field(default_factory=dict)

    def resolve(self, factory: type[Factory], **extras) -> Any:
        return factory.create(self.name, **self.overrides, **extras)

    def resolve_config(self, factory: type[Factory] | None) -> dict[str, Any]:
        if factory is None or not self.name:
            return {"name": self.name, **_serialize_config(self.overrides)}
        default = factory.get_default_config(self.name)
        if default is None:
            return {"name": self.name, **_serialize_config(self.overrides)}
        merged = replace(default, **self.overrides)   # same op Factory.create already uses
        return {"name": self.name, **_serialize_config(merged)}


def resolve_component(cfg: Any, field_name: str, **extras) -> Any:
    """
    Resolves cfg.<field_name> using the Factory declared via
    component_field() on that field. Same mapping resolve_config() reads --
    this is the single source of truth for both build and describe.
    """
    value = getattr(cfg, field_name)
    if value is None:
        return None
    if not isinstance(value, ComponentRef):
        raise TypeError(
            f"{type(cfg).__name__}.{field_name} is a {type(value).__name__}, not a ComponentRef "
            f"(got {value!r}). If you're constructing a config instance, use ComponentRef(...) "
            f"for the value -- component_field(...) is only for declaring the field in the "
            f"dataclass body, never for supplying a value."
        )
    f = next(f for f in dc_fields(cfg) if f.name == field_name)
    factory = f.metadata.get("factory")
    if factory is None:
        raise ValueError(f"'{field_name}' on {type(cfg).__name__} has no declared factory; use component_field().")
    return value.resolve(factory, **extras)


def _serialize_config(value: Any) -> Any:
    """Recursively flatten dataclasses/dicts/lists to YAML-safe values.
    Dataclass fields typed as ComponentRef are resolved via their own
    declared factory metadata -- no separate inference needed."""
    if isinstance(value, ComponentRef):
        return {"name": value.name, **_serialize_config(value.overrides)}
    if is_dataclass(value) and not isinstance(value, type):
        result = {}
        for f in dc_fields(value):
            v = getattr(value, f.name)
            if isinstance(v, ComponentRef):
                result[f.name] = v.resolve_config(f.metadata.get("factory"))
            else:
                result[f.name] = _serialize_config(v)
        return result
    if isinstance(value, dict):
        return {k: _serialize_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_config(v) for v in value]
    if isinstance(value, type):
        return value.__name__
    return value
