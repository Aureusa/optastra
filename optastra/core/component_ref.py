from __future__ import annotations
from dataclasses import dataclass, field, fields as dc_fields, is_dataclass, replace
from typing import Any

from .factory import Factory


def coerce_to_ref(value: Any) -> Any:
    """Accepts the friendly forms and normalizes to ComponentRef.
    None passes through (valid for optional fields)."""
    # Already a ComponentRef, or optional None
    if value is None or isinstance(value, ComponentRef):
        return value

    # No overrides, just a name string
    if isinstance(value, str):
        return ComponentRef(value)  # "fpn" -> ComponentRef("fpn")

    # (name, overrides) tuple 
    if isinstance(value, tuple) and len(value) == 2:
        name, overrides = value
        return ComponentRef(name, dict(overrides))  # ("resnet50", {"stem_channels": 32})

    # {"name": ..., "overrides": ...} dict
    if isinstance(value, dict):
        return ComponentRef(value["name"], dict(value.get("overrides", {})))  # {"name": resnet50, "overrides": {"stem_channels": 32}}
    
    raise TypeError(
        f"Cannot interpret {value!r} as a ComponentRef. Use a name string, "
        f"a (name, overrides) tuple, a {{'name':..., 'overrides':...}} dict, or ComponentRef(...) directly."
    )


def component_field(
        factory: type[Factory],
        *,
        default_name: str | None = None,
        default_overrides: dict | None = None,
        optional: bool = False,
        **kwargs
    ) -> Any:
    """
    Declares a ComponentRef field and which Factory describes it -- read by
    resolve_config()/_serialize_config() to expand this field's defaults
    when printing or dumping a config to YAML. Building is a separate,
    explicit step: the call site does `cfg.<field>.resolve(SomeFactory)`
    directly (it already knows which Factory it needs -- that's not
    something that has to be rediscovered through this metadata).
    """
    if optional and default_name is None:
        maker = lambda: None
    else:
        maker = lambda: ComponentRef(default_name or "", dict(default_overrides or {}))
    return field(default_factory=maker, metadata={"factory": factory}, **kwargs)


class ComponentRefConfigMixin:
    """
    Mixin: coerces every component_field() on this dataclass from a
    friendly shorthand into a real ComponentRef, once, right after
    construction -- so callers never have to import ComponentRef for the
    common case.

    This is used for configs of components that themselves have component fields,
    e.g. Faster RCNN has a backbone and a head field, both of which are ComponentRefs.
    """
    def __post_init__(self):
        for f in dc_fields(self):
            if "factory" not in f.metadata:
                continue
            current = getattr(self, f.name)
            coerced = coerce_to_ref(current)
            if getattr(type(self), "__dataclass_fields__", None) and self.__dataclass_params__.frozen:
                object.__setattr__(self, f.name, coerced)
            else:
                setattr(self, f.name, coerced)


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
