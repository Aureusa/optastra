from __future__ import annotations
from dataclasses import fields, is_dataclass, replace
from typing import Any, Callable

from .component_ref import ComponentRef


def _to_display(value: Any) -> Any:
    """
    Best-effort flatten for printing/dumping -- not meant to round-trip.
    Handles the one recurring offender: dataclass fields holding a class
    (e.g. ResNetConfig.block: type[ResidualBlock]).
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_display(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, (list, tuple)):
        return [_to_display(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_display(v) for k, v in value.items()}
    return value


def resolve_ref(ref: ComponentRef, config_lookup: Callable[[str], Any]) -> dict[str, Any]:
    """
    config_lookup is e.g. Backbone.config or Architecture.config --
    applies ref.overrides on top of the registered default, flattened.
    """
    default_cfg = config_lookup(ref.name)
    if default_cfg is None:
        return {"name": ref.name, **ref.overrides}
    resolved = replace(default_cfg, **ref.overrides)
    return {"name": ref.name, **_to_display(resolved)}
