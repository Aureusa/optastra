from typing import Any, Callable, List, Optional

from optastra.registry import ComponentRegistry


_registry = ComponentRegistry("backbone")


def register_backbone(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for registering backbones. The function name is used as the backbone name."""
    return _registry.register(fn)


def list_backbones(module: Optional[str] = None, filter: Optional[str] = None) -> List[str]:
    """List all registered backbone names, optionally filtered by module and/or a wildcard pattern."""
    return _registry.list(module=module, filter=filter)


def get_backbone_entrypoint(name: str) -> Callable[..., Any]:
    """Get the entrypoint function for a registered backbone by name."""
    return _registry.get_entrypoint(name)


def get_backbone_module(name: str) -> str:
    """Get the module name for a registered backbone by name."""
    return _registry.get_module(name)
