from typing import Any, Callable, List, Optional

from optastra.registry import ComponentRegistry


_registry = ComponentRegistry("neck")


def register_neck(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for registering necks. The function name is used as the neck name."""
    return _registry.register(fn)


def list_necks(module: Optional[str] = None, filter: Optional[str] = None) -> List[str]:
    """List all registered neck names, optionally filtered by module and/or a wildcard pattern."""
    return _registry.list(module=module, filter=filter)


def get_neck_entrypoint(name: str) -> Callable[..., Any]:
    """Get the entrypoint function for a registered neck by name."""
    return _registry.get_entrypoint(name)


def get_neck_module(name: str) -> str:
    """Get the module name for a registered neck by name."""
    return _registry.get_module(name)