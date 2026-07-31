from typing import Any, Callable, List, Optional

from optastra.registry import ComponentRegistry


_registry = ComponentRegistry("head")


def register_head(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for registering heads. The function name is used as the head name."""
    return _registry.register(fn)


def list_heads(module: Optional[str] = None, filter: Optional[str] = None) -> List[str]:
    """List all registered head names, optionally filtered by module and/or a wildcard pattern."""
    return _registry.list(module=module, filter=filter)


def get_head_entrypoint(name: str) -> Callable[..., Any]:
    """Get the entrypoint function for a registered head by name."""
    return _registry.get_entrypoint(name)


def get_head_module(name: str) -> str:
    """Get the module name for a registered head by name."""
    return _registry.get_module(name)