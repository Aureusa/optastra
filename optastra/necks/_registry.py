from typing import Any, Callable, List, Optional

from optastra.registry import ComponentRegistry


_registry = ComponentRegistry("neck")


def register_neck(
    fn: Optional[Callable[..., Any]] = None,
    *,
    config: Optional[Any] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]] | Callable[..., Any]:
    """Decorator for registering necks.

    Supports both usages:
    - @register_neck
    - @register_neck(config=...)
    """

    def decorator(inner_fn: Callable[..., Any]) -> Callable[..., Any]:
        return _registry.register(inner_fn, default_config=config)

    if fn is not None:
        return decorator(fn)
    return decorator


def list_necks(module: Optional[str] = None, filter: Optional[str] = None) -> List[str]:
    """List all registered neck names, optionally filtered by module and/or a wildcard pattern."""
    return _registry.list(module=module, filter=filter)


def get_neck_entrypoint(name: str) -> Callable[..., Any]:
    """Get the entrypoint function for a registered neck by name."""
    return _registry.get_entrypoint(name)


def get_neck_module(name: str) -> str:
    """Get the module name for a registered neck by name."""
    return _registry.get_module(name)

def get_neck_default_config(name: str) -> Any:
    """Get the default configuration for a registered neck by name."""
    return _registry.get_default_config(name)
