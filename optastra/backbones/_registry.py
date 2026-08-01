from typing import Any, Callable, List, Optional

from optastra.registry import ComponentRegistry


_registry = ComponentRegistry("backbone")


def register_backbone(
    fn: Optional[Callable[..., Any]] = None,
    *,
    config: Optional[Any] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]] | Callable[..., Any]:
    """Decorator for registering backbones.

    Supports both usages:
    - @register_backbone
    - @register_backbone(config=...)
    """

    def decorator(inner_fn: Callable[..., Any]) -> Callable[..., Any]:
        return _registry.register(inner_fn, default_config=config)

    if fn is not None:
        return decorator(fn)
    return decorator


def list_backbones(module: Optional[str] = None, filter: Optional[str] = None) -> List[str]:
    """List all registered backbone names, optionally filtered by module and/or a wildcard pattern."""
    return _registry.list_component(module=module, filter=filter)


def get_backbone_entrypoint(name: str) -> Callable[..., Any]:
    """Get the entrypoint function for a registered backbone by name."""
    return _registry.get_entrypoint(name)


def get_backbone_module(name: str) -> str:
    """Get the module name for a registered backbone by name."""
    return _registry.get_module(name)

def get_backbone_default_config(name: str) -> Any:
    """Get the default configuration for a registered backbone by name."""
    return _registry.get_default_config(name)

def check_backbone_registered(name: str) -> bool:
    """Check if a backbone is registered by name."""
    return _registry.is_registered(name)
