import fnmatch
import sys
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, TypeVar


T = TypeVar("T", bound=Callable[..., Any])


class ComponentRegistry:
    """A small generic registry for model components such as backbones, heads, and necks."""

    def __init__(self, component_name: str):
        self.component_name = component_name
        self._module_to_components: DefaultDict[str, Set[str]] = defaultdict(set)
        self._component_to_module: Dict[str, str] = {}
        self._component_entrypoints: Dict[str, Callable[..., Any]] = {}

    def register(self, fn: T) -> T:
        mod = sys.modules[fn.__module__]
        module_name = fn.__module__.split('.')[-1]
        component_name = fn.__name__

        if hasattr(mod, '__all__'):
            mod.__all__.append(component_name)  # type: ignore
        else:
            mod.__all__ = [component_name]  # type: ignore

        if component_name in self._component_entrypoints:
            raise ValueError(
                f'{self.component_name} {component_name} already registered by {self._component_to_module[component_name]}'
            )

        self._component_entrypoints[component_name] = fn
        self._component_to_module[component_name] = module_name
        self._module_to_components[module_name].add(component_name)
        return fn

    def list(self, module: Optional[str] = None, filter: Optional[str] = None) -> List[str]:
        if module is not None:
            components = list(self._module_to_components.get(module, []))
        else:
            components = list(self._component_entrypoints.keys())

        if filter not in (None, ""):
            components = fnmatch.filter(components, filter)
        return sorted(components)

    def get_entrypoint(self, name: str) -> Callable[..., Any]:
        if name not in self._component_entrypoints:
            raise ValueError(f'{self.component_name} {name} is not registered')
        return self._component_entrypoints[name]

    def get_module(self, name: str) -> str:
        if name not in self._component_to_module:
            raise ValueError(f'{self.component_name} {name} is not registered')
        return self._component_to_module[name]
