from optastra.core.registry import ComponentRegistry


_registry = ComponentRegistry("task")
register_task = _registry.make_decorator()
