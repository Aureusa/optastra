from optastra.core.registry import ComponentRegistry


_registry = ComponentRegistry("optimizer")
register_optimizer = _registry.make_decorator()
_scheduler_registry = ComponentRegistry("scheduler")
register_scheduler = _scheduler_registry.make_decorator()
