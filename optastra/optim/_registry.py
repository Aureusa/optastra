from optastra.core.registry import FamilyRegistry


_registry = FamilyRegistry("optimizer")
register_optimizer = _registry.make_decorator()
_scheduler_registry = FamilyRegistry("scheduler")
register_scheduler = _scheduler_registry.make_decorator()
