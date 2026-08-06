from optastra.core.registry import FamilyRegistry


_registry = FamilyRegistry("task")
register_task = _registry.make_decorator()
