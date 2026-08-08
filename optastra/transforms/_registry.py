from optastra.core.registry import FamilyRegistry


_registry = FamilyRegistry("transform")
register_transform = _registry.make_decorator()

_batch_registry = FamilyRegistry("batch_transform")
register_batch_transform = _batch_registry.make_decorator()
