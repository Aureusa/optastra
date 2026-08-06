from optastra.core.registry import FamilyRegistry


_registry = FamilyRegistry("architecture")
register_architecture = _registry.make_decorator()
