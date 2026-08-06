from optastra.core.registry import FamilyRegistry


_registry = FamilyRegistry("region_extractor")
register_region_extractor = _registry.make_decorator()
