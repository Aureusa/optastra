from optastra.core.registry import ComponentRegistry


_registry = ComponentRegistry("region_extractor")
register_region_extractor = _registry.make_decorator()
