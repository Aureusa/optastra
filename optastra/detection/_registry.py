from optastra.core.registry import FamilyRegistry


matcher_registry = FamilyRegistry("matcher")
sampler_registry = FamilyRegistry("sampler")
postprocessor_registry = FamilyRegistry("postprocessor")
criterion_registry = FamilyRegistry("criterion")

register_matcher = matcher_registry.make_decorator()
register_sampler = sampler_registry.make_decorator()
register_postprocessor = postprocessor_registry.make_decorator()
register_criterion = criterion_registry.make_decorator()
