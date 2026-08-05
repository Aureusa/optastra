from optastra.core.registry import ComponentRegistry


matcher_registry = ComponentRegistry("matcher")
sampler_registry = ComponentRegistry("sampler")
postprocessor_registry = ComponentRegistry("postprocessor")
criterion_registry = ComponentRegistry("criterion")

register_matcher = matcher_registry.make_decorator()
register_sampler = sampler_registry.make_decorator()
register_postprocessor = postprocessor_registry.make_decorator()
register_criterion = criterion_registry.make_decorator()
