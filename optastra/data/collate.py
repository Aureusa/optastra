from torch.utils.data._utils.collate import default_collate as torch_default_collate

from .sample import Sample
from optastra.core.registry import FamilyRegistry
from optastra.core.factory import Factory


__all__ = ["CollateFn"]


class CollateFn(Factory["CollateFn"]):
    """
    Registry for collate functions. Collate functions are used to combine a list of samples into a batch.
    """

    _registry = FamilyRegistry("collate")

    @classmethod
    def create(cls, name: str, **kwargs) -> "CollateFn":
        """
        Create a collate function by name.
        """
        return cls._registry.get_entrypoint(name, **kwargs)

    @classmethod
    def make_decorator(cls):
        """
        Returns a decorator that registers a collate function with the registry.
        """
        return cls._registry.make_decorator()


register_collate = CollateFn.make_decorator()


@register_collate
def default_collate(samples: list[Sample]) -> dict:
    """
    Default collate function that stacks images and targets into tensors.
    This is used when no specific collate function is registered for a task.
    """
    return dense(samples)  # Use the dense collate as the default behavior


@register_collate
def dense(samples: list[Sample]) -> dict:
    """
    Classification, regression, dense segmentation -- anything where
    every target field has a uniform shape across the batch.
    """
    images = torch_default_collate([s.image for s in samples])
    keys = samples[0].target.keys()
    targets = {k: torch_default_collate([s.target[k] for s in samples]) for k in keys}
    return {"inputs": images, "targets": targets}


@register_collate
def ragged(samples: list[Sample]) -> dict:
    """
    Detection, instance segmentation -- variable-length targets per image,
    can't be stacked into one tensor. Images must already be a fixed size
    (resize/pad in the transform), targets stay a list of per-image dicts.
    """
    images = torch_default_collate([s.image for s in samples])
    targets = [s.target for s in samples]  # list[dict], task's compute_losses handles the ragged-ness
    return {"inputs": images, "targets": targets}


@register_collate
def multiview(samples: list[Sample]) -> dict:
    """
    Self-supervised algorithms -- N augmented views per image, no labels.
    """
    num_views = len(samples[0].views)
    views = [torch_default_collate([s.views[v] for s in samples]) for v in range(num_views)]
    return {"views": views}
