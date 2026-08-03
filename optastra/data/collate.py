from ._registry import register_collate
from torch.utils.data._utils.collate import default_collate
from .sample import Sample


@register_collate("default_collate")
def default_collate_fn(samples: list[Sample]) -> dict:
    """
    Default collate function that stacks images and targets into tensors.
    This is used when no specific collate function is registered for a task.
    """
    return collate_dense(samples)  # Use the dense collate as the default behavior


@register_collate("dense")
def collate_dense(samples: list[Sample]) -> dict:
    """
    Classification, regression, dense segmentation -- anything where
    every target field has a uniform shape across the batch.
    """
    images = default_collate([s.image for s in samples])
    keys = samples[0].target.keys()
    targets = {k: default_collate([s.target[k] for s in samples]) for k in keys}
    return {"inputs": images, "targets": targets}


@register_collate("ragged")
def collate_ragged(samples: list[Sample]) -> dict:
    """
    Detection, instance segmentation -- variable-length targets per image,
    can't be stacked into one tensor. Images must already be a fixed size
    (resize/pad in the transform), targets stay a list of per-image dicts.
    """
    images = default_collate([s.image for s in samples])
    targets = [s.target for s in samples]  # list[dict], task's compute_losses handles the ragged-ness
    return {"inputs": images, "targets": targets}


@register_collate("multiview")
def collate_multiview(samples: list[Sample]) -> dict:
    """
    Self-supervised algorithms -- N augmented views per image, no labels.
    """
    num_views = len(samples[0].views)
    views = [default_collate([s.views[v] for s in samples]) for v in range(num_views)]
    return {"views": views}
