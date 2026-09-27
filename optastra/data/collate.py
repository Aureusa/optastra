import functools
import math

import torch
from torch.utils.data._utils.collate import default_collate as torch_default_collate

from .sample import Sample
from optastra.core.registry import FamilyRegistry
from optastra.core.factory import Factory


__all__ = ["CollateFn", "pad_images"]


class CollateFn(Factory["CollateFn"]):
    """
    Registry for collate functions. Collate functions are used to combine a list of samples into a batch.
    """

    _registry = FamilyRegistry("collate")

    @classmethod
    def create(cls, name: str, **kwargs) -> "CollateFn":
        """
        Create a collate function by name. Keyword arguments are bound to the
        function, e.g. ``CollateFn.create("ragged", size_divisibility=32)``.
        """
        entrypoint = cls._registry.get_entrypoint(name)
        return functools.partial(entrypoint, **kwargs) if kwargs else entrypoint


@CollateFn.register
def default_collate(samples: list[Sample]) -> dict:
    """
    Default collate function that stacks images and targets into tensors.
    This is used when no specific collate function is registered for a task.
    """
    return dense(samples)  # Use the dense collate as the default behavior


@CollateFn.register
def dense(samples: list[Sample]) -> dict:
    """
    Classification, regression, dense segmentation -- anything where
    every target field has a uniform shape across the batch.
    """
    images = torch_default_collate([s.image for s in samples])
    keys = samples[0].target.keys()
    targets = {k: torch_default_collate([s.target[k] for s in samples]) for k in keys}
    return {"inputs": images, "targets": targets}


def pad_images(
    images: list[torch.Tensor],
    size_divisibility: int = 0,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Stack (C, H_i, W_i) images into one (N, C, H, W) batch, padding bottom/right.

    :param size_divisibility: if > 0, round the padded H and W up to a multiple
        of it (e.g. 32, the coarsest stride of a ResNet + FPN).
    :return: the padded batch and each image's original (h, w).
    """
    image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in images]
    max_h = max(h for h, _ in image_sizes)
    max_w = max(w for _, w in image_sizes)
    if size_divisibility > 0:
        max_h = int(math.ceil(max_h / size_divisibility) * size_divisibility)
        max_w = int(math.ceil(max_w / size_divisibility) * size_divisibility)

    batch = images[0].new_full((len(images), images[0].shape[0], max_h, max_w), pad_value)
    for i, img in enumerate(images):
        batch[i, :, : img.shape[-2], : img.shape[-1]].copy_(img)
    return batch, image_sizes


@CollateFn.register
def ragged(samples: list[Sample], size_divisibility: int = 0, pad_value: float = 0.0) -> dict:
    """
    Detection, instance segmentation -- variable-length targets per image,
    can't be stacked into one tensor; targets stay a list of per-image dicts.

    Images may differ in size: they are padded (bottom/right, so box
    coordinates stay valid) to the largest H and W in the batch, rounded up to
    ``size_divisibility`` if > 0. Returns:

        inputs       (N, C, H, W) padded images
        targets      list of the samples' target dicts
        image_sizes  list of each image's (h, w) before padding
        rois         (R, 5) (batch_index, x1, y1, x2, y2) -- only if every
                     sample carries ``target["proposals"]`` (Fast R-CNN)
    """
    images, image_sizes = pad_images([s.image for s in samples], size_divisibility, pad_value)
    targets = [s.target for s in samples]  # list[dict], task's compute_losses handles the ragged-ness
    batch = {"inputs": images, "targets": targets, "image_sizes": image_sizes}

    if samples and all("proposals" in s.target for s in samples):
        rois = []
        for i, sample in enumerate(samples):
            proposals = torch.as_tensor(sample.target["proposals"], dtype=torch.float32).reshape(-1, 4)
            rois.append(torch.cat((torch.full((proposals.shape[0], 1), float(i)), proposals), dim=1))
        batch["rois"] = torch.cat(rois, dim=0)
    return batch


@CollateFn.register
def multiview(samples: list[Sample]) -> dict:
    """
    Self-supervised algorithms -- N augmented views per image, no labels.
    """
    num_views = len(samples[0].views)
    views = [torch_default_collate([s.views[v] for s in samples]) for v in range(num_views)]
    return {"views": views}
