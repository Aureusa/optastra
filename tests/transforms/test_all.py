import pytest
import torch
from optastra.transforms import Transform, BatchTransform
from optastra.data.sample import Sample

POSSIBLE_TYPES = [torch.float32, torch.float64, torch.uint8, torch.int32, torch.int64]

def _generate_sample(size=(3, 32, 32), dtype=torch.float32):
    """Generates a random sample image tensor for testing."""

    def random_image():
        if dtype == torch.uint8:
            return torch.randint(0, 256, size, dtype=dtype)
        return torch.rand(size, dtype=dtype)

    return Sample(
        image=random_image(),
        views=[random_image() for _ in range(2)],
        target={
            "labels": torch.randint(0, 10, (1,), dtype=torch.int64),
            "boxes": torch.rand((1, 4), dtype=torch.float32),
            "masks": torch.rand((1, size[1], size[2]), dtype=torch.float32),
        },
        meta={"this": "is a test sample"},
    )

def _generate_batch(batch_size=4, size=(3, 32, 32), dtype=torch.float32):
    """Generates a random batch of sample images for testing."""
    return {
        "inputs": torch.rand((batch_size, *size), dtype=dtype),
        "targets": {
            "labels": torch.randint(0, 10, (batch_size,), dtype=torch.int64),
            "boxes": torch.rand((batch_size, 4), dtype=torch.float32),
            "masks": torch.rand((batch_size, size[1], size[2]), dtype=torch.float32)
        }
    }

def test_all_registered_transforms_initialization():
    transforms = Transform.list_all()
    batch_transforms = BatchTransform.list_all()

    for transform_name in transforms:
        transform_class = Transform.create(transform_name)
        assert isinstance(transform_class, Transform), f"{transform_name} is not a valid Transform"

    for batch_transform_name in batch_transforms:
        batch_transform_class = BatchTransform.create(batch_transform_name)
        assert isinstance(batch_transform_class, BatchTransform), f"{batch_transform_name} is not a valid BatchTransform"

def test_all_transforms_functionality():
    transforms = Transform.list_all()

    for transform_name in transforms:
        transform_class = Transform.create(transform_name)
        sample = _generate_sample()
        transformed_sample = transform_class(sample)
        assert hasattr(transformed_sample, 'image'), f"{transform_name} did not return a Sample with an image"

# def test_all_transforms_with_various_dtypes():
#     transforms = Transform.list_all()

#     for transform_name in transforms:
#         transform_class = Transform.create(transform_name)
#         for dtype in POSSIBLE_TYPES:
#             sample = _generate_sample(dtype=dtype)
#             transformed_sample = transform_class(sample)
#             assert transformed_sample.image.dtype == sample.image.dtype, f"{transform_name} changed dtype from {sample.image.dtype} to {transformed_sample.image.dtype}"

def test_failure_on_unknown_transform():
    with pytest.raises(ValueError):
        Transform.create("unknown_transform")

def test_failure_on_unknown_batch_transform():
    with pytest.raises(ValueError):
        BatchTransform.create("unknown_batch_transform")
