from types import SimpleNamespace

import torch

from optastra.data.sample import Sample

# Import for registry side effects (default_collate, dense, ragged, multiview).
import optastra.data.collate  # noqa: F401
from optastra.data._registry import get_collate_entrypoint
from optastra.data.loader import build_dataloader


def test_dense_collate_stacks_images_and_targets():
    samples = [
        Sample(image=torch.randn(3, 8, 8), target={"targets": torch.tensor(1)}),
        Sample(image=torch.randn(3, 8, 8), target={"targets": torch.tensor(0)}),
    ]

    dense = get_collate_entrypoint("dense")
    batch = dense(samples)

    assert batch["inputs"].shape == (2, 3, 8, 8)
    assert batch["targets"]["targets"].shape == (2,)


def test_ragged_collate_keeps_targets_as_list():
    samples = [
        Sample(image=torch.randn(3, 8, 8), target={"boxes": torch.randn(2, 4)}),
        Sample(image=torch.randn(3, 8, 8), target={"boxes": torch.randn(5, 4)}),
    ]

    ragged = get_collate_entrypoint("ragged")
    batch = ragged(samples)

    assert batch["inputs"].shape == (2, 3, 8, 8)
    assert isinstance(batch["targets"], list)
    assert len(batch["targets"]) == 2
    assert batch["targets"][0]["boxes"].shape[0] == 2
    assert batch["targets"][1]["boxes"].shape[0] == 5


def test_multiview_collate_batches_each_view_index():
    samples = [
        Sample(views=[torch.randn(3, 6, 6), torch.randn(3, 6, 6)]),
        Sample(views=[torch.randn(3, 6, 6), torch.randn(3, 6, 6)]),
    ]

    multiview = get_collate_entrypoint("multiview")
    batch = multiview(samples)

    assert "views" in batch
    assert len(batch["views"]) == 2
    assert batch["views"][0].shape == (2, 3, 6, 6)
    assert batch["views"][1].shape == (2, 3, 6, 6)


def test_build_dataloader_uses_task_collate_name():
    dataset = [
        Sample(image=torch.randn(3, 4, 4), target={"targets": torch.tensor(1)}),
        Sample(image=torch.randn(3, 4, 4), target={"targets": torch.tensor(0)}),
    ]

    task = SimpleNamespace(collate="dense")
    dataloader = build_dataloader(dataset, task=task, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))

    assert batch["inputs"].shape == (2, 3, 4, 4)
    assert torch.equal(batch["targets"]["targets"], torch.tensor([1, 0]))


def test_sample_defaults_are_stable():
    sample = Sample()

    assert sample.image is None
    assert sample.views is None
    assert sample.target == {}
    assert sample.meta == {}
