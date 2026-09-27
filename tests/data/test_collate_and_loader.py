from types import SimpleNamespace

import torch

from optastra.data.sample import Sample

# Import for registry side effects (default_collate, dense, ragged, multiview).
import optastra.data.collate  # noqa: F401
from optastra.data.collate import CollateFn
from optastra.data.loader import build_dataloader


def test_dense_collate_stacks_images_and_targets():
    samples = [
        Sample(image=torch.randn(3, 8, 8), target={"targets": torch.tensor(1)}),
        Sample(image=torch.randn(3, 8, 8), target={"targets": torch.tensor(0)}),
    ]

    dense = CollateFn._registry.get_entrypoint("dense")
    batch = dense(samples)

    assert batch["inputs"].shape == (2, 3, 8, 8)
    assert batch["targets"]["targets"].shape == (2,)


def test_ragged_collate_keeps_targets_as_list():
    samples = [
        Sample(image=torch.randn(3, 8, 8), target={"boxes": torch.randn(2, 4)}),
        Sample(image=torch.randn(3, 8, 8), target={"boxes": torch.randn(5, 4)}),
    ]

    ragged = CollateFn._registry.get_entrypoint("ragged")
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

    multiview = CollateFn._registry.get_entrypoint("multiview")
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


def test_ragged_collate_pads_variable_size_images_and_records_their_sizes():
    first = torch.arange(3 * 5 * 7, dtype=torch.float32).view(3, 5, 7) + 1
    second = torch.arange(3 * 8 * 4, dtype=torch.float32).view(3, 8, 4) + 1
    samples = [
        Sample(image=first, target={"boxes": torch.zeros(1, 4)}),
        Sample(image=second, target={"boxes": torch.zeros(2, 4)}),
    ]

    batch = CollateFn._registry.get_entrypoint("ragged")(samples)

    assert batch["inputs"].shape == (2, 3, 8, 7)
    assert batch["image_sizes"] == [(5, 7), (8, 4)]
    # Content is placed top-left (so box coordinates stay valid), the rest is padding.
    assert torch.equal(batch["inputs"][0, :, :5, :7], first)
    assert torch.equal(batch["inputs"][1, :, :8, :4], second)
    assert batch["inputs"][0, :, 5:, :].abs().sum() == 0
    assert batch["inputs"][1, :, :, 4:].abs().sum() == 0
    assert "rois" not in batch


def test_ragged_collate_rounds_padded_size_up_to_size_divisibility():
    samples = [Sample(image=torch.ones(1, 33, 20), target={}), Sample(image=torch.ones(1, 10, 64), target={})]

    collate = CollateFn.create("ragged", size_divisibility=32)
    batch = collate(samples)

    assert batch["inputs"].shape == (2, 1, 64, 64)
    assert batch["image_sizes"] == [(33, 20), (10, 64)]


def test_build_dataloader_binds_collate_kwargs():
    dataset = [Sample(image=torch.ones(3, 5, 5), target={"boxes": torch.zeros(0, 4)})]
    task = SimpleNamespace(collate="ragged")

    batch = next(iter(build_dataloader(dataset, task=task, batch_size=1, collate_kwargs={"size_divisibility": 16})))

    assert batch["inputs"].shape == (1, 3, 16, 16)
    assert batch["image_sizes"] == [(5, 5)]


def test_ragged_collate_batches_per_sample_proposals_as_rois():
    samples = [
        Sample(image=torch.ones(3, 4, 4), target={"proposals": torch.tensor([[0.0, 0.0, 2.0, 2.0]])}),
        Sample(image=torch.ones(3, 4, 4), target={"proposals": torch.tensor([[1.0, 1.0, 3.0, 3.0], [0.0, 1.0, 2.0, 3.0]])}),
    ]

    batch = CollateFn._registry.get_entrypoint("ragged")(samples)

    assert batch["rois"].tolist() == [
        [0.0, 0.0, 0.0, 2.0, 2.0],
        [1.0, 1.0, 1.0, 3.0, 3.0],
        [1.0, 0.0, 1.0, 2.0, 3.0],
    ]
