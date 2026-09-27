import pytest
import torch
from torch.utils.data import Dataset

from optastra import Algorithm, Sample, Transform, build_dataloader
from optastra.transforms import Compose, MultiViewTransform


class _AddConstant(Transform):
    """Deterministic, in-place transform: makes each view identifiable and
    would leak between views if they shared one Sample."""

    def __init__(self, value: float):
        self.value = value

    def __call__(self, sample: Sample) -> Sample:
        sample.image += self.value
        sample.meta["seen"] = sample.meta.get("seen", 0) + 1
        return sample


def _sample() -> Sample:
    return Sample(image=torch.zeros(3, 4, 4), target={"label": torch.tensor(1)}, meta={"id": 7})


def test_one_pipeline_per_view():
    original = _sample()
    out = MultiViewTransform([_AddConstant(1.0), _AddConstant(2.0), _AddConstant(3.0)])(original)

    assert out.image is None
    assert [v.flatten()[0].item() for v in out.views] == [1.0, 2.0, 3.0]
    assert all(v.shape == (3, 4, 4) for v in out.views)
    # the input sample is not modified, views don't share memory, target/meta are kept
    assert torch.equal(original.image, torch.zeros(3, 4, 4))
    assert out.views[0].data_ptr() != out.views[1].data_ptr()
    assert out.meta == {"id": 7} and torch.equal(out.target["label"], torch.tensor(1))


def test_n_views_repeats_a_single_pipeline():
    out = MultiViewTransform(Compose([_AddConstant(1.0), _AddConstant(1.0)]), n_views=3)(_sample())
    assert len(out.views) == 3
    assert all(torch.equal(v, torch.full((3, 4, 4), 2.0)) for v in out.views)


def test_n_views_with_several_pipelines_is_rejected():
    with pytest.raises(ValueError, match="repeats a single pipeline"):
        MultiViewTransform([_AddConstant(1.0), _AddConstant(2.0)], n_views=2)


def test_registered_multi_view_builds_pipelines_from_refs():
    torch.manual_seed(0)
    transform = Transform.create(
        "multi_view",
        views=[
            ["to_float", ("random_hflip", {"p": 1.0})],
            {"name": "to_float", "overrides": {"scale": False}},
        ],
    )
    image = torch.arange(2 * 2 * 3, dtype=torch.uint8).reshape(3, 2, 2)
    out = transform(Sample(image=image))

    assert torch.allclose(out.views[0], image.float().flip(-1) / 255.0)
    assert torch.equal(out.views[1], image.float())


def test_registered_multi_view_default_is_two_identity_views():
    out = Transform.create("multi_view")(Sample(image=torch.ones(1, 2, 2)))
    assert len(out.views) == 2 and all(torch.equal(v, torch.ones(1, 2, 2)) for v in out.views)


class _ToyDataset(Dataset):
    def __init__(self, transform):
        self.images = torch.arange(6 * 3 * 4 * 4, dtype=torch.float32).reshape(6, 3, 4, 4)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(Sample(image=self.images[idx].clone()))


def test_multiview_samples_collate_into_algorithm_batches():
    dataset = _ToyDataset(MultiViewTransform([_AddConstant(0.0), _AddConstant(100.0)]))
    loader = build_dataloader(dataset, task=Algorithm.create("simclr"), batch_size=4, shuffle=False)
    batch = next(iter(loader))

    assert set(batch) == {"views"}
    assert len(batch["views"]) == 2
    torch.testing.assert_close(batch["views"][0], dataset.images[:4])
    torch.testing.assert_close(batch["views"][1], dataset.images[:4] + 100.0)
