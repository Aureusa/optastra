"""Seedable transform randomness."""
import random

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from optastra.data.sample import Sample
from optastra.transforms import Compose, Transform, rng, seed_transforms, worker_init_fn


def _pipeline():
    return Compose([
        Transform.create("random_resized_crop", size=24),
        Transform.create("random_hflip"),
        Transform.create("random_rotation", degrees=(-30.0, 30.0)),
        Transform.create("rand_augment_all_ops"),
        Transform.create("color_jitter"),
        Transform.create("augmix"),
        Transform.create("pixmix"),
    ])


def _sample():
    img = torch.linspace(0, 1, 3 * 32 * 40).reshape(3, 32, 40)
    return Sample(image=img, target={"boxes": torch.tensor([[4.0, 4.0, 30.0, 20.0]]), "labels": torch.tensor([1])})


def _run(seed, n=4):
    seed_transforms(seed)
    t = _pipeline()
    return [t(_sample()) for _ in range(n)]


def test_same_seed_same_output_different_seed_different_output():
    a, b, c = _run(0), _run(0), _run(1)
    for x, y in zip(a, b):
        assert torch.equal(x.image, y.image)
        assert torch.equal(x.target["boxes"], y.target["boxes"])
    assert any(not torch.equal(x.image, z.image) for x, z in zip(a, c))


def test_transforms_leave_global_rngs_alone():
    py_state, torch_state = random.getstate(), torch.get_rng_state()
    _run(0)
    assert random.getstate() == py_state
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_draw_helpers():
    seed_transforms(0)
    assert all(0 <= rng.randint(2, 4) - 2 <= 2 for _ in range(50))
    assert {rng.randint(0, 1) for _ in range(50)} == {0, 1}          # inclusive upper bound
    assert sorted(rng.sample("abcd", 4)) == list("abcd")
    w = rng.dirichlet(1.0, 3)
    assert w.shape == (3,) and abs(float(w.sum()) - 1.0) < 1e-5


class _AugmentedDataset(Dataset):
    def __init__(self):
        self.t = Transform.create("rand_augment", num_ops=3, magnitude=9)

    def __len__(self):
        return 4

    def __getitem__(self, i):
        return self.t(Sample(image=torch.linspace(0, 1, 3 * 8 * 8).reshape(3, 8, 8))).image


def _load(seed, **kwargs):
    loader = DataLoader(_AugmentedDataset(), batch_size=1, num_workers=2,
                        generator=torch.Generator().manual_seed(seed), **kwargs)
    return [batch[0] for batch in loader]


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_dataloader_workers_get_distinct_reproducible_streams():
    run_a, run_b = _load(0), _load(0)
    assert all(torch.equal(x, y) for x, y in zip(run_a, run_b))
    # items 0 and 1 are the first item of worker 0 and worker 1: identical inputs,
    # so they only differ if the forked workers don't share one generator state
    assert not torch.equal(run_a[0], run_a[1])
    # the explicit worker_init_fn gives the same streams
    run_c = _load(0, worker_init_fn=worker_init_fn)
    assert all(torch.equal(x, y) for x, y in zip(run_a, run_c))
