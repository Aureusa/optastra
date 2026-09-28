import pytest
import torch

from optastra import Task, build_dataloader
from optastra.data import Sample, ShardedSampler
from optastra.nn.features import HeadOutput


def _shards(n, world_size, **kwargs):
    return [list(ShardedSampler(n, rank=r, world_size=world_size, **kwargs)) for r in range(world_size)]


@pytest.mark.parametrize("n, world_size", [(8, 2), (7, 2), (10, 3), (1, 2), (0, 2)])
@pytest.mark.parametrize("shuffle", [False, True])
def test_shards_are_disjoint_and_cover_every_sample_once(n, world_size, shuffle):
    shards = _shards(n, world_size, shuffle=shuffle)
    flat = [i for shard in shards for i in shard]
    assert sorted(flat) == list(range(n))                                   # nothing dropped, nothing padded
    assert [len(s) for s in shards] == [len(ShardedSampler(n, rank=r, world_size=world_size)) for r in range(world_size)]
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_unshuffled_shards_are_strided():
    assert _shards(7, 2) == [[0, 2, 4, 6], [1, 3, 5]]


def test_shuffle_is_a_shared_permutation_that_changes_per_epoch():
    a, b = ShardedSampler(20, shuffle=True, seed=3, rank=0, world_size=2), ShardedSampler(20, shuffle=True, seed=3, rank=1, world_size=2)
    epoch0 = list(a), list(b)
    a.set_epoch(1), b.set_epoch(1)
    epoch1 = list(a), list(b)
    assert epoch0 != epoch1
    assert sorted(epoch1[0] + epoch1[1]) == list(range(20))
    order = torch.randperm(20, generator=torch.Generator().manual_seed(3 + 1)).tolist()
    assert epoch1 == (order[0::2], order[1::2])                            # seed + epoch, same on every rank


def test_invalid_rank_is_rejected():
    with pytest.raises(ValueError, match="rank"):
        ShardedSampler(10, rank=2, world_size=2)


def test_single_process_defaults_and_build_dataloader_is_unchanged():
    assert list(ShardedSampler(5)) == [0, 1, 2, 3, 4]       # rank 0 of 1 outside torchrun
    data = [Sample(image=torch.zeros(3), target={"labels": torch.tensor(0)}) for _ in range(5)]
    loader = build_dataloader(data, task=Task.create("classification_task"), batch_size=2, shuffle=True)
    assert not isinstance(loader.sampler, ShardedSampler)    # plain python run: ordinary DataLoader


class _EpochStream(torch.utils.data.IterableDataset):
    """Shards / shuffles itself; records the epochs the Trainer announces."""

    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)

    def __iter__(self):
        for i in range(8):
            yield Sample(image=torch.full((3,), float(i)), target={"labels": torch.tensor(i % 2)})


class _Logits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x):
        return HeadOutput(logits=self.linear(x))


def test_trainer_announces_epochs_to_an_iterable_dataset():
    from optastra import Trainer

    stream = _EpochStream()
    task = Task.create("classification_task")
    loader = build_dataloader(stream, task=task, batch_size=4, distributed=False)
    model = _Logits()
    Trainer(model, task, torch.optim.SGD(model.parameters(), lr=0.1), device="cpu").train(loader, max_iter=5)
    assert stream.epochs == [0, 1, 2]    # 2 batches per epoch -> epochs 0, 1, 2 start within 5 steps


def test_resume_and_evaluate_work_with_an_iterable_dataset(tmp_path):
    """A DataLoader over an IterableDataset has no length: resuming (epoch
    bookkeeping) and evaluate() must cope with that."""
    from optastra import Trainer
    from optastra.training.hooks import CheckpointHook, ResumeHook

    task = Task.create("classification_task")

    def run(hooks, max_iter):
        torch.manual_seed(0)
        model = _Logits()
        trainer = Trainer(model, task, torch.optim.SGD(model.parameters(), lr=0.1), hooks=hooks, device="cpu")
        loader = build_dataloader(_EpochStream(), task=task, batch_size=4, distributed=False)
        trainer.train(loader, max_iter=max_iter)
        return trainer, loader

    run([CheckpointHook(str(tmp_path), save_every=2)], max_iter=3)
    trainer, loader = run([ResumeHook(str(tmp_path))], max_iter=5)
    assert trainer.state.start_iter == 3
    assert set(trainer.evaluate(loader)) == {"accuracy", "total_loss"}
