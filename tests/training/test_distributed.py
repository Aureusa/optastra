"""Multi-process (DDP) tests.

Each test launches 2 processes the way `torchrun --nproc_per_node=2` does
(RANK / WORLD_SIZE / LOCAL_RANK / MASTER_* in the environment), on CPU with
the gloo backend. The code paths are the same as on 2 GPUs with NCCL:
DDP gradient averaging, sharded data, Evaluator.sync, rank-0-only hooks.
Every expectation is compared against an ordinary single-process run.
"""
import datetime
import json
import os
import socket

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from optastra import Optimizer, Sample, Task, Trainer, build_dataloader
from optastra.core import distributed as dist_utils
from optastra.data import ShardedSampler
from optastra.nn.features import HeadOutput
from optastra.tasks import ClassificationTask
from optastra.tasks.base import MeanMetricEvaluator
from optastra.training.hooks import CheckpointHook, Hook, JSONWriterHook, ResumeHook

WORLD_SIZE = 2
PER_RANK_BATCH = 4
NUM_TRAIN = 32   # 16 samples per rank -> 4 batches of 4 per epoch


# --- launching -------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _entry(rank: int, fn, port: int, out_dir: str, args: tuple) -> None:
    """What torchrun does for each process, then run `fn` inside the process group."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE),
    )
    torch.manual_seed(0)
    # gloo + a timeout: a desynchronised collective fails the test instead of hanging it.
    dist_utils.init_distributed(backend="gloo", timeout=datetime.timedelta(seconds=60))
    try:
        fn(rank, out_dir, *args)
    finally:
        dist_utils.cleanup()


def _launch(fn, out_dir, *args) -> None:
    mp.spawn(_entry, args=(fn, _free_port(), str(out_dir), args), nprocs=WORLD_SIZE, join=True)


# --- shared model / data ---------------------------------------------------------

class _Linear(nn.Module):
    def __init__(self, in_features: int = 3, out_features: int = 2, field: str = "values"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.field = field

    def forward(self, x: torch.Tensor) -> HeadOutput:
        return HeadOutput(**{self.field: self.linear(x)})


class _RegressionData(torch.utils.data.Dataset):
    def __init__(self, n: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, generator=g)
        self.y = torch.randn(n, 2, generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return Sample(image=self.x[i], target={"values": self.y[i]})


class _Stream(torch.utils.data.IterableDataset):
    def __iter__(self):
        yield from (_RegressionData(4)[i] for i in range(4))


class _ClassificationData(_RegressionData):
    def __getitem__(self, i):
        return Sample(image=self.x[i], target={"labels": torch.tensor(i % 4)})


class _MeanMetricClassification(ClassificationTask):
    """Classification scored by the generic default evaluator."""

    def build_evaluator(self):
        return MeanMetricEvaluator()


def _trainer(model, task, **kwargs) -> Trainer:
    return Trainer(model, task, Optimizer.create("sgd", model, lr=0.1), device="cpu", **kwargs)


# --- 1. training: 2 processes x batch 4 == 1 process x batch 8 --------------------

TRAIN_CONFIGS = [
    {"grad_accum_steps": 1, "shuffle": False},
    {"grad_accum_steps": 2, "shuffle": False},   # exercises DDP's no_sync on non-final micro-batches
    {"grad_accum_steps": 1, "shuffle": True},    # exercises set_epoch: a new, shared shuffle each epoch
    # torch.compile on top of the DDP wrapper, with accumulation (no_sync through the compiled module)
    {"grad_accum_steps": 2, "shuffle": False, "compile": {"backend": "eager"}},
]
TRAIN_ITERS = 6   # crosses epoch boundaries in every config


def _train_worker(rank: int, out_dir: str) -> None:
    results = []
    for cfg in TRAIN_CONFIGS:
        torch.manual_seed(0)
        model, task = _Linear(), Task.create("regression_task")
        loader = build_dataloader(_RegressionData(NUM_TRAIN), task=task, batch_size=PER_RANK_BATCH, shuffle=cfg["shuffle"])
        trainer = _trainer(model, task, grad_accum_steps=cfg["grad_accum_steps"], compile=cfg.get("compile", False))
        assert trainer.distributed and isinstance(loader.sampler, ShardedSampler)
        # An IterableDataset shards itself: build_dataloader must not attach a sampler.
        stream = build_dataloader(_Stream(), task=task, batch_size=PER_RANK_BATCH)
        assert not isinstance(stream.sampler, ShardedSampler)
        trainer.train(loader, max_iter=TRAIN_ITERS)
        results.append({"weights": model.state_dict(), "loss": trainer.storage.latest()["total_loss"]})
    torch.save(results, os.path.join(out_dir, f"train_rank{rank}.pt"))


def test_ddp_training_matches_single_process_with_the_global_batch(tmp_path):
    _launch(_train_worker, tmp_path)
    ranks = [torch.load(tmp_path / f"train_rank{r}.pt") for r in range(WORLD_SIZE)]

    for i, cfg in enumerate(TRAIN_CONFIGS):
        torch.manual_seed(0)
        model, task = _Linear(), Task.create("regression_task")
        # One process, the same global batch (2 x 4) and the same epoch order:
        # rank r's shard is order[r::2], so the ranks' batches together are exactly this batch.
        sampler = ShardedSampler(NUM_TRAIN, shuffle=cfg["shuffle"], seed=0, rank=0, world_size=1)
        loader = build_dataloader(_RegressionData(NUM_TRAIN), task=task, batch_size=PER_RANK_BATCH * WORLD_SIZE, sampler=sampler)
        trainer = _trainer(model, task, grad_accum_steps=cfg["grad_accum_steps"])
        assert not trainer.distributed
        trainer.train(loader, max_iter=TRAIN_ITERS)

        for rank_result in ranks:
            for key, value in model.state_dict().items():
                torch.testing.assert_close(rank_result[i]["weights"][key], value, msg=f"{cfg} {key}")
            # the logged loss is the mean over processes == the global-batch loss
            assert rank_result[i]["loss"] == pytest.approx(trainer.storage.latest()["total_loss"], rel=1e-5)


# --- 2. evaluation: sharded, exact, and safe with an empty shard ------------------

def _eval_setups():
    torch.manual_seed(1)
    return {
        "classification": (Task.create("classification_task"), _Linear(3, 4, field="logits"), _ClassificationData),
        "regression": (Task.create("regression_task"), _Linear(3, 2), _RegressionData),
        "mean_metric": (_MeanMetricClassification(), _Linear(3, 4, field="logits"), _ClassificationData),
    }


def _evaluate(num_samples: int) -> dict:
    results = {}
    for name, (task, model, data_cls) in _eval_setups().items():
        loader = build_dataloader(data_cls(num_samples, seed=2), task=task, batch_size=3)
        results[name] = _trainer(model, task).evaluate(loader)
    return results


def _eval_worker(rank: int, out_dir: str, sizes: list[int]) -> None:
    torch.save({n: _evaluate(n) for n in sizes}, os.path.join(out_dir, f"eval_rank{rank}.pt"))


def test_ddp_evaluation_is_exact_including_uneven_and_empty_shards(tmp_path):
    sizes = [7, 1]   # 7 -> shards of 4 and 3 samples; 1 -> rank 1 gets no batches at all
    _launch(_eval_worker, tmp_path, sizes)
    ranks = [torch.load(tmp_path / f"eval_rank{r}.pt") for r in range(WORLD_SIZE)]

    for n in sizes:
        expected = _evaluate(n)   # plain single-process evaluation over all n samples
        for rank_result in ranks:
            assert set(rank_result[n]) == set(expected)
            for name, metrics in expected.items():
                assert rank_result[n][name] == pytest.approx(metrics), (n, name)


# --- 3. rank-0-only side effects, and resuming a DDP run ------------------------------

class _StopAfter(Hook):
    def __init__(self, stop_after: int):
        self.stop_after = stop_after

    def after_step(self, state):
        if state.iter == self.stop_after:
            state.should_stop = True


def _run(out_dir: str, hooks: list, max_iter: int) -> nn.Module:
    torch.manual_seed(0)
    model, task = _Linear(), Task.create("regression_task")
    loader = build_dataloader(_RegressionData(NUM_TRAIN), task=task, batch_size=PER_RANK_BATCH, shuffle=True)
    _trainer(model, task, hooks=hooks).train(loader, max_iter=max_iter)
    return model


def _side_effects_worker(rank: int, out_dir: str) -> None:
    # Reference: 8 uninterrupted iterations (2 epochs), logging every step.
    reference = _run(os.path.join(out_dir, "reference"),
                     [JSONWriterHook(os.path.join(out_dir, "reference", "logs"), log_every=1)], max_iter=8)

    # Interrupted after iteration 3 (the end of epoch 0, where the checkpoint is) ...
    run_dir = os.path.join(out_dir, "run")
    _run(run_dir, [CheckpointHook(run_dir, save_every=3), _StopAfter(3)], max_iter=8)
    dist_utils.barrier()   # a restarted job would find the checkpoint on disk; here, wait for rank 0's write
    # ... and resumed by a fresh trainer on every rank.
    resumed = _run(run_dir, [ResumeHook(run_dir), CheckpointHook(run_dir, save_every=3)], max_iter=8)

    torch.save({"reference": reference.state_dict(), "resumed": resumed.state_dict()},
               os.path.join(out_dir, f"resume_rank{rank}.pt"))


def test_ddp_side_effects_happen_once_and_resume_is_exact(tmp_path):
    _launch(_side_effects_worker, tmp_path)

    # JSONWriterHook is main_process_only: one record per iteration, not one per process.
    with open(tmp_path / "reference" / "logs" / "metrics.jsonl") as f:
        train_iters = [json.loads(line)["iter"] for line in f if '"train"' in line]
    assert train_iters == list(range(8))

    assert sorted(os.listdir(tmp_path / "run")) == ["ckpt_3.pt", "ckpt_6.pt"]   # no stray .tmp files from racing writers

    for rank in range(WORLD_SIZE):
        result = torch.load(tmp_path / f"resume_rank{rank}.pt")
        for key, value in result["reference"].items():
            assert torch.equal(result["resumed"][key], value), (rank, key)
