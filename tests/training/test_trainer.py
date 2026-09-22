import pytest
import torch
import torch.nn as nn

from optastra.training.trainer import Trainer
from optastra.training.hooks.base import Hook
from optastra.training.hooks.checkpoint import CheckpointHook
from optastra.training.hooks.resume import ResumeHook
from optastra.training.hooks.scheduler import SchedulerHook
from optastra.tasks.base import TaskStepOutput


class _EvalOnlyTask:
    def run_step(self, model, batch, stage="train"):
        if stage == "val":
            # Return a deterministic metric from the batch.
            return TaskStepOutput(
                loss=None,
                metrics={"mae": float(batch["targets"].mean().item())},
            )

        x = batch["inputs"]
        preds = model(x)
        loss = ((preds - batch["targets"]) ** 2).mean()
        return TaskStepOutput(loss=loss, losses={"mse": loss})


class _RecorderHook:
    def __init__(self):
        self.events = []

    def before_eval(self, state):
        self.events.append(("before_eval", state.iter))

    def after_eval(self, state):
        self.events.append(("after_eval", state.iter))

    def before_eval_step(self, state):
        self.events.append(("before_eval_step", state.iter))

    def after_eval_step(self, state):
        self.events.append(("after_eval_step", state.iter))


def test_resolve_device_cpu_is_supported():
    resolved = Trainer._resolve_device("cpu")
    assert resolved.type == "cpu"


def test_resolve_device_cuda_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA device is available"):
        Trainer._resolve_device("cuda")


def test_move_to_device_handles_nested_containers():
    cpu = torch.device("cpu")
    nested = {
        "x": torch.randn(2, 3),
        "y": [torch.randn(1), (torch.randn(1), {"z": torch.randn(1)})],
        "meta": "keep-me",
    }

    moved = Trainer._move_to_device(nested, cpu)

    assert moved["x"].device.type == "cpu"
    assert moved["y"][0].device.type == "cpu"
    assert moved["y"][1][0].device.type == "cpu"
    assert moved["y"][1][1]["z"].device.type == "cpu"
    assert moved["meta"] == "keep-me"


def test_evaluate_averages_metrics_over_batches():
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model=model, task=_EvalOnlyTask(), optimizer=optimizer, device="cpu")

    dataloader = [
        {"inputs": torch.randn(2, 4), "targets": torch.tensor([[1.0], [3.0]])},
        {"inputs": torch.randn(2, 4), "targets": torch.tensor([[5.0], [7.0]])},
    ]

    metrics = trainer.evaluate(dataloader)

    # Mean of batch means: ((1+3)/2 + (5+7)/2) / 2 = (2 + 6) / 2 = 4
    assert metrics["mae"] == pytest.approx(4.0)


def test_evaluate_runs_eval_lifecycle_hooks():
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    recorder = _RecorderHook()
    trainer = Trainer(model=model, task=_EvalOnlyTask(), optimizer=optimizer, hooks=[recorder], device="cpu")

    dataloader = [
        {"inputs": torch.randn(2, 4), "targets": torch.tensor([[1.0], [3.0]])},
    ]

    trainer.evaluate(dataloader)

    assert recorder.events == [
        ("before_eval", 0),
        ("before_eval_step", 0),
        ("after_eval_step", 0),
        ("after_eval", 0),
    ]


class _StepRecorder(Hook):
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.iters = []
        self.scheduler_steps = []

    def after_step(self, state):
        self.iters.append(state.iter)
        self.scheduler_steps.append(self.scheduler.last_epoch)


def _build_trainer(hooks_fn):
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    trainer = Trainer(model=model, task=_EvalOnlyTask(), optimizer=optimizer, device="cpu")
    trainer.register_hooks(hooks_fn(scheduler))
    return trainer


def test_resume_continues_after_checkpointed_iter(tmp_path):
    dataloader = [
        {"inputs": torch.randn(2, 4), "targets": torch.randn(2, 1)} for _ in range(3)
    ]

    first = _build_trainer(lambda s: [CheckpointHook(str(tmp_path), save_every=2), SchedulerHook(s, log_lr=False)])
    first.train(dataloader, max_iter=5)  # iters 0..4, checkpoints at 2 and 4

    recorder = None

    def resumed_hooks(scheduler):
        nonlocal recorder
        recorder = _StepRecorder(scheduler)
        return [ResumeHook(str(tmp_path)), SchedulerHook(scheduler, log_lr=False), recorder]

    second = _build_trainer(resumed_hooks)
    second.train(dataloader, max_iter=8)

    assert recorder.iters == [5, 6, 7]
    assert recorder.scheduler_steps == [6, 7, 8]  # scheduler state restored in step with iter
    assert second.state.epoch == first.state.epoch == 1  # restored, and 3 steps fit in one fresh pass
