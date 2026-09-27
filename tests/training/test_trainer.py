import warnings

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


def test_evaluate_averages_metrics_weighted_by_batch_size():
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model=model, task=_EvalOnlyTask(), optimizer=optimizer, device="cpu")

    dataloader = [
        {"inputs": torch.randn(3, 4), "targets": torch.tensor([[1.0], [2.0], [3.0]])},
        {"inputs": torch.randn(1, 4), "targets": torch.tensor([[10.0]])},
    ]

    metrics = trainer.evaluate(dataloader)

    # Per-sample mean (1+2+3+10)/4 = 4, not the mean of batch means (2+10)/2 = 6.
    assert metrics["mae"] == pytest.approx(4.0)
    assert trainer.storage.latest()["val_mae"] == pytest.approx(4.0)
    assert trainer.state.eval_results == {"val_mae": pytest.approx(4.0)}


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


# --- runtime policy: precision, grad accumulation, clipping -------------------

class _RegressionTask:
    """Mean-squared-error task that records the dtype its forward ran in."""

    def __init__(self):
        self.pred_dtypes = []

    def run_step(self, model, batch, stage="train"):
        preds = model(batch["inputs"])
        self.pred_dtypes.append(preds.dtype)
        loss = ((preds.float() - batch["targets"]) ** 2).mean()
        return TaskStepOutput(loss=loss, losses={"mse": loss}, metrics={} if stage == "train" else {"mse": loss})


def _regression_batches(num_batches: int, batch_size: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return [
        {"inputs": torch.randn(batch_size, 4, generator=g), "targets": torch.randn(batch_size, 1, generator=g)}
        for _ in range(num_batches)
    ]


def _linear(seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(4, 1)


@pytest.mark.parametrize("precision, dtype", [("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)])
def test_trainer_precision_controls_autocast_on_cpu(precision, dtype):
    model = _linear()
    before = model.weight.detach().clone()
    task = _RegressionTask()
    trainer = Trainer(model, task, torch.optim.SGD(model.parameters(), lr=0.1), device="cpu", precision=precision)
    assert trainer.grad_scaler.is_enabled() == (precision == "fp16")
    if precision == "fp16":
        # The default init_scale (2**16) overflows fp16 grads on the first steps, which
        # GradScaler (correctly) skips; start small so the two steps here are real updates.
        trainer.grad_scaler = torch.amp.GradScaler("cpu", init_scale=1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # CPU runs must be warning-free
        trainer.train(_regression_batches(2, 4), max_iter=2)
        trainer.evaluate(_regression_batches(1, 4))

    assert task.pred_dtypes == [dtype, dtype, dtype]  # 2 train steps + 1 eval step
    assert not torch.equal(model.weight, before)
    assert model.weight.dtype == torch.float32  # master weights stay fp32


def test_trainer_rejects_unknown_precision():
    model = _linear()
    with pytest.raises(ValueError, match="precision"):
        Trainer(model, _RegressionTask(), torch.optim.SGD(model.parameters(), lr=0.1), device="cpu", precision="int8")


def test_grad_accumulation_matches_one_big_batch():
    big = _regression_batches(1, 8)[0]
    micro = [{k: v[:4] for k, v in big.items()}, {k: v[4:] for k, v in big.items()}]

    model_big, model_acc = _linear(), _linear()
    trainer_big = Trainer(model_big, _RegressionTask(), torch.optim.SGD(model_big.parameters(), lr=0.1), device="cpu")
    trainer_acc = Trainer(model_acc, _RegressionTask(), torch.optim.SGD(model_acc.parameters(), lr=0.1),
                          device="cpu", grad_accum_steps=2)

    trainer_big.train([big], max_iter=1)
    trainer_acc.train(micro, max_iter=1)  # one iter == one optimizer step over both micro-batches

    assert torch.allclose(model_acc.weight, model_big.weight, atol=1e-6)
    assert torch.allclose(model_acc.bias, model_big.bias, atol=1e-6)
    # Logged loss is the mean of the micro-batch losses == the big-batch loss (equal halves).
    assert trainer_acc.storage.latest()["total_loss"] == pytest.approx(trainer_big.storage.latest()["total_loss"], rel=1e-5)
    assert trainer_acc.state.iter == 0 and trainer_acc.state.epoch == 0


def test_grad_accumulation_counts_optimizer_steps_and_consumes_n_batches():
    model = _linear()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    recorder = _StepRecorder(scheduler)
    trainer = Trainer(model, _RegressionTask(), optimizer, hooks=[SchedulerHook(scheduler, log_lr=False), recorder],
                      device="cpu", grad_accum_steps=3)

    trainer.train(_regression_batches(6, 2), max_iter=4)  # 4 steps x 3 micro-batches = 12 batches = 2 epochs

    assert recorder.iters == [0, 1, 2, 3]
    assert recorder.scheduler_steps == [1, 2, 3, 4]
    assert trainer.state.epoch == 1  # the third pass is never started: 12 batches fit in 2 epochs


def test_clip_grad_norm_clips_and_logs_pre_clip_norm():
    model = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(10.0)
    batch = {"inputs": torch.ones(2, 4), "targets": torch.zeros(2, 1)}

    # Expected gradient of mean((w.x)^2) w.r.t. w for x=1: 2 * (w.x) * x = 2 * 40 = 80 per entry.
    expected_norm = (4 * 80.0 ** 2) ** 0.5
    trainer = Trainer(model, _RegressionTask(), torch.optim.SGD(model.parameters(), lr=1.0),
                      device="cpu", clip_grad_norm=1.0)
    trainer.train([batch], max_iter=1)

    assert trainer.storage.latest()["grad_norm"] == pytest.approx(expected_norm)
    # The applied update is the clipped gradient: unit norm, pointing along -grad.
    update = torch.full((1, 4), 10.0) - model.weight.detach()
    assert update.norm().item() == pytest.approx(1.0, rel=1e-4)
    assert torch.allclose(update, torch.full((1, 4), 0.5), atol=1e-4)


def test_no_grad_norm_logged_without_clipping():
    model = _linear()
    trainer = Trainer(model, _RegressionTask(), torch.optim.SGD(model.parameters(), lr=0.1), device="cpu")
    trainer.train(_regression_batches(1, 2), max_iter=1)
    assert "grad_norm" not in trainer.storage.keys()


# --- epochs and eval iteration -------------------------------------------------

class _EpochRecorder(Hook):
    def __init__(self):
        self.events = []

    def before_epoch(self, state):
        self.events.append(("before_epoch", state.epoch, state.iter))

    def after_epoch(self, state):
        self.events.append(("after_epoch", state.epoch, state.iter))


def test_epoch_hooks_fire_around_loader_rollover():
    model = _linear()
    recorder = _EpochRecorder()
    trainer = Trainer(model, _RegressionTask(), torch.optim.SGD(model.parameters(), lr=0.1),
                      hooks=[recorder], device="cpu")

    trainer.train(_regression_batches(3, 2), max_iter=7)  # batches: e0 = iters 0-2, e1 = 3-5, e2 = 6

    assert recorder.events == [
        ("before_epoch", 0, 0),
        ("after_epoch", 0, 3),    # detected when iter 3 asks for a batch
        ("before_epoch", 1, 3),
        ("after_epoch", 1, 6),
        ("before_epoch", 2, 6),   # epoch 2 is still in progress at the end: no after_epoch
    ]
    assert trainer.state.epoch == 2


def test_train_raises_on_empty_loader():
    model = _linear()
    trainer = Trainer(model, _RegressionTask(), torch.optim.SGD(model.parameters(), lr=0.1), device="cpu")
    with pytest.raises(ValueError, match="no batches"):
        trainer.train([], max_iter=1)


class _CountingLoader:
    """Iterable that counts how often it is iterated and how many batches it yields."""

    def __init__(self, batches):
        self.batches = batches
        self.iter_calls = 0
        self.yielded = 0

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        self.iter_calls += 1
        for b in self.batches:
            self.yielded += 1
            yield b


def test_evaluate_iterates_loader_exactly_once():
    model = _linear()
    trainer = Trainer(model, _RegressionTask(), torch.optim.SGD(model.parameters(), lr=0.1), device="cpu")
    loader = _CountingLoader(_regression_batches(2, 2))

    trainer.evaluate(loader)

    assert loader.iter_calls == 1
    assert loader.yielded == 2


def test_evaluate_handles_empty_loader():
    model = _linear()
    recorder = _RecorderHook()
    trainer = Trainer(model, _RegressionTask(), torch.optim.SGD(model.parameters(), lr=0.1),
                      hooks=[recorder], device="cpu")

    assert trainer.evaluate([]) == {}
    assert recorder.events == [("before_eval", 0), ("after_eval", 0)]
    assert not any(k.startswith("val_") for k in trainer.storage.keys())
    assert model.training


def test_evaluate_keeps_eval_step_scalars_out_of_train_storage_and_restores_state():
    model = _linear()
    trainer = Trainer(model, _RegressionTask(), torch.optim.SGD(model.parameters(), lr=0.1), device="cpu")
    trainer.train(_regression_batches(1, 2), max_iter=1)
    train_output = trainer.state.last_output
    train_loss = trainer.storage.smoothed("total_loss")

    trainer.evaluate(_regression_batches(2, 2, seed=1))

    assert trainer.storage.smoothed("total_loss") == train_loss  # smoothing window untouched
    assert set(trainer.storage.keys()) >= {"val_total_loss", "val_mse"}
    assert not any(k.startswith("val_step_") for k in trainer.storage.keys())
    assert "val_step_total_loss" in trainer.storage.keys(axis="eval_iter")
    assert trainer.state.last_output is train_output
