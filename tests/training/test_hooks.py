import json

import torch
import torch.nn as nn

from optastra.training.hooks.checkpoint import CheckpointHook
from optastra.training.hooks.early_stopping import EarlyStoppingHook
from optastra.training.hooks.ema import EMAHook
from optastra.training.hooks.eval import EvalHook
from optastra.training.hooks.writer import JSONWriterHook
from optastra.training.state import TrainerState
from optastra.training.storage import EventStorage


class _NoOpTask:
    pass


def _build_state() -> TrainerState:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    storage = EventStorage()
    return TrainerState(
        model=model,
        task=_NoOpTask(),
        optimizer=optimizer,
        storage=storage,
        device=torch.device("cpu"),
    )


def test_early_stopping_sets_should_stop_after_patience():
    state = _build_state()
    hook = EarlyStoppingHook(metric="loss", patience=2, mode="min")

    state.storage.put_scalar("loss", 1.0)
    hook.after_epoch(state)
    assert state.should_stop is False

    state.storage.put_scalar("loss", 1.1)
    hook.after_epoch(state)
    assert state.should_stop is False

    state.storage.put_scalar("loss", 1.2)
    hook.after_epoch(state)
    assert state.should_stop is True


def test_ema_hook_updates_teacher_parameters():
    student = nn.Linear(3, 3, bias=False)
    teacher = nn.Linear(3, 3, bias=False)

    with torch.no_grad():
        student.weight.fill_(1.0)
        teacher.weight.zero_()

    hook = EMAHook(student=student, teacher=teacher, momentum=0.5)
    state = _build_state()
    hook.after_step(state)

    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, 0.5))
    assert all(not p.requires_grad for p in teacher.parameters())


def test_eval_hook_pushes_prefixed_metrics_on_period_and_after_train():
    state = _build_state()
    state.max_iter = 10

    hook = EvalHook(eval_period=2, eval_fn=lambda: {"accuracy": 0.8}, prefix="val")

    state.iter = 0
    hook.after_step(state)
    assert "val_accuracy" not in state.storage.latest()

    state.iter = 2
    hook.after_step(state)
    assert "val_accuracy" not in state.storage.latest()

    state.storage.put_scalar("loss", 0.1)
    hook.after_train(state)
    assert "val_accuracy" not in state.storage.latest()
    assert state.storage.latest()["loss"] == 0.1


def test_checkpoint_hook_writes_expected_checkpoint_file(tmp_path):
    state = _build_state()
    state.iter = 4

    hook = CheckpointHook(output_dir=str(tmp_path), save_every=2)
    hook.after_step(state)

    ckpt = tmp_path / "ckpt_4.pt"
    assert ckpt.exists()


def test_json_writer_hook_appends_metrics_records(tmp_path):
    state = _build_state()
    state.iter = 7
    state.max_iter = 100
    state.storage.put_scalars(loss=1.23, val_accuracy=0.91)

    hook = JSONWriterHook(output_dir=str(tmp_path), filename="metrics.jsonl", log_every=5)
    hook.after_step(state)
    assert not (tmp_path / "metrics.jsonl").exists()

    state.iter = 10
    hook.after_step(state)

    path = tmp_path / "metrics.jsonl"
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["iter"] == 10
    assert record["phase"] == "train"
    assert record["max_iter"] == 100
    assert "scalars" in record
    assert record["loss"] == 1.23
    assert record["val_accuracy"] == 0.91


def test_json_writer_hook_writes_eval_record_without_loss_like_metrics(tmp_path):
    state = _build_state()
    state.iter = 10
    state.max_iter = 100
    state.storage.eval_iter = 0
    state.storage.max_eval_iter = 2
    state.storage.put_scalars(axis="eval_iter", val_step_accuracy=0.8, val_step_loss=0.2, eval_time=0.01, eval_data_time=0.001)

    hook = JSONWriterHook(output_dir=str(tmp_path), filename="metrics.jsonl", log_every=5)
    state.storage.eval_iter = 1
    hook.after_eval_step(state)
    assert not (tmp_path / "metrics.jsonl").exists()

    state.storage.eval_iter = 5
    state.storage.put_scalars(axis="eval_iter", val_step_accuracy=0.8, val_step_loss=0.2, eval_time=0.01, eval_data_time=0.001)
    hook.after_eval_step(state)

    path = tmp_path / "metrics.jsonl"
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["phase"] == "eval"
    assert record["iter"] == 10
    assert record["eval_iter"] == 6
    assert record["max_eval_iter"] == 2
    assert "val_step_accuracy" in record["metrics"]
    assert "val_step_loss" not in record["metrics"]
