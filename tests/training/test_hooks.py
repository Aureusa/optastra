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
    assert state.storage.latest()["val_accuracy"] == 0.8

    state.storage.put_scalar("val_accuracy", 0.1)
    hook.after_train(state)
    assert state.storage.latest()["val_accuracy"] == 0.8


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
    state.storage.put_scalars(loss=1.23, val_accuracy=0.91)

    hook = JSONWriterHook(output_dir=str(tmp_path), filename="metrics.jsonl")
    hook.after_step(state)

    path = tmp_path / "metrics.jsonl"
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["iter"] == 7
    assert record["loss"] == 1.23
    assert record["val_accuracy"] == 0.91
