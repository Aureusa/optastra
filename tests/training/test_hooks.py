import json
import logging
import random

import pytest
import torch
import torch.nn as nn

from optastra.nn.features import HeadOutput
from optastra.tasks.classification import ClassificationTask

from optastra.training.checkpointer import Checkpointer
from optastra.training.hooks.best_metric import BestMetricTracker
from optastra.training.hooks.checkpoint import BestCheckpointHook, CheckpointHook
from optastra.training.hooks.early_stopping import EarlyStoppingHook
from optastra.training.hooks.resume import ResumeHook
from optastra.training.hooks.ema import EMAHook
from optastra.training.hooks.eval import EvalHook
from optastra.training.hooks.logging import ConsoleLoggerHook
from optastra.training.trainer import Trainer
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


def _eval_at(state: TrainerState, it: int, **metrics: float) -> None:
    """Simulates Trainer.evaluate() writing val metrics at training iter `it`."""
    state.iter = state.storage.iter = it
    state.storage.put_scalars(**metrics)


def test_early_stopping_sets_should_stop_after_patience():
    state = _build_state()
    hook = EarlyStoppingHook(metric="loss", patience=2, mode="min")

    _eval_at(state, 10, loss=1.0)
    hook.after_eval(state)
    assert state.should_stop is False

    _eval_at(state, 20, loss=1.1)
    hook.after_eval(state)
    assert state.should_stop is False

    _eval_at(state, 30, loss=1.2)
    hook.after_eval(state)
    assert state.should_stop is True


def test_early_stopping_rejects_tracker_and_metric_together():
    with pytest.raises(TypeError):
        EarlyStoppingHook(metric="loss", tracker=BestMetricTracker("loss"))


def test_tracker_verdict_is_cached_per_iter_so_sharing_is_order_independent():
    state = _build_state()
    tracker = BestMetricTracker("val_loss", "min")

    _eval_at(state, 10, val_loss=1.0)
    assert tracker.update(state) is True
    assert tracker.update(state) is True  # second consumer, same eval
    assert tracker.best == 1.0

    _eval_at(state, 20, val_loss=2.0)
    assert tracker.update(state) is False
    assert tracker.update(state) is False


def test_tracker_ignores_stale_metric():
    state = _build_state()
    tracker = BestMetricTracker("val_loss", "min")
    _eval_at(state, 10, val_loss=1.0)
    tracker.update(state)

    state.iter = state.storage.iter = 20  # eval at 20 wrote nothing
    assert tracker.update(state) is None


def test_best_checkpoint_hook_saves_only_on_improvement(tmp_path):
    state = _build_state()
    tracker = BestMetricTracker("val_loss", "min")
    best_hook = BestCheckpointHook(str(tmp_path), tracker=tracker)
    stop_hook = EarlyStoppingHook(tracker=tracker, patience=5)
    state.hooks = [stop_hook, best_hook]  # consumer order must not matter
    best_path = tmp_path / "ckpt_best_val_loss.pt"

    _eval_at(state, 10, val_loss=1.0)
    for hook in state.hooks:
        hook.after_eval(state)
    assert best_path.exists()
    assert torch.load(best_path)["iter"] == 10
    assert stop_hook.bad_evals == 0

    _eval_at(state, 20, val_loss=2.0)
    for hook in state.hooks:
        hook.after_eval(state)
    assert torch.load(best_path)["iter"] == 10
    assert stop_hook.bad_evals == 1

    _eval_at(state, 30, val_loss=0.5)
    for hook in state.hooks:
        hook.after_eval(state)
    ckpt = torch.load(best_path)
    assert ckpt["iter"] == 30
    assert ckpt["extra"] == {"metric": "val_loss", "value": 0.5}
    assert stop_hook.bad_evals == 0


def test_resume_picks_latest_periodic_checkpoint_and_ignores_best(tmp_path):
    state = _build_state()
    checkpointer = Checkpointer(str(tmp_path))
    for it in (2, 10, 4):
        state.iter = it
        checkpointer.save(state, checkpointer.periodic_name(it))
    state.iter = 12
    checkpointer.save(state, checkpointer.best_name("val_loss"))

    fresh = _build_state()
    ResumeHook(checkpointer).before_train(fresh)
    assert fresh.iter == 10


def test_resume_without_checkpoints_or_dir_starts_from_scratch(tmp_path):
    state = _build_state()
    ResumeHook(str(tmp_path / "missing")).before_train(state)
    assert state.iter == 0


def test_resume_restores_same_class_hooks_in_order(tmp_path):
    state = _build_state()
    loss_hook = BestCheckpointHook(str(tmp_path), metric="val_loss", mode="min")
    acc_hook = BestCheckpointHook(str(tmp_path), metric="val_acc", mode="max")
    loss_hook.tracker.best, acc_hook.tracker.best = 0.3, 0.9
    state.hooks = [loss_hook, acc_hook]
    Checkpointer(str(tmp_path)).save(state, "ckpt_5.pt")

    fresh = _build_state()
    fresh_loss = BestCheckpointHook(str(tmp_path), metric="val_loss", mode="min")
    fresh_acc = BestCheckpointHook(str(tmp_path), metric="val_acc", mode="max")
    fresh.hooks = [ResumeHook(str(tmp_path)), fresh_loss, fresh_acc]
    fresh.hooks[0].before_train(fresh)
    assert fresh_loss.tracker.best == 0.3
    assert fresh_acc.tracker.best == 0.9


def test_early_stopping_loads_pre_tracker_checkpoint_state():
    hook = EarlyStoppingHook(metric="loss")
    hook.load_state_dict({"best": 0.7, "bad_evals": 3})
    assert hook.tracker.best == 0.7
    assert hook.bad_evals == 3


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


def test_eval_hook_runs_eval_fn_on_period_and_after_train_without_writing_storage():
    state = _build_state()
    state.max_iter = 10
    calls = []

    def eval_fn():
        calls.append(state.iter)
        return {"accuracy": 0.8}

    hook = EvalHook(eval_period=2, eval_fn=eval_fn, prefix="val")

    state.iter = 0
    hook.after_step(state)
    state.iter = 2
    hook.after_step(state)
    state.iter = 3
    hook.after_step(state)
    assert calls == [2]

    hook.after_train(state)  # final eval on the last weights
    assert calls == [2, 3]
    # Storage is written by Trainer.evaluate(), never re-written by the hook.
    assert "val_accuracy" not in state.storage.latest()

    state.iter = 4
    hook.after_step(state)
    hook.after_train(state)  # already evaluated at this iter -> skipped
    assert calls == [2, 3, 4]


def test_checkpoint_hook_writes_expected_checkpoint_file(tmp_path):
    state = _build_state()
    state.iter = 4

    hook = CheckpointHook(str(tmp_path), save_every=2)
    hook.after_step(state)

    ckpt = tmp_path / "ckpt_4.pt"
    assert ckpt.exists()


# def test_json_writer_hook_appends_metrics_records(tmp_path):
#     state = _build_state()
#     state.iter = 7
#     state.max_iter = 100
#     state.storage.put_scalars(loss=1.23, val_accuracy=0.91)

#     hook = JSONWriterHook(output_dir=str(tmp_path), filename="metrics.jsonl", log_every=5)
#     hook.after_step(state)
#     assert not (tmp_path / "metrics.jsonl").exists()

#     state.iter = 10
#     hook.after_step(state)

#     path = tmp_path / "metrics.jsonl"
#     lines = path.read_text().strip().splitlines()
#     assert len(lines) == 1

#     record = json.loads(lines[0])
#     assert record["iter"] == 10
#     assert record["phase"] == "train"
#     assert record["max_iter"] == 100
#     assert "scalars" in record
#     assert record["loss"] == 1.23
#     assert record["val_accuracy"] == 0.91


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


def test_json_writer_train_record_reads_iter_time_for_eta(tmp_path):
    state = _build_state()
    state.iter = state.storage.iter = 10
    state.max_iter = 110
    state.storage.put_scalars(total_loss=0.5, iter_time=2.0, data_time=0.5)

    hook = JSONWriterHook(output_dir=str(tmp_path), log_every=5)
    hook.after_step(state)

    record = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert record["eta_seconds"] == 200     # 2.0 s/iter * 100 remaining iters
    assert record["time"] == {"smoothed": 2.0, "last": 2.0}
    assert record["scalars"] == {"total_loss": 0.5}


def test_json_writer_writes_eval_summary_including_val_loss(tmp_path):
    state = _build_state()
    state.iter = 10
    state.eval_results = {"val_accuracy": 0.75, "val_total_loss": 0.4}

    hook = JSONWriterHook(output_dir=str(tmp_path))
    hook.after_eval(state)

    record = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert record["phase"] == "eval_summary"
    assert record["iter"] == 10
    assert record["metrics"] == {"val_accuracy": 0.75, "val_total_loss": 0.4}


def test_console_logger_logs_smoothed_total_loss(caplog):
    state = _build_state()
    state.max_iter = 10
    for it, loss in enumerate([1.0, 2.0, 3.0]):
        state.iter = state.storage.iter = it
        state.storage.put_scalar("total_loss", loss)

    hook = ConsoleLoggerHook(log_every=1)
    with caplog.at_level(logging.INFO, logger="optastra.train"):
        hook.after_step(state)
    assert "total_loss=2.0000" in caplog.text


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # torch.compile's own import-time warnings
def test_checkpoint_of_compiled_model_loads_into_uncompiled_model(tmp_path):
    state = _build_state()
    with torch.no_grad():
        state.model.weight.fill_(3.0)
    state.model = torch.compile(state.model)  # wrapper only; nothing is compiled until forward
    assert all(k.startswith("_orig_mod.") for k in state.model.state_dict())

    checkpointer = Checkpointer(str(tmp_path))
    path = checkpointer.save(state, "ckpt_1.pt")
    assert set(torch.load(path)["model"]) == {"weight", "bias"}

    fresh = _build_state()
    checkpointer.load(fresh, path)
    assert torch.equal(fresh.model.weight, torch.full((2, 2), 3.0))

    compiled = _build_state()
    compiled.model = torch.compile(compiled.model)
    checkpointer.load(compiled, path)
    assert torch.equal(compiled.model._orig_mod.weight, torch.full((2, 2), 3.0))


def test_checkpoint_round_trips_rng_state(tmp_path):
    state = _build_state()
    checkpointer = Checkpointer(str(tmp_path))
    torch.manual_seed(123)
    random.seed(123)
    path = checkpointer.save(state, "ckpt_1.pt")
    expected = (torch.rand(3), random.random())

    torch.manual_seed(0)
    random.seed(0)
    checkpointer.load(_build_state(), path)
    resumed = (torch.rand(3), random.random())

    assert torch.equal(resumed[0], expected[0])
    assert resumed[1] == expected[1]


def test_default_hooks_order_is_decided_by_priority(tmp_path):
    from optastra.training.hooks import default_hooks

    hooks = default_hooks(output_dir=str(tmp_path), best_metric="val_total_loss")
    assert isinstance(hooks[-1], ResumeHook)   # appended, not inserted
    model = nn.Linear(2, 2)
    trainer = Trainer(model, _NoOpTask(), torch.optim.SGD(model.parameters(), lr=0.1), hooks=hooks, device="cpu")
    assert isinstance(trainer.hooks[0], ResumeHook)   # ...but still runs first
    assert [type(h) for h in trainer.hooks[-2:]] == [CheckpointHook, BestCheckpointHook]  # priority 100


# --- end to end: train + EvalHook + JSONWriterHook ------------------------------

class _TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, inputs):
        return HeadOutput(logits=self.fc(inputs))


def test_eval_summaries_persist_in_storage_and_reach_metrics_jsonl(tmp_path):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    train = [{"inputs": torch.randn(4, 4, generator=g), "targets": torch.randint(0, 3, (4,), generator=g)}
             for _ in range(3)]
    # Unequal batch sizes: a mean of per-batch accuracies would be wrong here.
    val = [{"inputs": torch.randn(n, 4, generator=g), "targets": torch.randint(0, 3, (n,), generator=g)}
           for n in (5, 5, 2)]

    model = _TinyClassifier()
    task = ClassificationTask()
    trainer = Trainer(model, task, torch.optim.SGD(model.parameters(), lr=0.1), device="cpu")
    trainer.register_hooks([
        EvalHook(2, eval_fn=lambda: trainer.evaluate(val)),
        JSONWriterHook(str(tmp_path), log_every=1),
    ])
    trainer.train(train, max_iter=4)   # evals at iter 2 and after training (iter 3)

    # Expected values from the final weights, computed by hand over the whole val set.
    with torch.no_grad():
        inputs = torch.cat([b["inputs"] for b in val])
        labels = torch.cat([b["targets"] for b in val])
        logits = model(inputs).logits
        accuracy = (logits.argmax(1) == labels).float().mean().item()
        loss = torch.nn.functional.cross_entropy(logits, labels).item()

    latest = trainer.storage.latest()
    assert latest["val_accuracy"] == pytest.approx(accuracy)
    assert latest["val_total_loss"] == pytest.approx(loss, rel=1e-5)
    assert trainer.storage.history("val_accuracy")[-1][0] == 3   # tagged with the train iter

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    summaries = [r for r in records if r["phase"] == "eval_summary"]
    assert [r["iter"] for r in summaries] == [2, 3]
    assert summaries[-1]["metrics"]["val_accuracy"] == pytest.approx(accuracy)
    assert summaries[-1]["metrics"]["val_total_loss"] == pytest.approx(loss, rel=1e-5)
    # Train records keep reporting train scalars only.
    train_records = [r for r in records if r["phase"] == "train"]
    assert all(set(r["scalars"]) == {"total_loss", "ce_loss"} for r in train_records)
