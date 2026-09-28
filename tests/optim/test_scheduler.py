import pytest
import torch
import torch.nn as nn

from optastra.optim import Scheduler

# Import for registry side effects.
import optastra.optim.warmup_cosine  # noqa: F401


def _lr_curve(scheduler, optimizer, steps: int) -> list[float]:
    """LR in effect at steps 0..steps-1 (one optimizer.step() per step)."""
    lrs = []
    for _ in range(steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs


def test_warmup_cosine_lr_curve_values():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = Scheduler.create(
        "warmup_cosine", optimizer, total_steps=10, warmup_steps=2, warmup_start_factor=0.5, min_lr_factor=0.1,
    )

    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    lrs = _lr_curve(scheduler, optimizer, 12)

    assert lrs[0] == pytest.approx(0.05)              # warmup starts at start_factor * base_lr
    assert lrs[1] == pytest.approx(0.075)             # halfway through warmup
    assert lrs[2] == pytest.approx(0.1)               # warmup end = peak
    assert lrs[6] == pytest.approx(0.1 * (0.1 + 0.9 * 0.5))  # cosine midpoint: halfway to the floor
    assert lrs[10] == pytest.approx(0.01)             # total_steps reached: floor
    assert lrs[11] == pytest.approx(0.01)             # and stays there


def test_warmup_cosine_requires_total_steps():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="requires `total_steps`"):
        Scheduler.create("warmup_cosine", optimizer, warmup_steps=2)


def test_warmup_cosine_rejects_total_steps_shorter_than_warmup():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="must be >= warmup_steps"):
        Scheduler.create("warmup_cosine", optimizer, total_steps=10)  # default warmup_steps=500


def test_scheduler_create_rejects_unknown_name():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    try:
        Scheduler.create("missing_scheduler", optimizer)
        assert False, "Expected ValueError for missing scheduler"
    except ValueError as e:
        assert "not registered" in str(e)


def test_warmup_cosine_checkpoint_loads_with_weights_only(tmp_path):
    # torch.load defaults to weights_only=True (PyTorch >= 2.6): the scheduler
    # state inside a checkpoint must be plain data, or resuming a run and
    # export_backbone both fail.
    from optastra import Optimizer, Task, Trainer
    from optastra.training import Checkpointer
    from optastra.training.hooks import SchedulerHook

    model = torch.nn.Linear(2, 2)
    opt = Optimizer.create("sgd", model)
    sched = Scheduler.create("warmup_cosine", opt, warmup_steps=2, total_steps=10)
    for _ in range(3):
        opt.step()
        sched.step()
    trainer = Trainer(model, Task.create("regression_task"), opt, hooks=[SchedulerHook(sched)], device="cpu")
    path = Checkpointer(str(tmp_path)).save(trainer.state, "ckpt_3.pt")
    torch.load(path, weights_only=True)                      # must not raise

    fresh_opt = Optimizer.create("sgd", model)
    fresh = Scheduler.create("warmup_cosine", fresh_opt, warmup_steps=2, total_steps=10)
    fresh_trainer = Trainer(model, Task.create("regression_task"), fresh_opt, hooks=[SchedulerHook(fresh)], device="cpu")
    Checkpointer(str(tmp_path)).load(fresh_trainer.state, path)
    assert fresh.last_epoch == 3 and fresh.get_last_lr() == sched.get_last_lr()
    assert fresh.cfg.total_steps == 10                       # config still comes from the constructor
