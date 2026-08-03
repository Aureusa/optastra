import torch
import torch.nn as nn

from optastra.optim import Scheduler

# Import for registry side effects.
import optastra.optim.warmup_cosine  # noqa: F401


def test_scheduler_create_builds_warmup_cosine_and_steps_learning_rate():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    scheduler = Scheduler.create("warmup_cosine", optimizer, total_steps=10, warmup_steps=2)

    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

    lrs = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] > 0.0
    assert lrs[1] >= lrs[0]
    assert lrs[-1] <= lrs[1]


def test_scheduler_create_rejects_unknown_name():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    try:
        Scheduler.create("missing_scheduler", optimizer)
        assert False, "Expected ValueError for missing scheduler"
    except ValueError as e:
        assert "not registered" in str(e)