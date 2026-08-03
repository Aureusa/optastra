import pytest
import torch
import torch.nn as nn

from optastra.training.trainer import Trainer
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
