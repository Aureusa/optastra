import pytest
import torch
import torch.nn as nn

from optastra.algorithms.simclr.routine import SimCLRConfig, SimCLRTask


class _ToySimCLRModel(nn.Module):
    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.proj = nn.Linear(3 * 8 * 8, embed_dim)

    def forward(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.proj(v.flatten(1)) for v in views]


def test_simclr_task_train_step_returns_nt_xent_loss():
    model = _ToySimCLRModel()
    task = SimCLRTask(SimCLRConfig(temperature=0.5))
    views = [torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8)]

    output = task.run_step(model, {"views": views}, stage="train")

    assert output.loss is not None
    assert "nt_xent_loss" in output.losses
    assert output.losses["nt_xent_loss"].ndim == 0


def test_simclr_task_predict_step_decodes_raw_predictions():
    model = _ToySimCLRModel()
    task = SimCLRTask(SimCLRConfig())
    views = [torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)]

    output = task.run_step(model, {"views": views}, stage="predict")

    assert output.loss is None
    assert isinstance(output.predictions, list)
    assert len(output.predictions) == 2
    assert output.predictions[0].shape[0] == 2


def test_simclr_task_requires_minimum_number_of_views():
    model = _ToySimCLRModel()
    task = SimCLRTask(SimCLRConfig())

    with pytest.raises(ValueError, match="requires >= 2 views"):
        task.run_step(model, {"views": [torch.randn(2, 3, 8, 8)]}, stage="train")
