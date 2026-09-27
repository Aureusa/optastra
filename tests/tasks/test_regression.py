import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from optastra import Optimizer, Sample, Task, Trainer, build_dataloader
from optastra.nn.features import HeadOutput
from optastra.tasks import RegressionEvaluator, RegressionTask, RegressionTaskConfig
from optastra.tasks.base import TaskStepOutput


class LinearRegressor(nn.Module):
    """(B, F) inputs -> HeadOutput(values=(B, D)): the smallest model the task accepts."""

    def __init__(self, in_features: int = 3, num_outputs: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, num_outputs)

    def forward(self, x: torch.Tensor) -> HeadOutput:
        return HeadOutput(values=self.linear(x))


# --- registration / config -----------------------------------------------------

def test_regression_task_is_registered_with_mse_default():
    assert "regression_task" in Task.list_all()
    task = Task.create("regression_task")
    assert isinstance(task, RegressionTask)
    assert task.cfg.loss == "mse"
    assert task.collate == "dense"


def test_unknown_loss_is_rejected_at_construction():
    with pytest.raises(ValueError, match="loss must be one of"):
        Task.create("regression_task", loss="cross_entropy")


# --- targets ---------------------------------------------------------------------

def test_preprocess_targets_reshapes_scalars_to_column_and_casts_to_float():
    task = RegressionTask()
    out = task.preprocess_targets({"values": torch.tensor([1, 2, 3])})
    assert out["values"].dtype == torch.float32
    assert torch.equal(out["values"], torch.tensor([[1.0], [2.0], [3.0]]))


def test_preprocess_targets_keeps_multi_output_and_accepts_raw_tensor_and_custom_key():
    multi = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.equal(RegressionTask().preprocess_targets(multi)["values"], multi)
    task = RegressionTask(RegressionTaskConfig(target_key="redshift"))
    assert torch.equal(task.preprocess_targets({"redshift": multi})["values"], multi)


# --- losses ----------------------------------------------------------------------

@pytest.mark.parametrize("loss, reference", [
    ("mse", lambda p, t: F.mse_loss(p, t)),
    ("l1", lambda p, t: F.l1_loss(p, t)),
    ("huber", lambda p, t: F.huber_loss(p, t, delta=0.5)),
])
def test_losses_match_torch_reference(loss, reference):
    task = RegressionTask(RegressionTaskConfig(loss=loss, huber_delta=0.5))
    pred = torch.tensor([[0.0, 1.0], [2.0, -1.0]])
    true = torch.tensor([[0.5, 1.0], [0.0, 1.0]])
    losses = task.compute_losses(HeadOutput(values=pred), {"values": true})
    assert list(losses) == [f"{loss}_loss"]
    torch.testing.assert_close(losses[f"{loss}_loss"], reference(pred, true))


def test_mse_value_by_hand():
    task = RegressionTask()
    losses = task.compute_losses(HeadOutput(values=torch.tensor([[1.0], [3.0]])), {"values": torch.tensor([[0.0], [0.0]])})
    assert losses["mse_loss"].item() == pytest.approx((1 + 9) / 2)


def test_shape_mismatch_between_head_and_targets_is_a_clear_error():
    task = RegressionTask()
    with pytest.raises(ValueError, match="num_outputs"):
        task.compute_losses(HeadOutput(values=torch.zeros(4, 2)), {"values": torch.zeros(4, 1)})


# --- run_step stages --------------------------------------------------------------

def test_run_step_per_stage():
    torch.manual_seed(0)
    task, model = RegressionTask(), LinearRegressor()
    batch = {"inputs": torch.randn(4, 3), "targets": {"values": torch.randn(4)}}

    train = task.run_step(model, batch, stage="train")
    assert train.loss is not None and train.loss.requires_grad
    assert train.metrics == {} and train.predictions is None

    val = task.run_step(model, batch, stage="val")
    expected_mae = (model(batch["inputs"]).values - batch["targets"]["values"][:, None]).abs().mean().item()
    assert val.metrics["mae"] == pytest.approx(expected_mae)
    assert val.predictions.shape == (4, 1)

    predict = task.run_step(model, {"inputs": batch["inputs"]}, stage="predict")
    assert predict.loss is None and predict.predictions.shape == (4, 1)


def test_missing_targets_rejected_outside_predict():
    task = RegressionTask()
    with pytest.raises(ValueError, match="targets"):
        task.validate_batch({"inputs": torch.zeros(2, 3)}, stage="train")
    task.validate_batch({"inputs": torch.zeros(2, 3)}, stage="predict")   # fine


def test_head_without_values_is_rejected():
    task = RegressionTask()
    with pytest.raises(ValueError, match="values"):
        task.validate_predictions(HeadOutput(logits=torch.zeros(2, 3)))


# --- evaluator --------------------------------------------------------------------

def _output(pred, true, loss):
    return TaskStepOutput(loss=torch.tensor(loss), predictions=torch.tensor(pred), targets={"values": torch.tensor(true)})


def test_evaluator_is_exact_over_unequal_batches_and_multiple_outputs():
    preds = [[[1.0, 10.0], [2.0, 20.0], [4.0, 25.0]], [[3.0, 40.0]]]
    trues = [[[1.5, 12.0], [2.0, 18.0], [3.0, 30.0]], [[5.0, 40.0]]]
    losses = [0.3, 0.9]   # per-batch means

    ev = RegressionEvaluator()
    ev.reset()
    for p, t, l in zip(preds, trues, losses):
        ev.process(_output(p, t, l), batch={})
    got = ev.summarize()

    pred = torch.tensor(preds[0] + preds[1], dtype=torch.float64)
    true = torch.tensor(trues[0] + trues[1], dtype=torch.float64)
    err = pred - true
    r2_per_output = 1 - err.pow(2).sum(0) / (true - true.mean(0)).pow(2).sum(0)
    assert got["mae"] == pytest.approx(err.abs().mean().item())
    assert got["rmse"] == pytest.approx(err.pow(2).mean().sqrt().item())
    assert got["r2"] == pytest.approx(r2_per_output.mean().item())
    assert got["total_loss"] == pytest.approx((0.3 * 3 + 0.9 * 1) / 4)   # weighted by batch size, not 0.6


def test_evaluator_r2_is_one_for_perfect_predictions_and_omitted_for_constant_targets():
    ev = RegressionEvaluator()
    ev.process(_output([[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]], 0.0), batch={})
    assert ev.summarize()["r2"] == pytest.approx(1.0)
    assert ev.summarize()["mae"] == 0.0

    ev.reset()
    ev.process(_output([[1.0], [2.0]], [[5.0], [5.0]], 1.0), batch={})
    summary = ev.summarize()
    assert "r2" not in summary
    assert summary["rmse"] == pytest.approx(math.sqrt((16 + 9) / 2))


def test_evaluator_reset_clears_state():
    ev = RegressionEvaluator()
    ev.process(_output([[1.0]], [[0.0]], 1.0), batch={})
    ev.reset()
    assert ev.summarize() == {}


# --- end to end with the Trainer ---------------------------------------------------

class _LinearData(torch.utils.data.Dataset):
    """y = x @ w + b exactly, two outputs."""

    def __init__(self, n: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, generator=g)
        self.y = self.x @ torch.tensor([[1.0, -2.0], [0.5, 0.0], [-1.0, 3.0]]) + torch.tensor([0.5, -1.0])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return Sample(image=self.x[i], target={"values": self.y[i]})


def test_trainer_learns_linear_targets_and_reports_dataset_metrics():
    torch.manual_seed(0)
    task = Task.create("regression_task")
    model = LinearRegressor(num_outputs=2)
    train = build_dataloader(_LinearData(64, seed=0), task=task, batch_size=16, shuffle=True)
    val = build_dataloader(_LinearData(20, seed=1), task=task, batch_size=16)   # 16 + 4: unequal batches

    trainer = Trainer(model, task, Optimizer.create("adam", model, lr=0.05), device="cpu")
    before = trainer.evaluate(val)
    trainer.train(train, max_iter=300)
    after = trainer.evaluate(val)

    assert set(after) == {"total_loss", "mae", "rmse", "r2"}
    assert after["mae"] < 0.05 < before["mae"]
    assert after["r2"] > 0.99
    assert trainer.storage.latest()["val_r2"] == pytest.approx(after["r2"])
