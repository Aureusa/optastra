"""Regression: predict one or more continuous values per image -- e.g.
photometric redshift, galaxy size, stellar mass.

Pairs with `vanilla_regression_head` (or any head that returns
`HeadOutput.values` of shape (B, num_outputs)). Targets live in
`Sample.target["values"]` as a scalar or a (num_outputs,) tensor per sample.

This module is also a compact template for writing a new Task: the step
methods `Task.run_step` calls, plus an Evaluator for dataset-level metrics.
"""
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Mapping

from .base import Task, Stage, TaskStepOutput
from ..core.distributed import sum_across_processes
from ..nn.features import HeadOutput


__all__ = ["RegressionTask", "RegressionTaskConfig", "RegressionEvaluator"]


_LOSSES = ("mse", "l1", "huber")


@dataclass
class RegressionTaskConfig:
    loss: str = "mse"             # "mse" | "l1" | "huber"
    huber_delta: float = 1.0      # only used by the "huber" loss
    target_key: str = "values"    # where the targets live in Sample.target


class RegressionEvaluator:
    """Dataset-level MAE, RMSE and R^2 (plus the sample-weighted loss),
    exact regardless of batch sizes.

    Accumulates running sums in float64 rather than storing predictions, so
    memory stays O(num_outputs) however large the eval set is. R^2 is
    computed per output and averaged over outputs whose targets vary; it is
    left out when no output varies (it is undefined for constant targets).
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.loss_sum = 0.0
        self.loss_count = 0
        self.abs_err = None    # per-output sums, created on the first batch
        self.sq_err = None
        self.target_sum = None
        self.target_sq_sum = None

    def process(self, output: TaskStepOutput, batch: Mapping[str, Any]) -> None:
        pred = output.predictions.detach().to(torch.float64)
        true = output.targets["values"].detach().to(torch.float64)
        err = pred - true
        if self.abs_err is None:
            self.abs_err, self.sq_err, self.target_sum, self.target_sq_sum = (
                torch.zeros(true.shape[1], dtype=torch.float64, device=true.device) for _ in range(4)
            )
        self.abs_err += err.abs().sum(dim=0)
        self.sq_err += err.pow(2).sum(dim=0)
        self.target_sum += true.sum(dim=0)
        self.target_sq_sum += true.pow(2).sum(dim=0)
        self.count += true.shape[0]
        if output.loss is not None:
            # output.loss is a mean over the batch -> weight it by the batch size.
            self.loss_sum += float(output.loss) * true.shape[0]
            self.loss_count += true.shape[0]

    def sync(self) -> None:
        """Add up every process's running sums (multi-process evaluation).
        A process that saw no batches has no per-output sums yet: it sends
        only its zero counts, and receives everyone else's sums."""
        local = {"count": self.count, "loss_sum": self.loss_sum, "loss_count": self.loss_count}
        if self.abs_err is not None:
            local |= {"abs_err": self.abs_err, "sq_err": self.sq_err,
                      "target_sum": self.target_sum, "target_sq_sum": self.target_sq_sum}
        total = sum_across_processes(local)
        self.count, self.loss_sum, self.loss_count = total["count"], total["loss_sum"], total["loss_count"]
        if "abs_err" in total:
            self.abs_err, self.sq_err = total["abs_err"], total["sq_err"]
            self.target_sum, self.target_sq_sum = total["target_sum"], total["target_sq_sum"]

    def summarize(self) -> dict[str, float]:
        results = {}
        if self.loss_count > 0:
            results["total_loss"] = self.loss_sum / self.loss_count
        if self.count == 0:
            return results
        num_values = self.count * self.abs_err.numel()
        results["mae"] = (self.abs_err.sum() / num_values).item()
        results["rmse"] = (self.sq_err.sum() / num_values).sqrt().item()
        # Total sum of squares around each output's mean: sum(y^2) - (sum y)^2 / n.
        ss_tot = self.target_sq_sum - self.target_sum.pow(2) / self.count
        varies = ss_tot > 1e-12 * self.target_sq_sum.clamp_min(1.0)
        if varies.any():
            results["r2"] = (1 - self.sq_err[varies] / ss_tot[varies]).mean().item()
        return results


class RegressionTask(Task):
    required_fields = ("values",)   # the head must return HeadOutput.values
    collate = "dense"               # fixed-size images and targets -> stacked tensors

    def __init__(self, cfg: RegressionTaskConfig = RegressionTaskConfig()):
        if cfg.loss not in _LOSSES:
            raise ValueError(f"RegressionTask loss must be one of {_LOSSES}, got {cfg.loss!r}.")
        self.cfg = cfg

    def validate_batch(self, batch: Mapping[str, Any], stage: Stage = "train"):
        if "inputs" not in batch:
            raise ValueError("RegressionTask expects 'inputs' in the batch.")
        if stage != "predict" and "targets" not in batch:
            raise ValueError(f"RegressionTask expects 'targets' in the batch for stage '{stage}'.")

    def split_inputs_targets(self, batch: Mapping[str, Any], stage: Stage = "train"):
        return batch["inputs"], (None if stage == "predict" else batch["targets"])

    def preprocess_targets(self, raw_targets: Mapping[str, Any] | torch.Tensor) -> dict[str, torch.Tensor]:
        values = raw_targets[self.cfg.target_key] if isinstance(raw_targets, Mapping) else raw_targets
        values = values.float()
        if values.dim() == 1:       # one scalar per sample: (B,) -> (B, 1)
            values = values.unsqueeze(1)
        return {"values": values}

    def forward_model(self, model, inputs: torch.Tensor) -> HeadOutput:
        return model(inputs)

    def compute_losses(self, raw_preds: HeadOutput, targets: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pred, true = raw_preds.values.float(), targets["values"]
        if pred.shape != true.shape:
            raise ValueError(
                f"RegressionTask: predictions {tuple(pred.shape)} and targets {tuple(true.shape)} differ in shape. "
                f"Check the head's num_outputs against the number of target values per sample."
            )
        if self.cfg.loss == "mse":
            loss = F.mse_loss(pred, true)
        elif self.cfg.loss == "l1":
            loss = F.l1_loss(pred, true)
        else:
            loss = F.huber_loss(pred, true, delta=self.cfg.huber_delta)
        return {f"{self.cfg.loss}_loss": loss}

    def reduce_losses(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(losses.values())

    def compute_metrics(self, raw_preds: HeadOutput, targets: Mapping[str, torch.Tensor]) -> dict[str, float]:
        # Per-batch view (logged per eval step); dataset-level numbers come from the Evaluator.
        return {"mae": (raw_preds.values.float() - targets["values"]).abs().mean().item()}

    def decode_predictions(self, raw_preds: HeadOutput) -> torch.Tensor:
        return raw_preds.values.float()

    def build_evaluator(self) -> RegressionEvaluator:
        return RegressionEvaluator()


regression_task_configs = {
    "regression_task": RegressionTaskConfig(),
}


@Task.register(config=regression_task_configs["regression_task"])
def regression_task(cfg: RegressionTaskConfig) -> RegressionTask:
    return RegressionTask(cfg)
