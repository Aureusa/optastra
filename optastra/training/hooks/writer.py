import datetime
import json
import os

import torch

from .base import Hook
from ..state import TrainerState


class JSONWriterHook(Hook):
    """
    Appends one JSON line per logged step -- a durable, replay-able log file,
    decoupled from whatever console/tensorboard formatting other hooks do.

    Record kinds (field "phase"):
      - "train":        every `log_every` iters, smoothed train scalars.
      - "eval":         every `log_every` eval batches, per-batch eval metrics.
      - "eval_summary": once per Trainer.evaluate(), the dataset-level results
                        (e.g. val_accuracy, val_total_loss) from state.eval_results.

    Inspired by Detectron2:
    @misc{wu2019detectron2,
    author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                    Wan-Yen Lo and Ross Girshick},
    title =        {Detectron2},
    howpublished = {https://github.com/facebookresearch/detectron2},
    year =         {2019}
    }
    """
    main_process_only = True  # writes files / prints: rank 0 only in multi-process runs

    _SKIP = {"data_time", "iter_time"}
    _SMOOTH = {"total_loss", "iter_time", "data_time"}

    def __init__(self, output_dir: str, filename: str = "metrics.jsonl", log_every: int = 20):
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, filename)
        self.log_every = log_every

    def _write(self, record: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def after_step(self, state: TrainerState) -> None:
        if self.log_every > 0 and state.iter % self.log_every != 0:
            return
        self._write(self._build_train_record(state))

    def after_eval_step(self, state: TrainerState) -> None:
        if self.log_every > 0 and state.storage.eval_iter % self.log_every != 0:
            return
        record = self._build_eval_record(state)
        if record is not None:
            self._write(record)

    def after_eval(self, state: TrainerState) -> None:
        if state.eval_results:
            self._write({
                "phase": "eval_summary",
                "iter": state.iter,
                "max_iter": state.max_iter,
                "epoch": state.epoch,
                "metrics": dict(state.eval_results),
            })

    @staticmethod
    def _max_mem_mb() -> int | None:
        return torch.cuda.max_memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else None

    def _build_train_record(self, state: TrainerState) -> dict:
        s = state.storage
        latest = s.latest()

        eta_seconds = None
        eta_str = "N/A"
        if s.has("iter_time"):
            avg_time = s.smoothed("iter_time")
            remaining = state.max_iter - state.iter
            eta_seconds = int(avg_time * remaining)
            eta_str = str(datetime.timedelta(seconds=eta_seconds))

        # Train scalars only: eval summaries get their own "eval_summary" record.
        loss_names = [
            k for k in s.keys()
            if k not in self._SKIP and k not in ("lr",) and k not in state.eval_results
        ]
        scalars = {
            k: (s.smoothed(k) if k in self._SMOOTH else latest.get(k, float("nan")))
            for k in sorted(loss_names)
        }

        return {
            "phase": "train",
            "iter": state.iter,
            "max_iter": state.max_iter,
            "epoch": state.epoch,
            "eta": eta_str,
            "eta_seconds": eta_seconds,
            "scalars": scalars,
            "raw": latest,
            "time": {
                "smoothed": s.smoothed("iter_time") if s.has("iter_time") else None,
                "last": latest.get("iter_time"),
            },
            "data_time": {
                "smoothed": s.smoothed("data_time") if s.has("data_time") else None,
                "last": latest.get("data_time"),
            },
            "lr": latest.get("lr"),
            "max_mem_mb": self._max_mem_mb(),
        }

    def _build_eval_record(self, state: TrainerState) -> dict | None:
        s = state.storage
        fresh = s.latest_fresh(max_age=0, axis="eval_iter")
        if not fresh:
            return None

        eta_seconds = None
        eta_str = "N/A"
        if "eval_time" in fresh and s.max_eval_iter > 0:
            avg_time = s.smoothed("eval_time", axis="eval_iter")
            remaining = max(s.max_eval_iter - (s.eval_iter + 1), 0)
            eta_seconds = int(avg_time * remaining)
            eta_str = str(datetime.timedelta(seconds=eta_seconds))

        # Per-batch records show metrics only (mirrors CommonMetricPrinterHook);
        # the val loss is in the "eval_summary" record.
        metrics = {
            k: v for k, v in sorted(fresh.items())
            if k not in ("eval_time", "eval_data_time") and "loss" not in k.lower()
        }

        return {
            "phase": "eval",
            "iter": state.iter,
            "max_iter": state.max_iter,
            "eval_iter": s.eval_iter + 1,
            "max_eval_iter": s.max_eval_iter,
            "eta": eta_str,
            "eta_seconds": eta_seconds,
            "metrics": metrics,
            "time": {
                "smoothed": s.smoothed("eval_time", axis="eval_iter") if "eval_time" in fresh else None,
                "last": fresh.get("eval_time"),
            },
            "data_time": {
                "smoothed": s.smoothed("eval_data_time", axis="eval_iter") if "eval_data_time" in fresh else None,
                "last": fresh.get("eval_data_time"),
            },
            "max_mem_mb": self._max_mem_mb(),
        }
