import datetime
import json
import os

import torch

from .base import Hook
from ..state import TrainerState


class JSONWriterHook(Hook):
    """
    Appends one JSON line per step -- a durable, replay-able log file,
    decoupled from whatever console/tensorboard formatting other hooks do.
    
    Inspired by Detectron2:    
    @misc{wu2019detectron2,
    author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                    Wan-Yen Lo and Ross Girshick},
    title =        {Detectron2},
    howpublished = {https://github.com/facebookresearch/detectron2},
    year =         {2019}
    }
    """

    _SKIP = {"data_time", "time"}
    _SMOOTH = {"total_loss", "time", "data_time"}

    def __init__(self, output_dir: str, filename: str = "metrics.jsonl", log_every: int = 20):
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, filename)
        self.log_every = log_every

    def after_step(self, state: TrainerState) -> None:
        if self.log_every > 0 and state.iter % self.log_every != 0:
            return
        record = self._build_train_record(state)
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def after_eval_step(self, state: TrainerState) -> None:
        if self.log_every > 0 and state.storage.eval_iter % self.log_every != 0:
            return
        record = self._build_eval_record(state)
        if record is None:
            return
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _build_train_record(self, state: TrainerState) -> dict:
        s = state.storage
        latest = s.latest()

        eta_seconds = None
        eta_str = "N/A"
        if "time" in s._history:
            avg_time = s.smoothed("time")
            remaining = state.max_iter - state.iter
            eta_seconds = int(avg_time * remaining)
            eta_str = str(datetime.timedelta(seconds=eta_seconds))

        loss_names = [k for k in s._history if k not in self._SKIP and k not in ("lr",)]
        scalars = {
            k: (s.smoothed(k) if k in self._SMOOTH else latest.get(k, float("nan")))
            for k in sorted(loss_names)
        }

        record = {
            "phase": "train",
            "iter": state.iter,
            "max_iter": state.max_iter,
            "eta": eta_str,
            "eta_seconds": eta_seconds,
            "scalars": scalars,
            "raw": latest,
            "time": {
                "smoothed": s.smoothed("time") if "time" in s._history else None,
                "last": latest.get("time") if "time" in s._history else None,
            },
            "data_time": {
                "smoothed": s.smoothed("data_time") if "data_time" in s._history else None,
                "last": latest.get("data_time") if "data_time" in s._history else None,
            },
            "lr": latest.get("lr") if "lr" in s._history else None,
            "max_mem_mb": (
                torch.cuda.max_memory_allocated() // (1024 * 1024)
                if torch.cuda.is_available()
                else None
            ),
        }
        return record

    def _build_eval_record(self, state: TrainerState) -> dict | None:
        s = state.storage
        if s.max_eval_iter == 0:
            return None

        fresh = s.latest_fresh(max_age=0, axis="eval_iter")
        if not fresh:
            return None

        skip_keys = self._SKIP | {"eval_data_time", "eval_time"}
        time_key = "eval_time" if "eval_time" in fresh else "time"
        data_time_key = "eval_data_time" if "eval_data_time" in fresh else "data_time"

        eta_seconds = None
        eta_str = "N/A"
        if time_key in fresh:
            avg_time = s.smoothed(time_key)
            remaining = max(s.max_eval_iter - (s.eval_iter + 1), 0)
            eta_seconds = int(avg_time * remaining)
            eta_str = str(datetime.timedelta(seconds=eta_seconds))

        # Mirrors CommonMetricPrinterHook.after_eval_step: show eval metrics, not loss-like keys.
        metric_names = [
            k
            for k in fresh
            if k not in skip_keys and k not in ("lr",) and "loss" not in k.lower()
        ]
        metrics = {
            k: (s.smoothed(k) if k in self._SMOOTH else fresh.get(k, float("nan")))
            for k in sorted(metric_names)
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
                "smoothed": s.smoothed(time_key) if time_key in fresh else None,
                "last": fresh.get(time_key) if time_key in fresh else None,
            },
            "data_time": {
                "smoothed": s.smoothed(data_time_key) if data_time_key in fresh else None,
                "last": fresh.get(data_time_key) if data_time_key in fresh else None,
            },
            "lr": fresh.get("lr") if "lr" in fresh else None,
            "max_mem_mb": (
                torch.cuda.max_memory_allocated() // (1024 * 1024)
                if torch.cuda.is_available()
                else None
            ),
        }
