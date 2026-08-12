import datetime, logging
import torch
from .base import Hook
from ..state import TrainerState


class CommonMetricPrinterHook(Hook):
    """
    Reads everything currently in storage generically, plus computes
    ETA/memory itself -- mirrors d2's CommonMetricPrinter.
    
    Inspired by Detectron2:    
    @misc{wu2019detectron2,
    author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                    Wan-Yen Lo and Ross Girshick},
    title =        {Detectron2},
    howpublished = {https://github.com/facebookresearch/detectron2},
    year =         {2019}
    }
    """

    _SKIP = {"data_time", "time"}        # shown explicitly, not in the generic tail
    _SMOOTH = {"total_loss", "time", "data_time"}

    def __init__(self, log_every: int = 20):
        self.log_every = log_every
        self.logger = logging.getLogger("optastra.train")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = True

    def after_step(self, state: TrainerState) -> None:
        if state.iter % self.log_every != 0:
            return
        s = state.storage

        # ETA from smoothed step time * remaining iters
        eta_str = "N/A"
        if "iter_time" in s._history:
            avg_time = s.smoothed("iter_time")
            remaining = state.max_iter - state.iter
            eta_str = str(datetime.timedelta(seconds=int(avg_time * remaining)))

        # generic tail: every loss-like scalar currently tracked, in one pass,
        # no hardcoded names -- new loss components show up automatically
        loss_names = [
            k for k in s._history
            if k not in self._SKIP
            and k not in ("lr",)
        ]
        losses_str = "  ".join(
            f"{k}: {s.smoothed(k) if k in self._SMOOTH else s.latest().get(k, float('nan')):.4g}"
            for k in sorted(loss_names)
        )

        time_str = f"avg_iter_time: {s.smoothed('iter_time'):.4f} s" if "iter_time" in s._history else ""
        data_str = f"avg_data_time: {s.smoothed('data_time'):.4f} s" if "data_time" in s._history else ""
        lr_str = f"lr: {s.latest().get('lr', float('nan')):.2e}" if "lr" in s._history else ""

        mem_str = ""
        if torch.cuda.is_available():
            mem_str = f"max_mem: {torch.cuda.max_memory_allocated() // (1024 * 1024)}M"

        parts = [f"eta: {eta_str}", f"iter: {state.iter}/{state.max_iter}", losses_str, time_str, data_str, lr_str, mem_str]
        self.logger.info("  ".join(p for p in parts if p))

    def after_eval_step(self, state: TrainerState) -> None:
        s = state.storage
        if s.max_eval_iter == 0:
            return
        if s.eval_iter % self.log_every != 0:
            return

        # Use only metrics updated on this eval iteration to avoid mixing train-axis values.
        fresh = s.latest_fresh(max_age=0, axis="eval_iter")
        skip_keys = self._SKIP | {"eval_data_time", "eval_time"}

        time_key = "eval_time" if "eval_time" in fresh else "time"
        data_time_key = "eval_data_time" if "eval_data_time" in fresh else "data_time"

        # ETA from smoothed eval step time * remaining eval batches.
        eta_str = "N/A"
        if time_key in fresh:
            avg_time = s.smoothed(time_key)
            remaining = max(s.max_eval_iter - (s.eval_iter + 1), 0)
            eta_str = str(datetime.timedelta(seconds=int(avg_time * remaining)))

        # generic tail for fresh eval values, excluding loss-like keys.
        # Eval progress logs are intended to show metrics; losses are kept in train logs.
        metric_names = [
            k for k in fresh
            if k not in skip_keys
            and k not in ("lr",)
            and "loss" not in k.lower()
        ]
        metrics_str = "  ".join(
            f"{k}: {s.smoothed(k) if k in self._SMOOTH else fresh.get(k, float('nan')):.4g}"
            for k in sorted(metric_names)
        )

        time_str = (
            f"time: {s.smoothed(time_key):.4f}  last_time: {fresh.get(time_key, 0):.4f}"
            if time_key in fresh
            else ""
        )
        data_str = (
            f"data_time: {s.smoothed(data_time_key):.4f}  last_data_time: {fresh.get(data_time_key, 0):.4f}"
            if data_time_key in fresh
            else ""
        )
        lr_str = f"lr: {fresh.get('lr', float('nan')):.4g}" if "lr" in fresh else ""

        mem_str = ""
        if torch.cuda.is_available():
            mem_str = f"max_mem: {torch.cuda.max_memory_allocated() // (1024 * 1024)}M"

        parts = [
            f"[eval @ iter {state.iter}]",
            f"eta: {eta_str}",
            f"eval_iter: {s.eval_iter + 1}/{s.max_eval_iter}",
            metrics_str,
            time_str,
            data_str,
            lr_str,
            mem_str,
        ]
        self.logger.info("  ".join(p for p in parts if p))

    def after_eval(self, state: TrainerState) -> None:
        s = state.storage
        if s.max_eval_iter == 0:
            return

        # Log final eval metrics after evaluation is complete.
        # The avg metrics are here self.storage.put_scalars(axis="iter", **{f"val_{k}": v for k, v in averaged.items()})
        latest = s.latest()

        # Get only the val_ prefixed metrics for logging.
        val_metrics = {k: v for k, v in latest.items() if k.startswith("val_")}
        if not val_metrics:
            self.logger.info(f"[eval @ iter {state.iter}] No val_ metrics found in storage.")
            return

        metrics_str = "  ".join(f"{k}: {v:.4g}" for k, v in sorted(val_metrics.items()))
        self.logger.info(f"[eval @ iter {state.iter}] Final eval metrics: {metrics_str}")
