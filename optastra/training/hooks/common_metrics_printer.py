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
    main_process_only = True  # writes files / prints: rank 0 only in multi-process runs

    _SKIP = {"data_time", "iter_time"}        # shown explicitly, not in the generic tail
    _SMOOTH = {"total_loss", "iter_time", "data_time"}

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
        if s.has("iter_time"):
            avg_time = s.smoothed("iter_time")
            remaining = state.max_iter - state.iter
            eta_str = str(datetime.timedelta(seconds=int(avg_time * remaining)))

        # generic tail: every train scalar currently tracked, in one pass,
        # no hardcoded names -- new loss components show up automatically.
        # val_* summaries are printed by after_eval instead.
        loss_names = [
            k for k in s.keys()
            if k not in self._SKIP
            and k not in ("lr",)
            and k not in state.eval_results
        ]
        losses_str = "  ".join(
            f"{k}: {s.smoothed(k) if k in self._SMOOTH else s.latest().get(k, float('nan')):.4g}"
            for k in sorted(loss_names)
        )

        time_str = f"avg_iter_time: {s.smoothed('iter_time'):.4f} s" if s.has("iter_time") else ""
        data_str = f"avg_data_time: {s.smoothed('data_time'):.4f} s" if s.has("data_time") else ""
        lr_str = f"lr: {s.latest().get('lr', float('nan')):.2e}" if s.has("lr") else ""

        mem_str = ""
        if torch.cuda.is_available():
            mem_str = f"max_mem: {torch.cuda.max_memory_allocated() // (1024 * 1024)}M"

        parts = [f"eta: {eta_str}", f"iter: {state.iter}/{state.max_iter}", losses_str, time_str, data_str, lr_str, mem_str]
        self.logger.info("  ".join(p for p in parts if p))

    def after_eval_step(self, state: TrainerState) -> None:
        s = state.storage
        if s.eval_iter % self.log_every != 0:
            return

        # Per-batch eval scalars live on their own storage axis.
        fresh = s.latest_fresh(max_age=0, axis="eval_iter")
        skip_keys = {"eval_data_time", "eval_time"}

        # ETA from smoothed eval step time * remaining eval batches (unknown without len()).
        eta_str = "N/A"
        if "eval_time" in fresh and s.max_eval_iter > 0:
            avg_time = s.smoothed("eval_time", axis="eval_iter")
            remaining = max(s.max_eval_iter - (s.eval_iter + 1), 0)
            eta_str = str(datetime.timedelta(seconds=int(avg_time * remaining)))

        # generic tail for fresh eval values, excluding loss-like keys.
        # Eval progress logs are intended to show metrics; losses are kept in train logs.
        metric_names = [
            k for k in fresh
            if k not in skip_keys
            and k not in ("lr",)
            and "loss" not in k.lower()
            and "_dm" not in k.lower()  # exclude dmlab metrics
        ]
        metrics_str = "  ".join(f"{k}: {fresh[k]:.4g}" for k in sorted(metric_names))

        time_str = f"avg_iter_time: {s.smoothed('eval_time', axis='eval_iter'):.4f} s" if "eval_time" in fresh else ""
        data_str = f"avg_data_time: {s.smoothed('eval_data_time', axis='eval_iter'):.4f} s" if "eval_data_time" in fresh else ""

        mem_str = ""
        if torch.cuda.is_available():
            mem_str = f"max_mem: {torch.cuda.max_memory_allocated() // (1024 * 1024)}M"

        parts = [
            f"[eval @ iter {state.iter}]",
            f"eta: {eta_str}",
            f"eval_iter: {s.eval_iter + 1}/{s.max_eval_iter or '?'}",
            metrics_str,
            time_str,
            data_str,
            mem_str,
        ]
        self.logger.info("  ".join(p for p in parts if p))

    def after_eval(self, state: TrainerState) -> None:
        # Dataset-level results of the eval that just finished (written by Trainer.evaluate).
        val_metrics = state.eval_results
        if not val_metrics:
            self.logger.info(f"[eval @ iter {state.iter}] No eval metrics were produced.")
            return

        metrics_str = "  ".join(f"{k}: {v:.4g}" for k, v in sorted(val_metrics.items()))
        self.logger.info(f"[eval @ iter {state.iter}] Final eval metrics: {metrics_str}")
