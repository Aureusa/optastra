from .checkpoint import CheckpointHook, BestCheckpointHook
from .common_metrics_printer import CommonMetricPrinterHook
from .writer import JSONWriterHook
from .resume import ResumeHook
from ..checkpointer import Checkpointer


def default_hooks(
        log_every: int = 20,
        output_dir: str = "runs/exp1/",
        checkpoint_every: int = 500,
        resume: bool = True,
        checkpoint_name: str | None = None,
        best_metric: str | None = None,
        best_mode: str = "min",
    ):
    """
    Returns a list of default hooks for training, including logging and checkpointing.
    Set `best_metric` (e.g. "val_total_loss") to also keep a best-model checkpoint.
    """
    checkpointer = Checkpointer(output_dir)
    default_hooks_list = [
        CheckpointHook(checkpointer, save_every=checkpoint_every),
        CommonMetricPrinterHook(log_every=log_every),
        JSONWriterHook(output_dir=f"{output_dir}/logs", log_every=log_every)
    ]
    if best_metric is not None:
        default_hooks_list.insert(1, BestCheckpointHook(checkpointer, metric=best_metric, mode=best_mode))
    if resume:
        default_hooks_list.insert(0, ResumeHook(checkpointer, checkpoint_name=checkpoint_name))

    return default_hooks_list
