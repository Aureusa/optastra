from .checkpoint import CheckpointHook
from .common_metrics_printer import CommonMetricPrinterHook
from .writer import JSONWriterHook
from .resume import ResumeHook


def default_hooks(
        log_every: int = 20,
        output_dir: str = "runs/exp1/",
        checkpoint_every: int = 500,
        resume: bool = True,
        checkpoint_name: str | None = None
    ):
    """
    Returns a list of default hooks for training, including logging, evaluation, and checkpointing.
    """
    default_hooks_list = [
        CheckpointHook(output_dir=output_dir, save_every=checkpoint_every),
        CommonMetricPrinterHook(log_every=log_every),
        JSONWriterHook(output_dir=f"{output_dir}/logs", log_every=log_every)
    ]
    if resume:
        default_hooks_list.append(ResumeHook(output_dir=output_dir, checkpoint_name=checkpoint_name))

    return default_hooks_list
