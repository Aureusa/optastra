from .checkpoint import CheckpointHook
from .common_metrics_printer import CommonMetricPrinterHook
from .writer import JSONWriterHook


def default_hooks(
        log_every: int = 20,
        output_dir: str = "runs/exp1/",
        checkpoint_every: int = 500
    ):
    """
    Returns a list of default hooks for training, including logging, evaluation, and checkpointing.
    """
    return [
        CommonMetricPrinterHook(log_every=log_every),
        CheckpointHook(output_dir=output_dir, save_every=checkpoint_every),
        JSONWriterHook(output_dir=f"{output_dir}/logs", log_every=log_every),
    ]
