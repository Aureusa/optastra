import json, os
from .base import Hook
from ..state import TrainerState


class JSONWriterHook(Hook):
    """Appends one JSON line per step -- a durable, replay-able log file,
    decoupled from whatever console/tensorboard formatting other hooks do."""
    def __init__(self, output_dir: str, filename: str = "metrics.jsonl"):
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, filename)

    def after_step(self, state: TrainerState) -> None:
        record = {"iter": state.iter, **state.storage.latest()}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
