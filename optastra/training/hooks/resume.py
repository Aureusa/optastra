import logging

from .base import Hook
from ..checkpointer import Checkpointer


class ResumeHook(Hook):
    priority = 0  # restore state before any other hook's before_train reads it

    def __init__(self, checkpointer: Checkpointer | str, checkpoint_name: str | None = None):
        """
        Hook to resume training from a checkpoint.
        Runs first (priority 0), so every other hook's state is restored
        before their own before_train runs.

        :param checkpointer: Checkpointer (or output directory) that wrote the checkpoints.
        :param checkpoint_name: Name of the checkpoint file to resume from.
        If None, resumes from the latest periodic checkpoint (best-metric
        checkpoints are never picked up implicitly).
        """
        self.checkpointer = Checkpointer.resolve(checkpointer)
        self.checkpoint_name = checkpoint_name
        self.logger = logging.getLogger("optastra.train")

    def before_train(self, state):
        """
        Load the checkpoint and update the trainer state.
        Resolved here rather than in __init__, so checkpoints written between
        construction and training are seen and a missing output_dir is fine.
        """
        if self.checkpoint_name is not None:
            path = self.checkpointer.path(self.checkpoint_name)
        else:
            path = self.checkpointer.latest()
        if path is None:
            self.logger.info(f"No checkpoints found in {self.checkpointer.output_dir}. Starting training from scratch.")
            return
        self.checkpointer.load(state, path)
        self.logger.info(f"Resumed training from checkpoint: {path} at iteration {state.iter}")
