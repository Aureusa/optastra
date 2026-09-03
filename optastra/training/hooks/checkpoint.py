import torch, os
from .base import Hook
from ..state import TrainerState


class CheckpointHook(Hook):
    """
    Hook to save model checkpoints during training.
    Currently, it saves the model and optimizer state_dicts every `save_every` iterations.
    """
    def __init__(self, output_dir: str, save_every: int = 500):
        self.output_dir = output_dir
        self.save_every = save_every
        os.makedirs(output_dir, exist_ok=True)

    def after_step(self, state: TrainerState) -> None:
        if state.iter == 0 or state.iter % self.save_every != 0:
            return
        path = os.path.join(self.output_dir, f"ckpt_{state.iter}.pt")
        torch.save({
            "model": state.model.state_dict(),
            "optimizer": state.optimizer.state_dict(),
            "iter": state.iter,
            "hooks": [{"name": hook.__class__.__name__, "state": hook.state_dict()} for hook in state.hooks]
        }, path)
