import torch
import os
import logging

from .base import Hook


class ResumeHook(Hook):

    def __init__(self, output_dir: str, checkpoint_name: str = None):
        """
        Hook to resume training from a specific checkpoint.
        This hook should be added to the trainer before training starts.

        :param output_dir: Directory where checkpoints are saved.
        :param checkpoint_name: Name of the checkpoint file to resume from.
        If None, it will look for the latest checkpoint in the output_dir.
        """
        self.logger = logging.getLogger("optastra.train")
        if checkpoint_name is None:
            # Find the latest checkpoint in the output_dir
            checkpoints = [f for f in os.listdir(output_dir) if f.startswith("ckpt_") and f.endswith(".pt")]
            if not checkpoints:
                # No checkpoints found in {output_dir}.
                self.checkpoint_path = None
                self.logger.warning(f"No checkpoints found in {output_dir}. Starting training from scratch.")
                return
            
            # Sort by iteration number extracted from filename
            checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)
            self.checkpoint_path = os.path.join(output_dir, checkpoints[0])
        else:
            self.checkpoint_path = os.path.join(output_dir, checkpoint_name)

    def before_train(self, state):
        """
        Load the checkpoint and update the trainer state.
        """
        if self.checkpoint_path is None:
            self.logger.info("No checkpoint to resume from. Starting training from scratch.")
            return
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state.model.load_state_dict(checkpoint["model"])
        state.optimizer.load_state_dict(checkpoint["optimizer"])
        state.iter = checkpoint["iter"]
        for hook in state.hooks:
            if hasattr(hook, "load_state_dict") and "hooks" in checkpoint:
                hook_state = next((h for h in checkpoint["hooks"] if h.get("name") == hook.__class__.__name__), None)
                if hook_state:
                    hook.load_state_dict(hook_state["state"])
        self.logger.info(f"Resumed training from checkpoint: {self.checkpoint_path} at iteration {state.iter}")
        