from __future__ import annotations
from collections import defaultdict
from typing import Any, Iterable
import os
import re
import torch

from .state import TrainerState


__all__ = ["Checkpointer"]


class Checkpointer:
    """Owns the on-disk checkpoint format: file naming, what goes into a
    checkpoint, and how it is restored. Not a hook -- hooks only decide
    *when* to save/load (CheckpointHook, BestCheckpointHook, ResumeHook)
    and delegate the *how* here, so no hook hardcodes a filename pattern.

    Naming:
      periodic: `{prefix}_{iter}.pt`          -- the only files latest() considers
      best:     `{prefix}_best_{metric}.pt`   -- never picked up as a resume point
    """

    def __init__(self, output_dir: str, prefix: str = "ckpt"):
        self.output_dir = output_dir
        self.prefix = prefix
        self._periodic_re = re.compile(rf"^{re.escape(prefix)}_(\d+)\.pt$")

    @classmethod
    def resolve(cls, checkpointer: Checkpointer | str) -> Checkpointer:
        """Accepts a Checkpointer or an output directory, so hooks can take
        either (same idea as resolve_spec for FeatureSpec)."""
        return checkpointer if isinstance(checkpointer, Checkpointer) else cls(checkpointer)

    def periodic_name(self, iter: int) -> str:
        return f"{self.prefix}_{iter}.pt"

    def best_name(self, metric: str) -> str:
        return f"{self.prefix}_best_{metric}.pt"

    def path(self, name: str) -> str:
        return os.path.join(self.output_dir, name)

    def latest(self) -> str | None:
        """Path of the highest-iteration periodic checkpoint, or None."""
        if not os.path.isdir(self.output_dir):
            return None
        found = [
            (int(m.group(1)), f)
            for f in os.listdir(self.output_dir)
            if (m := self._periodic_re.match(f))
        ]
        return self.path(max(found)[1]) if found else None

    def save(self, state: TrainerState, name: str, extra: dict[str, Any] | None = None) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        path = self.path(name)
        tmp_path = f"{path}.tmp"
        torch.save({
            "model": state.model.state_dict(),
            "optimizer": state.optimizer.state_dict(),
            "iter": state.iter,
            "epoch": state.epoch,
            "hooks": [
                {"name": type(hook).__name__, "state": hook.state_dict() if hasattr(hook, "state_dict") else {}}
                for hook in state.hooks
            ],
            "extra": extra or {},
        }, tmp_path)
        # Atomic swap: a job killed mid-write never leaves a truncated checkpoint
        # (matters most for the best file, which is overwritten in place).
        os.replace(tmp_path, path)
        return path

    def load(self, state: TrainerState, path: str) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location="cpu")
        state.model.load_state_dict(checkpoint["model"])
        state.optimizer.load_state_dict(checkpoint["optimizer"])
        state.iter = checkpoint["iter"]
        # Checkpoints are written after step `iter` completed -- continue with the next one.
        state.start_iter = checkpoint["iter"] + 1
        state.epoch = checkpoint.get("epoch", state.epoch)
        self._restore_hooks(state.hooks, checkpoint.get("hooks", []))
        return checkpoint

    @staticmethod
    def _restore_hooks(hooks: Iterable[Any], saved: list[dict[str, Any]]) -> None:
        """Pairs the k-th hook of each class with the k-th saved entry of that
        class, so two hooks of the same type (e.g. two BestCheckpointHooks on
        different metrics) don't both receive the first one's state."""
        states_by_name: dict[str, list[dict]] = defaultdict(list)
        for entry in saved:
            states_by_name[entry["name"]].append(entry["state"])
        for hook in hooks:
            states = states_by_name.get(type(hook).__name__)
            if states and hasattr(hook, "load_state_dict"):
                hook.load_state_dict(states.pop(0))
