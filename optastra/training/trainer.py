from __future__ import annotations
from typing import Iterable, Mapping, Any
import time
import torch
import torch.nn as nn
from collections.abc import Mapping as MappingABC

from ..tasks.base import Task
from .state import TrainerState
from .storage import EventStorage
from .hooks.base import Hook


__all__ = ["Trainer"]


class Trainer:
    """Orchestrates model + task + optimizer + hooks. Knows nothing about
    what the task computes -- it only calls task.run_step and reads the
    generic TaskStepOutput fields (loss, losses, metrics)."""

    def __init__(
        self,
        model: nn.Module,
        task: Task,
        optimizer,
        hooks: Iterable[Hook] = (),
        device: str | torch.device = "cuda",
    ):
        resolved_device = self._resolve_device(device)
        model = model.to(resolved_device)
        self.storage = EventStorage()
        self.state = TrainerState(model=model, task=task, optimizer=optimizer, storage=self.storage, device=resolved_device)
        self.hooks: list[Hook] = list(hooks)

    def info(self) -> str:
        info_str = f"Trainer:\n"
        info_str += f"(Model) {self.state.model.info()}\n"
        info_str += f"(Task) {self.state.task.info()}\n"
        info_str += f"(Optimizer) {self.state.optimizer.info()}\n"
        for hook in self.hooks:
            info_str += f"(Hook) {hook.info()}\n"
        info_str += f"(Device) {self.state.device}\n"
        return info_str

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is selected as the training device, but no CUDA device is available. "
                "Set device='cpu' explicitly if you want to run on CPU."
            )
        return resolved

    @staticmethod
    def _move_to_device(data: Any, device: torch.device) -> Any:
        if torch.is_tensor(data):
            return data.to(device, non_blocking=True)
        if isinstance(data, MappingABC):
            return {k: Trainer._move_to_device(v, device) for k, v in data.items()}
        if isinstance(data, tuple):
            return tuple(Trainer._move_to_device(v, device) for v in data)
        if isinstance(data, list):
            return [Trainer._move_to_device(v, device) for v in data]
        return data

    def register_hooks(self, hooks: Iterable[Hook]) -> None:
        self.hooks.extend(hooks)

    def _run_hooks(self, method_name: str) -> None:
        for hook in self.hooks:
            getattr(hook, method_name)(self.state)

    def _next_batch(self, data_iter, dataloader):
        """Returns (batch, new_iter, epoch_advanced) and records data_time --
        isolated so both train() and evaluate() time fetching identically."""
        t0 = time.perf_counter()
        try:
            batch = next(data_iter)
            epoch_advanced = False
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            epoch_advanced = True
        data_time = time.perf_counter() - t0
        return data_iter, batch, epoch_advanced, data_time

    def train(self, dataloader: Iterable[Mapping[str, Any]], max_iter: int) -> None:
        self.state.max_iter = max_iter
        self._run_hooks("before_train")

        data_iter = iter(dataloader)
        for it in range(max_iter):
            if self.state.should_stop:
                break
            self.state.iter = self.storage.iter = it

            step_start = time.perf_counter()
            data_iter, batch, epoch_advanced, data_time = self._next_batch(data_iter, dataloader)
            if epoch_advanced:
                self.state.epoch += 1
            self.state.last_data_time = data_time
            self.storage.put_scalar("data_time", data_time)

            batch = self._move_to_device(batch, self.state.device)
            self.state.current_batch = batch

            self._run_hooks("before_step")

            self.state.optimizer.zero_grad(set_to_none=True)
            output = self.state.task.run_step(self.state.model, self.state.current_batch, stage="train")
            output.loss.backward()
            self.state.optimizer.step()

            self.state.last_output = output
            self.storage.put_scalar("iter_time", time.perf_counter() - step_start)
            self.storage.put_scalar("total_loss", output.loss.item())
            self.storage.put_scalars(**{k: v.item() for k, v in output.losses.items()})

            self._run_hooks("after_step")

        self._run_hooks("after_train")

    @torch.no_grad()
    def evaluate(self, dataloader: Iterable[Mapping[str, Any]]) -> dict[str, float]:
        self.state.model.eval()
        self.storage.eval_iter = 0
        max_eval_iter = len(dataloader)
        self.storage.max_eval_iter = max_eval_iter
        self._run_hooks("before_eval")

        all_metrics: dict[str, list[float]] = {}
        try:
            data_iter = iter(dataloader)
            eval_it = 0
            while True:
                step_start = time.perf_counter()
                data_iter, batch, epoch_advanced, data_time = self._next_batch(data_iter, dataloader)
                if epoch_advanced and eval_it > 0:
                    break  # single full pass over the val set, then stop
                self.storage.eval_iter = eval_it
                self._run_hooks("before_eval_step")
                self.storage.put_scalar("eval_data_time", data_time, axis="eval_iter")

                batch = self._move_to_device(batch, self.state.device)
                output = self.state.task.run_step(self.state.model, batch, stage="val")
                self.state.last_output = output

                # per-batch, tagged on the eval axis -- has its own history now
                self.storage.put_scalars(axis="eval_iter",
                                        **{f"val_step_{k}": v for k, v in output.metrics.items()})
                self.storage.put_scalar("eval_time", time.perf_counter() - step_start, axis="eval_iter")

                for k, v in output.metrics.items():
                    all_metrics.setdefault(k, []).append(float(v))
                self._run_hooks("after_eval_step")
                eval_it += 1

            averaged = {k: sum(v) / len(v) for k, v in all_metrics.items()}
            self.storage.put_scalars(axis="iter", **{f"val_{k}": v for k, v in averaged.items()})
            return averaged
        finally:
            self._run_hooks("after_eval")
            self.state.model.train()
        