from __future__ import annotations
from contextlib import nullcontext
from typing import Iterable, Iterator, Literal, Mapping, Any
import logging
import time
import torch
import torch.nn as nn
from collections.abc import Mapping as MappingABC
from torch.nn.parallel import DistributedDataParallel

from ..core.distributed import (
    all_reduce_mean, env_world_size, get_local_rank, init_distributed, is_main_process, sum_across_processes,
)
from ..tasks.base import Task, MeanMetricEvaluator
from .state import TrainerState
from .storage import EventStorage
from .hooks.base import Hook


__all__ = ["Trainer", "Precision"]


Precision = Literal["fp32", "bf16", "fp16"]


def _num_batches(dataloader: Iterable) -> int | None:
    """Batches per pass, or None when unknown -- e.g. a DataLoader over an
    IterableDataset, which has `__len__` but raises TypeError when called."""
    try:
        return len(dataloader)
    except TypeError:
        return None
_AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


class Trainer:
    """Orchestrates model + task + optimizer + hooks. Knows nothing about
    what the task computes -- it only calls task.run_step and reads the
    generic TaskStepOutput fields (loss, losses, metrics).

    The Trainer owns the *runtime policy* around each step, so Tasks stay pure:

    - ``precision``: "fp32" (default, no autocast), "bf16" or "fp16".
      bf16/fp16 wrap ``task.run_step`` in ``torch.autocast`` on the model's
      device type (works on CPU and CUDA); fp16 additionally uses a
      ``torch.amp.GradScaler`` to avoid gradient underflow.
    - ``grad_accum_steps``: one training iteration == one optimizer step,
      which consumes ``grad_accum_steps`` micro-batches from the loader. Each
      micro-batch loss is scaled by ``1 / grad_accum_steps`` so the
      accumulated gradient equals the mean over all micro-batches (identical
      to one big batch for mean-reduced losses). ``max_iter``, schedulers,
      checkpoints and every hook's ``state.iter`` count optimizer steps.
    - ``clip_grad_norm``: if set, the total gradient norm is clipped to this
      value before each optimizer step, and the pre-clip norm is logged to
      storage as "grad_norm".
    - ``distributed``: multi-process data-parallel training (DDP), one
      process per GPU. None (default) = on exactly when the script was
      launched with ``torchrun --nproc_per_node=N`` (N > 1), so the same
      script runs unchanged with ``python``. Each process uses GPU
      ``LOCAL_RANK``, trains on its own shard of the data (see
      ``build_dataloader``), and gradients are averaged across processes
      every optimizer step -- so ``batch_size`` is per process and the
      effective batch is ``batch_size * world_size``. ``state.model`` stays
      the plain module (hooks, checkpoints and evaluation never see the DDP
      wrapper); hooks with ``main_process_only = True`` (checkpoints, log
      files, console printing) run on rank 0 only, everything else on every
      process. Evaluation is sharded too and combined exactly by the task's
      Evaluator (``Evaluator.sync``).
    - ``sync_batchnorm``: DDP only -- convert BatchNorm layers to
      SyncBatchNorm so batch statistics are computed over the global batch
      (helps when the per-GPU batch is small, e.g. detection).
    - ``compile``: ``torch.compile`` the module the train step calls --
      after DDP wrapping, the order PyTorch recommends. True for the
      defaults, or a dict of ``torch.compile`` kwargs (e.g.
      ``{"mode": "max-autotune"}``). ``state.model`` stays uncompiled, so
      hooks, checkpoints and ``evaluate()`` are unaffected. Pass the model
      uncompiled: don't combine this with a model you compiled yourself.
    - ``find_unused_parameters``: DDP only -- set True if some parameters
      take no part in some forward passes (e.g. a branch that is skipped
      conditionally); DDP raises an error telling you so otherwise. It adds
      overhead, so leave it off unless that error appears.

    Hook lifecycle within ``train()``::

        before_train
        before_epoch                              # epoch state.epoch starts
        for each iteration:
            for each of grad_accum_steps micro-batches:
                [after_epoch, epoch += 1, before_epoch]   # only when the loader ran out
                before_step                       # state.current_batch / state.micro_step set
            (optimizer step)
            after_step                            # once per optimizer step
        after_train

    An epoch ends when the loader is exhausted, which is detected lazily when
    the next batch is requested -- so ``after_epoch`` for epoch k runs just
    before the first batch of epoch k+1 is fetched, and the (possibly
    partial) epoch in progress when training stops gets no ``after_epoch``.
    """

    def __init__(
        self,
        model: nn.Module,
        task: Task,
        optimizer,
        hooks: Iterable[Hook] = (),
        device: str | torch.device = "cuda",
        precision: Precision = "fp32",
        grad_accum_steps: int = 1,
        clip_grad_norm: float | None = None,
        distributed: bool | None = None,
        sync_batchnorm: bool = False,
        find_unused_parameters: bool = False,
        compile: bool | dict = False,
    ):
        if precision not in ("fp32", *_AUTOCAST_DTYPES):
            raise ValueError(f"precision must be 'fp32', 'bf16' or 'fp16', got {precision!r}")
        if grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")

        if distributed is None:
            distributed = env_world_size() > 1
        self.distributed = distributed and init_distributed()
        if self.distributed and torch.device(device).type == "cuda":
            device = torch.device("cuda", get_local_rank())   # one GPU per process
        self.sync_batchnorm = sync_batchnorm
        self.find_unused_parameters = find_unused_parameters
        self._ddp_model: DistributedDataParallel | None = None   # built lazily in train()
        self.compile = compile
        self._compiled_model: nn.Module | None = None

        resolved_device = self._resolve_device(device)
        if self.distributed and sync_batchnorm and resolved_device.type == "cpu":
            raise ValueError("sync_batchnorm=True needs GPUs: PyTorch's SyncBatchNorm does not run on CPU.")
        model = model.to(resolved_device)
        self.precision = precision
        self.grad_accum_steps = grad_accum_steps
        self.clip_grad_norm = clip_grad_norm
        # A disabled GradScaler is a transparent pass-through, so the train
        # step has one code path for every precision.
        self.grad_scaler = torch.amp.GradScaler(resolved_device.type, enabled=(precision == "fp16"))
        self.logger = logging.getLogger("optastra.train")

        self.storage = EventStorage()
        self.hooks: list[Hook] = list(hooks)
        self._sort_hooks()
        self.state = TrainerState(model=model, task=task, optimizer=optimizer, storage=self.storage, device=resolved_device, hooks=self.hooks)

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is selected as the training device, but no CUDA device is available. "
                "Set device='cpu' explicitly if you want to run on CPU."
            )
        if resolved.type == "cuda" and resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Device {resolved} does not exist: this process sees {torch.cuda.device_count()} GPU(s). "
                "With torchrun, --nproc_per_node must not exceed the number of GPUs."
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
        self._sort_hooks()
        self.state.hooks = self.hooks  # update state with new hooks

    def _sort_hooks(self) -> None:
        # Stable, in place: equal-priority hooks keep registration order.
        self.hooks.sort(key=lambda hook: getattr(hook, "priority", Hook.priority))

    def _run_hooks(self, method_name: str) -> None:
        main = is_main_process()
        for hook in self.state.hooks:
            if not main and getattr(hook, "main_process_only", False):
                continue   # checkpoints / log files / printing happen on rank 0 only
            # getattr default: duck-typed hooks don't have to define every event.
            getattr(hook, method_name, lambda state: None)(self.state)

    def _autocast(self):
        """Mixed-precision context for the forward pass + loss; a no-op for fp32."""
        if self.precision == "fp32":
            return nullcontext()
        return torch.autocast(device_type=self.state.device.type, dtype=_AUTOCAST_DTYPES[self.precision])

    def _next_batch(self, data_iter: Iterator, dataloader: Iterable) -> tuple[Iterator, Any, float]:
        """Returns (data_iter, batch, data_time). When the loader is
        exhausted, closes the epoch (after_epoch), advances state.epoch,
        opens the next one (before_epoch) and starts a fresh pass."""
        t0 = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            self._run_hooks("after_epoch")
            self.state.epoch += 1
            self.state.epoch_step = 0
            self._run_hooks("before_epoch")
            t0 = time.perf_counter()  # don't count hook time as data time
            self._set_sampler_epoch(dataloader)
            data_iter = iter(dataloader)
            try:
                batch = next(data_iter)
            except StopIteration:
                raise ValueError("The training dataloader yielded no batches.") from None
        self.state.epoch_step += 1
        return data_iter, batch, time.perf_counter() - t0

    def train(self, dataloader: Iterable[Mapping[str, Any]], max_iter: int) -> None:
        """Runs optimizer steps `start_iter .. max_iter - 1` (see class docstring)."""
        self.state.max_iter = max_iter
        self._run_hooks("before_train")
        # before_train hooks (ResumeHook) may have advanced start_iter / epoch,
        # and FreezeBackboneHook may have frozen parameters -- only now wrap for DDP.
        self._train_model = self._model_for_training()
        # Epochs roll over lazily, so a checkpoint written right after an
        # epoch's last batch still carries that epoch's number: resume at the
        # next one. (A checkpoint from mid-epoch restarts that epoch's pass.)
        num_batches = _num_batches(dataloader)
        if self.state.epoch_step and num_batches is not None and self.state.epoch_step >= num_batches:
            self.state.epoch += 1
        self.state.epoch_step = 0
        self._run_hooks("before_epoch")

        self._set_sampler_epoch(dataloader)
        data_iter = iter(dataloader)
        for it in range(self.state.start_iter, max_iter):
            if self.state.should_stop:
                break
            self.state.iter = self.storage.iter = it
            data_iter = self._train_step(data_iter, dataloader)
            self._run_hooks("after_step")

        self._run_hooks("after_train")

    def _model_for_training(self) -> nn.Module:
        """The module the train step calls: `state.model` itself, or -- in
        distributed runs -- a DistributedDataParallel wrapper around it, which
        averages gradients across processes during backward; compiled on top
        if `compile` is set.

        Built once, on the first train() call and after the before_train
        hooks, so ResumeHook and FreezeBackboneHook act on the plain model
        (DDP fixes the set of trainable parameters when it is created).
        """
        model = self._wrap_ddp() if self.distributed else self.state.model
        if not self.compile:
            return model
        if self._compiled_model is None:
            kwargs = self.compile if isinstance(self.compile, dict) else {}
            self._compiled_model = torch.compile(model, **kwargs)
        return self._compiled_model

    def _wrap_ddp(self) -> DistributedDataParallel:
        if self._ddp_model is None:
            if self.sync_batchnorm:
                # Swaps BatchNorm modules for SyncBatchNorm; the parameters are
                # the same objects, so the optimizer is unaffected.
                self.state.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.state.model)
            device_ids = [self.state.device.index] if self.state.device.type == "cuda" else None
            self._ddp_model = DistributedDataParallel(
                self.state.model, device_ids=device_ids, find_unused_parameters=self.find_unused_parameters,
            )
        return self._ddp_model

    def _set_sampler_epoch(self, dataloader: Iterable) -> None:
        """Tell a sharded sampler which epoch starts, so every epoch gets a
        new shuffle -- the same one on every process, keeping shards disjoint.
        An IterableDataset with a `set_epoch` method (it shards and shuffles
        itself) is told the same way."""
        sampler = getattr(dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.state.epoch)
        dataset = getattr(dataloader, "dataset", None)
        if isinstance(dataset, torch.utils.data.IterableDataset) and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.state.epoch)

    def _train_step(self, data_iter: Iterator, dataloader: Iterable) -> Iterator:
        """One optimizer step over `grad_accum_steps` micro-batches."""
        state = self.state
        model = self._train_model
        step_start = time.perf_counter()
        state.optimizer.zero_grad(set_to_none=True)

        total_data_time = 0.0
        loss_sum = 0.0
        losses_sum: dict[str, float] = {}
        for micro_step in range(self.grad_accum_steps):
            data_iter, batch, data_time = self._next_batch(data_iter, dataloader)
            total_data_time += data_time
            state.last_data_time = data_time
            state.micro_step = micro_step
            state.current_batch = self._move_to_device(batch, state.device)

            self._run_hooks("before_step")  # e.g. BatchTransformHook edits state.current_batch

            # DDP averages gradients across processes during backward. Under
            # gradient accumulation that only needs to happen once, on the last
            # micro-batch -- until then gradients just accumulate locally.
            last_micro_step = micro_step == self.grad_accum_steps - 1
            with nullcontext() if self._ddp_model is None or last_micro_step else self._ddp_model.no_sync():
                with self._autocast():
                    output = state.task.run_step(model, state.current_batch, stage="train")
                # Mean over micro-batches: each contributes 1/N of the gradient.
                self.grad_scaler.scale(output.loss / self.grad_accum_steps).backward()

            loss_sum += output.loss.item()
            for k, v in output.losses.items():
                losses_sum[k] = losses_sum.get(k, 0.0) + v.item()

        if self.clip_grad_norm is not None:
            self.grad_scaler.unscale_(state.optimizer)  # clip the real gradients, not the scaled ones
            grad_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), self.clip_grad_norm)
            self.storage.put_scalar("grad_norm", grad_norm.item())
        self.grad_scaler.step(state.optimizer)  # skips the step if fp16 grads overflowed
        self.grad_scaler.update()

        n = self.grad_accum_steps
        losses = {"total_loss": loss_sum / n, **{k: v / n for k, v in losses_sum.items()}}
        if self.distributed:   # log the mean over all processes, not just this shard's
            mean = all_reduce_mean(torch.tensor(list(losses.values()), dtype=torch.float64, device=state.device))
            losses = dict(zip(losses, mean.tolist()))

        state.last_output = output
        self.storage.put_scalar("data_time", total_data_time)
        self.storage.put_scalar("iter_time", time.perf_counter() - step_start)
        self.storage.put_scalars(**losses)
        return data_iter

    @torch.no_grad()
    def evaluate(self, dataloader: Iterable[Mapping[str, Any]], prefix: str = "val") -> dict[str, float]:
        """One full pass over `dataloader` with stage="val".

        Per-batch scalars go to the storage's "eval_iter" axis (reset every
        call): `{prefix}_step_total_loss`, `{prefix}_step_<metric>`,
        `eval_time`, `eval_data_time`. The dataset-level results come from
        the task's Evaluator (`task.build_evaluator()`, sample-weighted) and
        are written once on the "iter" axis as `{prefix}_<name>` -- e.g.
        `val_accuracy`, `val_total_loss` -- where they persist, and also
        mirrored in `state.eval_results` before the after_eval hooks run.

        Distributed runs: every process must call this (as EvalHook does);
        each evaluates its own shard of a `build_dataloader` loader, and the
        Evaluator's `sync()` combines the shards before `summarize()`, so all
        processes get the same, exact dataset-level results.

        Returns the un-prefixed results (e.g. {"accuracy": ..., "total_loss": ...}).
        """
        state, storage = self.state, self.storage
        build_evaluator = getattr(state.task, "build_evaluator", None)
        evaluator = build_evaluator() if build_evaluator is not None else MeanMetricEvaluator()
        if self.distributed and not hasattr(evaluator, "sync"):
            raise TypeError(
                f"{type(evaluator).__name__} has no sync() method, which distributed evaluation needs to "
                "combine the processes' results -- see the Evaluator protocol in optastra/tasks/base.py."
            )
        was_training = state.model.training
        train_output = state.last_output
        state.model.eval()
        storage.reset_eval()
        storage.max_eval_iter = _num_batches(dataloader) or 0
        state.eval_results = {}
        evaluator.reset()
        self._run_hooks("before_eval")

        try:
            num_batches = 0
            fetch_start = time.perf_counter()
            for eval_it, batch in enumerate(dataloader):
                step_start = time.perf_counter()
                storage.eval_iter = eval_it
                storage.put_scalar("eval_data_time", step_start - fetch_start, axis="eval_iter")
                self._run_hooks("before_eval_step")

                batch = self._move_to_device(batch, state.device)
                with self._autocast():
                    output = state.task.run_step(state.model, batch, stage="val")
                state.last_output = output
                evaluator.process(output, batch)

                if output.loss is not None:
                    storage.put_scalar(f"{prefix}_step_total_loss", output.loss.item(), axis="eval_iter")
                storage.put_scalars(axis="eval_iter", **{f"{prefix}_step_{k}": float(v) for k, v in output.metrics.items()})
                storage.put_scalar("eval_time", time.perf_counter() - step_start, axis="eval_iter")

                self._run_hooks("after_eval_step")
                num_batches += 1
                fetch_start = time.perf_counter()

            if self.distributed:   # combine every process's shard (also when this one had no batches)
                evaluator.sync()
                num_batches = int(sum_across_processes({"batches": num_batches})["batches"])
            if num_batches == 0:
                self.logger.warning("evaluate(): the dataloader yielded no batches; no metrics were written.")
                return {}

            results = evaluator.summarize()
            state.eval_results = {f"{prefix}_{k}": v for k, v in results.items()}
            storage.put_scalars(axis="iter", **state.eval_results)
            return results
        finally:
            self._run_hooks("after_eval")
            state.last_output = train_output
            state.model.train(was_training)
