"""Hooks, checkpoints and resuming an interrupted run.

The Trainer is a plain loop; everything else -- logging, evaluation,
checkpointing, LR schedules, early stopping -- is a Hook reading and writing
`TrainerState` / `EventStorage`. This example shows:

    1. Writing custom hooks: one that adds a metric to storage (and so to the
       console log and `metrics.jsonl`, with no other changes), and one that
       simulates a job being killed mid-training.
    2. Periodic checkpoints (`CheckpointHook`) and resuming (`ResumeHook`):
       the checkpoint holds model + optimizer + every hook's state + RNG
       state, so a resumed run continues *exactly* where the interrupted one
       stopped -- the example checks the final weights are bit-identical to
       an uninterrupted run.
    3. Reading the structured `logs/metrics.jsonl` log afterwards.

Note: resuming is exact here because the checkpoint falls on an epoch
boundary and the data pipeline has no random augmentation. Mid-epoch, the
resumed run restarts the epoch's data order; and the transforms' own random
generator (`optastra.transforms.rng`) is not stored in checkpoints yet.

Run with:
    python examples/11_hooks_checkpoint_resume.py
"""
import json
import os
import shutil
import tempfile

import torch
from torch.utils.data import TensorDataset

from optastra import Optimizer, Sample, Task, Trainer, build_dataloader, build_sequential_model
from optastra.training import Checkpointer
from optastra.training.hooks import CheckpointHook, EvalHook, Hook, JSONWriterHook, ResumeHook
from optastra.training.state import TrainerState

NUM_SAMPLES, BATCH_SIZE = 64, 16   # 4 iterations per epoch
MAX_ITER = 16
SAVE_EVERY = 7                     # iteration 7 is the last one of epoch 1 -> an epoch-boundary checkpoint


# --- 1. Custom hooks -------------------------------------------------------

class ThroughputHook(Hook):
    """Adds "samples_per_sec" to storage after every step. Anything put in
    storage shows up in the console printer and metrics.jsonl automatically."""

    def after_step(self, state: TrainerState) -> None:
        batch_size = state.current_batch["inputs"].shape[0]
        state.storage.put_scalar("samples_per_sec", batch_size / state.storage.latest()["iter_time"])


class SimulatedCrashHook(Hook):
    """Stops training after `stop_after` -- stands in for a preempted / killed job.
    The Trainer only sees `state.should_stop`; it never knows why."""

    def __init__(self, stop_after: int):
        self.stop_after = stop_after

    def after_step(self, state: TrainerState) -> None:
        if state.iter == self.stop_after:
            print(f"  !! simulated crash after iteration {state.iter}")
            state.should_stop = True


# --- Setup shared by every run ---------------------------------------------

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.images = torch.rand(NUM_SAMPLES, 3, 32, 32, generator=g)
        self.labels = torch.randint(0, 4, (NUM_SAMPLES,), generator=g)

    def __len__(self) -> int:
        return NUM_SAMPLES

    def __getitem__(self, idx: int) -> Sample:
        return Sample(image=self.images[idx], target={"labels": self.labels[idx]})


def make_trainer(hooks: list[Hook], seed: int) -> tuple[Trainer, object, object]:
    torch.manual_seed(seed)
    task = Task.create("classification_task")
    model = build_sequential_model(
        "resnet18", ["global_avg_pool"], ("vanilla_classification_head", {"num_classes": 4, "hidden_features": 64}),
    )
    trainer = Trainer(model, task, Optimizer.create("adamw", model, lr=1e-3), hooks=hooks, device="cpu")
    loader = build_dataloader(ToyDataset(), task=task, batch_size=BATCH_SIZE, shuffle=True)
    return trainer, loader, model


def main() -> None:
    root = tempfile.mkdtemp(prefix="optastra_resume_")
    try:
        # --- Reference: one uninterrupted run ------------------------------
        ref_dir = os.path.join(root, "reference")
        reference, loader, ref_model = make_trainer([CheckpointHook(ref_dir, save_every=SAVE_EVERY)], seed=0)
        reference.train(loader, max_iter=MAX_ITER)
        print(f"reference run: {MAX_ITER} iterations, final loss {reference.storage.latest()['total_loss']:.4f}")

        # --- 2a. Same run, killed after the checkpoint at iteration 7 -------
        run_dir = os.path.join(root, "run")
        log_dir = os.path.join(run_dir, "logs")
        crashed, loader, _ = make_trainer([
            CheckpointHook(run_dir, save_every=SAVE_EVERY),
            SimulatedCrashHook(stop_after=SAVE_EVERY + 2),   # dies 2 iterations after the checkpoint
            ThroughputHook(),
            JSONWriterHook(log_dir, log_every=1),
        ], seed=0)
        crashed.train(loader, max_iter=MAX_ITER)
        print(f"  checkpoints on disk: {sorted(os.listdir(run_dir))}")

        # --- 2b. A fresh process would do exactly this: new objects (even a
        # different init seed -- the weights come from the checkpoint), plus
        # a ResumeHook that loads the latest periodic checkpoint.
        resumed, loader, resumed_model = make_trainer([
            ResumeHook(Checkpointer(run_dir)),
            CheckpointHook(run_dir, save_every=SAVE_EVERY),
            ThroughputHook(),
            JSONWriterHook(log_dir, log_every=1),
        ], seed=123)
        resumed.register_hooks([EvalHook(eval_period=0, eval_fn=lambda: resumed.evaluate(loader))])   # eval once at the end
        resumed.train(loader, max_iter=MAX_ITER)
        print(f"resumed run: continued from iteration {resumed.state.start_iter} to {MAX_ITER}, "
              f"final loss {resumed.storage.latest()['total_loss']:.4f}")

        identical = all(torch.equal(a, b) for a, b in zip(ref_model.state_dict().values(),
                                                           resumed_model.state_dict().values()))
        print(f"final weights identical to the uninterrupted run: {identical}")

        # --- 3. The structured log ------------------------------------------
        with open(os.path.join(log_dir, "metrics.jsonl")) as f:
            records = [json.loads(line) for line in f]
        train_records = [r for r in records if r["phase"] == "train"]
        summary = next(r for r in records if r["phase"] == "eval_summary")
        print(f"metrics.jsonl: {len(records)} records; train iterations logged: {[r['iter'] for r in train_records]}")
        print(f"  (8 and 9 appear twice: logged by the crashed run, then redone after resuming from iteration 7)")
        print(f"  a train record's scalars: {sorted(train_records[-1]['scalars'])}")
        print(f"  eval summary: { {k: round(v, 4) for k, v in summary['metrics'].items()} }")
        print("  (eval runs in eval mode with BatchNorm running statistics, which barely warm up in 16 steps --"
              " hence the gap to the training loss)")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
