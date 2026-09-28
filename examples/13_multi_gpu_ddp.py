"""Multi-GPU training with DistributedDataParallel (DDP): one process per
GPU, each training on its own shard of the data, gradients averaged across
processes every step. The *same* script runs either way:

    python examples/13_multi_gpu_ddp.py                           # 1 process
    torchrun --nproc_per_node=2 examples/13_multi_gpu_ddp.py      # 2 processes (e.g. 2 GPUs)

Nothing in the training code mentions DDP: `Trainer` and `build_dataloader`
detect torchrun and set it up (process group, one GPU per process, sharded
data, gradient averaging, exact sharded evaluation, checkpoints and logs
written once by rank 0). On a machine with fewer GPUs than processes it runs
on CPU instead, so you can try it anywhere.

On a SLURM cluster (e.g. one node with 2 A100s), in your batch script:

    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --gpus-per-node=2
    #SBATCH --cpus-per-task=16
    srun torchrun --standalone --nproc_per_node=2 train.py

Things that change with N processes:
    - `batch_size` is per process: the effective batch is batch_size * N,
      so scale the learning rate (or halve batch_size to keep the old one).
    - `max_iter` still counts optimizer steps; each step now sees N times
      more data, so an epoch takes N times fewer iterations.

Run with:
    python examples/13_multi_gpu_ddp.py
"""
import os
import shutil
import tempfile

import torch
from torch.utils.data import Dataset

from optastra import Optimizer, Sample, Task, Trainer, build_dataloader, build_sequential_model
from optastra.core import distributed
from optastra.training.hooks import CheckpointHook, EvalHook, JSONWriterHook

NUM_CLASSES = 4
SIZE = 32
PER_PROCESS_BATCH = 16
MAX_ITER = 30


class ToyBrightness(Dataset):
    """Class k = images of mean brightness ~ k / 4, plus noise: learnable in a few steps."""

    def __init__(self, num: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, NUM_CLASSES, (num,), generator=g)
        level = (self.labels.float() + 0.5) / NUM_CLASSES
        self.images = level[:, None, None, None] + 0.2 * torch.randn(num, 3, SIZE, SIZE, generator=g)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Sample:
        return Sample(image=self.images[idx], target={"labels": self.labels[idx]})


def say(*args) -> None:
    """print() on rank 0 only -- otherwise every process prints the same line."""
    if distributed.is_main_process():
        print(*args, flush=True)


def main() -> None:
    torch.manual_seed(0)   # same seed on every process -> same initial weights (DDP also syncs them)
    distributed.init_distributed()   # optional: Trainer / build_dataloader would do it; done early to know the rank
    world_size = distributed.get_world_size()
    use_gpu = torch.cuda.device_count() >= world_size
    say(f"{world_size} process(es) on {'GPU' if use_gpu else 'CPU'}; "
        f"batch {PER_PROCESS_BATCH} per process = {PER_PROCESS_BATCH * world_size} per optimizer step")

    # Every process must write to the same run directory: rank 0 picks it.
    output_dir = distributed.broadcast_object(tempfile.mkdtemp(prefix="optastra_ddp_") if distributed.is_main_process() else None)

    task = Task.create("classification_task")
    model = build_sequential_model(
        "resnet18", ["global_avg_pool"], ("vanilla_classification_head", {"num_classes": NUM_CLASSES, "hidden_features": 64}),
    )
    # Under torchrun each loader yields only this process's shard (disjoint across processes).
    train_loader = build_dataloader(ToyBrightness(512, seed=0), task=task, batch_size=PER_PROCESS_BATCH, shuffle=True)
    val_loader = build_dataloader(ToyBrightness(200, seed=1), task=task, batch_size=PER_PROCESS_BATCH)
    print(f"  [rank {distributed.get_rank()}] {len(train_loader.dataset)} training samples in total, "
          f"{len(train_loader.sampler)} in this process's shard", flush=True)

    trainer = Trainer(
        model, task, Optimizer.create("adamw", model, lr=1e-3 * world_size),   # linear LR scaling with the global batch
        device="cuda" if use_gpu else "cpu",
        precision="bf16" if use_gpu else "fp32",   # A100s: bf16 is the fast path
        sync_batchnorm=use_gpu and world_size > 1,  # BatchNorm statistics over the global batch (GPU only)
    )
    trainer.register_hooks([
        EvalHook(eval_period=10, eval_fn=lambda: trainer.evaluate(val_loader)),   # runs on every process
        CheckpointHook(output_dir, save_every=10),                                 # writes on rank 0 only
        JSONWriterHook(os.path.join(output_dir, "logs"), log_every=5),             # writes on rank 0 only
    ])
    trainer.train(train_loader, max_iter=MAX_ITER)

    # Sharded evaluation is combined exactly: every process holds the same dataset-level result.
    results = {k: round(v, 4) for k, v in trainer.state.eval_results.items()}
    print(f"  [rank {distributed.get_rank()}] final eval: {results}", flush=True)

    distributed.barrier()
    if distributed.is_main_process():
        say(f"files (written once, by rank 0): {sorted(os.listdir(output_dir))}")
        shutil.rmtree(output_dir, ignore_errors=True)
    distributed.cleanup()


if __name__ == "__main__":
    main()
