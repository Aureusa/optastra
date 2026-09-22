"""Advanced: wire together every training-time piece -- Backbone/Neck/Head,
Task, Optimizer, Scheduler, DataLoader, Trainer, and hooks -- the same way
a real training script does (see e.g. `rand_augment_all_ops.py` in a
downstream project), just toy-sized so it runs on CPU in a few seconds
with synthetic data instead of a real dataset.

Run with:
    python examples/05_full_training_pipeline.py
"""
import shutil
import tempfile

import torch
from torch.utils.data import Dataset

from optastra import (
    Optimizer,
    Scheduler,
    Trainer,
    Task,
    Transform,
    Sample,
    build_dataloader,
    build_sequential_model,
)
from optastra.transforms import Compose
from optastra.training.hooks import (
    BestCheckpointHook,
    BestMetricTracker,
    default_hooks,
    EvalHook,
    SchedulerHook,
    EarlyStoppingHook,
)
from optastra.training.logging_config import setup_logging

NUM_CLASSES = 5
IMAGE_SIZE = 32
NUM_TRAIN = 256
NUM_VAL = 64
BATCH_SIZE = 16
EPOCHS = 3
LOG_EVERY = 5


class ToyImageDataset(Dataset):
    """Stands in for a real dataset (e.g. DECaLSDataset) -- random images
    and labels, but the same `Sample` + `transform` contract every real
    Optastra dataset follows."""

    def __init__(self, num_samples: int, transform: Transform, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randint(
            0, 256, (num_samples, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8, generator=generator
        )
        self.labels = torch.randint(0, NUM_CLASSES, (num_samples,), generator=generator)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Sample:
        sample = Sample(image=self.images[idx].clone(), target={"labels": self.labels[idx].clone()})
        return self.transform(sample)


def main() -> None:
    output_dir = tempfile.mkdtemp(prefix="optastra_toy_pipeline_")
    setup_logging(output_dir, filename="log.log")
    torch.manual_seed(42)

    # MODEL -- Backbone -> Neck -> Head, same composition as example 01.
    model = build_sequential_model(
        backbone="resnet18",
        necks=["global_avg_pool"],
        head=("vanilla_classification_head", {"num_classes": NUM_CLASSES}),
    )

    # TASK -- owns losses/metrics/target formatting, Trainer never sees them.
    task = Task.create("classification_task")

    # OPTIMIZER
    optimizer = Optimizer.create("adamw", model, lr=1e-3)

    # TRAINER -- orchestrates model + task + optimizer + hooks.
    trainer = Trainer(model=model, task=task, optimizer=optimizer, device="cpu")

    # DATA -- augment at train time only, same to_float-first convention
    # every real pipeline follows.
    train_transform = Compose([
        Transform.create("to_float"),
        Transform.create("rand_augment", num_ops=2, magnitude=5),
    ])
    val_transform = Compose([
        Transform.create("to_float"),
    ])
    train_dataset = ToyImageDataset(NUM_TRAIN, train_transform, seed=0)
    val_dataset = ToyImageDataset(NUM_VAL, val_transform, seed=1)

    train_loader = build_dataloader(train_dataset, task=task, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = build_dataloader(val_dataset, task=task, batch_size=BATCH_SIZE, shuffle=False)

    max_iter = EPOCHS * (len(train_dataset) // BATCH_SIZE)
    eval_period = max_iter // EPOCHS  # once per epoch

    # SCHEDULER -- stepped by a hook, Trainer/Task never know it exists.
    scheduler = Scheduler.create("warmup_cosine", optimizer=optimizer, warmup_steps=5, total_steps=max_iter)

    # HOOKS -- behavior is added entirely through hooks, not by editing Trainer.
    # One shared tracker: best-model checkpointing and early stopping agree on
    # what "best" means, regardless of hook order.
    best_loss = BestMetricTracker("val_total_loss", mode="min")
    hooks = default_hooks(
        log_every=LOG_EVERY,
        output_dir=output_dir,
        checkpoint_every=eval_period,
    ) + [
        SchedulerHook(scheduler),
        EvalHook(eval_period, eval_fn=lambda: trainer.evaluate(val_loader)),
        BestCheckpointHook(output_dir, tracker=best_loss),
        EarlyStoppingHook(tracker=best_loss, patience=3),
    ]
    trainer.register_hooks(hooks)

    print(f"Training {type(model).__name__} for {max_iter} iterations ({EPOCHS} epochs)...")
    trainer.train(train_loader, max_iter=max_iter)
    print(f"Done. Logs/checkpoints written to {output_dir}")

    shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
