"""Regression: predict continuous values per image -- e.g. photometric
redshift, galaxy size, stellar mass. Toy-sized, synthetic, CPU-only.

Scenario: each image holds one blob; predict two numbers per image, its
radius (2..12 px) and its brightness (0.5..1.5).

What it shows:
    1. `vanilla_regression_head` (pooled features -> `num_outputs` values)
       with the built-in `regression_task` (loss "mse" | "l1" | "huber").
    2. Targets are `Sample.target["values"]`: a scalar, or a
       (num_outputs,) tensor for several values per image.
    3. Evaluation reports dataset-level MAE, RMSE and R^2 through the task's
       Evaluator -- exact over the whole validation set, not an average of
       per-batch numbers (R^2 can't be averaged that way).

Writing a *new* kind of task? `optastra/tasks/regression.py` is a compact
template: the step methods `Task.run_step` calls, plus an Evaluator.

Run with:
    python examples/10_regression.py
"""
import torch
from torch.utils.data import Dataset

from optastra import Optimizer, Sample, Task, Trainer, build_dataloader, build_sequential_model

SIZE = 32
TARGETS = ("radius", "brightness")
NUM_TRAIN, NUM_VAL = 512, 128
BATCH_SIZE = 32
MAX_ITER = 150


class ToyBlobs(Dataset):
    """A disc of random radius and brightness at a random position, plus noise."""

    def __init__(self, num: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        radius = 2 + 10 * torch.rand(num, generator=g)
        brightness = 0.5 + torch.rand(num, generator=g)
        centres = 8 + (SIZE - 16) * torch.rand(num, 2, generator=g)
        yy, xx = torch.meshgrid(torch.arange(SIZE), torch.arange(SIZE), indexing="ij")
        dist = ((yy[None] - centres[:, 0, None, None]) ** 2 + (xx[None] - centres[:, 1, None, None]) ** 2).sqrt()
        discs = brightness[:, None, None] * (dist <= radius[:, None, None]).float()
        self.images = (discs[:, None] + 0.2 * torch.randn(num, 1, SIZE, SIZE, generator=g)).expand(-1, 3, -1, -1)
        self.values = torch.stack([radius, brightness], dim=1)   # (num, 2)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, idx: int) -> Sample:
        return Sample(image=self.images[idx].clone(), target={"values": self.values[idx]})


def main() -> None:
    torch.manual_seed(0)
    task = Task.create("regression_task", loss="huber")   # robust to the occasional large error
    model = build_sequential_model(
        backbone="resnet18",
        necks=["global_avg_pool"],
        head=("vanilla_regression_head", {"num_outputs": len(TARGETS)}),
    )
    train_loader = build_dataloader(ToyBlobs(NUM_TRAIN, seed=0), task=task, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = build_dataloader(ToyBlobs(NUM_VAL, seed=1), task=task, batch_size=BATCH_SIZE)

    trainer = Trainer(model, task, Optimizer.create("adamw", model, lr=1e-3), device="cpu")
    print("before training:", {k: round(v, 3) for k, v in trainer.evaluate(val_loader).items()})
    trainer.train(train_loader, max_iter=MAX_ITER)
    print(f"after {MAX_ITER} iterations:", {k: round(v, 3) for k, v in trainer.evaluate(val_loader).items()})

    # Inference: stage="predict" needs no targets and returns the decoded (B, num_outputs) values.
    model.eval()
    batch = next(iter(val_loader))
    with torch.no_grad():
        predicted = task.run_step(model, {"inputs": batch["inputs"][:3]}, stage="predict").predictions
    for pred, true in zip(predicted.tolist(), batch["targets"]["values"][:3].tolist()):
        print("  predicted", {n: round(p, 2) for n, p in zip(TARGETS, pred)},
              " true", {n: round(t, 2) for n, t in zip(TARGETS, true)})


if __name__ == "__main__":
    main()
