"""Compare backbones on the same data -- where the only thing that changes
between runs is a backbone name (plus its overrides). This is the core
workflow the framework is designed for: ResNet vs EfficientNet vs ConvNeXt
vs ViT becomes a loop over config values, not four training scripts.

Everything else -- neck, head, task, optimizer, trainer, data, seed -- is
shared, so differences in the results come from the backbone alone.

Scenario: classify four synthetic shapes (disc, square, ring, cross) at
random positions and sizes. Toy-sized: 32x32 images, a few dozen steps per
model on CPU. Expect the numbers to be noisy -- the point is the workflow.

Run with:
    python examples/12_compare_backbones.py
"""
import time

import torch
from torch.utils.data import Dataset

from optastra import Optimizer, Sample, Task, Trainer, build_dataloader, build_sequential_model, count_parameters

SIZE = 32
SHAPES = ("disc", "square", "ring", "cross")
NUM_TRAIN, NUM_VAL = 512, 256
BATCH_SIZE = 32
MAX_ITER = 40

# The experiment grid: (label, backbone ref). In a real study this list would
# come from a YAML file or the command line.
BACKBONES = [
    ("resnet18", "resnet18"),
    ("efficientnet_b0", "efficientnet_b0"),
    ("convnext_tiny", "convnext_tiny"),
    # A ViT sized for 32 px inputs: 4x4 patches -> an 8x8 token grid.
    ("vit_tiny/4 (6 blocks)", ("vit_tiny", {"img_size": SIZE, "patch_size": 4, "depth": 6})),
]


class ToyShapes(Dataset):
    def __init__(self, num: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        yy, xx = torch.meshgrid(torch.arange(SIZE).float(), torch.arange(SIZE).float(), indexing="ij")
        self.labels = torch.randint(0, len(SHAPES), (num,), generator=g)
        images = []
        for label in self.labels.tolist():
            r = 4 + 6 * float(torch.rand((), generator=g))
            cy, cx = (r + (SIZE - 2 * r) * torch.rand(2, generator=g)).tolist()
            dy, dx = (yy - cy).abs(), (xx - cx).abs()
            dist = (dy ** 2 + dx ** 2).sqrt()
            mask = [dist <= r, (dy <= r) & (dx <= r), (dist <= r) & (dist >= r - 2),
                    ((dy <= 1.5) | (dx <= 1.5)) & (dy <= r) & (dx <= r)][label]
            images.append(mask.float() + 0.3 * torch.randn(SIZE, SIZE, generator=g))
        self.images = torch.stack(images)[:, None].expand(-1, 3, -1, -1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Sample:
        return Sample(image=self.images[idx].clone(), target={"labels": self.labels[idx]})


def run(backbone) -> dict:
    """One experiment: identical except for `backbone`."""
    torch.manual_seed(0)
    model = build_sequential_model(
        backbone=backbone,
        necks=["global_avg_pool"],   # works for CNNs and ViT alike (ViT exposes a spatial token map)
        head=("vanilla_classification_head", {"num_classes": len(SHAPES), "hidden_features": 128}),
    )
    task = Task.create("classification_task")
    train_loader = build_dataloader(ToyShapes(NUM_TRAIN, seed=0), task=task, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = build_dataloader(ToyShapes(NUM_VAL, seed=1), task=task, batch_size=BATCH_SIZE)

    trainer = Trainer(model, task, Optimizer.create("adamw", model, lr=1e-3), device="cpu", clip_grad_norm=1.0)
    start = time.perf_counter()
    trainer.train(train_loader, max_iter=MAX_ITER)
    ms_per_iter = 1000 * (time.perf_counter() - start) / MAX_ITER
    val = trainer.evaluate(val_loader)
    return {"params_m": count_parameters(model) / 1e6, "ms_per_iter": ms_per_iter, **val}


def main() -> None:
    print(f"{'backbone':<24}{'params':>9}{'ms/iter':>10}{'val_acc':>10}{'val_loss':>10}")
    for label, backbone in BACKBONES:
        r = run(backbone)
        print(f"{label:<24}{r['params_m']:>8.1f}M{r['ms_per_iter']:>10.0f}{r['accuracy']:>10.3f}{r['total_loss']:>10.3f}")
    print(f"({MAX_ITER} steps each on CPU; chance accuracy is {1 / len(SHAPES):.2f})")


if __name__ == "__main__":
    main()
