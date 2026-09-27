"""Self-supervised pretraining -> export the backbone -> fine-tune it on a
labelled task. Toy-sized (synthetic data, CPU, a few seconds) but the same
shape as a real SSL transfer experiment:

    1. BYOL pretraining. The algorithm builds its own model from config, so
       swapping the backbone is a config change, and `algo.hooks()` keeps the
       EMA target network updated (and checkpointed) after every step.
    2. `export_backbone` pulls the pretrained online backbone out of the
       training checkpoint as a plain state_dict.
    3. Fine-tuning: `weights=` on the backbone ref loads it into an ordinary
       Backbone -> Neck -> Head classification model.

Run with:
    python examples/06_ssl_pretrain_then_finetune.py
"""
import os
import shutil
import tempfile

import torch
from torch.utils.data import Dataset

from optastra import (
    Algorithm,
    Optimizer,
    Sample,
    Task,
    Trainer,
    Transform,
    build_dataloader,
    build_sequential_model,
)
from optastra.backbones import export_backbone
from optastra.training import Checkpointer
from optastra.training.hooks import CheckpointHook
from optastra.transforms import Compose, MultiViewTransform

IMAGE_SIZE = 32
NUM_IMAGES = 64
NUM_CLASSES = 4
BATCH_SIZE = 16
PRETRAIN_ITERS = 8
FINETUNE_ITERS = 8


class ToyImageDataset(Dataset):
    """Random uint8 images + labels, returned as `Sample`s run through `transform`.
    The same dataset serves both stages -- only the transform differs."""

    def __init__(self, transform: Transform, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randint(0, 256, (NUM_IMAGES, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8, generator=g)
        self.labels = torch.randint(0, NUM_CLASSES, (NUM_IMAGES,), generator=g)
        self.transform = transform

    def __len__(self) -> int:
        return NUM_IMAGES

    def __getitem__(self, idx: int) -> Sample:
        return self.transform(Sample(image=self.images[idx].clone(), target={"labels": self.labels[idx]}))


def pretrain(output_dir: str) -> str:
    """BYOL on unlabelled views; returns the last checkpoint's path."""
    # Swap the backbone here, e.g. ("vit_tiny", {"img_size": 32, "patch_size": 8}):
    # CNNs get a global_avg_pool neck automatically, ViTs pool with their CLS token.
    small_mlp = {"hidden_dim": 128, "out_dim": 64}
    algo = Algorithm.create("byol", backbone="resnet18", projector=small_mlp, predictor=small_mlp)
    model = algo.build_model()

    # Two independently augmented views per image -> Sample.views -> batch["views"].
    view = Compose([
        Transform.create("to_float"),
        Transform.create("random_resized_crop", size=IMAGE_SIZE, scale=(0.5, 1.0)),
        Transform.create("random_hflip"),
    ])
    dataset = ToyImageDataset(MultiViewTransform(view, n_views=2))
    loader = build_dataloader(dataset, task=algo, batch_size=BATCH_SIZE, shuffle=True)   # multiview collate

    optimizer = Optimizer.create("adamw", model, lr=1e-3)
    hooks = algo.hooks() + [CheckpointHook(output_dir, save_every=PRETRAIN_ITERS - 1)]
    trainer = Trainer(model, algo, optimizer, hooks=hooks, device="cpu")
    trainer.train(loader, max_iter=PRETRAIN_ITERS)
    print(f"[pretrain] {PRETRAIN_ITERS} BYOL steps, final loss {trainer.storage.latest()['total_loss']:.4f}, "
          f"target momentum {trainer.storage.latest()['byol_momentum']:.4f}")
    return Checkpointer(output_dir).latest()


def finetune(backbone_weights: str) -> None:
    """Supervised classification starting from the pretrained backbone."""
    model = build_sequential_model(
        backbone=("resnet18", {"weights": backbone_weights}),
        necks=["global_avg_pool"],
        head=("vanilla_classification_head", {"num_classes": NUM_CLASSES}),
    )
    task = Task.create("classification_task")
    dataset = ToyImageDataset(Transform.create("to_float"))
    loader = build_dataloader(dataset, task=task, batch_size=BATCH_SIZE, shuffle=True)

    trainer = Trainer(model, task, Optimizer.create("adamw", model, lr=1e-3), device="cpu")
    trainer.train(loader, max_iter=FINETUNE_ITERS)
    metrics = trainer.evaluate(loader)
    print(f"[finetune] {FINETUNE_ITERS} steps, " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


def main() -> None:
    torch.manual_seed(0)
    output_dir = tempfile.mkdtemp(prefix="optastra_ssl_")
    try:
        checkpoint = pretrain(output_dir)
        backbone_path = os.path.join(output_dir, "byol_resnet18_backbone.pt")
        weights = export_backbone(checkpoint, backbone_path)   # "online_backbone." prefix auto-detected
        print(f"[export] {len(weights)} backbone tensors from {os.path.basename(checkpoint)} -> {backbone_path}")
        finetune(backbone_path)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
