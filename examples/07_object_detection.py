"""Object detection with Faster R-CNN: synthetic "sources" on a noisy sky,
variable image sizes, box-aware augmentation, training, evaluation and
decoded predictions -- toy-sized so it runs on CPU in well under a minute.

What it shows:
    1. A detection dataset is just `Sample(image, target={"boxes", "labels"})`
       -- boxes are (N, 4) XYXY in absolute pixels, labels are 0..C-1.
    2. Geometric transforms move the boxes together with the pixels.
    3. The "ragged" collate (picked automatically from `task.collate`) pads
       images of different sizes into one batch and records `image_sizes`,
       so predictions are clipped to each image's real extent.
    4. `Architecture.create("faster_rcnn_r18_fpn", num_classes=...)` is the
       whole model: backbone + FPN + RPN + ROIAlign + box head, each one a
       ComponentRef in its config that can be swapped by name.
    5. `task.run_step(..., stage="predict")` returns one `Sample` per image
       with the final boxes / scores / labels after NMS.

Run with:
    python examples/07_object_detection.py
"""
import torch
from torch.utils.data import Dataset

from optastra import Architecture, Optimizer, Sample, Task, Trainer, Transform, build_dataloader
from optastra.transforms import Compose

CLASSES = ("compact", "extended")   # label 0: small round-ish source, label 1: elongated source
NUM_TRAIN = 24
NUM_VAL = 8
BATCH_SIZE = 4
MAX_ITER = 12


class ToySkyDetection(Dataset):
    """Noisy sky images of varying size with 1-3 bright sources each.
    "compact" sources are small squares, "extended" ones are wide boxes."""

    def __init__(self, num_images: int, transform: Transform | None, seed: int):
        self.samples = []
        g = torch.Generator().manual_seed(seed)
        for _ in range(num_images):
            h, w = (int(v) for v in torch.randint(96, 145, (2,), generator=g))   # sizes differ per image
            image = 0.1 * torch.rand(3, h, w, generator=g)                      # sky background
            boxes, labels = [], []
            for _ in range(int(torch.randint(1, 4, (), generator=g))):
                label = int(torch.randint(0, 2, (), generator=g))
                bw, bh = (16, 16) if label == 0 else (40, 14)
                x0 = int(torch.randint(0, w - bw, (), generator=g))
                y0 = int(torch.randint(0, h - bh, (), generator=g))
                image[:, y0:y0 + bh, x0:x0 + bw] += 0.8                          # the source itself
                boxes.append([x0, y0, x0 + bw, y0 + bh])
                labels.append(label)
            self.samples.append(Sample(
                image=image.clamp(0, 1),
                target={"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels)},
            ))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        sample = self.samples[idx]
        sample = Sample(image=sample.image.clone(), target={k: v.clone() for k, v in sample.target.items()})
        return self.transform(sample) if self.transform else sample


def main() -> None:
    torch.manual_seed(0)

    # Flips move target["boxes"] with the image (so do crops, resizes and rotations).
    train_transform = Compose([
        Transform.create("random_hflip"),
        Transform.create("random_vflip"),
    ])
    train_set = ToySkyDetection(NUM_TRAIN, train_transform, seed=0)
    val_set = ToySkyDetection(NUM_VAL, None, seed=1)

    model = Architecture.create("faster_rcnn_r18_fpn", num_classes=len(CLASSES))
    task = Task.create("detection_task", num_classes=len(CLASSES))   # must agree with the model -- checked every step

    # task.collate == "ragged": images are padded to a common size (rounded up to
    # a multiple of 32 here, which FPN-style strides like) and "image_sizes"
    # records each original (H, W).
    loader_kwargs = dict(task=task, batch_size=BATCH_SIZE, collate_kwargs={"size_divisibility": 32})
    train_loader = build_dataloader(train_set, shuffle=True, **loader_kwargs)
    val_loader = build_dataloader(val_set, shuffle=False, **loader_kwargs)

    batch = next(iter(train_loader))
    print(f"padded batch: {tuple(batch['inputs'].shape)}, original sizes: {batch['image_sizes']}")

    optimizer = Optimizer.create("sgd", model, lr=0.01)
    trainer = Trainer(model, task, optimizer, device="cpu", clip_grad_norm=10.0)
    trainer.train(train_loader, max_iter=MAX_ITER)
    losses = {k: round(v, 3) for k, v in trainer.storage.latest().items() if k.endswith("loss")}
    print(f"after {MAX_ITER} iterations: {losses}")

    # Dataset-level validation loss (+ the criterion's metrics), sample-weighted.
    print("val:", {k: round(v, 4) for k, v in trainer.evaluate(val_loader).items()})

    # Inference: stage="predict" skips losses and returns decoded detections.
    model.eval()
    val_batch = next(iter(val_loader))
    with torch.no_grad():
        output = task.run_step(model, {"inputs": val_batch["inputs"], "image_sizes": val_batch["image_sizes"]},
                               stage="predict")
    first = output.predictions[0]
    print(f"image 0 ({first.meta['image_size']}): {len(first.target['boxes'])} detections")
    for box, score, label in list(zip(first.target["boxes"], first.target["scores"], first.target["labels"]))[:3]:
        print(f"  {CLASSES[label]:>8}  score={score:.2f}  box={[round(v) for v in box.tolist()]}")
    print("  (a real run trains for thousands of iterations -- after 12 steps the scores are still near-uniform)")


if __name__ == "__main__":
    main()
