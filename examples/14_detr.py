"""DETR in optastra: set prediction for points and boxes, with swappable query types.

DETR (Carion et al., 2020) is a composition here, every part a ComponentRef:

    Backbone -> (Neck, e.g. FPN) -> (detr_encoder) -> Decoder -> Head on every decoder layer

What it shows:
    1. Point detection: `point_detr_r18_fpn` finds faint sources on single-band
       sky images and is trained with `point_detection_task` (Hungarian matching
       + L1 / focal objectness). Predictions are pixel positions with scores.
    2. Box detection: the same architecture with `detr_box_head`, trained by the
       ordinary `detection_task` with `criterion="detr_criterion"`. The decoder's
       query type is an ablation axis -- vanilla DETR, Conditional DETR or
       DAB-DETR -- swapped by name.
    3. External queries: decoders whose queries have reference points
       (conditional / DAB) also take queries placed at known positions.

Everything is shrunk (64-wide transformer, 1-2 layers) so it runs on CPU in
well under a minute; the presets' defaults are the paper-sized models.

Run with:
    python examples/14_detr.py
"""
import torch
from torch.utils.data import Dataset

from optastra import Architecture, Optimizer, Sample, Task, Trainer, build_dataloader
from optastra.decoders import QuerySet
from optastra.detection import keys

SMALL = {"num_heads": 4, "dim_feedforward": 128, "dropout": 0.0}


def tiny_detr(name: str, decoder: str, **overrides):
    """A registered DETR preset with a small transformer -- every part is still swappable."""
    defaults = dict(
        backbone=("resnet18", {"layers": [1, 1, 1, 1], "in_channels": 1}),   # single-band images
        d_model=64,                                                          # injected into encoder and decoder
        encoder=("detr_encoder", {"num_layers": 1, **SMALL}),
        decoder=(decoder, {"num_layers": 2, "num_queries": 10, **SMALL}),
    )
    return Architecture.create(name, **{**defaults, **overrides})


class ToySky(Dataset):
    """Single-band images of different sizes with 2-5 Gaussian sources each.
    Targets: "points" (G, 2) pixel (x, y) for point detection, and "boxes" /
    "labels" (a +-2 sigma box; label 0 = round, 1 = elongated) for boxes."""

    def __init__(self, num_images: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.samples = []
        for _ in range(num_images):
            h, w = (int(v) for v in torch.randint(48, 65, (2,), generator=g))
            n = int(torch.randint(2, 6, (), generator=g))
            xy = torch.rand(n, 2, generator=g) * torch.tensor([w - 12.0, h - 12.0]) + 6
            elongated = torch.rand(n, generator=g) < 0.5
            sx = torch.where(elongated, torch.tensor(3.0), torch.tensor(1.5))
            sy = torch.full((n,), 1.5)
            yy, xx = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
            image = 0.05 * torch.randn(1, h, w, generator=g)
            for (x, y), a, b in zip(xy.tolist(), sx.tolist(), sy.tolist()):
                image[0] += torch.exp(-0.5 * (((xx - x) / a) ** 2 + ((yy - y) / b) ** 2))
            boxes = torch.stack([xy[:, 0] - 2 * sx, xy[:, 1] - 2 * sy, xy[:, 0] + 2 * sx, xy[:, 1] + 2 * sy], dim=1)
            self.samples.append(Sample(image=image, target={
                "points": xy, "boxes": boxes.clamp(min=0), "labels": elongated.long(),
            }))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def task_batch(task, sample: Sample) -> dict:
    """One sample collated the way the task's dataloader would (here: "ragged")."""
    return build_dataloader([sample], task=task, batch_size=1).collate_fn([sample])


def point_detection() -> None:
    print("1. Point detection with point_detr_r18_fpn")
    model = tiny_detr(
        "point_detr_r18_fpn", "detr_decoder",                               # vanilla DETR queries
        neck=("fpn", {"out_channels": 64, "in_features": ("C3", "C4", "C5")}),
        encoder=("detr_encoder", {"num_layers": 1, "in_features": ("P3", "P4", "P5"), **SMALL}),
    )
    # The postprocessor keeps points whose objectness exceeds score_thresh (default 0.5);
    # here we keep the 4 best-scoring ones whatever their score, to show the output format.
    task = Task.create("point_detection_task", postprocessor=("point_postprocessor", {"score_thresh": 0.0, "max_points": 4}))
    train = build_dataloader(ToySky(256, seed=0), task=task, batch_size=8, shuffle=True)   # ragged collate: sizes differ
    val = build_dataloader(ToySky(16, seed=1), task=task, batch_size=8)

    trainer = Trainer(model, task, Optimizer.create("adamw", model, lr=1e-3), device="cpu", clip_grad_norm=1.0)
    print("   val before:", {k: round(v, 2) for k, v in trainer.evaluate(val).items()})
    trainer.train(train, max_iter=150)
    print("   val after: ", {k: round(v, 2) for k, v in trainer.evaluate(val).items()},
          "  (loc_error_px: mean distance of matched queries to their source)")

    model.eval()
    sample = ToySky(16, seed=1)[0]
    with torch.no_grad():
        found = task.run_step(model, task_batch(task, sample), "predict").predictions[0].target
    print(f"   true sources: {[[round(v, 1) for v in p] for p in sample.target['points'].tolist()]}")
    print(f"   top queries:  {[[round(v, 1) for v in p] for p in found['points'].tolist()]}"
          f"  scores {[round(s, 2) for s in found['scores'].tolist()]}")
    print("   (DETR converges slowly -- the paper trains for 300-500 epochs; after 150 CPU steps the\n"
          "    queries have only started to move towards the sources and objectness is not yet confident)")


def box_detection() -> None:
    print("2. Box detection: one task, three query types")
    task = Task.create("detection_task", num_classes=2, criterion="detr_criterion", postprocessor="detr_postprocessor")
    loader = build_dataloader(ToySky(8, seed=0), task=task, batch_size=4, collate_kwargs={"size_divisibility": 32})
    for name, decoder in [("detr_r50", "detr_decoder"),
                          ("conditional_detr_r50", "conditional_detr_decoder"),
                          ("dab_detr_r50", "dab_detr_decoder")]:
        torch.manual_seed(0)
        model = tiny_detr(name, decoder, num_classes=2)
        trainer = Trainer(model, task, Optimizer.create("adamw", model, lr=1e-3), device="cpu")
        trainer.train(loader, max_iter=4)
        losses = trainer.storage.latest()
        final = {k: round(losses[k], 3) for k in ("loss_ce", "loss_bbox", "loss_giou")}
        print(f"   {model.decoder.cfg.query_type:>11} queries: {final}  (+ the same terms per earlier layer, *_aux0)")
    model.eval()
    with torch.no_grad():
        detections = task.run_step(model, next(iter(loader)), "predict").predictions[0].target
    print(f"   decoded: {len(detections['boxes'])} boxes (XYXY pixels) with scores and labels -- no NMS needed")


def external_queries() -> None:
    print("3. External queries at known positions (DAB-DETR)")
    torch.manual_seed(0)
    model = tiny_detr("point_detr_r18_fpn", "dab_detr_decoder").eval()
    images = torch.rand(1, 1, 64, 64)
    known = torch.tensor([[[10.0, 20.0], [40.5, 33.0]]])                    # pixel (x, y)
    with torch.no_grad():
        out = model(images, queries=QuerySet(reference_points=known / 64))   # fractions of the image size
    points = out.extra[keys.POINTS][0, :2] * 64                              # external queries come first
    print(f"   {out.extra[keys.DECODER_TOKENS].shape[1]} tokens = 2 external + 10 learned; "
          f"external queries start at {points.round(decimals=2).tolist()}")


if __name__ == "__main__":
    torch.manual_seed(0)
    point_detection()
    box_detection()
    external_queries()
