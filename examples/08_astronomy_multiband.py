"""Astronomy-style data: multi-band, background-subtracted, high-dynamic-range
images -- the case Optastra was built for. Toy-sized, synthetic, CPU-only.

Survey cutouts are not 8-bit RGB photos: they have 1..N bands (here 5,
like ugriz), float fluxes with values far above 1 (bright cores) and below
0 (background-subtracted noise), or raw integer counts. This example shows:

    1. Ingestion: `to_float` is dtype-aware -- uint8 / uint16 counts are
       scaled to [0, 1], float fluxes are passed through untouched.
    2. A domain-specific transform registered in a few lines: a per-band
       asinh stretch (the standard way to compress astronomical dynamic range).
    3. Stock augmentations on 5-band float data: rotations by any angle
       (galaxies have no "up"), flips, PSF-like blur, and photometric
       RandAugment ops -- none of them clip to [0, 1] or assume 3 channels.
    4. Any backbone takes the band count as `in_channels`.

Run with:
    python examples/08_astronomy_multiband.py
"""
import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from optastra import Optimizer, Sample, Task, Trainer, Transform, build_dataloader, build_sequential_model
from optastra.transforms import Compose, seed_transforms

BANDS = ("u", "g", "r", "i", "z")
CLASSES = ("point_source", "extended")
SIZE = 48
NUM_TRAIN, NUM_VAL = 128, 32
BATCH_SIZE = 16
MAX_ITER = 16


# --- 1. A domain-specific transform: per-band asinh stretch -----------------

@dataclass
class AsinhStretchConfig:
    softening_sigmas: float = 3.0   # the stretch turns from linear to logarithmic at this many noise sigmas


class AsinhStretch(Transform):
    """x -> asinh(x / (k * sigma_band)): linear around the noise level,
    logarithmic for bright sources. sigma is estimated per band with the
    median absolute deviation, so it adapts to each band's noise."""

    def __init__(self, cfg: AsinhStretchConfig = AsinhStretchConfig()):
        self.cfg = cfg

    def __call__(self, sample: Sample) -> Sample:
        flat = sample.image.flatten(1)                                    # (C, H*W)
        mad = (flat - flat.median(dim=1, keepdim=True).values).abs().median(dim=1).values
        sigma = (1.4826 * mad).clamp_min(1e-6)                            # MAD -> Gaussian sigma
        sample.image = torch.asinh(sample.image / (self.cfg.softening_sigmas * sigma[:, None, None]))
        return sample


@Transform.register(config=AsinhStretchConfig())
def asinh_stretch(cfg: AsinhStretchConfig) -> AsinhStretch:
    return AsinhStretch(cfg)


# --- Synthetic survey cutouts ------------------------------------------------

def render_source(label: int, g: torch.Generator) -> torch.Tensor:
    """(5, SIZE, SIZE) float fluxes: a point source (PSF-sized Gaussian) or an
    extended, elongated exponential profile, with band-dependent brightness
    ("colour") and background-subtracted Gaussian noise (can be negative)."""
    yy, xx = torch.meshgrid(torch.arange(SIZE) - SIZE / 2, torch.arange(SIZE) - SIZE / 2, indexing="ij")
    angle = float(torch.rand((), generator=g)) * math.pi
    u = xx * math.cos(angle) + yy * math.sin(angle)
    v = -xx * math.sin(angle) + yy * math.cos(angle)
    if label == 0:
        profile = torch.exp(-(xx ** 2 + yy ** 2) / (2 * 1.5 ** 2))                      # PSF, sigma 1.5 px
    else:
        profile = torch.exp(-torch.sqrt((u / 6.0) ** 2 + (v / 2.5) ** 2))               # elongated disk
    peak = 10 ** (1 + 2.5 * float(torch.rand((), generator=g)))                          # 10 .. ~3000
    colour = torch.linspace(0.3, 1.0, len(BANDS)) if label else torch.linspace(1.0, 0.4, len(BANDS))
    noise = torch.randn(len(BANDS), SIZE, SIZE, generator=g)                              # sigma = 1
    return peak * colour[:, None, None] * profile + noise


class ToySurveyCutouts(Dataset):
    def __init__(self, num: int, transform: Transform, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, len(CLASSES), (num,), generator=g)
        self.images = torch.stack([render_source(int(l), g) for l in self.labels])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Sample:
        return self.transform(Sample(image=self.images[idx].clone(), target={"labels": self.labels[idx]}))


def main() -> None:
    torch.manual_seed(0)
    seed_transforms(0)

    # --- Ingestion: to_float is dtype-aware -------------------------------------
    to_float = Transform.create("to_float")
    raw_counts = torch.randint(0, 65536, (len(BANDS), 4, 4), dtype=torch.int32).to(torch.uint16)
    fluxes = torch.tensor([-3.0, 0.5, 2500.0]).view(3, 1, 1).expand(3, 4, 4).clone()
    print(f"uint16 counts -> [{to_float(Sample(image=raw_counts)).image.min():.3f}, "
          f"{to_float(Sample(image=raw_counts)).image.max():.3f}]  (scaled by 1/65535)")
    print(f"float fluxes  -> {sorted(set(to_float(Sample(image=fluxes)).image.flatten().tolist()))}  (unchanged)")

    # --- Augmentation on 5-band float data -------------------------------------
    stretch = Transform.create("asinh_stretch")
    train_transform = Compose([
        Transform.create("to_float"),
        stretch,
        Transform.create("random_rotation", degrees=(0.0, 360.0)),   # rotation-invariant targets
        Transform.create("random_hflip"),
        Transform.create("random_vflip"),
        Transform.create("gaussian_blur", p=0.3),                     # seeing / PSF variation
        # Photometric ops infer each image's own value range, so nothing is clipped to [0, 1].
        Transform.create("rand_augment", num_ops=1, magnitude=5, ops=["Identity", "Contrast", "Brightness", "Sharpness"]),
    ])
    val_transform = Compose([Transform.create("to_float"), stretch])

    raw = render_source(1, torch.Generator().manual_seed(8))   # a bright one: peak ~1300, noise sigma 1
    out = train_transform(Sample(image=raw.clone(), target={"labels": torch.tensor(1)})).image
    print(f"raw cutout:       shape {tuple(raw.shape)}, range [{raw.min():.1f}, {raw.max():.1f}]")
    print(f"after train aug:  shape {tuple(out.shape)}, range [{out.min():.2f}, {out.max():.2f}], dtype {out.dtype}")

    train_set = ToySurveyCutouts(NUM_TRAIN, train_transform, seed=0)
    val_set = ToySurveyCutouts(NUM_VAL, val_transform, seed=1)

    # --- Model: the band count is just in_channels -----------------------------
    model = build_sequential_model(
        backbone=("resnet18", {"in_channels": len(BANDS)}),
        necks=["global_avg_pool"],
        head=("vanilla_classification_head", {"num_classes": len(CLASSES), "hidden_features": 128}),
    )
    task = Task.create("classification_task")
    train_loader = build_dataloader(train_set, task=task, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = build_dataloader(val_set, task=task, batch_size=BATCH_SIZE)

    trainer = Trainer(model, task, Optimizer.create("adamw", model, lr=1e-3), device="cpu")
    trainer.train(train_loader, max_iter=MAX_ITER)
    print(f"after {MAX_ITER} iterations: train loss {trainer.storage.latest()['total_loss']:.3f}, "
          + ", ".join(f"val_{k}={v:.3f}" for k, v in trainer.evaluate(val_loader).items()))


if __name__ == "__main__":
    main()
