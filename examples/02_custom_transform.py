"""Extending Optastra: register a brand-new Transform in ~15 lines.

This is the same three-piece pattern every family in Optastra follows
(config dataclass + component class + `<Base>.register` factory function).
See docs/extending.md for the full write-up.

Run with:
    python examples/02_custom_transform.py
"""
from dataclasses import dataclass

import torch

from optastra import Transform
from optastra.data.sample import Sample
from optastra.transforms import rng, seed_transforms
from optastra.transforms.functional import infer_value_range


# 1. A plain dataclass holding the transform's tunable parameters.
@dataclass
class InvertConfig:
    probability: float = 0.5
    value_range: tuple[float, float] | None = None   # None -> (0, 1) or the image's own (min, max)


# 2. The Transform itself -- operates on one `Sample` and returns it.
class Invert(Transform):
    """Randomly inverts pixel values: x -> lo + hi - x. Photometric only, so
    the target is untouched; works for any channel count and value range."""

    def __init__(self, cfg: InvertConfig = InvertConfig()):
        self.cfg = cfg

    def __call__(self, sample: Sample) -> Sample:
        # draw from the transforms' seedable generator, never `random` / torch's global RNG
        if rng.rand() < self.cfg.probability:
            lo, hi = infer_value_range(sample.image, self.cfg.value_range)
            sample.image = lo + hi - sample.image
        return sample


# 3. The registration entrypoint -- this is what `Transform.create(name)` calls.
@Transform.register(config=InvertConfig())
def invert(cfg: InvertConfig) -> Invert:
    return Invert(cfg)


def main() -> None:
    # Registered the moment this module is imported -- no separate
    # "plugin" list to update.
    assert "invert" in Transform.list_all()

    seed_transforms(0)   # reproducible augmentations
    transform = Transform.create("invert", probability=1.0)
    sample = Sample(image=torch.zeros(3, 8, 8))
    out = transform(sample)

    print(f"registered transforms containing 'invert': {Transform.list_all(filter='invert')}")
    print(f"mean pixel value after inversion (expect 1.0): {out.image.mean().item()}")

    # A 5-band float image in physical units: pass the known range explicitly.
    hdr = Transform.create("invert", probability=1.0, value_range=(-10.0, 1000.0))
    out = hdr(Sample(image=torch.full((5, 8, 8), 100.0)))
    print(f"5-band HDR pixel after inversion (expect 890.0): {out.image[0, 0, 0].item()}")


if __name__ == "__main__":
    main()
