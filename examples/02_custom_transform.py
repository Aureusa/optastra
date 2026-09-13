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


# 1. A plain dataclass holding the transform's tunable parameters.
@dataclass
class InvertConfig:
    probability: float = 0.5


# 2. The Transform itself -- operates on one `Sample` and returns it.
class Invert(Transform):
    """Randomly inverts pixel values (assumes a float image in [0, 1])."""

    def __init__(self, cfg: InvertConfig = InvertConfig()):
        self.cfg = cfg

    def __call__(self, sample: Sample) -> Sample:
        if torch.rand(()) < self.cfg.probability:
            sample.image = 1.0 - sample.image
        return sample


# 3. The registration entrypoint -- this is what `Transform.create(name)` calls.
@Transform.register(config=InvertConfig())
def invert(cfg: InvertConfig) -> Invert:
    return Invert(cfg)


def main() -> None:
    # Registered the moment this module is imported -- no separate
    # "plugin" list to update.
    assert "invert" in Transform.list_all()

    transform = Transform.create("invert", probability=1.0)
    sample = Sample(image=torch.zeros(3, 8, 8))
    out = transform(sample)

    print(f"registered transforms containing 'invert': {Transform.list_all(filter='invert')}")
    print(f"mean pixel value after inversion (expect 1.0): {out.image.mean().item()}")


if __name__ == "__main__":
    main()
