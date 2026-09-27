"""Preset factories (`*_weak`, `*_strong`, `rand_augment_all_ops`): their
defaults live in the registered config, and user overrides win."""
import subprocess
import sys

import pytest
import torch

from optastra.data.sample import Sample
from optastra.transforms import BatchTransform, Transform
from optastra.transforms import functional as FN
from optastra.transforms.ops import ALL_OP_NAMES, PHOTOMETRIC_OPS

# (family, name, field, preset value, override value)
PRESETS = [
    (Transform, "auto_augment_weak", "magnitude_scale", 0.5, 0.9),
    (Transform, "auto_augment_strong", "magnitude_scale", 1.3, 2.0),
    (Transform, "trivial_augment_weak", "magnitude_max", 4.0, 9.0),
    (Transform, "trivial_augment_strong", "magnitude_min", 6.0, 1.0),
    (Transform, "augmix_weak", "num_chains", 2, 5),
    (Transform, "augmix_strong", "magnitude", 7.0, 2.0),
    (Transform, "rand_augment_all_ops", "ops", ALL_OP_NAMES, ("Identity",)),
    (BatchTransform, "mixup_weak", "alpha", 0.05, 0.4),
    (BatchTransform, "mixup_strong", "p", 0.8, 0.1),
    (BatchTransform, "cutmix_weak", "alpha", 0.5, 3.0),
    (BatchTransform, "cutmix_strong", "p", 0.8, 0.3),
]


@pytest.mark.parametrize("family,name,field,preset,override", PRESETS)
def test_preset_defaults_are_visible_and_overridable(family, name, field, preset, override):
    assert getattr(family.get_default_config(name), field) == preset   # what describe() shows
    assert getattr(family.create(name).cfg, field) == preset
    assert getattr(family.create(name, **{field: override}).cfg, field) == override


def test_trivial_augment_weak_actually_uses_overridden_magnitude():
    # magnitude 9 on Solarize -> threshold 1 - 0.9 = 0.1: everything >= 0.1 is inverted.
    # The old factory reset magnitude_max to 4 (threshold >= 0.6).
    img = torch.linspace(0, 1, 11).reshape(1, 1, 11)
    t = Transform.create("trivial_augment_weak", ops=("Solarize",), magnitude_min=9.0, magnitude_max=9.0)
    out = t(Sample(image=img.clone())).image
    assert torch.allclose(out, FN.solarize(img, 0.1))


def test_auto_augment_scale_override_reaches_the_policy():
    t = Transform.create("auto_augment_weak", magnitude_scale=1.0)
    assert t._policy[0] == [("Posterize", 0.4, 8.0), ("Rotate", 0.6, 9.0)]


def test_rand_augment_default_ops_are_an_ordered_tuple():
    ops = Transform.get_default_config("rand_augment").ops
    assert ops == tuple(PHOTOMETRIC_OPS)


_HASHSEED_SCRIPT = """
import torch
from optastra.data.sample import Sample
from optastra.transforms import Transform, seed_transforms
seed_transforms(0)
t = Transform.create("rand_augment", num_ops=2, magnitude=9)
img = torch.linspace(0, 1, 3 * 8 * 8).reshape(3, 8, 8)
print([round(float(t(Sample(image=img.clone())).image.sum()), 4) for _ in range(5)])
"""


def test_rand_augment_does_not_depend_on_pythonhashseed():
    # the op list used to come from a `set` of strings, whose order changes with PYTHONHASHSEED
    import os
    outputs = []
    for hashseed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        res = subprocess.run([sys.executable, "-c", _HASHSEED_SCRIPT], env=env,
                             capture_output=True, text=True, check=True)
        outputs.append(res.stdout.strip())
    assert outputs[0] == outputs[1]
