"""MixUp / CutMix: configurable keys, other target entries preserved, correct values."""
import pytest
import torch

from optastra.transforms import BatchTransform, seed_transforms


def _batch(b=4, h=10, w=10):
    # image i is constant i, label i -> the partner of image i can be read back
    images = torch.arange(b, dtype=torch.float32).reshape(b, 1, 1, 1).expand(b, 3, h, w).clone()
    return {"x": images, "y": {"cls": torch.arange(b), "boxes": torch.ones(b, 4)}, "meta": "kept"}


@pytest.mark.parametrize("seed", range(3))
def test_mixup_values_and_custom_keys(seed):
    seed_transforms(seed)
    batch = _batch()
    images = batch["x"].clone()
    t = BatchTransform.create("mixup", p=1.0, inputs_key="x", targets_key="y", labels_key="cls")
    out = t(batch)
    y = out["y"]
    assert set(y) == {"cls", "cls_b", "lam", "boxes"}
    assert torch.equal(y["boxes"], torch.ones(4, 4)) and out["meta"] == "kept"
    lam, partner = y["lam"], y["cls_b"]
    expected = lam * images + (1 - lam) * images[partner]
    assert torch.allclose(out["x"], expected)


@pytest.mark.parametrize("seed", range(3))
def test_cutmix_patch_area_matches_lam(seed):
    seed_transforms(seed)
    batch = _batch(h=20, w=20)
    inputs = batch["x"]
    t = BatchTransform.create("cutmix", p=1.0, inputs_key="x", targets_key="y", labels_key="cls")
    out = t(batch)
    y = out["y"]
    assert torch.equal(y["boxes"], torch.ones(4, 4))
    assert torch.equal(inputs, _batch(h=20, w=20)["x"])   # caller's tensor not modified in place
    for i, j in enumerate(y["cls_b"].tolist()):
        pasted = (out["x"][i] == j).all(dim=0)
        if i != j:
            assert abs(pasted.float().mean().item() - (1 - y["lam"])) < 1e-6
        assert ((out["x"][i] == i).all(dim=0) | pasted).all()   # every pixel is from i or j


def test_default_keys_still_match_classification_task():
    seed_transforms(0)
    batch = {"inputs": torch.rand(2, 3, 4, 4), "targets": {"labels": torch.tensor([0, 1])}}
    out = BatchTransform.create("mixup", p=1.0)(batch)
    assert set(out["targets"]) == {"labels", "labels_b", "lam"}


def test_non_dict_targets_raise_clear_error():
    batch = {"inputs": torch.rand(2, 3, 4, 4), "targets": [{"labels": torch.tensor([0])}] * 2}
    with pytest.raises(TypeError, match="dict"):
        BatchTransform.create("cutmix", p=1.0)(batch)
