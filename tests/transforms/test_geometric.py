"""Target-correct geometry: boxes and masks must move exactly like the image.

Box convention: XYXY absolute pixels, pixel i covers [i, i+1).
"""
import pytest
import torch

from optastra.data.sample import Sample
from optastra.transforms import Transform, seed_transforms


def _block_mask(h, w, box, dtype=torch.uint8):
    """(H, W) mask that is 1 exactly inside the integer box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    m = torch.zeros(h, w, dtype=dtype)
    m[y1:y2, x1:x2] = 1
    return m


def _mask_bbox(mask):
    ys, xs = torch.nonzero(mask, as_tuple=True)
    return [xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1]


def _sample(h, w, boxes, c=3):
    """Image whose channel 0 equals the first instance mask (so the image's
    geometry can be compared to the mask's), plus boxes/labels/masks."""
    masks = torch.stack([_block_mask(h, w, b) for b in boxes])
    image = torch.zeros(c, h, w)
    image[0] = masks[0].float()
    return Sample(
        image=image,
        target={
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.arange(len(boxes)),
            "masks": masks,
        },
    )


def _assert_binary(masks):
    assert masks.dtype == torch.uint8
    assert set(masks.unique().tolist()) <= {0, 1}


# ---- deterministic transforms, hand-computed boxes --------------------------

def test_resize_scales_boxes_by_old_size():
    # H=100, W=200 -> 50x50: x scale 0.25, y scale 0.5
    s = _sample(100, 200, [(20, 10, 60, 50)])
    out = Transform.create("resize", size=50)(s)
    assert out.image.shape == (3, 50, 50)
    assert out.target["boxes"].tolist() == [[5.0, 5.0, 15.0, 25.0]]
    _assert_binary(out.target["masks"])
    assert _mask_bbox(out.target["masks"][0]) == [5, 5, 15, 25]


def test_center_crop_shifts_clips_and_drops_boxes():
    # 100x120 -> 60: top = 20, left = 30
    s = _sample(100, 120, [(40, 30, 100, 70), (0, 0, 20, 10), (50, 40, 60, 50)])
    out = Transform.create("crop", size=60)(s)
    assert out.image.shape == (3, 60, 60)
    # box 0 clipped on the right; box 1 lies entirely outside -> dropped with its label and mask
    assert out.target["boxes"].tolist() == [[10.0, 10.0, 60.0, 50.0], [20.0, 20.0, 30.0, 30.0]]
    assert out.target["labels"].tolist() == [0, 2]
    assert out.target["masks"].shape == (2, 60, 60)
    assert _mask_bbox(out.target["masks"][0]) == [10, 10, 60, 50]
    assert _mask_bbox(out.target["masks"][1]) == [20, 20, 30, 30]
    assert torch.equal(out.image[0].to(torch.uint8), out.target["masks"][0])


def test_hflip_and_vflip_move_boxes_and_masks():
    s = _sample(50, 100, [(10, 20, 30, 40)])
    out = Transform.create("random_hflip", p=1.0)(s)
    assert out.target["boxes"].tolist() == [[70.0, 20.0, 90.0, 40.0]]
    assert _mask_bbox(out.target["masks"][0]) == [70, 20, 90, 40]

    s = _sample(50, 100, [(10, 20, 30, 40)])
    out = Transform.create("random_vflip", p=1.0)(s)
    assert out.target["boxes"].tolist() == [[10.0, 10.0, 30.0, 30.0]]
    assert _mask_bbox(out.target["masks"][0]) == [10, 10, 30, 30]


def test_hflip_supports_uint16_images():
    s = Sample(image=torch.tensor([[[0, 1, 2, 3]]]).to(torch.uint16))
    out = Transform.create("random_hflip", p=1.0)(s)
    assert out.image.dtype == torch.uint16
    assert out.image.to(torch.int32).flatten().tolist() == [3, 2, 1, 0]


def test_random_crop_moves_boxes_with_the_window():
    # encode coordinates in the image so the chosen window can be read back
    h, w = 80, 90
    s = _sample(h, w, [(30, 20, 70, 60)])
    s.image[1] = torch.arange(h, dtype=torch.float32)[:, None].expand(h, w)
    s.image[2] = torch.arange(w, dtype=torch.float32)[None, :].expand(h, w)
    seed_transforms(3)
    out = Transform.create("random_crop", size=40)(s)
    top, left = int(out.image[1, 0, 0]), int(out.image[2, 0, 0])
    expected = torch.tensor([[30.0 - left, 20.0 - top, 70.0 - left, 60.0 - top]]).clamp(0, 40)
    assert torch.equal(out.target["boxes"], expected)
    assert _mask_bbox(out.target["masks"][0]) == expected[0].int().tolist()


def test_random_resized_crop_crops_then_scales_boxes(monkeypatch):
    from optastra.transforms.geometric import RandomResizedCrop
    s = _sample(100, 100, [(30, 20, 60, 40)])
    t = Transform.create("random_resized_crop", size=40)
    # window: top=10, left=20, h=40, w=80 -> x scale 0.5, y scale 1.0
    monkeypatch.setattr(RandomResizedCrop, "_sample_crop_box", lambda self, H, W: (10, 20, 40, 80))
    out = t(s)
    assert out.image.shape == (3, 40, 40)
    assert out.target["boxes"].tolist() == [[5.0, 10.0, 20.0, 30.0]]
    _assert_binary(out.target["masks"])
    assert _mask_bbox(out.target["masks"][0]) == [5, 10, 20, 30]


def test_random_rotation_90_degrees_rotates_boxes_and_masks():
    # 90 deg counter-clockwise about the centre (32, 32): (dx, dy) -> (dy, -dx)
    # x in [10, 30] -> dx in [-22, -2];  y in [20, 24] -> dy in [-12, -8]
    # -> x' = 32 + dy in [20, 24],  y' = 32 - dx in [34, 54]
    s = _sample(64, 64, [(10, 20, 30, 24)])
    out = Transform.create("random_rotation", degrees=(90.0, 90.0))(s)
    assert torch.allclose(out.target["boxes"], torch.tensor([[20.0, 34.0, 24.0, 54.0]]), atol=1e-4)
    _assert_binary(out.target["masks"])
    assert _mask_bbox(out.target["masks"][0]) == [20, 34, 24, 54]
    # the image rotated the same way as the mask
    assert torch.equal((out.image[0] > 0.5).to(torch.uint8), out.target["masks"][0])


def test_random_rotation_drops_boxes_rotated_out_of_the_image():
    # a corner box rotated by 45 deg ends up entirely left of x=0
    s = _sample(64, 64, [(0, 0, 4, 4), (28, 28, 36, 36)])
    out = Transform.create("random_rotation", degrees=(45.0, 45.0))(s)
    assert out.target["labels"].tolist() == [1]
    assert out.target["masks"].shape[0] == 1
    # the centred square's bounding box grows to side 8 * sqrt(2)
    half = 4 * 2 ** 0.5
    assert torch.allclose(out.target["boxes"], torch.tensor([[32 - half, 32 - half, 32 + half, 32 + half]]), atol=1e-4)


def test_semantic_mask_and_empty_targets_are_supported():
    s = Sample(
        image=torch.rand(3, 20, 30),
        target={
            "mask": torch.randint(0, 5, (20, 30)),        # (H, W) semantic map with class ids
            "boxes": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.long),
            "masks": torch.zeros(0, 20, 30, dtype=torch.uint8),
        },
    )
    semantic = s.target["mask"].clone()
    out = Transform.create("resize", size=10)(s)
    assert out.target["mask"].shape == (10, 10)
    assert out.target["mask"].dtype == semantic.dtype
    assert set(out.target["mask"].unique().tolist()) <= set(semantic.unique().tolist())
    assert out.target["masks"].shape == (0, 10, 10)
    assert out.target["boxes"].shape == (0, 4)


# ---- geometric ops inside augmentation policies ------------------------------

@pytest.mark.parametrize("op", ["Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY"])
@pytest.mark.parametrize("seed", [0, 1])
def test_policy_geometric_ops_keep_boxes_on_masks(op, seed):
    # the box of a warped rectangle must match the warped (nearest) mask to within a pixel
    seed_transforms(seed)
    s = _sample(64, 80, [(30, 20, 50, 44)], c=5)
    out = Transform.create("trivial_augment", ops=(op,), magnitude_min=7.0, magnitude_max=7.0)(s)
    box = out.target["boxes"][0]
    mask_box = torch.tensor(_mask_bbox(out.target["masks"][0]), dtype=torch.float32)
    _assert_binary(out.target["masks"])
    assert torch.allclose(box, mask_box, atol=1.5), (box, mask_box)
    assert torch.equal((out.image[0] > 0.5).to(torch.uint8), out.target["masks"][0])


def test_translate_moves_box_by_exact_shift():
    seed_transforms(0)
    s = _sample(50, 100, [(40, 10, 50, 20)])
    out = Transform.create("trivial_augment", ops=("TranslateX",), magnitude_min=10.0, magnitude_max=10.0)(s)
    # shift = +/- int(1.0 * 100 * 0.3) = +/- 30
    assert out.target["boxes"].tolist() in ([[70.0, 10.0, 80.0, 20.0]], [[10.0, 10.0, 20.0, 20.0]])
    assert _mask_bbox(out.target["masks"][0]) == out.target["boxes"][0].int().tolist()


# ---- end to end ------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_multiband_hdr_sample_survives_full_pipeline(seed):
    from optastra.transforms import Compose
    seed_transforms(seed)
    s = _sample(96, 128, [(10, 10, 40, 30), (50, 40, 90, 80), (100, 60, 120, 90)], c=5)
    s.image = torch.rand(5, 96, 128, generator=torch.Generator().manual_seed(seed)) * 1010 - 10   # HDR in [-10, 1000]
    pipeline = Compose([
        Transform.create("to_float"),
        Transform.create("random_resized_crop", size=64, scale=(0.5, 1.0)),
        Transform.create("random_hflip"),
        Transform.create("random_vflip"),
        Transform.create("random_rotation", degrees=(-20.0, 20.0)),
        Transform.create("rand_augment_all_ops"),
        Transform.create("trivial_augment"),
        Transform.create("auto_augment"),
        Transform.create("color_jitter"),
        Transform.create("gaussian_blur"),
        Transform.create("augmix"),
    ])
    out = pipeline(s)
    t = out.target
    assert out.image.shape == (5, 64, 64) and out.image.dtype == torch.float32
    assert out.image.max() > 1.0
    n = t["boxes"].shape[0]
    assert t["labels"].shape == (n,) and t["masks"].shape == (n, 64, 64)
    _assert_binary(t["masks"])
    assert (t["boxes"] >= 0).all() and (t["boxes"] <= 64).all()
    assert (t["boxes"][:, 2:] > t["boxes"][:, :2]).all()
    for box, mask in zip(t["boxes"], t["masks"]):
        if mask.any():
            # every mask pixel lies inside its box (boxes get looser after rotations). Each
            # nearest-neighbour warp may push a mask up to half a pixel past the exact shape,
            # and this pipeline chains several, hence the 2 px tolerance.
            mx1, my1, mx2, my2 = _mask_bbox(mask)
            assert mx1 >= box[0] - 2 and my1 >= box[1] - 2 and mx2 <= box[2] + 2 and my2 <= box[3] + 2
