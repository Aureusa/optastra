"""Photometric ops: dtype/channel/range-agnostic, and identical to torchvision
on standard [0, 1] RGB images."""
import pytest
import torch
import torchvision.transforms.functional as TF

from optastra.data.sample import Sample
from optastra.transforms import Transform, seed_transforms
from optastra.transforms import functional as FN
from optastra.transforms.pixmix import PixMix, PixMixConfig


# ---- parity with torchvision on standard images ----------------------------

@pytest.fixture
def rgb_uint8():
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (3, 16, 20), dtype=torch.uint8, generator=g)


def test_posterize_equalize_match_torchvision_uint8(rgb_uint8):
    x = rgb_uint8.float() / 255
    for bits in (1, 3, 6, 8):
        expected = TF.posterize(rgb_uint8, bits).float() / 255
        assert torch.allclose(FN.posterize(x, bits), expected, atol=1e-6)
    assert torch.allclose(FN.equalize(x), TF.equalize(rgb_uint8).float() / 255, atol=1e-6)


def test_blend_ops_match_torchvision_float(rgb_uint8):
    x = rgb_uint8.float() / 255
    for factor in (0.3, 1.7):
        assert torch.allclose(FN.adjust_brightness(x, factor), TF.adjust_brightness(x, factor), atol=1e-6)
        assert torch.allclose(FN.adjust_contrast(x, factor), TF.adjust_contrast(x, factor), atol=1e-5)
        assert torch.allclose(FN.adjust_saturation(x, factor), TF.adjust_saturation(x, factor), atol=1e-5)
        assert torch.allclose(FN.adjust_sharpness(x, factor), TF.adjust_sharpness(x, factor), atol=1e-5)
    assert torch.allclose(FN.adjust_hue(x, 0.2), TF.adjust_hue(x, 0.2), atol=1e-5)
    assert torch.allclose(FN.grayscale(x), TF.rgb_to_grayscale(x, num_output_channels=3), atol=1e-6)


# ---- value-range semantics -------------------------------------------------

def test_infer_value_range():
    assert FN.infer_value_range(torch.tensor([0.2, 0.5])) == (0.0, 1.0)
    assert FN.infer_value_range(torch.tensor([-10.0, 1000.0])) == (-10.0, 1000.0)
    assert FN.infer_value_range(torch.tensor([0.2, 0.5]), (0, 255)) == (0.0, 255.0)


def test_solarize_threshold_is_not_clamped_to_image_max():
    # regression: the threshold used to be clamped to max(img) - 1e-6, which
    # inverted the brightest pixel even when the threshold was above it
    img = torch.tensor([[[0.1, 0.3, 0.5]]])
    assert torch.equal(FN.solarize(img, 0.8), img)
    assert torch.allclose(FN.solarize(img, 0.3), torch.tensor([[[0.1, 0.7, 0.5]]]))


def test_solarize_in_image_units():
    img = torch.tensor([[[-10.0, 200.0, 600.0]]])
    out = FN.solarize(img, 500.0, value_range=(-10.0, 1000.0))
    assert out.tolist() == [[[-10.0, 200.0, 390.0]]]   # -10 + 1000 - 600

    # the Solarize transform's threshold is a fraction of the value range
    s = Sample(image=img.clone())
    out = Transform.create("solarize", p=1.0, threshold=0.5, value_range=(-10.0, 1000.0))(s).image
    assert out.tolist() == [[[-10.0, 200.0, 390.0]]]   # threshold -10 + 0.5 * 1010 = 495


def test_brightness_clamps_to_value_range_not_unit_interval():
    img = torch.tensor([[[-10.0, 100.0, 800.0]]])
    out = FN.adjust_brightness(img, 1.5, value_range=(-10.0, 1000.0))
    assert out.tolist() == [[[-10.0, 150.0, 1000.0]]]


def test_autocontrast_stretches_each_channel_to_the_range():
    img = torch.tensor([[[2.0, 4.0, 6.0]], [[3.0, 3.0, 3.0]]])
    out = FN.autocontrast(img, value_range=(0.0, 10.0))
    assert out.tolist() == [[[0.0, 5.0, 10.0]], [[3.0, 3.0, 3.0]]]   # constant channel untouched


def test_posterize_quantizes_over_value_range():
    img = torch.tensor([[[-10.0, 490.0, 1000.0]]])
    out = FN.posterize(img, 1, value_range=(-10.0, 1000.0))   # 1 bit -> levels 0 and 128 of 256
    assert torch.allclose(out, torch.tensor([[[-10.0, -10.0, -10.0 + 128 / 255 * 1010]]]))


def test_multiband_saturation_and_grayscale_use_band_mean():
    img = torch.arange(5.0).reshape(5, 1, 1).expand(5, 2, 2)
    assert torch.allclose(FN.adjust_saturation(img, 0.0, (0.0, 10.0)), torch.full((5, 2, 2), 2.0))
    assert torch.allclose(FN.grayscale(img), torch.full((5, 2, 2), 2.0))


def test_hue_rejects_non_rgb():
    with pytest.raises(ValueError, match="3-channel"):
        FN.adjust_hue(torch.rand(5, 4, 4), 0.1)


# ---- ToFloat -----------------------------------------------------------------

def test_to_float_is_dtype_aware():
    def run(img, **kw):
        return Transform.create("to_float", **kw)(Sample(image=img)).image

    assert run(torch.tensor([0, 255], dtype=torch.uint8)).tolist() == [0.0, 1.0]
    assert run(torch.tensor([0, 65535], dtype=torch.uint16)).tolist() == [0.0, 1.0]
    assert run(torch.tensor([-5, 300], dtype=torch.int16)).tolist() == [-5.0, 300.0]
    assert run(torch.tensor([-5, 300], dtype=torch.int32), signed_scale=10.0).tolist() == [-0.5, 30.0]
    assert run(torch.tensor([7, 255], dtype=torch.uint8), scale=False).tolist() == [7.0, 255.0]
    hdr = torch.tensor([-10.0, 1000.0])
    assert run(hdr.clone()).tolist() == [-10.0, 1000.0]


# ---- every photometric transform on non-standard images ---------------------

PHOTOMETRIC = [
    ("rand_augment", {}),
    ("rand_augment_all_ops", {}),
    ("auto_augment", {}),
    ("trivial_augment", {}),
    ("augmix", {}),
    ("pixmix", {}),
    ("color_jitter", {"p": 1.0}),
    ("gaussian_blur", {"p": 1.0}),
    ("solarize", {"p": 1.0}),
    ("random_grayscale", {"p": 1.0}),
]


def _hdr_image(channels=5, h=24, w=32):
    g = torch.Generator().manual_seed(1)
    img = torch.rand(channels, h, w, generator=g) * 1010 - 10     # values in [-10, 1000]
    img[0, 0, 0], img[0, 0, 1] = -10.0, 1000.0
    return img


@pytest.mark.parametrize("name,kwargs", PHOTOMETRIC)
@pytest.mark.parametrize("seed", range(3))
def test_hdr_multiband_not_clipped_to_unit_range(name, kwargs, seed):
    seed_transforms(seed)
    img = _hdr_image()
    out = Transform.create(name, **kwargs)(Sample(image=img.clone())).image
    assert out.dtype == torch.float32
    assert out.shape == img.shape
    assert torch.isfinite(out).all()
    assert out.max() > 1.0          # not squashed / clipped into [0, 1]
    assert out.min() >= -10.0 - 1e-3 and out.max() <= 1000.0 + 1e-3


# gaussian_blur / random_grayscale need no value range: their output is a convex combination of the input
@pytest.mark.parametrize("name,kwargs", [p for p in PHOTOMETRIC if p[0] not in ("gaussian_blur", "random_grayscale")])
def test_explicit_value_range_is_respected(name, kwargs):
    seed_transforms(0)
    img = _hdr_image()
    out = Transform.create(name, value_range=(-10.0, 1000.0), **kwargs)(Sample(image=img)).image
    assert out.min() >= -10.0 - 1e-3 and out.max() <= 1000.0 + 1e-3


@pytest.mark.parametrize("name,kwargs", PHOTOMETRIC)
@pytest.mark.parametrize("image", [
    torch.rand(1, 24, 32),                                                  # single band
    torch.randint(0, 65536, (1, 24, 32), dtype=torch.int32).to(torch.uint16),
    torch.randint(0, 65536, (3, 24, 32), dtype=torch.int32).to(torch.uint16),
], ids=["1ch-float", "1ch-uint16", "3ch-uint16"])
def test_single_band_and_uint16_inputs(name, kwargs, image):
    seed_transforms(0)
    out = Transform.create(name, **kwargs)(Sample(image=image.clone())).image
    assert out.dtype == torch.float32
    assert out.shape == image.shape
    assert out.min() >= 0.0 and out.max() <= 1.0   # uint16 -> [0, 1] via to_float_image


def test_color_jitter_skips_hue_for_multiband_but_changes_image():
    seed_transforms(0)
    img = _hdr_image()
    out = Transform.create("color_jitter", p=1.0)(Sample(image=img.clone())).image
    assert not torch.allclose(out, img)


# ---- PixMix mixing images -----------------------------------------------------

def test_pixmix_non_square_multiband_and_mixing_set_checks():
    seed_transforms(0)
    img = _hdr_image(channels=5, h=20, w=36)
    cfg = PixMixConfig(num_mixing_rounds=4)
    # 1-channel mixing images of another size are broadcast and resized
    out = PixMix(cfg, mixing_set=[torch.rand(1, 8, 8)])(Sample(image=img.clone())).image
    assert out.shape == img.shape and out.max() > 1.0

    wrong = PixMix(PixMixConfig(num_mixing_rounds=20), mixing_set=[torch.rand(3, 8, 8)])
    with pytest.raises(ValueError, match="channels"):
        wrong(Sample(image=img.clone()))


def test_unknown_op_name_fails_at_construction():
    with pytest.raises(ValueError, match="Unknown op"):
        Transform.create("rand_augment", ops=("Identity", "Solarise"))
    with pytest.raises(ValueError, match="Unknown op"):
        Transform.create("augmix", ops=("Rotate",))   # AugMix is photometric-only
