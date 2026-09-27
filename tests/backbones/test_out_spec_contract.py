"""
Every registered backbone (and backbone + FPN) must produce exactly what its
out_spec declares: same feature names, channel counts and strides.

The stride is *measured* as (change in input size) / (change in output size)
between two input sizes. This is the true sampling interval of the output
grid, and it is independent of border effects from unpadded convs/pools
(e.g. AlexNet maps 224 -> 6, but 256 -> 7 and 512 -> 15: stride 32).
"""
import pytest
import torch

from optastra.backbones import Backbone
from optastra.necks import Neck

SIZES = (64, 128)


def _forward_at_sizes(module, sizes=SIZES):
    outputs = []
    with torch.no_grad():
        for size in sizes:
            outputs.append(module(torch.randn(1, 3, size, size)).feature_maps)
    return outputs


def _assert_matches_spec(spec, small, large, sizes=SIZES):
    assert set(small) == set(spec.channels) == set(spec.strides)
    for name in spec.channels:
        assert small[name].shape[1] == spec.channels[name], name
        measured_stride = (sizes[1] - sizes[0]) / (large[name].shape[-1] - small[name].shape[-1])
        assert measured_stride == spec.strides[name], name


@pytest.mark.parametrize("name", Backbone.list_all())
def test_backbone_and_fpn_outputs_match_out_spec(name):
    torch.manual_seed(0)
    backbone = Backbone.create(name).eval()
    small, large = _forward_at_sizes(backbone)
    _assert_matches_spec(backbone.out_spec, small, large)

    fpn = Neck.create("fpn", backbone.out_spec, out_channels=32).eval()
    with torch.no_grad():
        fpn_small = fpn(backbone(torch.randn(1, 3, SIZES[0], SIZES[0]))).feature_maps
        fpn_large = fpn(backbone(torch.randn(1, 3, SIZES[1], SIZES[1]))).feature_maps
    _assert_matches_spec(fpn.out_spec, fpn_small, fpn_large)
