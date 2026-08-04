from optastra.region_extractors import RegionExtractor
from optastra.nn.features import FeatureSpec


def test_region_extractor_config_returns_roi_align_defaults():
    cfg = RegionExtractor.config("roi_align")

    assert cfg.output_size == 7
    assert cfg.aligned is True


def test_region_extractor_create_builds_roi_align_and_sets_out_spec():
    model = RegionExtractor.create(
        "roi_align",
        in_spec=FeatureSpec(channels={"P2": 256}, strides={"P2": 4}),
        stage="P2",
        output_size=5,
    )

    assert model.out_spec.embed_dim == 256
    assert model.out_spec.num_tokens == 25
