from optastra.architectures import Architecture
from optastra.architectures.faster_rcnn import FasterRCNN


def test_architecture_config_returns_faster_rcnn_defaults():
    cfg = Architecture.get_default_config("faster_rcnn_r50_fpn")

    assert cfg.backbone.name == "resnet50"
    assert cfg.neck.name == "fpn"
    assert cfg.num_classes == 91


def test_architecture_create_builds_registered_faster_rcnn_variant():
    model = Architecture.create("faster_rcnn_r50_c5", num_classes=12)

    assert isinstance(model, FasterRCNN)
    assert model.neck is None
    assert model.cfg.backbone.name == "resnet50"
    assert model.cfg.num_classes == 12
