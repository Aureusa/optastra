import torch.nn as nn

from optastra.models import factory


class _DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channels = {"C5": 32}


class _DummyNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channels = {"P5": 64}


class _DummyHead(nn.Module):
    def __init__(self):
        super().__init__()


def test_create_helpers_call_registry_entrypoints_with_expected_kwargs(monkeypatch):
    calls = {"backbone": None, "neck": None, "head": None}

    def fake_get_backbone_entrypoint(name):
        assert name == "dummy_backbone"
        return lambda: calls.__setitem__("backbone", {}) or _DummyBackbone()

    def fake_get_neck_entrypoint(name):
        assert name == "dummy_neck"
        return lambda **kwargs: calls.__setitem__("neck", kwargs) or _DummyNeck()

    def fake_get_head_entrypoint(name):
        assert name == "dummy_head"
        return lambda **kwargs: calls.__setitem__("head", kwargs) or _DummyHead()

    monkeypatch.setattr(factory, "get_backbone_entrypoint", fake_get_backbone_entrypoint)
    monkeypatch.setattr(factory, "get_neck_entrypoint", fake_get_neck_entrypoint)
    monkeypatch.setattr(factory, "get_head_entrypoint", fake_get_head_entrypoint)

    model = factory.create_model("dummy_backbone", "dummy_neck", "dummy_head")

    assert isinstance(model, nn.Sequential)
    assert isinstance(model[0], _DummyBackbone)
    assert isinstance(model[1], _DummyNeck)
    assert isinstance(model[2], _DummyHead)
    assert calls["backbone"] == {}
    assert calls["neck"] == {"in_channels": {"C5": 32}}
    assert calls["head"] == {"in_features": 64, "hidden_features": 256, "stage": "P5"}
