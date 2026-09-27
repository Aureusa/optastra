import re
import warnings

import pytest
import torch

from optastra import Algorithm, Backbone, Optimizer, Trainer, build_sequential_model
from optastra.backbones import export_backbone, load_backbone_weights
from optastra.training import Checkpointer
from optastra.training.hooks import CheckpointHook

SMALL_MLP = {"hidden_dim": 32, "out_dim": 16}


def _assert_same_weights(a: torch.nn.Module, b: torch.nn.Module) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for key in sa:
        assert torch.equal(sa[key], sb[key]), key


def _pretrain_byol(output_dir) -> tuple[torch.nn.Module, str]:
    """2 BYOL steps on random data, checkpointed by the standard CheckpointHook."""
    torch.manual_seed(0)
    algo = Algorithm.create("byol", backbone="resnet18", projector=SMALL_MLP, predictor=SMALL_MLP)
    model = algo.build_model()
    trainer = Trainer(
        model, algo, Optimizer.create("sgd", model, lr=0.1),
        hooks=algo.hooks() + [CheckpointHook(str(output_dir), save_every=1)], device="cpu",
    )
    loader = [{"views": [torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)]} for _ in range(2)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.train(loader, max_iter=2)
    return model, Checkpointer(str(output_dir)).latest()


def test_byol_pretrain_export_and_finetune_carry_the_online_backbone(tmp_path):
    model, ckpt = _pretrain_byol(tmp_path)
    assert ckpt is not None and ckpt.endswith("ckpt_1.pt")

    # Freshly built backbones differ from the pretrained one...
    fresh = Backbone.create("resnet18")
    assert not all(torch.equal(v, model.online_backbone.state_dict()[k]) for k, v in fresh.state_dict().items())

    # ...export_backbone pulls out the *online* backbone, prefix-free...
    out = tmp_path / "byol_resnet18.pt"
    exported = export_backbone(ckpt, out)
    assert set(exported) == set(model.online_backbone.state_dict())

    # ...and it loads into a downstream classification model via a ComponentRef override.
    clf = build_sequential_model(
        backbone=("resnet18", {"weights": str(out)}),
        necks=["global_avg_pool"],
        head=("vanilla_classification_head", {"num_classes": 3}),
    )
    _assert_same_weights(clf[0], model.online_backbone)

    # The raw training checkpoint also works directly (prefix auto-detected).
    _assert_same_weights(Backbone.create("resnet18", weights=ckpt), model.online_backbone)
    # An explicit prefix picks a different sub-module.
    _assert_same_weights(
        Backbone.create("resnet18", weights=ckpt, weights_prefix="target_backbone."), model.target_backbone
    )


def test_prefixes_from_sequential_and_compiled_models_are_detected(tmp_path):
    torch.manual_seed(0)
    clf = build_sequential_model("resnet18", ["global_avg_pool"], ("vanilla_classification_head", {"num_classes": 3}))
    compiled_style = {f"_orig_mod.{k}": v for k, v in clf.state_dict().items()}   # what torch.compile saves
    torch.save({"model": compiled_style, "iter": 0}, tmp_path / "ckpt.pt")

    loaded = Backbone.create("resnet18", weights=str(tmp_path / "ckpt.pt"))
    _assert_same_weights(loaded, clf[0])

    exported = export_backbone(tmp_path / "ckpt.pt", tmp_path / "bb.pt")
    assert set(exported) == set(clf[0].state_dict())


def test_raw_state_dict_loads_without_prefix():
    torch.manual_seed(0)
    source = Backbone.create("resnet18")
    target = load_backbone_weights(Backbone.create("resnet18"), source.state_dict())
    _assert_same_weights(target, source)


def test_mismatched_weights_fail_strictly_with_missing_and_unexpected_keys():
    state = Backbone.create("resnet18").state_dict()
    missing_key = next(iter(state))
    state.pop(missing_key)
    state["extra_layer.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match=rf"missing keys \(1\): \['{re.escape(missing_key)}'\]\n"
                                         r"  unexpected keys \(1\): \['extra_layer.weight'\]"):
        Backbone.create("resnet18", weights=state)


def test_unknown_prefix_raises_a_helpful_error():
    state = {f"encoder.{k}": v for k, v in Backbone.create("resnet18").state_dict().items()}
    with pytest.raises(ValueError, match=r"top-level prefixes are \['encoder.'\]"):
        Backbone.create("resnet18", weights=state)
    # ...and works once the prefix is given.
    Backbone.create("resnet18", weights=state, weights_prefix="encoder.")
