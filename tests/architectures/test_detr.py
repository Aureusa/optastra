import pytest
import torch

from optastra import ComponentRef, ExperimentConfig
from optastra.architectures import DETR, Architecture
from optastra.data.collate import ragged
from optastra.data.sample import Sample
from optastra.decoders import QuerySet
from optastra.decoders import keys as decoder_keys
from optastra.detection import keys
from optastra.tasks import Task


BOX_PRESETS = {"detr_r50": "detr_decoder", "conditional_detr_r50": "conditional_detr_decoder",
               "dab_detr_r50": "dab_detr_decoder"}
ALL_PRESETS = [*BOX_PRESETS, "point_detr_r18_fpn"]


def _tiny(name, decoder=None, num_layers=2, num_queries=10, decoder_overrides=None, **overrides):
    torch.manual_seed(0)
    decoder = decoder or BOX_PRESETS.get(name, "detr_decoder")
    dec = {"num_layers": num_layers, "num_queries": num_queries, "dim_feedforward": 64, "num_heads": 4,
           "dropout": 0.0, **(decoder_overrides or {})}
    enc = {"num_layers": 1, "dim_feedforward": 64, "num_heads": 4, "dropout": 0.0}
    kw = dict(backbone=("resnet18", {"layers": [1, 1, 1, 1]}), d_model=64, decoder=(decoder, dec))
    if name == "point_detr_r18_fpn":
        enc["in_features"] = ("P3", "P4", "P5")
        kw["neck"] = ("fpn", {"in_features": ("C3", "C4", "C5"), "out_channels": 32})
    else:
        kw["num_classes"] = 3
    kw["encoder"] = ("detr_encoder", enc)
    kw.update(overrides)
    return Architecture.create(name, **kw)


SIZES = [(64, 96), (50, 70)]


def _images():
    torch.manual_seed(1)
    return torch.randn(2, 3, 64, 96)


# --- construction --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_PRESETS)
def test_presets_are_registered_with_expected_parts(name):
    assert name in Architecture.list_all()
    cfg = Architecture.get_default_config(name)
    if name in BOX_PRESETS:
        assert cfg.decoder.name == BOX_PRESETS[name]
        assert cfg.head.name == "detr_box_head" and cfg.num_classes == 80 and cfg.neck is None
        assert cfg.backbone.name == "resnet50"
    else:
        assert cfg.head.name == "keypoint_head" and cfg.num_classes is None
        assert cfg.neck.name == "fpn" and cfg.neck.overrides["in_features"] == ("C3", "C4", "C5")
        assert cfg.encoder.overrides["in_features"] == ("P3", "P4", "P5")


@pytest.mark.parametrize("name", ALL_PRESETS)
def test_presets_build_with_tiny_settings(name):
    model = _tiny(name)
    assert isinstance(model, DETR)
    assert model.encoder.cfg.d_model == 64 and model.decoder.cfg.d_model == 64
    if name in BOX_PRESETS:
        assert model.encoder.feature_names == ["C5"]
        assert model.head.cfg.num_classes == 3
    else:
        assert model.neck.stage_names == ["C3", "C4", "C5"]
        assert model.encoder.feature_names == ["P3", "P4", "P5"] and model.encoder.level_embed is not None
        assert model.decoder.feature_names == ["P3", "P4", "P5"]


def test_d_model_conflict_is_an_error():
    with pytest.raises(ValueError, match="d_model"):
        _tiny("detr_r50", decoder_overrides={"d_model": 32})
    with pytest.raises(ValueError, match="d_model"):
        _tiny("detr_r50", encoder=("detr_encoder", {"d_model": 128, "num_heads": 4}))
    _tiny("detr_r50", decoder_overrides={"d_model": 64})          # agreeing values are fine


def test_without_encoder_the_decoder_reads_the_backbone_maps():
    model = _tiny("detr_r50", encoder=None, decoder_overrides={"in_features": ("C5",)})
    assert model.encoder is None
    out = model(_images(), SIZES)
    assert out.logits.shape == (2, 10, 4)


# --- forward contract ------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_PRESETS)
def test_forward_contract(name):
    model = _tiny(name, num_layers=3).eval()
    out = model(_images(), SIZES)
    if name in BOX_PRESETS:
        assert out.logits.shape == (2, 10, 4) and out.boxes.shape == (2, 10, 4)
    else:
        assert out.scores.shape == (2, 10) and out.extra[keys.POINTS].shape == (2, 10, 2)
    aux = out.extra[keys.AUX_OUTPUTS]
    assert len(aux) == 2                                           # num_layers - 1
    assert out.extra[keys.IMAGE_SIZES] == SIZES
    assert out.extra[keys.QUERY_MASK].shape == (2, 10) and out.extra[keys.QUERY_MASK].all()
    assert out.extra[keys.DECODER_TOKENS].shape == (2, 10, 64)
    has_refs = model.decoder.uses_reference_points
    assert (decoder_keys.REFERENCE_POINTS in out.extra) == has_refs
    # the head ran on every layer: the aux outputs are the head on the earlier layer tokens
    feats = model.backbone(_images())
    if model.neck is not None:
        feats = model.neck(feats)
    feats.extra[keys.IMAGE_SIZES] = SIZES
    decoded = model.decoder(model.encoder(feats))
    first = model.head(type(feats)(patch_tokens=decoded.extra[decoder_keys.LAYER_TOKENS][0], extra=decoded.extra))
    field = "logits" if name in BOX_PRESETS else "scores"
    torch.testing.assert_close(getattr(aux[0], field), getattr(first, field), rtol=1e-4, atol=1e-5)


def test_without_aux_loss_there_are_no_aux_outputs():
    out = _tiny("detr_r50", aux_loss=False).eval()(_images(), SIZES)
    assert out.extra[keys.AUX_OUTPUTS] == []


def test_default_image_sizes_fill_the_batch():
    model = _tiny("detr_r50").eval()
    assert model(_images()).extra[keys.IMAGE_SIZES] == [(64, 96), (64, 96)]


def test_external_queries_through_the_architecture():
    model = _tiny("dab_detr_r50").eval()
    q = QuerySet(reference_points=torch.tensor([[[0.2, 0.3]], [[0.7, 0.4]]]))
    out = model(_images(), SIZES, queries=q)
    assert out.logits.shape == (2, 11, 4)
    torch.testing.assert_close(out.extra[decoder_keys.REFERENCE_POINTS][:, 0], q.reference_points[:, 0])


@pytest.mark.parametrize("name", list(BOX_PRESETS))
def test_every_parameter_gets_a_gradient(name):
    model = _tiny(name).train()
    q = None
    if name == "conditional_detr_r50":
        model = _tiny(name, decoder_overrides={"external_queries": True}).train()
    if name != "detr_r50":
        q = QuerySet(content=torch.randn(2, 2, 64), reference_points=torch.rand(2, 2, 2) * 0.8 + 0.1)
    out = model(_images(), SIZES, queries=q)
    g = torch.Generator().manual_seed(0)
    loss = sum((o.logits * torch.randn(o.logits.shape, generator=g)).sum() + (o.boxes * 3).sum()
               for o in [out, *out.extra[keys.AUX_OUTPUTS]])
    loss.backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


def test_point_preset_every_parameter_gets_a_gradient():
    """FPN on C3-C5 and an encoder over all of P3-P5: no parameter is left without a gradient (DDP)."""
    model = _tiny("point_detr_r18_fpn").train()
    out = model(_images(), SIZES)
    loss = sum((o.extra[keys.POINTS] * 3).sum() + o.scores.sum() for o in [out, *out.extra[keys.AUX_OUTPUTS]])
    loss.backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


# --- tasks ---------------------------------------------------------------------------------------

def _detection_batch():
    g = torch.Generator().manual_seed(2)
    samples = [
        Sample(image=torch.randn(3, 64, 96, generator=g), target={
            "boxes": torch.tensor([[10.0, 12.0, 40.0, 50.0], [50.0, 5.0, 90.0, 30.0]]), "labels": torch.tensor([0, 2])}),
        Sample(image=torch.randn(3, 50, 70, generator=g), target={
            "boxes": torch.tensor([[5.0, 5.0, 30.0, 45.0]]), "labels": torch.tensor([1])}),
    ]
    return ragged(samples)


@pytest.mark.parametrize("name", list(BOX_PRESETS))
def test_detection_task_train_steps(name):
    model = _tiny(name)
    task = Task.create("detection_task", num_classes=3, criterion="detr_criterion", postprocessor="detr_postprocessor")
    batch = _detection_batch()
    assert batch["image_sizes"] == SIZES and batch["inputs"].shape == (2, 3, 64, 96)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for _ in range(2):
        out = task.run_step(model, batch, stage="train")
        assert torch.isfinite(out.loss)
        assert {"loss_ce", "loss_bbox", "loss_giou", "loss_ce_aux0"} <= set(out.losses)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    assert not torch.equal(before["decoder.layers.0.cross_attn.proj.weight"],
                           model.decoder.layers[0].cross_attn.proj.weight)

    model.eval()
    val = task.run_step(model, batch, stage="val")
    assert "matched_class_accuracy" in val.metrics
    assert len(val.predictions) == 2
    for sample, (h, w) in zip(val.predictions, SIZES):
        boxes = sample.target["boxes"]
        assert boxes.shape[1] == 4 and len(boxes) <= 10
        assert (boxes[:, [0, 2]] <= w).all() and (boxes[:, [1, 3]] <= h).all() and (boxes >= 0).all()


def test_point_detection_task_end_to_end():
    assert "point_detection_task" in Task.list_all()
    task = Task.create("point_detection_task")
    assert task.collate == "ragged"
    model = _tiny("point_detr_r18_fpn")
    g = torch.Generator().manual_seed(3)
    batch = ragged([
        Sample(image=torch.randn(3, 64, 96, generator=g), target={"points": torch.tensor([[10.0, 20.0], [80.0, 40.0]])}),
        Sample(image=torch.randn(3, 50, 70, generator=g), target={"points": [[30.0, 25.0]]}),
    ])
    out = task.run_step(model, batch, stage="train")
    assert {"loss_loc", "loss_objectness", "loss_loc_aux0", "loss_objectness_aux0"} == set(out.losses)
    assert torch.isfinite(out.loss)
    out.loss.backward()

    model.eval()
    val = task.run_step(model, batch, stage="val")
    assert "loc_error_px" in val.metrics
    pred = task.run_step(model, {"inputs": batch["inputs"], "image_sizes": batch["image_sizes"]}, stage="predict")
    assert pred.loss is None and len(pred.predictions) == 2
    for sample in pred.predictions:
        assert sample.target["points"].shape[1] == 2
    with pytest.raises(ValueError, match="points"):
        task.preprocess_targets([{"boxes": torch.zeros(1, 4)}])
    with pytest.raises(ValueError, match="targets"):
        task.validate_batch({"inputs": batch["inputs"]}, stage="train")


# --- overfitting -------------------------------------------------------------------------------------

def _dots_batch():
    """Two 64 x 64 images, each with three bright Gaussian dots on faint noise."""
    g = torch.Generator().manual_seed(0)
    yy, xx = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    samples = []
    for _ in range(2):
        pts = torch.rand(3, 2, generator=g) * 48 + 8
        img = 0.05 * torch.randn(64, 64, generator=g)
        for x, y in pts:
            img = img + torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5 ** 2))
        samples.append(Sample(image=img.expand(3, 64, 64).clone(), target={"points": pts}))
    return ragged(samples)


def test_point_detr_overfits_one_batch():
    torch.manual_seed(0)
    model = _tiny("point_detr_r18_fpn", num_queries=6)
    task = Task.create("point_detection_task")
    batch = _dots_batch()
    # lr tuned on CPU: across init seeds the ratio after 50 steps is 0.07-0.18
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    first = None
    for _ in range(50):
        out = task.run_step(model, batch, stage="train")
        first = first if first is not None else out.losses["loss_loc"].item()
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    last = task.run_step(model.eval(), batch, stage="val")
    assert last.losses["loss_loc"].item() < 0.3 * first


# --- config -------------------------------------------------------------------------------------------

def test_experiment_config_yaml_round_trip_for_detr(tmp_path):
    cfg = ExperimentConfig(
        architecture=ComponentRef("point_detr_r18_fpn", {
            "backbone": ComponentRef("resnet18", {"layers": [1, 1, 1, 1]}),
            "neck": ComponentRef("fpn", {"in_features": ["C3", "C4"], "out_channels": 32}),
            "encoder": ComponentRef("detr_encoder", {"in_features": ["P3", "P4"], "num_layers": 1, "num_heads": 4,
                                                     "dim_feedforward": 64}),
            "decoder": ComponentRef("dab_detr_decoder", {"num_layers": 2, "num_queries": 10, "num_heads": 4,
                                                         "dim_feedforward": 64}),
            "head": ComponentRef("keypoint_head", {"predict_uncertainty": True}),
            "d_model": 64,
        }),
        task=ComponentRef("point_detection_task", {
            "criterion": ComponentRef("point_set_criterion", {
                "loc_loss": "laplace_nll", "loc_scale": 2.0,
                "matcher": ComponentRef("hungarian_matcher", {"point_scale": 3.0}),
            }),
        }),
        optimizer=ComponentRef("adamw", {"lr": 1e-4}),
        seed=1,
    )
    path = tmp_path / "detr.yaml"
    cfg.to_yaml(str(path))
    loaded = ExperimentConfig.from_yaml(str(path))
    assert loaded == cfg
    arch = loaded.architecture.overrides
    assert isinstance(arch["decoder"], ComponentRef) and arch["decoder"].name == "dab_detr_decoder"
    matcher = loaded.task.overrides["criterion"].overrides["matcher"]
    assert isinstance(matcher, ComponentRef) and matcher.overrides == {"point_scale": 3.0}

    # and it builds what it describes
    model = Architecture.create(loaded.architecture.name, **arch)
    assert model.decoder.cfg.query_type == "dab" and model.head.cfg.predict_uncertainty
    task = Task.create(loaded.task.name, **loaded.task.overrides)
    assert task.criterion.cfg.loc_loss == "laplace_nll" and task.criterion.matcher.cfg.point_scale == 3.0
