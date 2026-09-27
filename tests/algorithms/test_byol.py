import warnings

import pytest
import torch

from optastra import Algorithm, Optimizer, Trainer
from optastra.algorithms import AlgorithmHook, BYOLModel, BYOLTask
from optastra.training import Checkpointer

SMALL_MLP = {"hidden_dim": 32, "out_dim": 16}


def _byol(**overrides) -> tuple[BYOLTask, BYOLModel]:
    algo = Algorithm.create("byol", backbone="resnet18", projector=SMALL_MLP, predictor=SMALL_MLP, **overrides)
    return algo, algo.build_model()


def _loader(num_batches: int, seed: int = 0) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    return [{"views": [torch.randn(4, 3, 32, 32, generator=g) for _ in range(2)]} for _ in range(num_batches)]


def _train(trainer: Trainer, loader, max_iter: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # Task.run_step's CUDA autocast warns on CPU
        trainer.train(loader, max_iter=max_iter)


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def test_byol_builds_model_from_config():
    algo, model = _byol()
    assert isinstance(model, BYOLModel)
    assert type(model.online_backbone).__name__ == "ResNet"
    assert type(model.online_neck).__name__ == "GlobalPool"
    assert model.online_projector.mlp[0].in_features == 512
    assert model.predictor.mlp[0].in_features == 16
    assert all(not p.requires_grad for p in model._target_parameters())
    # target starts as an exact copy of the online encoder
    online, target = _snapshot(model.online_backbone), _snapshot(model.target_backbone)
    assert all(torch.equal(online[k], target[k]) for k in online)


def test_byol_swaps_backbone_by_config_only():
    algo = Algorithm.create(
        "byol", backbone=("vit_tiny", {"img_size": 32, "patch_size": 8, "depth": 1}),
        projector=SMALL_MLP, predictor=SMALL_MLP,
    )
    model = algo.build_model()
    assert type(model.target_backbone).__name__ == "ViT"
    assert model.online_neck is None and model.target_neck is None
    out = algo.run_step(model, {"views": [torch.randn(2, 3, 32, 32)] * 2}, stage="train")
    assert torch.isfinite(out.loss)


def test_byol_loss_is_symmetric_negative_cosine():
    algo = BYOLTask()
    a, b = torch.randn(3, 8), torch.randn(3, 8)
    preds = {"online_z1": a, "online_z2": b, "target_z1": a * 2.0, "target_z2": b}
    expected = -0.5 * (torch.cosine_similarity(a, b, dim=-1) + torch.cosine_similarity(b, a, dim=-1)).mean()
    assert algo.compute_losses(preds, None)["byol_loss"].item() == pytest.approx(expected.item(), rel=1e-6)


def test_momentum_schedule_is_cosine_from_base_to_final():
    algo = BYOLTask()   # base 0.996 -> final 1.0
    assert algo.momentum_at(0, 100) == pytest.approx(0.996)
    assert algo.momentum_at(50, 100) == pytest.approx(0.998)
    assert algo.momentum_at(100, 100) == pytest.approx(1.0)
    assert algo.momentum_at(25, 100) == pytest.approx(1 - 0.004 * (1 + 2 ** -0.5) / 2)
    assert algo.momentum_at(500, 100) == pytest.approx(1.0)   # clamped past the end
    assert algo.momentum_at(7, None) == pytest.approx(0.996)  # no horizon -> constant


def test_target_moves_by_exactly_the_ema_formula_after_a_step():
    torch.manual_seed(0)
    algo, model = _byol(base_momentum=0.9, final_momentum=0.9)
    optimizer = Optimizer.create("sgd", model, lr=0.1)
    # No algorithm hooks: take one plain optimizer step, then apply the hook by hand.
    trainer = Trainer(model, algo, optimizer, device="cpu")
    target_before = _snapshot(model.target_backbone)
    _train(trainer, _loader(1), max_iter=1)

    online = _snapshot(model.online_backbone)
    target = _snapshot(model.target_backbone)   # includes BN stats the target updated in its own forward
    assert any(not torch.equal(online[k], target_before[k]) for k in online), "optimizer step changed nothing"

    AlgorithmHook(algo).after_step(trainer.state)

    updated = _snapshot(model.target_backbone)
    m = 0.9
    buffers = {name for name, _ in model.target_backbone.named_buffers()}
    assert any(k.endswith("running_mean") for k in buffers)
    for key, value in updated.items():
        if value.is_floating_point():   # parameters AND BN running stats
            torch.testing.assert_close(value, m * target[key] + (1 - m) * online[key])
        else:                           # num_batches_tracked is copied
            assert torch.equal(value, online[key])
    assert any(not torch.equal(updated[k], target_before[k]) for k in updated)
    # online network untouched by the EMA
    assert all(torch.equal(v, online[k]) for k, v in _snapshot(model.online_backbone).items())
    assert algo.step == 1
    assert trainer.storage.latest()["byol_momentum"] == pytest.approx(m)


def test_algorithm_hooks_update_the_target_during_training():
    torch.manual_seed(0)
    algo, model = _byol()
    trainer = Trainer(model, algo, Optimizer.create("sgd", model, lr=0.1), hooks=algo.hooks(), device="cpu")
    before = _snapshot(model.target_projector)
    _train(trainer, _loader(3), max_iter=3)
    after = _snapshot(model.target_projector)
    assert algo.step == 3
    assert any(not torch.equal(before[k], after[k]) for k in before)


def test_ema_state_round_trips_through_checkpointer(tmp_path):
    """4 uninterrupted steps == 2 steps + checkpoint + resume into fresh objects + 2 steps."""
    loader = _loader(4, seed=1)

    torch.manual_seed(0)
    algo_ref, model_ref = _byol(total_steps=4)
    trainer_ref = Trainer(model_ref, algo_ref, Optimizer.create("sgd", model_ref, lr=0.1),
                          hooks=algo_ref.hooks(), device="cpu")
    _train(trainer_ref, loader, max_iter=4)

    torch.manual_seed(0)
    algo_a, model_a = _byol(total_steps=4)
    trainer_a = Trainer(model_a, algo_a, Optimizer.create("sgd", model_a, lr=0.1), hooks=algo_a.hooks(), device="cpu")
    _train(trainer_a, loader[:2], max_iter=2)
    path = Checkpointer(str(tmp_path)).save(trainer_a.state, "ckpt_1.pt")

    torch.manual_seed(123)   # different init, overwritten by the checkpoint
    algo_b, model_b = _byol(total_steps=4)
    trainer_b = Trainer(model_b, algo_b, Optimizer.create("sgd", model_b, lr=0.1), hooks=algo_b.hooks(), device="cpu")
    Checkpointer(str(tmp_path)).load(trainer_b.state, path)
    assert algo_b.step == 2
    _train(trainer_b, loader[2:], max_iter=4)   # resumes at iter 2

    assert algo_b.step == algo_ref.step == 4
    ref, resumed = _snapshot(model_ref), _snapshot(model_b)
    for key in ref:
        torch.testing.assert_close(resumed[key], ref[key], msg=key)


def test_after_optimizer_step_rejects_a_non_byol_model():
    algo = BYOLTask()

    class _State:
        model = torch.nn.Linear(2, 2)
        max_iter = 1

    with pytest.raises(TypeError, match="expects a BYOLModel"):
        algo.after_optimizer_step(_State())
