import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from optastra.nn.features import HeadOutput
from optastra.tasks.base import Task
from optastra.tasks.classification import ClassificationTask, ClassificationTaskConfig


class _ToyClassifier(nn.Module):
    def __init__(self, in_features: int = 5, num_classes: int = 3):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, inputs: torch.Tensor) -> HeadOutput:
        return HeadOutput(logits=self.fc(inputs))


def test_task_factory_creates_registered_classification_task():
    task = Task.create("classification_task")

    assert isinstance(task, ClassificationTask)


def test_task_factory_raises_for_unknown_task():
    with pytest.raises(ValueError, match="task 'unknown_task' is not registered"):
        Task.create("unknown_task")


def test_classification_task_train_step_returns_ce_loss():
    task = ClassificationTask(ClassificationTaskConfig(label_smoothing=0.0, reduction="mean"))
    model = _ToyClassifier(in_features=4, num_classes=3)

    batch = {
        "inputs": torch.randn(6, 4),
        "targets": torch.tensor([0, 1, 2, 0, 1, 2]),
    }
    output = task.run_step(model, batch, stage="train")

    assert output.loss is not None
    assert "ce_loss" in output.losses
    assert output.predictions is None


def test_classification_task_val_step_computes_accuracy_and_predictions():
    task = ClassificationTask(ClassificationTaskConfig())

    class _DeterministicModel(nn.Module):
        def forward(self, inputs):
            logits = torch.tensor([[10.0, 0.0], [0.0, 10.0], [9.0, 1.0]])
            return HeadOutput(logits=logits)

    batch = {
        "inputs": torch.randn(3, 4),
        "targets": torch.tensor([0, 1, 1]),
    }
    output = task.run_step(_DeterministicModel(), batch, stage="val")

    assert output.loss is not None
    assert "accuracy" in output.metrics
    assert output.metrics["accuracy"] == pytest.approx(2 / 3)
    assert torch.equal(output.predictions, torch.tensor([0, 1, 0]))


def test_classification_task_validate_batch_rejects_missing_keys():
    task = ClassificationTask(ClassificationTaskConfig())

    with pytest.raises(ValueError, match="Batch must contain 'inputs' and 'targets' keys"):
        task.validate_batch({"inputs": torch.randn(2, 4)}, stage="train")


def test_registered_preset_matches_dataclass_defaults():
    task = Task.create("classification_task")
    assert task.cfg == ClassificationTaskConfig()
    assert task.cfg.label_smoothing == 0.0


def test_run_step_applies_no_precision_policy_of_its_own():
    # Precision is the Trainer's job: run_step must not autocast (and must not
    # warn on CPU), so on CPU the logits and loss stay fp32.
    task = ClassificationTask()
    model = _ToyClassifier(in_features=4, num_classes=3)
    batch = {"inputs": torch.randn(2, 4), "targets": torch.tensor([0, 2])}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        output = task.run_step(model, batch, stage="val")

    assert output.raw_predictions.logits.dtype == torch.float32
    assert output.loss.dtype == torch.float32


def test_classification_evaluator_uses_counts_across_unequal_batches():
    task = ClassificationTask()
    evaluator = task.build_evaluator()
    evaluator.reset()

    class _FixedLogits(nn.Module):
        def forward(self, inputs):
            return HeadOutput(logits=inputs)   # inputs are the logits

    model = _FixedLogits()
    # Batch 1: 3/3 correct. Batch 2: 0/1 correct.
    b1 = {"inputs": torch.tensor([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]]), "targets": torch.tensor([0, 1, 0])}
    b2 = {"inputs": torch.tensor([[5.0, 0.0]]), "targets": torch.tensor([1])}
    out1, out2 = task.run_step(model, b1, stage="val"), task.run_step(model, b2, stage="val")
    evaluator.process(out1, b1)
    evaluator.process(out2, b2)
    results = evaluator.summarize()

    assert results["accuracy"] == pytest.approx(3 / 4)  # not (1.0 + 0.0) / 2
    expected_loss = F.cross_entropy(torch.cat([b1["inputs"], b2["inputs"]]), torch.tensor([0, 1, 0, 1]))
    assert results["total_loss"] == pytest.approx(expected_loss.item(), rel=1e-6)

    evaluator.reset()
    assert evaluator.summarize() == {}
