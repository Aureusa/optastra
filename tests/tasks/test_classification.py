import pytest
import torch
import torch.nn as nn

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
