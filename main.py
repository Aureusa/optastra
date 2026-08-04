import torch
import torch.nn as nn

from optastra.backbones import Backbone
from optastra.necks import Neck
from optastra.heads import Head
from optastra.training.logging_config import setup_logging
from optastra.training.hooks.checkpoint import CheckpointHook
from optastra.training.hooks.eval import EvalHook
from optastra.training.hooks.common_metrics_printer import CommonMetricPrinterHook
from optastra.training.hooks.timer import IterTimerHook
from optastra.training.hooks.scheduler import SchedulerHook
from optastra.training.hooks.defaults import default_hooks
from optastra.algorithms import Algorithm, SimCLRModel
from optastra.training.trainer import Trainer
from optastra.tasks import Task
from optastra.optim import Optimizer, Scheduler, ParamGroupConfig

def model_test():
    backbone = Backbone.create("resnet50")
    neck = Neck.create("fpn", in_spec=backbone.out_spec)
    pool_neck = Neck.create("global_max_pool", in_spec=neck.out_spec)
    head = Head.create("vanilla_classification_head", in_spec=pool_neck.out_spec, num_classes=10)
    model = torch.nn.Sequential(backbone, neck, pool_neck, head)

    classification_task = Task.create("classification_task")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model=model, task=classification_task, optimizer=optimizer, device="cuda")

    # Create dummy dataloaders for training and validation
    train_loader = [({"inputs": torch.randn(1, 3, 224, 224), "targets": torch.tensor([1])}) for _ in range(1000)]
    val_loader = [({"inputs": torch.randn(1, 3, 224, 224), "targets": torch.tensor([1])}) for _ in range(200)]

    trainer.register_hooks([
        CommonMetricPrinterHook(log_every=20),
        EvalHook(eval_period=200, eval_fn=lambda: trainer.evaluate(val_loader)),
        CheckpointHook(output_dir="runs/exp1/ckpts", save_every=500),
    ])

    trainer.train(train_loader, max_iter=1000)

def simclr_test():
    # Create the model components: backbone, neck, and projection head
    backbone = Backbone.create("resnet50")
    neck = Neck.create("fpn", in_spec=backbone.out_spec)
    backbone_fpn = torch.nn.Sequential(backbone, neck)
    pool_neck = Neck.create("global_max_pool", in_spec=neck.out_spec)
    head = nn.Sequential(
        torch.nn.Linear(pool_neck.out_spec.embed_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    )
    head = Head.create("vanilla_classification_head", in_spec=pool_neck.out_spec, num_classes=10)
    model = SimCLRModel(backbone=backbone_fpn, neck=pool_neck, projector=head)
    model = nn.Sequential(backbone_fpn, pool_neck, head)

    # Create dummy dataloaders for training and validation
    train_loader = [({"views": [torch.randn(32, 3, 224, 224), torch.randn(32, 3, 224, 224)]}) for _ in range(10)]
    val_loader = [({"views": [torch.randn(32, 3, 224, 224), torch.randn(32, 3, 224, 224)]}) for _ in range(5)]

    # Create dummy dataloaders for training and validation
    train_loader = [({"inputs": torch.randn(1, 3, 224, 224), "targets": torch.tensor([1])}) for _ in range(1000)]
    val_loader = [({"inputs": torch.randn(1, 3, 224, 224), "targets": torch.tensor([1])}) for _ in range(500)]

    # Total number of steps
    total_iters_per_epoch = len(train_loader)
    epochs = 10
    total_steps = total_iters_per_epoch * epochs
    print(f"Total training steps: {total_steps}")
    print(f"Scheduler warmup steps: {int(0.1*total_steps)}")

    # Create the SimCLR task
    simclr_task = Algorithm.create("classification_task")

    # Define the optimizer and scheduler
    optimizer = Optimizer.create(
        "adamw", model, lr=1e-3,
        param_groups=ParamGroupConfig(lr_multipliers={"backbone": 0.1}),
    )
    scheduler = Scheduler.create("warmup_cosine", optimizer, total_steps=int(0.1 * total_steps), warmup_steps=50)
    
    trainer = Trainer(model=model, task=simclr_task, optimizer=optimizer, device="cuda")

    hooks = default_hooks(
        log_every=20,
        output_dir="runs/exp1/",
        checkpoint_every=500
    )
    hooks.append(SchedulerHook(scheduler=scheduler))
    hooks.append(EvalHook(eval_period=200, eval_fn=lambda: trainer.evaluate(val_loader)))

    trainer.register_hooks(hooks)

    trainer.train(train_loader, max_iter=total_steps)

def test_faster_rcnn():
    from optastra.architectures import Architecture

    faster_rcnn = Architecture.create("mask_rcnn_r50_fpn")

    print(faster_rcnn)

if __name__ == "__main__":
    setup_logging("runs/exp1/logs", filename="train.log", color=True)
    simclr_test()
    # test_faster_rcnn()
