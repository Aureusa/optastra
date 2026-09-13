"""Quickstart: compose a backbone + neck + head into a trainable model.

Runs on CPU with synthetic tensors -- no dataset, no GPU, no cluster
required. This is the "Backbone -> Neck -> Head" composition pattern
described in the README's Philosophy section, using the actual
`build_sequential_model` helper from `optastra.core.build`.

Run with:
    python examples/01_quickstart_classification.py
"""
import torch

from optastra import Optimizer, build_sequential_model

NUM_CLASSES = 10


def main() -> None:
    # ComponentRefs can be given as plain strings, (name, overrides) tuples,
    # or {"name": ..., "overrides": ...} dicts -- coerced automatically.
    model = build_sequential_model(
        backbone="resnet18",
        necks=["global_avg_pool"],
        head=("vanilla_classification_head", {"num_classes": NUM_CLASSES}),
    )

    optimizer = Optimizer.create("adamw", model, lr=1e-3)

    images = torch.randn(4, 3, 64, 64)
    labels = torch.randint(0, NUM_CLASSES, (4,))

    output = model(images)  # Backbone -> GlobalPool -> ClassificationHead
    loss = torch.nn.functional.cross_entropy(output.logits, labels)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(f"logits shape: {tuple(output.logits.shape)}")
    print(f"loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
