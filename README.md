## Optastra

> **A modular computer vision framework built on PyTorch, focused on composable architectures, clean implementations, and research reproducibility.**

## Motivation

The idea for this project grew out of my work in computer vision for astronomy. While there are many excellent computer vision frameworks available, they generally fall into one of three categories:

- research code accompanying a single paper,
- deployment-oriented libraries,
- or large ecosystems that prioritize stability over rapid experimentation.

For astronomy, one of the most widely used frameworks is [ZooBot](https://github.com/mwalmsley/zoobot). ZooBot addresses an important problem: most astronomers are not computer vision experts. Its design therefore emphasizes ease of use. Users can select a pretrained backbone, replace the task-specific prediction head, fine-tune on their own data, and obtain a competitive model with relatively little effort. This significantly lowers the barrier to applying deep learning to astronomical datasets and has in turn advanced the field significantly.

However, this design also comes with trade-offs. ZooBot is primarily intended for downstream fine-tuning rather than developing new computer vision methods. It builds upon the excellent [`timm`](https://github.com/huggingface/pytorch-image-models) (PyTorch Image Models) library, making it straightforward to reuse existing architectures but comparatively cumbersome to introduce entirely new architectures, training paradigms, or pretraining strategies.

This distinction is particularly important in astronomy because datasets from different surveys often exhibit substantial domain shifts. Images often differ in wavelength, spatial resolution, point spread function, noise characteristics, and preprocessing pipelines. Consequently, representations learned on one survey do not necessarily transfer well to another. For example, a model pretrained on **Galaxy Zoo Hubble** or **CANDELS** may not transfer optimally to **Euclid** imagery without further adaptation.

While fine-tuning pretrained models is often sufficient, there are many situations where pretraining directly on survey-specific data—or experimenting with entirely new self-supervised learning methods—is desirable. However, existing frameworks are not designed to facilitate this kind of experimentation.

Existing libraries such as `timm` and `torchvision` provide outstanding implementations of modern vision architectures, but they are intentionally general-purpose libraries rather than research frameworks for rapidly prototyping new models and pretraining algorithms. Likewise, astronomy-focused frameworks such as ZooBot prioritize accessibility for downstream users over flexibility for computer vision research.

## Goal

This project aims to fill that gap.

Its goal is to provide a modular, research-oriented computer vision framework in which implementing new architectures, experimenting with novel pretraining methods, and assembling models from reusable components is straightforward.

Rather than treating architectures and training pipelines as fixed entities, the framework encourages composing models from interchangeable building blocks. The focus is on making research fast: new ideas should require implementing only the novel components, not rewriting an entire training stack.

Although motivated by astronomical applications, the framework is **general-purpose** and is intended to be useful for computer vision research on any specialized imaging dataset.

---

# Philosophy

## Models are compositions, not monoliths

Modern vision models are built from recurring ideas.

Instead of treating models like independent implementations:

```text
ResNet
ViT
ConvNeXt
DINOv2
SAM
```

the framework treats them as compositions of reusable building blocks.

For example:

```text
PatchEmbedding
↓
TransformerBlock × N
↓
LayerNorm
↓
Classification Head
```

or

```text
Residual Block
↓
Residual Stage × 4
↓
FPN
↓
Detection Head
```

New papers should primarily be assembled from existing components rather than implemented from scratch every time.

---

## Organize around ideas, not papers

The repository should emphasize reusable concepts such as:

* residual connections
* attention
* patch embeddings
* normalization layers
* stochastic depth
* deformable convolutions
* positional embeddings
* feature pyramids

rather than duplicating code for every architecture.

---

## Separate architecture from algorithms

This distinction is important.

Examples:

**Architectures**

* ResNet
* ConvNeXt
* ViT
* Swin

**Algorithms**

* DINOv2
* MAE
* SimCLR
* CLIP
* BYOL

A ViT is an architecture.

DINOv2 is a pretraining algorithm that produces pretrained ViTs.

Keeping these concepts separate makes the framework significantly more extensible.

---

## Separate architecture from tasks

A backbone should not know whether it is being used for

* classification
* regression
* segmentation
* detection
* representation learning

Tasks attach their own heads and losses.

---

## Composition over inheritance

Avoid deep inheritance trees.

Instead of

```text
VisionModel
    ↑
Detector
    ↑
YOLO
```

prefer

```text
Backbone

Neck

Head

Loss

Postprocessor
```

assembled into complete models.

---

## Progressive abstraction

The repository should also serve as an educational resource.

Someone should be able to learn by navigating through layers of abstraction:

```text
Tensor

↓

Layer

↓

Block

↓

Backbone

↓

Model

↓

Task

↓

Training
```

Every implementation should remain readable.

---

# Repository Structure

```text
optastra/

    nn/
        blocks/
            /ideas

    backbones/
        resnet.py
        vit.py
        ...

    necks/

    heads/

    tasks/
        classification/
        regression/
        detection/
        segmentation/

    algorithms/
        mae/
        dinov2/
        clip/
        simclr/

    models/
        classification/
        detection/
        segmentation/
        regression/

    training/

    losses/

    datasets/

    transforms/

    metrics/

    visualization/

```

---

# Core Abstractions

## Backbone

Responsible only for producing feature representations.

```python
Backbone
    images
        ↓
BackboneFeatures
```

Backbones should not know anything about downstream tasks.

---

## BackboneFeatures

Backbones should not return raw tensors.

Instead they return a structured object that can represent outputs from many architectures.

Possible contents include:

* feature maps
* global embeddings
* patch tokens
* CLS token
* metadata

This allows CNNs and transformers to share a common interface without forcing identical tensor shapes.

---

## Head

Converts features into task-specific outputs.

Examples:

* ClassificationHead
* RegressionHead
* DetectionHead
* SegmentationHead
* EmbeddingHead

---

## Task

A task owns:

* losses
* metrics
* target formatting
* prediction decoding

Examples:

* Classification
* Regression
* Detection
* Segmentation

---

## Trainer

The trainer should know almost nothing about individual models.

Conceptually:

```text
Batch
    ↓
Model
    ↓
Task
    ↓
Loss
```

No model-specific logic should appear inside the training loop.

---

# Research Direction

The framework should make it easy to compare:

Different backbones

* ResNet
* ConvNeXt
* ViT
* Swin

Different pretraining

* ImageNet
* MAE
* DINOv2
* Astronomy self-supervised
* Euclid self-supervised

Different tasks

* classification
* regression
* detection
* segmentation

Ideally these become configuration changes rather than code changes.

---

# Initial Models

## Backbones

Essential

* ResNet
* ConvNeXt
* Vision Transformer
* Swin Transformer

Later

* EfficientNet
* DINOv2
* EVA
* SigLIP

---

## Detection

* Faster R-CNN
* YOLO
* DETR
* RT-DETR

---

## Segmentation

* UNet
* DeepLabV3+
* Mask R-CNN
* SAM

---

## Representation Learning

* MAE
* DINOv2
* SimCLR
* CLIP

---

# Long-Term Goal

The framework should eventually make experimentation look like this:

```python
backbone = Backbone.create(name="resnet50")

task = Regression()

trainer.fit(
    backbone,
    task,
    dataset,
)
```

or

```python
model = DetectionModel(
    backbone=ConvNeXt(),
    neck=FPN(),
    head=RTDETRHead(),
)
```

rather than requiring every architecture to have its own training pipeline.

---

# Astronomy Vision

Astronomy is the original motivation, not the limitation.

The long-term vision is to support astronomy as a first-class domain through plugins:

```text
astro/

    euclid/

    galaxy_zoo/

    simulations/

    visualization/
```

while keeping the core framework completely domain-independent.

---

# Design Principles

* Clean implementations over clever abstractions.
* Readability over premature optimization.
* Composition over inheritance.
* Ideas over papers.
* Algorithms separated from architectures.
* Tasks separated from models.
* Backbones as reusable feature extractors.
* Research reproducibility by default.
* Educational codebase suitable for learning state-of-the-art computer vision.

---

# Vision Statement

The ambition is **not** to build another model zoo.

The ambition is to build a framework where implementing, understanding, pretraining, and composing modern computer vision models feels natural.

It should be equally useful to:

* researchers implementing new papers,
* machine learning engineers building production models,
* students learning computer vision,
* and domain scientists (e.g., astronomers) applying state-of-the-art vision methods to scientific problems.
