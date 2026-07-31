## Optastra

> **A modular computer vision framework built on PyTorch, focused on composable architectures, clean implementations, and research reproducibility.**

## Motivation

The idea for this project came out of my interest in computer vision primarily due to the fact that most of my research is centered around computer vision in application to astronomy. The problem arrises from the fact that most computer vision frameworks are either:
* research code tied to a single paper,
* deployment-focused,
* or large monolithic ecosystems that are difficult to extend.

The most prevalent framework for doing CV in Astronomy is [ZooBot](https://github.com/mwalmsley/zoobot), which solves an important problem -- namely that astronomers are often not computer vision experts. Thus, the design strategy of ZooBot is centered around making things easy for non-experts. Swap a few final layers, train on your data, and get your model. It also allows loading pretrained models from huggingface, and uses the [`timm`](https://github.com/huggingface/pytorch-image-models) library for model implementation, which can then be finetuned on a specific use case.

Of course this is of great advantage to most astronomers as they don't have to implement complex models themselves, hack up some pre-training routine, and so on. However, this results in a framework that is not very flexible. It is constrained to the architectures implemented in `timm` (which granted are a lot), and is difficult to implement new (or any) pretraining methods. Another issue is that astronomers sometimes forget that the pretrained models they are using might have been pretrained on data that is very different from their own. For example, a model pretrained on `Galaxy Zoo Hubble and CANDELS` may not perform well on astronomical images from `Euclid` due to different noise characteristics, different resolutions, and different data distributions. This is not necessarily a fault of the astronomers using the framework, but rather a design choice of the framework itself -- which inexperienced non-experts in the field of CV can easily miss.

The data distribution of astronomical images is often very different depending on the survey. The ability to pretrain models on new data is important. Other libraries such as `timm`, `torchvision`, `ZooBot`, etc., are not designed for this. This is a problem that needs to be solved in order to make computer vision more accessible both for astronomers and for other researchers working with specialized datasets.

This project aims to be different!

It is intended to become a framework where implementing new vision architectures, experimenting with new pretraining methods, and assembling models from reusable components is straightforward.

Although motivated by astronomy, the framework is **general-purpose** and should be useful for any computer vision research.

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
