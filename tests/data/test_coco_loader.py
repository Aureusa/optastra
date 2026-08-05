import json

import torch
from torchvision.io import write_png

from optastra.data import build_dataloader, load_data_from_coco_json
from optastra.tasks import Task


def test_coco_loader_returns_samples_without_proposals(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    image_path = images_dir / "sample.png"
    write_png(torch.zeros((3, 8, 8), dtype=torch.uint8), str(image_path))

    coco_json = {
        "images": [{"id": 1, "file_name": "sample.png", "height": 8, "width": 8}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 5, "bbox": [1, 2, 3, 4]},
        ],
        "categories": [{"id": 5, "name": "object"}],
    }

    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(coco_json), encoding="utf-8")

    dataset = load_data_from_coco_json(json_path)
    sample = dataset[0]

    assert sample.image.shape == (3, 8, 8)
    assert sample.target["boxes"].shape == (1, 4)
    assert torch.equal(sample.target["boxes"], torch.tensor([[1.0, 2.0, 4.0, 6.0]]))
    assert torch.equal(sample.target["labels"], torch.tensor([0]))
    assert "proposals" not in sample.target


def test_coco_loader_works_with_build_dataloader(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    image_path = images_dir / "sample.png"
    write_png(torch.zeros((3, 8, 8), dtype=torch.uint8), str(image_path))

    coco_json = {
        "images": [{"id": 1, "file_name": "sample.png", "height": 8, "width": 8}],
        "annotations": [],
        "categories": [],
    }

    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(coco_json), encoding="utf-8")

    dataset = load_data_from_coco_json(json_path)
    task = Task.create("detection_task", num_classes=1)
    dataloader = build_dataloader(dataset, task=task, batch_size=1, shuffle=False)

    batch = next(iter(dataloader))

    assert batch["inputs"].shape == (1, 3, 8, 8)
    assert batch["targets"][0]["boxes"].shape == (0, 4)


def test_coco_loader_decodes_rle_masks(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    image_path = images_dir / "sample.png"
    write_png(torch.zeros((3, 4, 4), dtype=torch.uint8), str(image_path))

    coco_json = {
        "images": [{"id": 1, "file_name": "sample.png", "height": 4, "width": 4}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 5,
                "bbox": [0, 0, 1, 1],
                "segmentation": {"size": [4, 4], "counts": [0, 1, 15]},
            },
        ],
        "categories": [{"id": 5, "name": "object"}],
    }

    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(coco_json), encoding="utf-8")

    dataset = load_data_from_coco_json(json_path)
    sample = dataset[0]

    assert "masks" in sample.target
    assert sample.target["masks"].shape == (1, 4, 4)
    assert sample.target["masks"].sum().item() == 1.0