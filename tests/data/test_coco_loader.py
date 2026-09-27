import json

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.io import write_png

from optastra.data import CocoDetectionDataset, build_dataloader, load_data_from_coco_json
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

def _write_coco(tmp_path, annotations, categories, height=4, width=6, image=None):
    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)
    if image is None:
        write_png(torch.full((3, height, width), 255, dtype=torch.uint8), str(images_dir / "sample.png"))
    else:
        Image.fromarray(image).save(images_dir / "sample.png")
    coco_json = {
        "images": [{"id": 1, "file_name": "sample.png", "height": height, "width": width}],
        "annotations": annotations,
        "categories": categories,
    }
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(coco_json), encoding="utf-8")
    return json_path


def test_coco_loader_scales_uint8_by_255(tmp_path):
    sample = load_data_from_coco_json(_write_coco(tmp_path, [], []))[0]

    assert sample.image.dtype == torch.float32
    assert torch.all(sample.image == 1.0)


def test_coco_loader_scales_uint16_by_65535_and_keeps_channels_unchanged(tmp_path):
    raw = (np.arange(24, dtype=np.uint16).reshape(4, 6) * 2000)
    json_path = _write_coco(tmp_path, [], [], image=raw)

    sample = load_data_from_coco_json(json_path, channels="unchanged")[0]

    assert sample.image.shape == (1, 4, 6)
    expected = torch.from_numpy(raw.astype(np.float32) / 65535.0)
    assert torch.allclose(sample.image[0], expected)


def test_coco_loader_explicit_scale_and_channel_mode(tmp_path):
    json_path = _write_coco(tmp_path, [], [])

    gray = load_data_from_coco_json(json_path, channels="gray", scale=1.0)[0]

    assert gray.image.shape == (1, 4, 6)
    assert torch.all(gray.image == 255.0)
    with pytest.raises(ValueError, match="channels"):
        load_data_from_coco_json(json_path, channels="bgr")


def test_coco_loader_rejects_unknown_category_ids_at_construction(tmp_path):
    annotations = [{"id": 7, "image_id": 1, "category_id": 3, "bbox": [0, 0, 2, 2]}]
    json_path = _write_coco(tmp_path, annotations, [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}])

    with pytest.raises(ValueError, match="category_id 3"):
        load_data_from_coco_json(json_path)


def test_coco_loader_maps_sparse_category_ids_to_contiguous_labels(tmp_path):
    annotations = [
        {"id": 1, "image_id": 1, "category_id": 90, "bbox": [0, 0, 2, 2]},
        {"id": 2, "image_id": 1, "category_id": 7, "bbox": [1, 1, 2, 2]},
    ]
    dataset = load_data_from_coco_json(_write_coco(tmp_path, annotations, [{"id": 7}, {"id": 90}]))

    assert dataset.num_classes == 2
    assert dataset[0].target["labels"].tolist() == [1, 0]


def test_coco_loader_passes_the_whole_sample_with_original_size_masks_to_the_transform(tmp_path):
    # Column-major (Fortran) RLE of a 4x6 mask whose first column is on: runs [0 zeros, 4 ones, 20 zeros].
    annotations = [{
        "id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 4],
        "segmentation": {"size": [4, 6], "counts": [0, 4, 20]},
    }]
    json_path = _write_coco(tmp_path, annotations, [{"id": 1}])
    seen = {}

    def hflip_and_upscale(sample):
        seen["image"] = tuple(sample.image.shape)
        seen["masks"] = sample.target["masks"].clone()
        width = sample.image.shape[-1]
        boxes = sample.target["boxes"].clone()
        boxes[:, [0, 2]] = width - sample.target["boxes"][:, [2, 0]]
        sample.image = sample.image.flip(-1).repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
        sample.target["boxes"] = boxes * 2
        sample.target["masks"] = sample.target["masks"].flip(-1).repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
        return sample

    sample = load_data_from_coco_json(json_path, transform=hflip_and_upscale)[0]

    expected_mask = torch.zeros((4, 6), dtype=torch.uint8)
    expected_mask[:, 0] = 1
    assert seen["image"] == (3, 4, 6)
    assert torch.equal(seen["masks"][0], expected_mask)  # decoded at the original size, before the transform
    assert sample.target["boxes"].tolist() == [[10.0, 0.0, 12.0, 8.0]]
    assert sample.target["masks"].shape == (1, 8, 12)
    assert sample.target["masks"][0, :, 10:].all()


def test_coco_loader_decodes_compressed_rle_strings():
    # pycocotools' compressed form of the runs [3, 4, 5, 6, 6]: from the 4th run on, each
    # value is stored as a difference to the run two places earlier.
    assert CocoDetectionDataset._decode_coco_rle_counts("34521") == [3, 4, 5, 6, 6]
    assert CocoDetectionDataset._decode_coco_rle_counts("04d0") == [0, 4, 20]

    mask = CocoDetectionDataset._decode_rle({"size": [4, 6], "counts": "34521"}, 4, 6)
    expected = torch.zeros(24, dtype=torch.uint8)
    expected[3:7] = 1
    expected[12:18] = 1
    assert torch.equal(mask, expected.view(6, 4).t())  # runs are column-major
