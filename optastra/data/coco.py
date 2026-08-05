from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image, ImageDraw

from .sample import Sample


class CocoDetectionDataset(Dataset[Sample]):
    """Minimal COCO detection dataset that yields Sample objects."""

    def __init__(self, json_path: str | Path, image_root: str | Path | None = None, transform: Callable[[torch.Tensor], torch.Tensor] | None = None):
        self.json_path = Path(json_path)
        self.image_root = self._resolve_image_root(image_root)
        self.transform = transform
        self._has_warned_mask_decode = False

        with self.json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])

        self._images = sorted(images, key=lambda item: item["id"])
        self._annotations_by_image_id: dict[int, list[dict[str, Any]]] = {}
        for annotation in annotations:
            image_id = int(annotation["image_id"])
            self._annotations_by_image_id.setdefault(image_id, []).append(annotation)

        category_ids = [int(category["id"]) for category in categories] or [int(annotation["category_id"]) for annotation in annotations]
        self._category_to_index = {category_id: index for index, category_id in enumerate(sorted(set(category_ids)))}

    @staticmethod
    def _decode_coco_rle_counts(encoded: str) -> list[int]:
        counts: list[int] = []
        p = 0
        m = 0
        while p < len(encoded):
            x = 0
            shift = 0
            more = True
            while more:
                c = ord(encoded[p]) - 48
                p += 1
                x |= (c & 0x1F) << shift
                more = (c & 0x20) != 0
                shift += 5
                if not more and (c & 0x10):
                    x |= -1 << shift
            if m > 1:
                x += counts[m - 2]
            counts.append(int(x))
            m += 1
        return counts

    @staticmethod
    def _decode_rle(segmentation: dict[str, Any], height: int, width: int) -> torch.Tensor | None:
        counts = segmentation.get("counts")
        size = segmentation.get("size", [height, width])
        if not isinstance(size, list) or len(size) != 2:
            return None

        mask_h, mask_w = int(size[0]), int(size[1])
        if isinstance(counts, str):
            run_lengths = CocoDetectionDataset._decode_coco_rle_counts(counts)
        elif isinstance(counts, list):
            run_lengths = [int(v) for v in counts]
        else:
            return None

        total = mask_h * mask_w
        flat = torch.zeros((total,), dtype=torch.uint8)
        cursor = 0
        value = 0
        for run in run_lengths:
            if run <= 0:
                value = 1 - value
                continue
            end = min(cursor + run, total)
            if value == 1:
                flat[cursor:end] = 1
            cursor = end
            value = 1 - value
            if cursor >= total:
                break

        # COCO RLE is column-major (Fortran order).
        decoded = flat.view(mask_w, mask_h).t().contiguous()
        if decoded.shape != (height, width):
            return None
        return decoded

    @staticmethod
    def _decode_polygons(segmentation: list[Any], height: int, width: int) -> torch.Tensor | None:
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        drew = False
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6:
                continue
            xy = [float(v) for v in polygon]
            points = [(xy[i], xy[i + 1]) for i in range(0, len(xy), 2)]
            draw.polygon(points, outline=1, fill=1)
            drew = True
        if not drew:
            return None
        return torch.tensor(list(canvas.getdata()), dtype=torch.uint8).view(height, width)

    def _decode_segmentation(self, segmentation: Any, height: int, width: int) -> torch.Tensor | None:
        if isinstance(segmentation, dict):
            return self._decode_rle(segmentation, height, width)
        if isinstance(segmentation, list):
            return self._decode_polygons(segmentation, height, width)
        return None

    def _resolve_image_root(self, image_root: str | Path | None) -> Path:
        if image_root is not None:
            return Path(image_root)

        sibling_images = self.json_path.parent / "images"
        if sibling_images.is_dir():
            return sibling_images
        return self.json_path.parent

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> Sample:
        image_record = self._images[index]
        image_id = int(image_record["id"])

        image_path = self.image_root / image_record["file_name"]
        image = read_image(str(image_path)).to(torch.float32).div(255.0)
        if self.transform is not None:
            image = self.transform(image)

        annotations = self._annotations_by_image_id.get(image_id, [])
        boxes: list[list[float]] = []
        labels: list[int] = []
        masks: list[torch.Tensor] = []
        has_any_segmentation = False
        all_masks_decoded = True
        img_h, img_w = int(image.shape[-2]), int(image.shape[-1])

        for annotation in annotations:
            if annotation.get("iscrowd", 0):
                continue

            x, y, width, height = annotation["bbox"]
            if width <= 0 or height <= 0:
                continue

            boxes.append([float(x), float(y), float(x + width), float(y + height)])
            labels.append(self._category_to_index[int(annotation["category_id"])])

            segmentation = annotation.get("segmentation")
            if segmentation is not None:
                has_any_segmentation = True
                decoded = self._decode_segmentation(segmentation, img_h, img_w)
                if decoded is None:
                    all_masks_decoded = False
                else:
                    masks.append(decoded)

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        target: dict[str, Any] = {"boxes": boxes_tensor, "labels": labels_tensor}
        if has_any_segmentation and all_masks_decoded and len(masks) == len(boxes):
            target["masks"] = torch.stack(masks, dim=0).to(torch.float32)
        elif has_any_segmentation and not self._has_warned_mask_decode:
            warnings.warn(
                "Some COCO segmentations could not be decoded, so masks were omitted for this sample. "
                "roi_mask_loss can remain zero unless mask decoding succeeds for all instances.",
                stacklevel=2,
            )
            self._has_warned_mask_decode = True

        return Sample(
            image=image,
            target=target,
            meta={
                "image_id": image_id,
                "file_name": image_record["file_name"],
                "height": int(image_record.get("height", image.shape[-2])),
                "width": int(image_record.get("width", image.shape[-1])),
            },
        )


def load_data_from_coco_json(json_path: str | Path, image_root: str | Path | None = None, transform: Callable[[torch.Tensor], torch.Tensor] | None = None) -> CocoDetectionDataset:
    return CocoDetectionDataset(json_path=json_path, image_root=image_root, transform=transform)