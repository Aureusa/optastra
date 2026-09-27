from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
from PIL import Image, ImageDraw

from .sample import Sample


__all__ = ["CocoDetectionDataset", "load_data_from_coco_json", "image_to_float"]


# channels= options -> torchvision read modes. "unchanged" keeps whatever the file stores
# (e.g. 1-channel 16-bit astronomy PNGs); "rgb" forces 3 channels (COCO has some grayscale JPEGs).
_READ_MODES = {
    "unchanged": ImageReadMode.UNCHANGED,
    "gray": ImageReadMode.GRAY,
    "gray_alpha": ImageReadMode.GRAY_ALPHA,
    "rgb": ImageReadMode.RGB,
    "rgb_alpha": ImageReadMode.RGB_ALPHA,
}


def image_to_float(image: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    """Convert an image to float32.

    :param scale: divide the pixel values by this. ``None`` infers it from the
        dtype: integer images are divided by their dtype's maximum (uint8 -> 255,
        uint16 -> 65535) so they land in [0, 1]; float images are left as they are.
    """
    if scale is None:
        if image.is_floating_point():
            return image.to(torch.float32)
        scale = float(torch.iinfo(image.dtype).max)
    return image.to(torch.float32) / scale


class CocoDetectionDataset(Dataset[Sample]):
    """Minimal COCO detection dataset that yields Sample objects.

    Each item is ``Sample(image, target={"boxes", "labels"[, "masks"]}, meta)``:
    boxes are XYXY in pixels, labels are contiguous indices ``0..num_categories-1``
    (sorted COCO category ids), masks are ``(G, H, W)`` uint8 decoded at the
    original image size. If ``transform`` is given it receives the whole
    Sample (image *and* target), so geometric transforms can move boxes/masks.

    :param channels: one of ``"unchanged"``, ``"gray"``, ``"gray_alpha"``,
        ``"rgb"``, ``"rgb_alpha"`` -- how many channels to decode.
    :param scale: divisor applied to pixel values; ``None`` = infer from dtype
        (see :func:`image_to_float`).
    """

    def __init__(
        self,
        json_path: str | Path,
        image_root: str | Path | None = None,
        transform: Callable[[Sample], Sample] | None = None,
        *,
        channels: str = "rgb",
        scale: float | None = None,
    ):
        if channels not in _READ_MODES:
            raise ValueError(f"channels must be one of {sorted(_READ_MODES)}, got {channels!r}.")
        self.json_path = Path(json_path)
        self.image_root = self._resolve_image_root(image_root)
        self.transform = transform
        self.channels = channels
        self.scale = scale
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

        # Fail at construction, not deep inside a DataLoader worker.
        for annotation in annotations:
            category_id = int(annotation["category_id"])
            if category_id not in self._category_to_index:
                raise ValueError(
                    f"Annotation {annotation.get('id')} (image {annotation['image_id']}) has category_id "
                    f"{category_id}, which is not listed in 'categories' of {self.json_path}. "
                    f"Known category ids: {sorted(self._category_to_index)}."
                )

    @property
    def num_classes(self) -> int:
        """Number of foreground categories -- pass this as num_classes to the architecture and task."""
        return len(self._category_to_index)

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
            # Runs from the 4th on are stored as deltas to the run two places earlier (pycocotools rleFrString).
            if m > 2:
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

        # Runs alternate 0, 1, 0, 1, ... starting with background.
        total = mask_h * mask_w
        runs = torch.tensor(run_lengths, dtype=torch.long).clamp(min=0)
        values = (torch.arange(runs.numel()) % 2).to(torch.uint8)
        flat = torch.repeat_interleave(values, runs)[:total]
        if flat.numel() < total:
            flat = torch.cat((flat, flat.new_zeros(total - flat.numel())))

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
        return torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).view(height, width)

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

    def _build_target(self, image_id: int, img_h: int, img_w: int) -> dict[str, Any]:
        """Boxes/labels/masks of one image, in original-image pixel coordinates."""
        annotations = self._annotations_by_image_id.get(image_id, [])
        boxes: list[list[float]] = []
        labels: list[int] = []
        masks: list[torch.Tensor] = []
        has_any_segmentation = False
        all_masks_decoded = True

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
            target["masks"] = torch.stack(masks, dim=0)
        elif has_any_segmentation and not self._has_warned_mask_decode:
            warnings.warn(
                "Some COCO segmentations could not be decoded, so masks were omitted for this sample. "
                "roi_mask_loss can remain zero unless mask decoding succeeds for all instances.",
                stacklevel=2,
            )
            self._has_warned_mask_decode = True
        return target

    def __getitem__(self, index: int) -> Sample:
        image_record = self._images[index]
        image_id = int(image_record["id"])

        image_path = self.image_root / image_record["file_name"]
        raw = decode_image(str(image_path), mode=_READ_MODES[self.channels])
        image = image_to_float(raw, self.scale)

        # Targets (masks included) are built at the original image size, *before*
        # any transform; the transform then moves image and target together.
        img_h, img_w = int(raw.shape[-2]), int(raw.shape[-1])
        sample = Sample(
            image=image,
            target=self._build_target(image_id, img_h, img_w),
            meta={
                "image_id": image_id,
                "file_name": image_record["file_name"],
                "height": img_h,
                "width": img_w,
            },
        )
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def load_data_from_coco_json(
    json_path: str | Path,
    image_root: str | Path | None = None,
    transform: Callable[[Sample], Sample] | None = None,
    **kwargs,
) -> CocoDetectionDataset:
    """Build a :class:`CocoDetectionDataset`; ``kwargs`` are ``channels`` / ``scale``."""
    return CocoDetectionDataset(json_path=json_path, image_root=image_root, transform=transform, **kwargs)
