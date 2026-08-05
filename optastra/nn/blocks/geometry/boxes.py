from __future__ import annotations

import math

import torch
from torchvision.ops import batched_nms as tv_batched_nms
from torchvision.ops import box_iou


def _centers_and_sizes(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    widths = (x2 - x1).clamp(min=1e-6)
    heights = (y2 - y1).clamp(min=1e-6)
    ctr_x = x1 + 0.5 * widths
    ctr_y = y1 + 0.5 * heights
    return ctr_x, ctr_y, widths, heights


def generate_anchors(
    height: int,
    width: int,
    stride: int,
    scales: tuple[float, ...],
    aspect_ratios: tuple[float, ...],
    device: torch.device,
) -> torch.Tensor:
    """Generate XYXY anchors for one FPN level."""
    if not scales:
        raise ValueError("'scales' must not be empty.")
    if not aspect_ratios:
        raise ValueError("'aspect_ratios' must not be empty.")

    base_anchors = []
    for scale in scales:
        for ratio in aspect_ratios:
            area = (stride * scale) ** 2
            w = math.sqrt(area / ratio)
            h = ratio * w
            base_anchors.append(torch.tensor([-0.5 * w, -0.5 * h, 0.5 * w, 0.5 * h], device=device))
    base_anchors = torch.stack(base_anchors, dim=0)  # (A, 4)

    shifts_x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride
    shifts_y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=-1).reshape(-1, 4)  # (H*W, 4)

    anchors = base_anchors[None, :, :] + shifts[:, None, :]
    return anchors.reshape(-1, 4)


def encode_boxes(
    anchors: torch.Tensor,
    target_boxes: torch.Tensor,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Encode target boxes as deltas relative to anchors."""
    wx, wy, ww, wh = weights
    ax, ay, aw, ah = _centers_and_sizes(anchors)
    tx, ty, tw, th = _centers_and_sizes(target_boxes)

    dx = wx * (tx - ax) / aw
    dy = wy * (ty - ay) / ah
    dw = ww * torch.log(tw / aw)
    dh = wh * torch.log(th / ah)
    return torch.stack((dx, dy, dw, dh), dim=-1)


def apply_deltas_to_anchors(
    deltas: torch.Tensor,
    anchors: torch.Tensor,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    scale_clamp: float = math.log(1000.0 / 16.0),
) -> torch.Tensor:
    """Decode predicted deltas back to XYXY boxes."""
    wx, wy, ww, wh = weights
    ax, ay, aw, ah = _centers_and_sizes(anchors)

    dx = deltas[..., 0] / wx
    dy = deltas[..., 1] / wy
    dw = (deltas[..., 2] / ww).clamp(max=scale_clamp)
    dh = (deltas[..., 3] / wh).clamp(max=scale_clamp)

    pred_cx = dx * aw + ax
    pred_cy = dy * ah + ay
    pred_w = torch.exp(dw) * aw
    pred_h = torch.exp(dh) * ah

    x1 = pred_cx - 0.5 * pred_w
    y1 = pred_cy - 0.5 * pred_h
    x2 = pred_cx + 0.5 * pred_w
    y2 = pred_cy + 0.5 * pred_h
    return torch.stack((x1, y1, x2, y2), dim=-1)


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    return box_iou(boxes1, boxes2)


def clip_boxes_to_image(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    x_coords = boxes[..., 0::2].clamp(min=0.0, max=float(width))
    y_coords = boxes[..., 1::2].clamp(min=0.0, max=float(height))
    return torch.stack((x_coords[..., 0], y_coords[..., 0], x_coords[..., 1], y_coords[..., 1]), dim=-1)


def remove_small_boxes(boxes: torch.Tensor, min_size: float) -> torch.Tensor:
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    return torch.where((ws >= min_size) & (hs >= min_size))[0]


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, level_ids: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    return tv_batched_nms(boxes, scores, level_ids, iou_threshold)