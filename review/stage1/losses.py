"""Box-supervised Stage I targets and objective."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def gaussian_box_targets(
    boxes_xyxy: torch.Tensor,
    positive: torch.Tensor,
    grid_size: int = 28,
    image_size: int = 224,
    scale: float = 0.25,
) -> torch.Tensor:
    """Convert resized-image boxes to smooth grid targets; negatives remain zero."""
    device = boxes_xyxy.device
    coords = torch.arange(grid_size, device=device, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    targets = torch.zeros((boxes_xyxy.shape[0], 1, grid_size, grid_size), device=device)
    grid_scale = grid_size / image_size
    for index, (box, is_positive) in enumerate(zip(boxes_xyxy, positive)):
        if not bool(is_positive):
            continue
        x1, y1, x2, y2 = box * grid_scale
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        sx = ((x2 - x1).clamp_min(1.0) * scale).clamp_min(1.0)
        sy = ((y2 - y1).clamp_min(1.0) * scale).clamp_min(1.0)
        targets[index, 0] = torch.exp(
            -0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)
        )
    return targets


def stage1_objective(
    slice_logits: torch.Tensor,
    attention_logits: torch.Tensor,
    slice_labels: torch.Tensor,
    attention_targets: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 1.5,
    negative_patch_weight: float = 0.3,
    existence_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    labels = slice_labels.float()
    slice_loss = F.binary_cross_entropy_with_logits(slice_logits, labels)
    attention = torch.sigmoid(attention_logits)
    sample_weights = torch.where(labels > 0.5, 1.0, negative_patch_weight).view(-1, 1, 1, 1)
    patch_loss = (sample_weights * (attention - attention_targets).square()).mean()
    positive = labels > 0.5
    if positive.any():
        max_response = attention[positive].flatten(1).amax(dim=1)
        existence_loss = (1.0 - max_response).mean()
    else:
        existence_loss = attention.sum() * 0.0
    total = alpha * slice_loss + beta * patch_loss + existence_weight * existence_loss
    return total, {
        "slice": slice_loss.detach(),
        "patch": patch_loss.detach(),
        "existence": existence_loss.detach(),
    }

