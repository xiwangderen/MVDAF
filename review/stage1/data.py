"""De-identified Stage I manifest loader."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
import numpy as np


class SliceWindowDataset(Dataset):
    """Load a center slice, three-slice window, label, and resized-image box."""

    def __init__(self, manifest: str | Path, augment: bool = False) -> None:
        self.rows = pd.read_csv(manifest)
        self.augment = augment
        required = {
            "slice_path", "prev_path", "next_path", "slice_label",
            "x1", "y1", "x2", "y2",
        }
        missing = required.difference(self.rows.columns)
        if missing:
            raise ValueError(f"Missing Stage I manifest columns: {sorted(missing)}")
        self.spatial = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _load(path: str) -> torch.Tensor:
        array = np.load(path).astype("float32")
        if array.shape != (224, 224):
            raise ValueError(f"Expected [224,224], got {array.shape} for {path}")
        return torch.from_numpy(array)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows.iloc[index]
        center = self._load(row.slice_path).unsqueeze(0)
        window = torch.stack(
            [self._load(row.prev_path), center.squeeze(0), self._load(row.next_path)], dim=0
        )
        label = torch.tensor(float(row.slice_label), dtype=torch.float32)
        box = torch.tensor([[row.x1, row.y1, row.x2, row.y2]], dtype=torch.float32)
        boxes = tv_tensors.BoundingBoxes(box, format="XYXY", canvas_size=(224, 224))
        if self.augment:
            center, window, boxes = self.spatial(center, window, boxes)
            gain = random.uniform(0.9, 1.1)
            bias = random.uniform(-0.05, 0.05)
            center = (center * gain + bias).clamp(0, 1)
            window = (window * gain + bias).clamp(0, 1)
        return {
            "center": center,
            "window": window.unsqueeze(0),
            "slice_label": label,
            "box": boxes.as_subclass(torch.Tensor).squeeze(0),
        }

