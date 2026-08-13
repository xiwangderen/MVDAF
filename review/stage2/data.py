"""Feature-manifest datasets for Stage II."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def select_candidates(p_l: np.ndarray, threshold: float = 0.75) -> np.ndarray:
    selected = np.flatnonzero(p_l >= threshold)
    if selected.size == 0:
        selected = np.asarray([int(np.argmax(p_l))])
    return selected


class CandidateROIDataset(Dataset):
    """Flatten score-selected ROI vectors from a de-identified feature manifest."""

    def __init__(self, manifest: str | Path, modality: str, threshold: float = 0.75) -> None:
        rows = pd.read_csv(manifest)
        rows = rows[rows.modality == modality]
        features: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for path in rows.feature_path:
            payload = np.load(path)
            indices = select_candidates(payload["p_l"], threshold)
            features.append(payload["roi"][indices].astype("float32"))
            labels.append(np.full(indices.size, int(payload["label"]), dtype="int64"))
        if not features:
            raise ValueError(f"No feature files found for modality {modality}")
        self.features = torch.from_numpy(np.concatenate(features))
        self.labels = torch.from_numpy(np.concatenate(labels))

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]

