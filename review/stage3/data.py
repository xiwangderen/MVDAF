"""Variable-length three-sequence patient bags for Stage III."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MODALITIES = ("T1A", "T2A", "T2C")


class PatientBagDataset(Dataset):
    def __init__(self, manifest: str | Path) -> None:
        frame = pd.read_csv(manifest)
        self.patients: list[dict] = []
        for subject_key, rows in frame.groupby("subject_key", sort=False):
            by_modality = {row.modality: row.feature_path for row in rows.itertuples()}
            if any(modality not in by_modality for modality in MODALITIES):
                continue
            payloads = {m: np.load(by_modality[m]) for m in MODALITIES}
            labels = {int(payloads[m]["label"]) for m in MODALITIES}
            if len(labels) != 1:
                raise ValueError(f"Inconsistent labels for {subject_key}")
            self.patients.append(
                {
                    "bags": {m: torch.from_numpy(payloads[m]["roi"].astype("float32")) for m in MODALITIES},
                    "label": labels.pop(),
                }
            )

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, index: int) -> dict:
        return self.patients[index]


def collate_patient_bags(batch: list[dict]) -> dict:
    bags: dict[str, torch.Tensor] = {}
    lengths: dict[str, list[int]] = {}
    for modality in MODALITIES:
        pieces = [patient["bags"][modality] for patient in batch]
        bags[modality] = torch.cat(pieces, dim=0)
        lengths[modality] = [piece.shape[0] for piece in pieces]
    return {
        "bags": bags,
        "lengths": lengths,
        "labels": torch.tensor([patient["label"] for patient in batch], dtype=torch.float32),
    }

