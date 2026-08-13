"""Export fold-specific Stage I ROI vectors and slice tumor probabilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .data import SliceWindowDataset
from .model import Stage1MVDAF


def run(config: dict) -> None:
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    manifest = pd.read_csv(config["manifest"])
    dataset = SliceWindowDataset(config["manifest"], augment=False)
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 64)), shuffle=False)
    model = Stage1MVDAF(imagenet_pretrained=False)
    payload = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model", payload))
    model.to(device).eval()
    rois: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["center"].to(device), batch["window"].to(device))
            rois.append(output.roi.cpu().numpy())
            scores.append(torch.sigmoid(output.slice_logits).cpu().numpy())
    roi = np.concatenate(rois)
    p_l = np.concatenate(scores)
    destination = Path(config["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    for subject_key, indices in manifest.groupby("subject_key", sort=False).indices.items():
        rows = manifest.iloc[indices]
        label = int(rows.patient_label.iloc[0])
        modality = str(rows.modality.iloc[0])
        np.savez_compressed(
            destination / f"{subject_key}_{modality}.npz",
            roi=roi[indices].astype("float32"),
            p_l=p_l[indices].astype("float32"),
            label=np.int64(label),
            slice_index=rows.slice_index.to_numpy(dtype="int64"),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        run(yaml.safe_load(handle))


if __name__ == "__main__":
    main()

