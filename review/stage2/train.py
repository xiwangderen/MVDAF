"""Train a fold- and modality-specific Stage II encoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import silhouette_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import CandidateROIDataset
from .losses import DualQueueSupConCompactness
from .model import ModalityROIEncoder, ProjectionHead


@torch.no_grad()
def geometry_score(
    encoder: ModalityROIEncoder,
    projection: ProjectionHead,
    loader: DataLoader,
    device: torch.device,
) -> float:
    encoder.eval()
    projection.eval()
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for roi, target in loader:
        embeddings.append(projection(encoder(roi.to(device))).cpu().numpy())
        labels.append(target.numpy())
    embedding = np.concatenate(embeddings)
    target = np.concatenate(labels)
    return float(silhouette_score(embedding, target)) if np.unique(target).size == 2 else -1.0


def run(config: dict) -> None:
    torch.manual_seed(int(config.get("seed", 2026)))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    modality = config["modality"]
    threshold = float(config.get("candidate_threshold", 0.75))
    train_set = CandidateROIDataset(config["train_manifest"], modality, threshold)
    val_set = CandidateROIDataset(config["val_manifest"], modality, threshold)
    counts = torch.bincount(train_set.labels, minlength=2).float()
    weights = (1.0 / counts.clamp_min(1))[train_set.labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_set), replacement=True)
    batch_size = int(config.get("batch_size", 256))
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    encoder = ModalityROIEncoder().to(device)
    projection = ProjectionHead().to(device)
    objective = DualQueueSupConCompactness(
        capacity=int(config.get("queue_capacity", 4096)),
        temperature=float(config.get("temperature", 0.07)),
        queue_samples_per_class=(
            int(config["queue_samples_per_class"])
            if config.get("queue_samples_per_class") is not None
            else None
        ),
        malignant_compact_weight=float(config.get("malignant_compact_weight", 1.25)),
    ).to(device)
    optimizer = AdamW(
        [*encoder.parameters(), *projection.parameters()],
        lr=float(config.get("lr", 1e-4)), weight_decay=1e-4,
    )
    epochs = int(config.get("epochs", 200))
    compact_start_epoch = int(config.get("compact_start_epoch", 4))
    target_weight = float(config.get("compact_weight", 0.16))
    best = -np.inf
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        encoder.train()
        projection.train()
        compact_weight = target_weight if epoch + 1 >= compact_start_epoch else 0.0
        for roi, labels in train_loader:
            roi, labels = roi.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            z = projection(encoder(roi))
            loss, _ = objective(z, labels, compact_weight)
            loss.backward()
            optimizer.step()
        score = geometry_score(encoder, projection, val_loader, device)
        if score > best:
            best = score
            torch.save(
                {"encoder": encoder.state_dict(), "projection": projection.state_dict(), "val_geometry": score},
                output / "best.pt",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        run(yaml.safe_load(handle))


if __name__ == "__main__":
    main()

