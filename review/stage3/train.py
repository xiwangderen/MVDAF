"""Train Stage III while keeping the Stage II encoders frozen."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import MODALITIES, PatientBagDataset, collate_patient_bags
from .model import MVDAFPatientClassifier, binary_focal_loss


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    labels: list[float] = []
    probabilities: list[float] = []
    for batch in loader:
        bags = {m: batch["bags"][m].to(device) for m in MODALITIES}
        output = model(bags, batch["lengths"])
        labels.extend(batch["labels"].tolist())
        probabilities.extend(output["probabilities"].cpu().tolist())
    return float(roc_auc_score(labels, probabilities))


def run(config: dict) -> None:
    torch.manual_seed(int(config.get("seed", 2026)))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    train_set = PatientBagDataset(config["train_manifest"])
    val_set = PatientBagDataset(config["val_manifest"])
    batch_size = int(config.get("batch_size", 8))
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_patient_bags
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_patient_bags
    )
    model = MVDAFPatientClassifier(
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_heads=int(config.get("num_heads", 4)),
    )
    model.load_and_freeze_encoders(config["stage2_checkpoints"])
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(parameters, lr=float(config.get("lr", 1e-4)), weight_decay=1e-4)
    best = -np.inf
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(int(config.get("epochs", 40))):
        model.train()
        for batch in train_loader:
            bags = {m: batch["bags"][m].to(device) for m in MODALITIES}
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(bags, batch["lengths"])
            loss = binary_focal_loss(output["logits"], labels, alpha=0.75, gamma=2.0)
            loss.backward()
            optimizer.step()
        val_auc = evaluate(model, val_loader, device)
        if val_auc > best:
            best = val_auc
            torch.save({"model": model.state_dict(), "val_auc": val_auc}, output_dir / "best.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        run(yaml.safe_load(handle))


if __name__ == "__main__":
    main()

