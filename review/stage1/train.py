"""Train one fold-specific Stage I model from a YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .data import SliceWindowDataset
from .losses import gaussian_box_targets, stage1_objective
from .model import Stage1MVDAF


def evaluate(model: Stage1MVDAF, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    labels: list[float] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["center"].to(device), batch["window"].to(device))
            labels.extend(batch["slice_label"].tolist())
            probabilities.extend(torch.sigmoid(output.slice_logits).cpu().tolist())
    return float(roc_auc_score(labels, probabilities))


def run(config: dict) -> None:
    torch.manual_seed(int(config.get("seed", 2026)))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    train_set = SliceWindowDataset(config["train_manifest"], augment=True)
    val_set = SliceWindowDataset(config["val_manifest"], augment=False)
    train_loader = DataLoader(
        train_set, batch_size=int(config.get("batch_size", 64)), shuffle=True,
        num_workers=int(config.get("num_workers", 4)), pin_memory=True,
    )
    val_loader = DataLoader(val_set, batch_size=int(config.get("batch_size", 64)))
    model = Stage1MVDAF(imagenet_pretrained=bool(config.get("imagenet_pretrained", True)))
    if config.get("medicalnet_checkpoint"):
        model.encoder_3d.load_medicalnet(config["medicalnet_checkpoint"])
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=float(config.get("lr", 1e-4)), weight_decay=1e-4)
    epochs = int(config.get("epochs", 200))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_auc = -np.inf
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = float(config.get("gaussian_scale", 0.25))
    max_existence = float(config.get("existence_weight", 0.05))
    warmup = max(int(config.get("existence_warmup_epochs", 30)), 1)
    for epoch in range(epochs):
        model.train()
        existence_weight = max_existence * min((epoch + 1) / warmup, 1.0)
        for batch in train_loader:
            center = batch["center"].to(device)
            window = batch["window"].to(device)
            labels = batch["slice_label"].to(device)
            boxes = batch["box"].to(device)
            targets = gaussian_box_targets(boxes, labels > 0.5, scale=scale)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(center, window)
                loss, _ = stage1_objective(
                    output.slice_logits, output.attention_logits, labels, targets,
                    alpha=0.5, beta=1.5, negative_patch_weight=0.3,
                    existence_weight=existence_weight,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        val_auc = evaluate(model, val_loader, device)
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(), "val_auc": val_auc}, output_dir / "best.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        run(yaml.safe_load(handle))


if __name__ == "__main__":
    main()

