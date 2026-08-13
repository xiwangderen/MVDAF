"""Modality-specific encoder and training-only projection head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ModalityROIEncoder(nn.Module):
    """Residual 1024-D ROI encoder retained for Stage III."""

    def __init__(self, dim: int = 1024, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_norm = nn.BatchNorm1d(dim)
        self.enhancer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, roi: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(roi)
        return self.activation(normalized + self.enhancer(normalized))


class ProjectionHead(nn.Module):
    """Training-only 1024 -> 512 -> 128 projection."""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, output_dim: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(encoded), dim=1)

