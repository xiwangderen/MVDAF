"""Frozen slice encoders, attention MIL, and patient-specific cross-attention."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from review.stage2.model import ModalityROIEncoder

MODALITIES = ("T1A", "T2A", "T2C")


class AttentionMIL(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 128) -> None:
        super().__init__()
        self.value = nn.Linear(input_dim, hidden_dim)
        self.attention_v = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(
        self, features: torch.Tensor, lengths: list[int]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        representations: list[torch.Tensor] = []
        reliance: list[torch.Tensor] = []
        offset = 0
        for length in lengths:
            bag = features[offset : offset + length]
            offset += length
            scores = self.attention(self.attention_v(bag) * self.attention_u(bag)).squeeze(1)
            weights = torch.softmax(scores, dim=0)
            representations.append((weights[:, None] * self.value(bag)).sum(dim=0))
            reliance.append(weights)
        return torch.stack(representations), reliance


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int = 128, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.modality_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim)
        )

    def forward(self, modalities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query.expand(modalities.shape[0], -1, -1)
        attended, weights = self.attention(
            self.query_norm(query),
            self.modality_norm(modalities),
            self.modality_norm(modalities),
            need_weights=True,
            average_attn_weights=True,
        )
        residual = query + attended
        fused = residual + self.ffn(self.output_norm(residual))
        return fused.squeeze(1), weights.squeeze(1)


class MVDAFPatientClassifier(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict({m: ModalityROIEncoder() for m in MODALITIES})
        self.aggregators = nn.ModuleDict({m: AttentionMIL(hidden_dim=hidden_dim) for m in MODALITIES})
        self.fusion = CrossAttentionFusion(hidden_dim, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(64, 1)
        )

    def load_and_freeze_encoders(self, checkpoints: Mapping[str, str]) -> None:
        for modality in MODALITIES:
            payload = torch.load(checkpoints[modality], map_location="cpu", weights_only=False)
            self.encoders[modality].load_state_dict(payload["encoder"])
            self.encoders[modality].eval()
            for parameter in self.encoders[modality].parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        for encoder in self.encoders.values():
            encoder.eval()
        return self

    def forward(
        self,
        bags: Mapping[str, torch.Tensor],
        lengths: Mapping[str, list[int]],
    ) -> dict:
        modality_features: list[torch.Tensor] = []
        slice_weights: dict[str, list[torch.Tensor]] = {}
        for modality in MODALITIES:
            with torch.no_grad():
                encoded = self.encoders[modality](bags[modality])
            patient_features, weights = self.aggregators[modality](encoded, lengths[modality])
            modality_features.append(patient_features)
            slice_weights[modality] = weights
        stacked = torch.stack(modality_features, dim=1)
        fused, modality_weights = self.fusion(stacked)
        logits = self.classifier(fused).squeeze(1)
        return {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "slice_weights": slice_weights,
            "modality_weights": modality_weights,
            "fused": fused,
        }


def binary_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability_true = torch.exp(-bce)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - probability_true).pow(gamma) * bce).mean()

