"""Final-protocol Stage I model.

The 2D branch supplies C3--C5 features and the 1024-dimensional C4 ROI;
the 3D branch supplies local inter-slice context for the slice classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50

from .fpn import FPNSpatialAttention
from .models.resnet3d import generate_model


@dataclass
class Stage1Output:
    slice_logits: torch.Tensor
    attention_logits: torch.Tensor
    attention: torch.Tensor
    roi: torch.Tensor
    c4: torch.Tensor
    gate: torch.Tensor


class ResNet2DFeatures(nn.Module):
    def __init__(self, imagenet_pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if imagenet_pretrained else None
        backbone = resnet50(weights=weights)
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            1, old_conv.out_channels, old_conv.kernel_size, old_conv.stride,
            old_conv.padding, bias=False
        )
        if imagenet_pretrained:
            with torch.no_grad():
                backbone.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.layer1(self.stem(x))
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


class ResNet3DFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = generate_model(
            model_type="resnet", model_depth=50, resnet_shortcut="B", num_classes=2
        )

    def load_medicalnet(self, checkpoint: str | Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        clean: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            key = key.removeprefix("module.").removeprefix("model.")
            clean[key] = value
        if "conv1.weight" in clean and clean["conv1.weight"].shape[1] == 3:
            clean["conv1.weight"] = clean["conv1.weight"].mean(dim=1, keepdim=True)
        self.backbone.load_state_dict(clean, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x = b.layer4(b.layer3(b.layer2(b.layer1(x))))
        return F.adaptive_avg_pool3d(x, 1).flatten(1)


class GatedSliceClassifier(nn.Module):
    def __init__(self, fusion_dim: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.project_2d = nn.Linear(2048, fusion_dim)
        self.project_3d = nn.Linear(2048, fusion_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(fusion_dim, 1)
        )

    def forward(
        self, g2d: torch.Tensor, g3d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z2 = self.project_2d(g2d)
        z3 = self.project_3d(g3d)
        gate = self.gate(torch.cat([z2, z3], dim=1))
        fused = gate * z2 + (1.0 - gate) * z3
        return self.classifier(fused).squeeze(1), gate


def normalized_attention_pool(
    feature: torch.Tensor, attention: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    attention = F.interpolate(
        attention, size=feature.shape[-2:], mode="bilinear", align_corners=False
    )
    numerator = (feature * attention).sum(dim=(-2, -1))
    denominator = attention.sum(dim=(-2, -1)).clamp_min(eps)
    return numerator / denominator


class Stage1MVDAF(nn.Module):
    """Dual-stream slice scorer with box-supervised FPN attention pooling."""

    def __init__(self, imagenet_pretrained: bool = True) -> None:
        super().__init__()
        self.encoder_2d = ResNet2DFeatures(imagenet_pretrained=imagenet_pretrained)
        self.encoder_3d = ResNet3DFeatures()
        self.attention_head = FPNSpatialAttention()
        self.slice_classifier = GatedSliceClassifier()

    def forward(self, center: torch.Tensor, window: torch.Tensor) -> Stage1Output:
        """Args: center `[B,1,224,224]`; window `[B,1,3,224,224]`."""
        c3, c4, c5 = self.encoder_2d(center)
        g2d = F.adaptive_avg_pool2d(c5, 1).flatten(1)
        g3d = self.encoder_3d(window)
        slice_logits, gate = self.slice_classifier(g2d, g3d)
        attention_logits = self.attention_head(c3, c4, c5)
        attention = torch.sigmoid(attention_logits)
        roi = normalized_attention_pool(c4, attention)
        return Stage1Output(
            slice_logits=slice_logits,
            attention_logits=attention_logits,
            attention=attention,
            roi=roi,
            c4=c4,
            gate=gate,
        )

