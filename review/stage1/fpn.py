"""Feature-pyramid spatial attention head used in Stage I."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FPNSpatialAttention(nn.Module):
    """Fuse ResNet C3--C5 features and predict 28x28 attention logits."""

    def __init__(self, fpn_dim: int = 256) -> None:
        super().__init__()
        self.lateral3 = nn.Conv2d(512, fpn_dim, 1)
        self.lateral4 = nn.Conv2d(1024, fpn_dim, 1)
        self.lateral5 = nn.Conv2d(2048, fpn_dim, 1)
        self.smooth4 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.prediction = nn.Sequential(
            nn.Conv2d(3 * fpn_dim, fpn_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(
        self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor
    ) -> torch.Tensor:
        p5 = self.lateral5(c5)
        p4 = self.smooth4(
            self.lateral4(c4)
            + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        )
        p3 = self.smooth3(
            self.lateral3(c3)
            + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        )
        pyramid = torch.cat(
            [
                p3,
                F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(p5, size=p3.shape[-2:], mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        return self.prediction(pyramid)

