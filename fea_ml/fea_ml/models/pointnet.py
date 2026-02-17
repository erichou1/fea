from __future__ import annotations

import torch
from torch import nn


class SurrogatePointNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        target_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, target_dim),
        )

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = points.transpose(1, 2)
        x = self.mlp(x)
        x = torch.max(x, dim=2).values
        f = self.feature_mlp(features)
        x = torch.cat([x, f], dim=1)
        return self.head(x)
