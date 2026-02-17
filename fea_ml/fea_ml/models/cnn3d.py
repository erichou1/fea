"""
3D CNN surrogate model for voxel-based FEA prediction (v2).

Architecture improvements over v1:
  - GELU activations (smoother gradients)
  - Squeeze-and-Excitation (SE) channel attention
  - Stochastic depth (drop-path) for regularisation
  - Pre-norm residual blocks
  - Wider prediction head with skip connection
  - Multi-scale global pooling (avg + max)
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D feature maps."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1,1)
        return x * w


class DropPath(nn.Module):
    """Stochastic depth (drop whole residual branch)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device)) / keep
        return x * mask


class ConvBlock3D(nn.Module):
    """Conv3D -> BN -> GELU -> optional pool -> optional dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                       padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        ]
        if pool:
            layers.append(nn.MaxPool3d(kernel_size=2))
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock3D(nn.Module):
    """Pre-norm residual block with SE attention and stochastic depth."""

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.se = SEBlock3D(channels) if use_se else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(x))
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.act(self.bn2(out))
        out = self.conv2(out)
        out = self.se(out)
        out = self.drop_path(out)
        return out + residual


# ---------------------------------------------------------------------------
# Main model: Surrogate3DCNN  (for 64^3 inputs)
# ---------------------------------------------------------------------------

class Surrogate3DCNN(nn.Module):
    """
    3D CNN with SE-ResBlocks, multi-scale pooling, and wide prediction head.

    Supports MC Dropout for uncertainty estimation.
    """

    def __init__(
        self,
        in_channels: int = 7,
        feature_dim: int = 8,
        target_dim: int = 4,
        base_channels: int = 32,
        dropout: float = 0.15,
        drop_path: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout
        c = base_channels

        # Encoder with progressive downsampling
        self.encoder = nn.Sequential(
            ConvBlock3D(in_channels, c, pool=True, dropout=dropout),       # /2
            ConvBlock3D(c, c * 2, pool=True, dropout=dropout),             # /4
            ConvBlock3D(c * 2, c * 4, pool=True, dropout=dropout),         # /8
            ConvBlock3D(c * 4, c * 8, pool=True, dropout=dropout),         # /16
        )

        # Residual refinement with SE + stochastic depth
        if use_residual:
            self.residual = nn.Sequential(
                ResBlock3D(c * 8, dropout=dropout, drop_path=drop_path, use_se=True),
                ResBlock3D(c * 8, dropout=dropout, drop_path=drop_path, use_se=True),
                ResBlock3D(c * 8, dropout=dropout, drop_path=drop_path, use_se=True),
            )
        else:
            self.residual = nn.Identity()

        # Multi-scale pooling (avg + max -> 2x channels)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # Feature MLP (non-spatial inputs)
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # Prediction head with skip connection
        combined_dim = c * 8 * 2 + 128  # avg + max pool + features
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_skip = nn.Linear(combined_dim, 256)  # skip connection
        self.head_out = nn.Linear(256, target_dim)

    def forward(self, voxel: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = self.encoder(voxel)
        x = self.residual(x)

        # Multi-scale pooling
        x_avg = self.avg_pool(x).flatten(1)
        x_max = self.max_pool(x).flatten(1)
        x_pool = torch.cat([x_avg, x_max], dim=1)

        # Features
        f = self.feature_mlp(features)

        # Combine
        combined = torch.cat([x_pool, f], dim=1)

        # Head with skip
        h = self.head(combined)
        h = h + self.head_skip(combined)  # residual
        return self.head_out(h)

    def enable_mc_dropout(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout3d)):
                m.train()

    def disable_mc_dropout(self) -> None:
        self.eval()


# ---------------------------------------------------------------------------
# Larger model: Surrogate3DResNet  (for 128^3 inputs)
# ---------------------------------------------------------------------------

class Surrogate3DResNet(nn.Module):
    """
    ResNet-style 3D CNN for 128^3 voxel inputs.

    Uses gradient checkpointing for memory efficiency.
    """

    def __init__(
        self,
        in_channels: int = 7,
        feature_dim: int = 8,
        target_dim: int = 4,
        base_channels: int = 64,
        dropout: float = 0.15,
        drop_path: float = 0.1,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpointing = use_checkpointing
        c = base_channels

        # Stem: aggressive initial downsample
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(c),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # Stochastic depth schedule (linearly increasing)
        dp_rates = [drop_path * i / 7 for i in range(8)]

        # Residual stages
        self.stage1 = self._make_stage(c, c, 2, stride=1, dropout=dropout,
                                        dp_rates=dp_rates[0:2])
        self.stage2 = self._make_stage(c, c * 2, 2, stride=2, dropout=dropout,
                                        dp_rates=dp_rates[2:4])
        self.stage3 = self._make_stage(c * 2, c * 4, 2, stride=2, dropout=dropout,
                                        dp_rates=dp_rates[4:6])
        self.stage4 = self._make_stage(c * 4, c * 8, 2, stride=2, dropout=dropout,
                                        dp_rates=dp_rates[6:8])

        # Multi-scale pooling
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # Feature MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        combined_dim = c * 8 * 2 + 128
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_skip = nn.Linear(combined_dim, 256)
        self.head_out = nn.Linear(256, target_dim)

    def _make_stage(self, in_ch: int, out_ch: int, n_blocks: int, stride: int,
                     dropout: float, dp_rates: list) -> nn.Sequential:
        layers = []
        if stride > 1 or in_ch != out_ch:
            layers.append(nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.GELU(),
            ))
        for i in range(n_blocks):
            layers.append(ResBlock3D(out_ch, dropout=dropout,
                                     drop_path=dp_rates[i] if i < len(dp_rates) else 0,
                                     use_se=True))
        return nn.Sequential(*layers)

    def forward(self, voxel: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = self.stem(voxel)

        if self.use_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            x = checkpoint(self.stage1, x, use_reentrant=False)
            x = checkpoint(self.stage2, x, use_reentrant=False)
            x = checkpoint(self.stage3, x, use_reentrant=False)
            x = checkpoint(self.stage4, x, use_reentrant=False)
        else:
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)

        # Multi-scale pooling
        x_avg = self.avg_pool(x).flatten(1)
        x_max = self.max_pool(x).flatten(1)
        x_pool = torch.cat([x_avg, x_max], dim=1)

        f = self.feature_mlp(features)
        combined = torch.cat([x_pool, f], dim=1)

        h = self.head(combined)
        h = h + self.head_skip(combined)
        return self.head_out(h)

    def enable_mc_dropout(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout3d)):
                m.train()

    def disable_mc_dropout(self) -> None:
        self.eval()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_surrogate_model(
    in_channels: int,
    feature_dim: int,
    target_dim: int,
    resolution: int = 64,
    dropout: float = 0.15,
    drop_path: float = 0.1,
    backbone: str = "cnn3d",
    base_channels: int | None = None,
) -> nn.Module:
    """
    Factory function to create surrogate model.

    Args:
        in_channels: Voxel input channels
        feature_dim: Non-spatial feature dimension
        target_dim: Number of output targets
        resolution: Voxel grid resolution (64 or 128)
        dropout: Dropout rate
        drop_path: Stochastic depth rate
        backbone: "cnn3d" or "resnet3d"
        base_channels: Override base channel count
    """
    if backbone == "resnet3d" or resolution >= 128:
        c = base_channels or (32 if resolution < 128 else 64)
        return Surrogate3DResNet(
            in_channels=in_channels,
            feature_dim=feature_dim,
            target_dim=target_dim,
            base_channels=c,
            dropout=dropout,
            drop_path=drop_path,
            use_checkpointing=resolution >= 128,
        )
    else:
        c = base_channels or 32
        return Surrogate3DCNN(
            in_channels=in_channels,
            feature_dim=feature_dim,
            target_dim=target_dim,
            base_channels=c,
            dropout=dropout,
            drop_path=drop_path,
            use_residual=True,
        )
