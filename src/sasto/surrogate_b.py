"""Architecture B: a residual 3-D CNN with squeeze-and-excitation gating.

This module exists to test whether the K6 shelf-life and mu-over-sigma results
are properties of the benchmark or of the single dense CNN that produced them.
It is governed by K6_AMENDMENT_07_SECOND_ARCHITECTURE.md, sha256
a23d88e2156f0bc002625d9670d4206fbfeaa474be42e343f5120c2cc27fdb8d, frozen
before any weight here was trained.

It is deliberately NOT in SOURCE_BUNDLE_PATHS. Adding it would change
source_bundle_sha256 and invalidate the provenance of all 3,134 existing
trajectory records. Architecture B reuses the frozen split, the frozen roles,
and the frozen FEA targets; only the network differs.
"""
from __future__ import annotations

from typing import Sequence

SEED_NAMESPACE_B = "sasto-v-g2b-residual-ensemble-v1"
TARGET_NAMES = ("compliance", "max_von_mises", "max_displacement")


class SurrogateBError(ValueError):
    """An architecture-B construction contract was violated."""


def _require_torch():
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as error:  # pragma: no cover
        raise SurrogateBError("PyTorch is required for architecture B") from error
    return torch, nn


def ResidualSurrogateCNN(
    *,
    target_names: Sequence[str] = TARGET_NAMES,
    width: int = 12,
    se_reduction: int = 4,
):
    """Residual 3-D CNN with SE channel attention and a heteroscedastic head.

    Differs from ``DenseSurrogateCNN`` along every axis that matters for the
    comparison: residual blocks instead of a plain stack, BatchNorm instead of
    GroupNorm(1), SE channel attention with no counterpart in A, and a deeper
    head with dropout. Emits ``(mu, dispersion)`` with the same softplus
    parameterization as A and trains under the same Gaussian NLL, so the
    comparison isolates the network.
    """
    torch, nn = _require_torch()
    names = tuple(target_names)
    if not names or len(set(names)) != len(names):
        raise SurrogateBError("model target names must be unique and non-empty")
    if not isinstance(width, int) or isinstance(width, bool) or width < 4:
        raise SurrogateBError("width must be an integer of at least four")
    if not isinstance(se_reduction, int) or se_reduction < 1:
        raise SurrogateBError("se_reduction must be a positive integer")

    class _SE(nn.Module):
        """Squeeze-and-excitation gate over channels."""

        def __init__(self, channels: int) -> None:
            super().__init__()
            hidden = max(2, channels // se_reduction)
            self.fc1 = nn.Linear(channels, hidden)
            self.fc2 = nn.Linear(hidden, channels)

        def forward(self, x):
            w = x.mean(dim=(2, 3, 4))
            w = torch.sigmoid(self.fc2(torch.relu(self.fc1(w))))
            return x * w[:, :, None, None, None]

    class _ResBlock(nn.Module):
        """Two 3x3x3 convolutions with an identity or projected skip.

        ``stride`` downsamples inside the block. 3-D pooling is deliberately
        avoided: neither ``max_pool3d`` nor ``avg_pool3d`` is implemented for
        the MPS backend in torch 2.8, and the CPU fallback is far slower than
        the whole run. Strided convolution is native and keeps the residual
        topology, which is what distinguishes B from A.
        """

        def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
            super().__init__()
            self.conv1 = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm3d(cout)
            self.conv2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm3d(cout)
            self.se = _SE(cout)
            self.skip = (
                nn.Identity() if cin == cout and stride == 1
                else nn.Sequential(nn.Conv3d(cin, cout, 1, stride=stride, bias=False),
                                   nn.BatchNorm3d(cout))
            )
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            h = self.act(self.bn1(self.conv1(x)))
            h = self.se(self.bn2(self.conv2(h)))
            return self.act(h + self.skip(x))

    class _ResidualSurrogateCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            w = width
            self.stem = nn.Sequential(
                nn.Conv3d(2, w, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(w),
                nn.ReLU(inplace=True),
            )
            self.stage1 = _ResBlock(w, w, stride=2)
            self.stage2 = _ResBlock(w, 2 * w, stride=2)
            self.stage3 = nn.Sequential(_ResBlock(2 * w, 4 * w, stride=2),
                                        nn.AdaptiveAvgPool3d(1))
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(4 * w, 4 * w),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(4 * w, 2 * len(names)),
            )
            self.target_names = names

        @property
        def parameter_count(self) -> int:
            return sum(p.numel() for p in self.parameters())

        def forward(self, channels):
            if channels.ndim != 5 or channels.shape[1] != 2:
                raise SurrogateBError("model input must be [batch, 2, 64, 64, 64]")
            h = self.stage3(self.stage2(self.stage1(self.stem(channels))))
            out = self.head(h)
            k = len(self.target_names)
            mu, raw_scale = out[:, :k], out[:, k:]
            # Softplus dispersion, identical parameterization to architecture A,
            # so the Gaussian NLL objective and the sigma semantics match exactly.
            return mu, torch.nn.functional.softplus(raw_scale) + 1e-6

    return _ResidualSurrogateCNN()

