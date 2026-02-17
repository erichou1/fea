from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MeshModifierConfig:
    scale_min: float = 0.7
    scale_max: float = 1.0


class MeshModifier:
    def __init__(self, config: MeshModifierConfig) -> None:
        self.config = config

    def parameter_dim(self) -> int:
        return 3

    def apply(self, points: np.ndarray, params: np.ndarray) -> np.ndarray:
        scales = self._scales(params)
        center = points.mean(axis=0, keepdims=True)
        scaled = (points - center) * scales + center
        return scaled.astype(np.float32)

    def _scales(self, params: np.ndarray) -> np.ndarray:
        params = np.clip(params[:3], 0.0, 1.0)
        scale = self.config.scale_min + params * (self.config.scale_max - self.config.scale_min)
        return scale.astype(np.float32)

    def volume_proxy(self, points: np.ndarray) -> float:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        extent = np.maximum(maxs - mins, 1e-6)
        return float(np.prod(extent))
