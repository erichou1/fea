from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import meshio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def normalize_points(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    points = points - center
    scale = np.max(np.linalg.norm(points, axis=1)) + 1e-6
    return points / scale


def sample_points(points: np.ndarray, num_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if points.shape[0] >= num_points:
        idx = rng.choice(points.shape[0], size=num_points, replace=False)
    else:
        idx = rng.choice(points.shape[0], size=num_points, replace=True)
    return points[idx]


@dataclass
class NormalizationStats:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


class FEADataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        target_names: Tuple[str, ...],
        material_types: Tuple[str, ...],
        load_cases: Tuple[str, ...],
        stats: Optional[NormalizationStats] = None,
        num_points: int = 2048,
        normalize: bool = True,
        seed: int = 0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.target_names = target_names
        self.material_types = material_types
        self.load_cases = load_cases
        self.records = self._load_manifest(self.manifest_path)
        self.stats = stats
        self.num_points = num_points
        self.normalize = normalize
        self.seed = seed

    def _load_manifest(self, path: Path) -> List[Dict[str, object]]:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.suffix == ".json":
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                return payload["samples"]
            return payload
        if path.suffix in {".csv", ".tsv"}:
            sep = "	" if path.suffix == ".tsv" else ","
            return pd.read_csv(path, sep=sep).to_dict(orient="records")
        raise ValueError(f"Unsupported manifest extension: {path.suffix}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        points = self._load_points(record, idx)
        feature_vec = self._build_feature_vector(record)
        targets = self._load_targets(record)

        if self.stats:
            feature_vec = (feature_vec - self.stats.feature_mean) / self.stats.feature_std
            targets = (targets - self.stats.target_mean) / self.stats.target_std

        return {
            "points": torch.from_numpy(points).float(),
            "features": torch.from_numpy(feature_vec).float(),
            "targets": torch.from_numpy(targets).float(),
        }

    def _load_points(self, record: Dict[str, object], idx: int) -> np.ndarray:
        mesh_path = Path(record["mesh_path"])
        mesh = meshio.read(mesh_path)
        points = mesh.points.astype(np.float32)
        if points.shape[1] > 3:
            points = points[:, :3]
        if self.normalize:
            points = normalize_points(points)
        return sample_points(points, self.num_points, self.seed + idx)

    def _build_feature_vector(self, record: Dict[str, object]) -> np.ndarray:
        material_props = np.array(
            [
                record["youngs_modulus"],
                record["poisson_ratio"],
                record["density"],
                record["yield_stress"],
            ],
            dtype=np.float32,
        )
        material_onehot = np.zeros(len(self.material_types), dtype=np.float32)
        material_label = record["material_type"]
        material_onehot[self.material_types.index(material_label)] = 1.0

        load_onehot = np.zeros(len(self.load_cases), dtype=np.float32)
        load_label = record["load_case"]
        load_onehot[self.load_cases.index(load_label)] = 1.0

        return np.concatenate([material_props, material_onehot, load_onehot], axis=0)

    def _load_targets(self, record: Dict[str, object]) -> np.ndarray:
        return np.array([record[name] for name in self.target_names], dtype=np.float32)


def compute_normalization_stats(
    dataset: FEADataset,
) -> NormalizationStats:
    feature_list = []
    target_list = []
    for idx in range(len(dataset)):
        record = dataset.records[idx]
        feature_list.append(dataset._build_feature_vector(record))
        target_list.append(dataset._load_targets(record))
    features = np.stack(feature_list, axis=0)
    targets = np.stack(target_list, axis=0)
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0) + 1e-6
    target_mean = targets.mean(axis=0)
    target_std = targets.std(axis=0) + 1e-6
    return NormalizationStats(
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
    )
