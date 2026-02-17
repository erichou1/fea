"""
Voxel-based dataset for FEA surrogate training.

Loads voxel grids (occupancy, SDF, part labels, masks) and FEA targets
from the standardized data/runs/<run_id>/ directory structure.

Normalization strategy (v2):
  ALL targets are log1p-transformed before z-score normalizing.
  Targets span 5–12 orders of magnitude; without log transform the loss
  is dominated by extreme outliers and the model learns nothing useful.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class VoxelNormalizationStats:
    """Normalization statistics for features and targets.

    After log1p transform and winsorisation the stats are:
      z = (log1p(clamp(raw)) - target_mean) / target_std
    """
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray       # computed in log1p space
    target_std: np.ndarray        # computed in log1p space
    # Which targets get log transform (now default=all)
    log_transform_targets: List[str] = field(default_factory=list)
    # Winsorization bounds (in log1p space)
    target_clip_low: Optional[np.ndarray] = None
    target_clip_high: Optional[np.ndarray] = None
    # Robust stats for reference
    target_median: Optional[np.ndarray] = None
    target_iqr: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
            "log_transform_targets": self.log_transform_targets,
            "target_clip_low": self.target_clip_low.tolist() if self.target_clip_low is not None else None,
            "target_clip_high": self.target_clip_high.tolist() if self.target_clip_high is not None else None,
            "target_median": self.target_median.tolist() if self.target_median is not None else None,
            "target_iqr": self.target_iqr.tolist() if self.target_iqr is not None else None,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "VoxelNormalizationStats":
        """Load from dict."""
        return cls(
            feature_mean=np.array(d["feature_mean"], dtype=np.float32),
            feature_std=np.array(d["feature_std"], dtype=np.float32),
            target_mean=np.array(d["target_mean"], dtype=np.float32),
            target_std=np.array(d["target_std"], dtype=np.float32),
            log_transform_targets=d.get("log_transform_targets", []),
            target_clip_low=np.array(d["target_clip_low"]) if d.get("target_clip_low") else None,
            target_clip_high=np.array(d["target_clip_high"]) if d.get("target_clip_high") else None,
            target_median=np.array(d["target_median"]) if d.get("target_median") else None,
            target_iqr=np.array(d["target_iqr"]) if d.get("target_iqr") else None,
        )


# Part label constants (same as voxelize.py)
NUM_PARTS = 6  # 0=empty, 1-5 for parts


class VoxelFEADataset(Dataset):
    """
    Dataset for voxel-based FEA surrogate training.
    
    Each sample is a directory containing:
    - occ.npz: occupancy grid (D,H,W) uint8
    - sdf.npz: SDF grid (D,H,W) float32 (optional)
    - part.npz: part labels (D,H,W) uint8
    - edit_mask.npz: editable regions (D,H,W) uint8
    - protected_mask.npz: protected regions (D,H,W) uint8
    - meta.json: material/load case metadata
    - targets.json: FEA scalar outputs
    """
    
    def __init__(
        self,
        run_dirs: List[Path],
        target_names: Tuple[str, ...],
        material_types: Tuple[str, ...],
        load_cases: Tuple[str, ...],
        resolution: int = 64,
        use_sdf: bool = False,
        stats: Optional[VoxelNormalizationStats] = None,
        augment: bool = False,
    ):
        """
        Args:
            run_dirs: List of paths to run directories
            target_names: Names of FEA targets to predict
            material_types: List of material type strings for encoding
            load_cases: List of load case IDs for encoding
            resolution: Expected voxel grid resolution
            use_sdf: Whether to include SDF as input channel
            stats: Normalization statistics (None to skip normalization)
            augment: Whether to apply data augmentation
        """
        self.run_dirs = [Path(d) for d in run_dirs]
        self.target_names = target_names
        self.material_types = material_types
        self.load_cases = load_cases
        self.resolution = resolution
        self.use_sdf = use_sdf
        self.stats = stats
        self.augment = augment
        
        # Validate directories exist
        self.valid_runs = []
        for d in self.run_dirs:
            if (d / "occ.npz").exists() and (d / "targets.json").exists():
                self.valid_runs.append(d)
        
        if len(self.valid_runs) < len(self.run_dirs):
            print(f"Warning: {len(self.run_dirs) - len(self.valid_runs)} invalid run directories skipped")
    
    def __len__(self) -> int:
        return len(self.valid_runs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        run_dir = self.valid_runs[idx]
        
        # Load voxel grids
        occ = np.load(run_dir / "occ.npz")["data"].astype(np.float32)
        part = np.load(run_dir / "part.npz")["data"]
        
        # Optional SDF
        sdf = None
        if self.use_sdf:
            sdf_path = run_dir / "sdf.npz"
            if sdf_path.exists():
                sdf = np.load(sdf_path)["data"].astype(np.float32)
        
        # Load masks (for reference, may be used in optimization)
        edit_mask = np.load(run_dir / "edit_mask.npz")["data"]
        protected_mask = np.load(run_dir / "protected_mask.npz")["data"]
        
        # Load metadata
        with open(run_dir / "meta.json", "r") as f:
            meta = json.load(f)
        
        # Load targets
        with open(run_dir / "targets.json", "r") as f:
            targets_dict = json.load(f)
        
        # Build input tensor: [occ, part_onehot, (optional sdf)]
        voxel_input = self._build_voxel_input(occ, part, sdf)
        
        # Build feature vector (material + load case)
        features = self._build_feature_vector(meta)
        
        # Build target vector
        targets = self._build_targets(targets_dict)
        
        # Apply augmentation
        if self.augment:
            voxel_input = self._augment(voxel_input)
        
        # Normalize features and targets
        if self.stats is not None:
            features = (features - self.stats.feature_mean) / (self.stats.feature_std + 1e-8)
            
            # Apply log1p transform to specified targets (should be ALL of them)
            for i, name in enumerate(self.target_names):
                if name in self.stats.log_transform_targets:
                    targets[i] = np.log1p(np.abs(targets[i]))
            
            # Clip outliers (in log space)
            if self.stats.target_clip_low is not None:
                targets = np.maximum(targets, self.stats.target_clip_low)
            if self.stats.target_clip_high is not None:
                targets = np.minimum(targets, self.stats.target_clip_high)
            
            # Z-score normalization (in log space)
            targets = (targets - self.stats.target_mean) / (self.stats.target_std + 1e-8)
        
        return {
            "voxel": torch.from_numpy(voxel_input).float(),
            "features": torch.from_numpy(features).float(),
            "targets": torch.from_numpy(targets).float(),
            "run_id": run_dir.name,
        }
    
    def _build_voxel_input(
        self,
        occ: np.ndarray,
        part: np.ndarray,
        sdf: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Build multi-channel voxel input.
        
        Channels: [occ, part_0, part_1, ..., part_5, (sdf)]
        """
        channels = [occ[None, ...]]  # (1, D, H, W)
        
        # One-hot encode part labels
        for p in range(NUM_PARTS):
            part_channel = (part == p).astype(np.float32)[None, ...]
            channels.append(part_channel)
        
        # Optional SDF
        if sdf is not None:
            # Normalize SDF to roughly [-1, 1]
            sdf_norm = np.tanh(sdf / 10.0)
            channels.append(sdf_norm[None, ...])
        
        return np.concatenate(channels, axis=0)
    
    def _build_feature_vector(self, meta: dict) -> np.ndarray:
        """Build non-spatial feature vector from metadata."""
        # Material properties (normalized to reasonable scale)
        youngs = meta.get("youngs_modulus", meta.get("E", 2e11)) / 1e11
        poisson = meta.get("poisson_ratio", meta.get("nu", 0.3))
        density = meta.get("density", 2400.0) / 1000.0
        yield_stress = meta.get("yield_stress", 30e6) / 1e7
        
        material_props = np.array([youngs, poisson, density, yield_stress], dtype=np.float32)
        
        # Material type one-hot
        material_label = meta.get("material_type", meta.get("material_label", "concrete"))
        material_onehot = np.zeros(len(self.material_types), dtype=np.float32)
        if material_label in self.material_types:
            material_onehot[self.material_types.index(material_label)] = 1.0
        else:
            # Unknown material - use first as default
            material_onehot[0] = 1.0
        
        # Load case one-hot
        load_case = str(meta.get("load_case_id", meta.get("load_case", "case_a")))
        load_onehot = np.zeros(len(self.load_cases), dtype=np.float32)
        if load_case in self.load_cases:
            load_onehot[self.load_cases.index(load_case)] = 1.0
        else:
            load_onehot[0] = 1.0
        
        return np.concatenate([material_props, material_onehot, load_onehot])
    
    def _build_targets(self, targets_dict: dict) -> np.ndarray:
        """Build target vector from targets dict."""
        targets = []
        for name in self.target_names:
            value = targets_dict.get(name, 0.0)
            targets.append(float(value))
        return np.array(targets, dtype=np.float32)
    
    def _augment(self, voxel: np.ndarray) -> np.ndarray:
        """Apply random augmentation (rotations, flips, noise).

        Augmentations that preserve structural meaning:
        - 90° rotations around vertical (Z) axis
        - Horizontal flips (mirror symmetry)
        - Small additive Gaussian noise on continuous channels
        - Random channel dropout (zero a part channel with low prob)
        """
        # Random 90-degree rotations around Z axis (axes 2,3 = H,W)
        k = np.random.randint(4)
        if k > 0:
            voxel = np.rot90(voxel, k=k, axes=(2, 3))
        
        # Random flips (horizontal)
        if np.random.rand() > 0.5:
            voxel = np.flip(voxel, axis=2)
        if np.random.rand() > 0.5:
            voxel = np.flip(voxel, axis=3)
        
        # Small Gaussian noise on all channels (std=0.02)
        noise = np.random.normal(0, 0.02, voxel.shape).astype(np.float32)
        voxel = voxel + noise
        
        # Random channel dropout: zero out one part one-hot channel with 10% prob
        if np.random.rand() < 0.1 and voxel.shape[0] > 2:
            ch = np.random.randint(1, min(voxel.shape[0], 7))  # skip occ channel
            voxel[ch] = 0.0
        
        return voxel.copy()
    
    def get_feature_dim(self) -> int:
        """Get dimension of non-spatial feature vector."""
        return 4 + len(self.material_types) + len(self.load_cases)
    
    def get_voxel_channels(self) -> int:
        """Get number of voxel input channels."""
        channels = 1 + NUM_PARTS  # occ + part one-hot
        if self.use_sdf:
            channels += 1
        return channels


def compute_voxel_normalization_stats(
    dataset: VoxelFEADataset,
    log_transform_targets: Optional[List[str]] = None,
    winsorize_percentile: float = 2.0,
) -> VoxelNormalizationStats:
    """
    Compute normalization statistics from a dataset.
    
    Strategy (v2):
      1. log1p-transform every target marked in log_transform_targets
         (default = ALL targets, since FEA outputs span many orders of magnitude)
      2. Winsorise at *winsorize_percentile* / (100 - winsorize_percentile)
      3. Compute mean/std on the winsorised log-transformed values
    
    Args:
        dataset: Dataset to compute stats from
        log_transform_targets: Target names to apply log1p transform.
            Pass None to log-transform ALL targets (recommended).
        winsorize_percentile: Percentile for outlier clipping (e.g., 2.0 = 2nd/98th)
        
    Returns:
        VoxelNormalizationStats
    """
    # Default: log-transform ALL targets
    if log_transform_targets is None:
        log_transform_targets = list(dataset.target_names)
    
    feature_list = []
    target_list = []
    
    # Temporarily disable normalization to get raw values
    original_stats = dataset.stats
    dataset.stats = None
    
    for idx in range(len(dataset)):
        run_dir = dataset.valid_runs[idx]
        
        # Load metadata
        with open(run_dir / "meta.json", "r") as f:
            meta = json.load(f)
        with open(run_dir / "targets.json", "r") as f:
            targets_dict = json.load(f)
        
        features = dataset._build_feature_vector(meta)
        targets = dataset._build_targets(targets_dict)
        
        # Apply log1p transform BEFORE computing stats
        for i, name in enumerate(dataset.target_names):
            if name in log_transform_targets:
                targets[i] = np.log1p(np.abs(targets[i]))
        
        feature_list.append(features)
        target_list.append(targets)
    
    dataset.stats = original_stats
    
    features = np.stack(feature_list, axis=0)
    targets = np.stack(target_list, axis=0)
    
    # Compute winsorization bounds (in log space)
    target_clip_low = np.percentile(targets, winsorize_percentile, axis=0)
    target_clip_high = np.percentile(targets, 100 - winsorize_percentile, axis=0)
    
    # Clip for stats computation
    targets_clipped = np.clip(targets, target_clip_low, target_clip_high)
    
    # Robust statistics
    target_median = np.median(targets_clipped, axis=0).astype(np.float32)
    q75 = np.percentile(targets_clipped, 75, axis=0)
    q25 = np.percentile(targets_clipped, 25, axis=0)
    target_iqr = (q75 - q25).astype(np.float32)
    
    return VoxelNormalizationStats(
        feature_mean=features.mean(axis=0).astype(np.float32),
        feature_std=features.std(axis=0).astype(np.float32) + 1e-8,
        target_mean=targets_clipped.mean(axis=0).astype(np.float32),
        target_std=targets_clipped.std(axis=0).astype(np.float32) + 1e-8,
        log_transform_targets=log_transform_targets,
        target_clip_low=target_clip_low.astype(np.float32),
        target_clip_high=target_clip_high.astype(np.float32),
        target_median=target_median,
        target_iqr=target_iqr,
    )


def load_run_dirs_from_manifest(manifest_path: Path) -> List[Path]:
    """Load list of run directories from a manifest file."""
    with open(manifest_path, "r") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return [Path(d) for d in data]
    elif "runs" in data:
        return [Path(d) for d in data["runs"]]
    else:
        raise ValueError(f"Invalid manifest format: {manifest_path}")


def create_data_splits(
    runs_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    split_by_family: bool = True,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Create train/val/test splits from run directories.
    
    Args:
        runs_dir: Directory containing run subdirectories
        train_ratio, val_ratio, test_ratio: Split ratios
        seed: Random seed
        split_by_family: If True, split by design family (prefix) to avoid leakage
        
    Returns:
        (train_dirs, val_dirs, test_dirs) tuples
    """
    np.random.seed(seed)
    
    # Find all valid run directories
    all_runs = sorted([
        d for d in runs_dir.iterdir()
        if d.is_dir() and (d / "occ.npz").exists()
    ])
    
    if not all_runs:
        raise ValueError(f"No valid runs found in {runs_dir}")
    
    if split_by_family:
        # Group runs by design family (assume prefix before underscore)
        families: Dict[str, List[Path]] = {}
        for run in all_runs:
            # Extract family from run name (e.g., "house_001_v1" -> "house_001")
            parts = run.name.split("_")
            if len(parts) >= 2:
                family = "_".join(parts[:-1])
            else:
                family = run.name
            
            if family not in families:
                families[family] = []
            families[family].append(run)
        
        # Split families
        family_names = list(families.keys())
        np.random.shuffle(family_names)
        
        n_train = int(len(family_names) * train_ratio)
        n_val = int(len(family_names) * val_ratio)
        
        train_families = family_names[:n_train]
        val_families = family_names[n_train:n_train + n_val]
        test_families = family_names[n_train + n_val:]
        
        train_dirs = [r for f in train_families for r in families[f]]
        val_dirs = [r for f in val_families for r in families[f]]
        test_dirs = [r for f in test_families for r in families[f]]
    else:
        # Simple random split
        np.random.shuffle(all_runs)
        
        n_train = int(len(all_runs) * train_ratio)
        n_val = int(len(all_runs) * val_ratio)
        
        train_dirs = all_runs[:n_train]
        val_dirs = all_runs[n_train:n_train + n_val]
        test_dirs = all_runs[n_train + n_val:]
    
    return train_dirs, val_dirs, test_dirs
