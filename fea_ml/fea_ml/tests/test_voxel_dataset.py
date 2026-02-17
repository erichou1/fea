"""Tests for voxel dataset loading."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fea_ml.data.voxel_dataset import (
    VoxelFEADataset,
    VoxelNormalizationStats,
    compute_voxel_normalization_stats,
    create_data_splits,
)


@pytest.fixture
def sample_run_dir():
    """Create a temporary run directory with sample data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "sample_001"
        run_dir.mkdir()
        
        resolution = 32
        
        # Create sample voxel data
        occ = np.random.randint(0, 2, (resolution, resolution, resolution), dtype=np.uint8)
        part = np.random.randint(0, 6, (resolution, resolution, resolution), dtype=np.uint8)
        edit_mask = (part > 0).astype(np.uint8)
        protected_mask = np.zeros_like(occ)
        
        np.savez_compressed(run_dir / "occ.npz", data=occ)
        np.savez_compressed(run_dir / "part.npz", data=part)
        np.savez_compressed(run_dir / "edit_mask.npz", data=edit_mask)
        np.savez_compressed(run_dir / "protected_mask.npz", data=protected_mask)
        
        # Create metadata
        meta = {
            "youngs_modulus": 2e11,
            "poisson_ratio": 0.3,
            "density": 2400.0,
            "yield_stress": 30e6,
            "material_type": "concrete",
            "load_case_id": "gravity",
            "length_unit": "meters",
            "baseline_run_id": "sample_000",
        }
        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f)
        
        # Create targets
        targets = {
            "max_von_mises": 15.5e6,
            "max_displacement": 0.002,
            "min_safety_factor": 2.1,
            "compliance": 0.05,
        }
        with open(run_dir / "targets.json", "w") as f:
            json.dump(targets, f)
        
        # Create voxel metadata
        voxel_meta = {
            "bounds_min": [0, 0, 0],
            "bounds_max": [10, 10, 10],
            "voxel_size": 10 / resolution,
            "resolution": resolution,
        }
        with open(run_dir / "voxel_meta.json", "w") as f:
            json.dump(voxel_meta, f)
        
        yield run_dir


def test_dataset_loading(sample_run_dir):
    """Test that dataset loads correctly."""
    dataset = VoxelFEADataset(
        run_dirs=[sample_run_dir],
        target_names=("max_von_mises", "max_displacement", "min_safety_factor", "compliance"),
        material_types=("concrete", "mortar"),
        load_cases=("gravity", "wind"),
        resolution=32,
        use_sdf=False,
        stats=None,
        augment=False,
    )
    
    assert len(dataset) == 1
    
    sample = dataset[0]
    assert "voxel" in sample
    assert "features" in sample
    assert "targets" in sample
    
    # Check shapes
    assert sample["voxel"].shape[0] == 1 + 6  # occ + 6 part channels
    assert sample["voxel"].shape[1] == 32
    assert sample["features"].shape[0] == 4 + 2 + 2  # props + materials + load_cases
    assert sample["targets"].shape[0] == 4


def test_normalization_stats(sample_run_dir):
    """Test normalization statistics computation."""
    dataset = VoxelFEADataset(
        run_dirs=[sample_run_dir],
        target_names=("max_von_mises", "max_displacement", "min_safety_factor", "compliance"),
        material_types=("concrete", "mortar"),
        load_cases=("gravity", "wind"),
        resolution=32,
        stats=None,
    )
    
    stats = compute_voxel_normalization_stats(dataset)
    
    assert stats.feature_mean.shape[0] == 8
    assert stats.target_mean.shape[0] == 4
    assert np.all(stats.feature_std > 0)
    assert np.all(stats.target_std > 0)


def test_feature_encoding(sample_run_dir):
    """Test that material and load case encoding works."""
    dataset = VoxelFEADataset(
        run_dirs=[sample_run_dir],
        target_names=("min_safety_factor",),
        material_types=("concrete", "mortar", "polymer"),
        load_cases=("gravity", "wind", "combined"),
        resolution=32,
        stats=None,
    )
    
    sample = dataset[0]
    features = sample["features"].numpy()
    
    # First 4 are material props
    # Next 3 are material one-hot (concrete should be [1, 0, 0])
    material_onehot = features[4:7]
    assert material_onehot[0] == 1.0  # concrete
    assert material_onehot[1] == 0.0
    assert material_onehot[2] == 0.0
    
    # Next 3 are load case one-hot (gravity should be [1, 0, 0])
    load_onehot = features[7:10]
    assert load_onehot[0] == 1.0  # gravity


def test_data_splits():
    """Test train/val/test splitting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runs_dir = Path(tmpdir)
        
        # Create multiple fake runs
        for i in range(10):
            run_dir = runs_dir / f"house_00{i % 3}_v{i}"
            run_dir.mkdir()
            
            np.savez_compressed(run_dir / "occ.npz", data=np.zeros((8, 8, 8), dtype=np.uint8))
            np.savez_compressed(run_dir / "part.npz", data=np.zeros((8, 8, 8), dtype=np.uint8))
            np.savez_compressed(run_dir / "edit_mask.npz", data=np.zeros((8, 8, 8), dtype=np.uint8))
            np.savez_compressed(run_dir / "protected_mask.npz", data=np.zeros((8, 8, 8), dtype=np.uint8))
            
            with open(run_dir / "meta.json", "w") as f:
                json.dump({}, f)
            with open(run_dir / "targets.json", "w") as f:
                json.dump({}, f)
        
        train, val, test = create_data_splits(
            runs_dir,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
            split_by_family=True,
        )
        
        # Check all runs are accounted for
        total = len(train) + len(val) + len(test)
        assert total == 10
        
        # Check no overlap
        all_runs = set(train + val + test)
        assert len(all_runs) == 10
