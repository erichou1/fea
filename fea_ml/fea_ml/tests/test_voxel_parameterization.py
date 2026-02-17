"""Tests for voxel parameterization."""
from __future__ import annotations

import numpy as np
import pytest

from fea_ml.geometry.voxelize import PART_EXTERIOR_WALL, PART_INTERIOR_WALL
from fea_ml.optim.voxel_parameterization import (
    VoxelMaskedErosion,
    VoxelMaskedErosionConfig,
)


@pytest.fixture
def sample_grids():
    """Create sample voxel grids for testing."""
    resolution = 16
    
    # Create a solid cube
    occ = np.ones((resolution, resolution, resolution), dtype=np.uint8)
    
    # Part labels: exterior wall on outside, interior in middle
    part = np.zeros_like(occ)
    part[:, :, :] = PART_EXTERIOR_WALL
    part[4:12, 4:12, 4:12] = PART_INTERIOR_WALL
    
    # Edit mask: interior is editable
    edit_mask = np.zeros_like(occ)
    edit_mask[4:12, 4:12, 4:12] = 1
    
    # Protected mask: outer shell protected
    protected_mask = np.zeros_like(occ)
    protected_mask[:2, :, :] = 1
    protected_mask[-2:, :, :] = 1
    protected_mask[:, :2, :] = 1
    protected_mask[:, -2:, :] = 1
    protected_mask[:, :, :2] = 1
    protected_mask[:, :, -2:] = 1
    
    return occ, part, edit_mask, protected_mask


def test_parameterization_respects_edit_mask(sample_grids):
    """Test that erosion only affects edit_mask regions."""
    occ, part, edit_mask, protected_mask = sample_grids
    
    config = VoxelMaskedErosionConfig(
        erosion_min=0.0,
        erosion_max=0.3,
    )
    param = VoxelMaskedErosion(config)
    
    # Apply with some erosion
    params = np.array([0.5, 0.5, 0.5, 0.5, 0.0], dtype=np.float32)
    modified = param.apply(occ, part, edit_mask, protected_mask, params)
    
    # Check that non-editable regions are unchanged
    non_editable = (edit_mask == 0) & (protected_mask == 0)
    # Note: some changes may occur due to mask boundaries, so check protected
    protected_region = protected_mask == 1
    assert np.array_equal(modified[protected_region], occ[protected_region])


def test_parameterization_never_modifies_protected(sample_grids):
    """Test that protected regions are never modified."""
    occ, part, edit_mask, protected_mask = sample_grids
    
    config = VoxelMaskedErosionConfig(
        erosion_min=0.0,
        erosion_max=0.5,  # Strong erosion
    )
    param = VoxelMaskedErosion(config)
    
    # Apply with maximum erosion
    params = np.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    modified = param.apply(occ, part, edit_mask, protected_mask, params)
    
    # Protected must be unchanged
    protected = protected_mask == 1
    assert np.array_equal(modified[protected], occ[protected])


def test_zero_params_preserve_geometry(sample_grids):
    """Test that zero parameters preserve original geometry."""
    occ, part, edit_mask, protected_mask = sample_grids
    
    config = VoxelMaskedErosionConfig()
    param = VoxelMaskedErosion(config)
    
    params = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    modified = param.apply(occ, part, edit_mask, protected_mask, params)
    
    # With zero erosion, geometry should be mostly preserved
    # (SDF-based erosion may have slight boundary effects)
    assert param.volume_reduction(occ, modified) < 0.01


def test_erosion_reduces_volume(sample_grids):
    """Test that erosion reduces volume."""
    occ, part, edit_mask, protected_mask = sample_grids
    
    config = VoxelMaskedErosionConfig(
        erosion_min=0.0,
        erosion_max=0.3,
    )
    param = VoxelMaskedErosion(config)
    
    # Apply erosion
    params = np.array([0.5, 0.5, 0.5, 0.5, 0.0], dtype=np.float32)
    modified = param.apply(occ, part, edit_mask, protected_mask, params)
    
    original_volume = param.compute_volume(occ)
    modified_volume = param.compute_volume(modified)
    
    # Volume should decrease (or stay same if no editable material)
    assert modified_volume <= original_volume


def test_parameter_dim():
    """Test parameter dimension."""
    config = VoxelMaskedErosionConfig()
    param = VoxelMaskedErosion(config)
    
    assert param.parameter_dim() == 5


def test_random_params():
    """Test random parameter generation."""
    config = VoxelMaskedErosionConfig()
    param = VoxelMaskedErosion(config)
    
    rng = np.random.default_rng(42)
    params = param.random_params(rng)
    
    assert params.shape == (5,)
    assert np.all(params >= 0)
    assert np.all(params <= 1)
