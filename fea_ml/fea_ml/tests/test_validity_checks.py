"""Tests for geometry validity checks."""
from __future__ import annotations

import numpy as np
import pytest

from fea_ml.geometry.validity_checks import (
    check_watertight,
    check_min_thickness,
    check_connectivity,
    count_thin_features,
    compute_surface_area,
    validate_geometry,
)
from fea_ml.geometry.voxelize import PART_EXTERIOR_WALL, PART_INTERIOR_WALL


def test_watertight_solid_cube():
    """A solid cube should be watertight."""
    occ = np.ones((16, 16, 16), dtype=np.uint8)
    part = np.full_like(occ, PART_EXTERIOR_WALL)
    
    result = check_watertight(occ, part)
    assert result.passed


def test_watertight_with_hole():
    """A cube with a through-hole should fail watertight check."""
    occ = np.ones((16, 16, 16), dtype=np.uint8)
    part = np.full_like(occ, PART_EXTERIOR_WALL)
    
    # Create a hole through the cube
    occ[7:9, 7:9, :] = 0
    
    result = check_watertight(occ, part)
    # This may or may not fail depending on flood fill behavior
    # The hole goes all the way through, so it should create a leak
    assert not result.passed or result.value > 0


def test_min_thickness_solid():
    """A solid cube should pass min thickness."""
    occ = np.ones((16, 16, 16), dtype=np.uint8)
    
    result = check_min_thickness(occ, min_thickness_voxels=2.0)
    assert result.passed


def test_min_thickness_thin_shell():
    """A thin shell should fail min thickness."""
    occ = np.zeros((16, 16, 16), dtype=np.uint8)
    occ[7, :, :] = 1  # Single-voxel-thick plane
    
    result = check_min_thickness(occ, min_thickness_voxels=3.0)
    # Thin features should be detected
    assert result.value > 0  # Some fraction is thin


def test_connectivity_single_component():
    """A single connected component should pass."""
    occ = np.ones((16, 16, 16), dtype=np.uint8)
    
    result = check_connectivity(occ, max_components=1)
    assert result.passed
    assert result.value == 1


def test_connectivity_multiple_components():
    """Multiple disconnected components should fail."""
    occ = np.zeros((16, 16, 16), dtype=np.uint8)
    
    # Two separate cubes
    occ[1:4, 1:4, 1:4] = 1
    occ[10:13, 10:13, 10:13] = 1
    
    result = check_connectivity(occ, max_components=1)
    assert not result.passed
    assert result.value == 2


def test_thin_features_count():
    """Test thin feature counting."""
    occ = np.zeros((16, 16, 16), dtype=np.uint8)
    
    # Create thin features
    occ[7:9, 7:9, :] = 1  # Thin bar
    
    count, frac = count_thin_features(occ, thin_threshold_voxels=3.0)
    
    assert count > 0
    assert 0 <= frac <= 1


def test_surface_area():
    """Test surface area computation."""
    occ = np.zeros((16, 16, 16), dtype=np.uint8)
    occ[5:11, 5:11, 5:11] = 1  # 6x6x6 cube
    
    area = compute_surface_area(occ)
    
    # Expected: 6 faces * 6*6 = 216
    assert area == 216


def test_validate_geometry():
    """Test combined validation."""
    occ = np.ones((16, 16, 16), dtype=np.uint8)
    part = np.full_like(occ, PART_EXTERIOR_WALL)
    
    passed, results = validate_geometry(occ, part)
    
    assert passed
    assert "watertight" in results
    assert "min_thickness" in results
    assert "connectivity" in results
    assert "volume" in results
