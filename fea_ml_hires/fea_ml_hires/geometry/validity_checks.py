"""
Validity checks for voxel geometries.

Implements watertight detection, minimum thickness, connectivity,
and thin feature counting for structural optimization constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from fea_ml_hires.geometry.voxelize import PART_EXTERIOR_WALL


@dataclass
class ValidityResult:
    """Result of a validity check."""
    passed: bool
    value: float  # Metric value (e.g., min thickness, component count)
    message: str
    details: Optional[dict] = None


def check_watertight(
    occ: np.ndarray,
    part: np.ndarray,
    check_exterior_only: bool = True,
) -> ValidityResult:
    """
    Check if geometry is watertight (no through-holes in exterior walls).
    
    Uses flood fill from outside to detect if exterior can reach interior cavities
    that should be enclosed.
    
    Args:
        occ: (D,H,W) occupancy grid
        part: (D,H,W) part labels
        check_exterior_only: If True, only check exterior wall voxels
        
    Returns:
        ValidityResult with pass/fail and leak locations
    """
    resolution = occ.shape[0]
    
    if check_exterior_only:
        # Create mask of exterior wall voxels only
        target_mask = (part == PART_EXTERIOR_WALL) & (occ > 0)
    else:
        target_mask = occ > 0
    
    if not np.any(target_mask):
        return ValidityResult(
            passed=True,
            value=0.0,
            message="No exterior wall voxels to check",
        )
    
    # Pad with zeros to create "outside" region
    padded_occ = np.pad(occ, pad_width=1, mode='constant', constant_values=0)
    
    # Flood fill from corner (guaranteed outside)
    outside_mask = np.zeros_like(padded_occ, dtype=bool)
    outside_mask[0, 0, 0] = True
    
    # Use binary dilation with connectivity check
    # "Outside" = empty space reachable from corner
    structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    
    # Iterative flood fill (more memory efficient than recursive)
    empty_space = padded_occ == 0
    prev_count = 0
    current_outside = outside_mask.copy()
    
    for _ in range(resolution * 3):  # Max iterations
        # Dilate current outside region
        dilated = ndimage.binary_dilation(current_outside, structure)
        # Constrain to empty space
        current_outside = dilated & empty_space
        # Check convergence
        new_count = np.sum(current_outside)
        if new_count == prev_count:
            break
        prev_count = new_count
    
    # Remove padding
    outside_reachable = current_outside[1:-1, 1:-1, 1:-1]
    
    # Find interior cavities (empty but not reachable from outside)
    interior_empty = (occ == 0) & (~outside_reachable)
    
    # Check if any interior cavities touch exterior walls (leak)
    if check_exterior_only:
        # Dilate interior cavities to find adjacent voxels
        dilated_interior = ndimage.binary_dilation(interior_empty, structure)
        # Leak = exterior wall voxel adjacent to interior cavity
        leaks = dilated_interior & target_mask
    else:
        # Any hole that goes through is a leak
        leaks = interior_empty
    
    leak_count = np.sum(leaks)
    
    if leak_count > 0:
        # Find leak locations
        leak_coords = np.argwhere(leaks)
        return ValidityResult(
            passed=False,
            value=float(leak_count),
            message=f"FAIL: {leak_count} leak voxels detected in exterior walls",
            details={"leak_coords": leak_coords[:10].tolist()},  # First 10
        )
    
    return ValidityResult(
        passed=True,
        value=0.0,
        message="OK: Exterior walls are watertight",
    )


def check_min_thickness(
    occ: np.ndarray,
    min_thickness_voxels: float = 2.0,
    sdf: Optional[np.ndarray] = None,
) -> ValidityResult:
    """
    Check if geometry meets minimum thickness requirement.
    
    Uses distance transform or SDF to find thin regions.
    
    Args:
        occ: (D,H,W) occupancy grid
        min_thickness_voxels: Minimum required thickness in voxels
        sdf: Optional pre-computed SDF (negative inside)
        
    Returns:
        ValidityResult with pass/fail and thin region info
    """
    if sdf is None:
        # Compute distance to surface
        dist_inside = ndimage.distance_transform_edt(occ)
    else:
        # Use negative SDF values (inside geometry)
        dist_inside = np.maximum(-sdf, 0)
    
    # Thickness is 2x distance to nearest surface
    # (distance from center to surface on both sides)
    max_depth = dist_inside.max()
    
    # Find thin regions
    half_min = min_thickness_voxels / 2.0
    thin_mask = (occ > 0) & (dist_inside < half_min)
    thin_count = np.sum(thin_mask)
    thin_fraction = thin_count / max(np.sum(occ), 1)
    
    # Allow small amount of thin features (e.g., at edges)
    threshold = 0.05  # 5% thin is acceptable
    
    if thin_fraction > threshold:
        return ValidityResult(
            passed=False,
            value=float(thin_fraction),
            message=f"FAIL: {thin_fraction:.1%} of voxels are thinner than {min_thickness_voxels} voxels",
            details={"thin_count": int(thin_count), "max_depth": float(max_depth)},
        )
    
    return ValidityResult(
        passed=True,
        value=float(thin_fraction),
        message=f"OK: {thin_fraction:.1%} thin regions (threshold: {threshold:.0%})",
        details={"max_depth": float(max_depth)},
    )


def check_connectivity(
    occ: np.ndarray,
    max_components: int = 1,
) -> ValidityResult:
    """
    Check if geometry is connected (no floating islands).
    
    Args:
        occ: (D,H,W) occupancy grid
        max_components: Maximum allowed connected components
        
    Returns:
        ValidityResult with pass/fail and component count
    """
    if not np.any(occ):
        return ValidityResult(
            passed=True,
            value=0.0,
            message="OK: Empty grid",
        )
    
    # Label connected components (6-connectivity)
    structure = ndimage.generate_binary_structure(3, 1)
    labeled, num_components = ndimage.label(occ, structure=structure)
    
    if num_components <= max_components:
        return ValidityResult(
            passed=True,
            value=float(num_components),
            message=f"OK: {num_components} connected component(s)",
        )
    
    # Find component sizes
    component_sizes = ndimage.sum(occ, labeled, range(1, num_components + 1))
    main_component = np.argmax(component_sizes) + 1
    main_size = component_sizes[main_component - 1]
    
    # Floating islands are small components
    islands = []
    for i in range(1, num_components + 1):
        if i != main_component:
            islands.append({
                "id": i,
                "size": int(component_sizes[i - 1]),
            })
    
    return ValidityResult(
        passed=False,
        value=float(num_components),
        message=f"FAIL: {num_components} components (max: {max_components}). Main: {int(main_size)} voxels",
        details={"islands": islands[:5]},  # First 5
    )


def count_thin_features(
    occ: np.ndarray,
    thin_threshold_voxels: float = 1.5,
    sdf: Optional[np.ndarray] = None,
) -> Tuple[int, float]:
    """
    Count voxels that are part of thin features.
    
    Used as a penalty term in optimization to discourage unprintable geometry.
    
    Args:
        occ: (D,H,W) occupancy grid
        thin_threshold_voxels: Threshold below which features are "thin"
        sdf: Optional pre-computed SDF
        
    Returns:
        (thin_count, thin_fraction) tuple
    """
    if sdf is None:
        dist_inside = ndimage.distance_transform_edt(occ)
    else:
        dist_inside = np.maximum(-sdf, 0)
    
    # Count voxels closer to surface than threshold
    half_threshold = thin_threshold_voxels / 2.0
    thin_mask = (occ > 0) & (dist_inside < half_threshold)
    thin_count = int(np.sum(thin_mask))
    thin_fraction = thin_count / max(np.sum(occ), 1)
    
    return thin_count, float(thin_fraction)


def compute_surface_area(occ: np.ndarray) -> int:
    """
    Compute approximate surface area of voxel geometry.
    
    Counts exposed faces between occupied and empty voxels.
    
    Args:
        occ: (D,H,W) occupancy grid
        
    Returns:
        Number of exposed voxel faces
    """
    # Pad with zeros
    padded = np.pad(occ, pad_width=1, mode='constant', constant_values=0)
    
    # Count transitions in each axis direction
    surface_area = 0
    
    # X direction
    surface_area += np.sum(np.abs(np.diff(padded, axis=0)))
    # Y direction  
    surface_area += np.sum(np.abs(np.diff(padded, axis=1)))
    # Z direction
    surface_area += np.sum(np.abs(np.diff(padded, axis=2)))
    
    return int(surface_area)


def validate_geometry(
    occ: np.ndarray,
    part: np.ndarray,
    min_thickness_voxels: float = 2.0,
    max_components: int = 1,
    sdf: Optional[np.ndarray] = None,
) -> Tuple[bool, dict]:
    """
    Run all validity checks on a geometry.
    
    Args:
        occ: (D,H,W) occupancy grid
        part: (D,H,W) part labels
        min_thickness_voxels: Minimum thickness requirement
        max_components: Maximum connected components
        sdf: Optional pre-computed SDF
        
    Returns:
        (all_passed, results_dict) tuple
    """
    results = {}
    
    # Watertight check (exterior walls only)
    wt_result = check_watertight(occ, part, check_exterior_only=True)
    results["watertight"] = {
        "passed": wt_result.passed,
        "value": wt_result.value,
        "message": wt_result.message,
    }
    
    # Min thickness check
    th_result = check_min_thickness(occ, min_thickness_voxels, sdf)
    results["min_thickness"] = {
        "passed": th_result.passed,
        "value": th_result.value,
        "message": th_result.message,
    }
    
    # Connectivity check
    cn_result = check_connectivity(occ, max_components)
    results["connectivity"] = {
        "passed": cn_result.passed,
        "value": cn_result.value,
        "message": cn_result.message,
    }
    
    # Thin features count
    thin_count, thin_frac = count_thin_features(occ, sdf=sdf)
    results["thin_features"] = {
        "count": thin_count,
        "fraction": thin_frac,
    }
    
    # Surface area
    surface_area = compute_surface_area(occ)
    results["surface_area"] = surface_area
    
    # Volume
    volume = int(np.sum(occ))
    results["volume"] = volume
    
    all_passed = wt_result.passed and th_result.passed and cn_result.passed
    
    return all_passed, results
