"""Geometry utilities for voxelization, SDF computation, and mesh conversion."""

from fea_ml_hires.geometry.voxelize import (
    mesh_to_voxels,
    stl_to_voxels,
    compute_sdf,
    compute_distance_to_outside,
    generate_masks,
    voxels_to_mesh,
)
from fea_ml_hires.geometry.validity_checks import (
    check_watertight,
    check_min_thickness,
    check_connectivity,
    count_thin_features,
)

__all__ = [
    "mesh_to_voxels",
    "stl_to_voxels", 
    "compute_sdf",
    "compute_distance_to_outside",
    "generate_masks",
    "voxels_to_mesh",
    "check_watertight",
    "check_min_thickness",
    "check_connectivity",
    "count_thin_features",
]
