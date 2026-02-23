"""
Voxelization utilities for converting meshes to voxel grids.

Supports occupancy grids, SDF computation, part labels, and mask generation
for the structural optimization pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage


@dataclass
class VoxelizationConfig:
    """Configuration for voxelization."""
    resolution: int = 64  # Grid resolution (64 or 128)
    padding: float = 0.05  # Padding fraction around geometry
    shell_thickness_voxels: int = 3  # Protected exterior skin band


@dataclass
class VoxelGrids:
    """Collection of voxel grids for a geometry."""
    occ: np.ndarray  # Occupancy (D,H,W) uint8 {0,1}
    sdf: Optional[np.ndarray]  # Signed distance field (D,H,W) float32
    part: np.ndarray  # Part labels (D,H,W) uint8 {0-5}
    edit_mask: np.ndarray  # Editable regions (D,H,W) uint8
    protected_mask: np.ndarray  # Protected regions (D,H,W) uint8
    bounds: Tuple[np.ndarray, np.ndarray]  # (min_corner, max_corner)
    voxel_size: float  # Size of each voxel in world units


# Part label constants
PART_EMPTY = 0
PART_EXTERIOR_WALL = 1
PART_INTERIOR_WALL = 2
PART_ROOF = 3
PART_FLOOR = 4
PART_OTHER = 5

PART_NAMES = {
    PART_EMPTY: "empty",
    PART_EXTERIOR_WALL: "exterior_wall",
    PART_INTERIOR_WALL: "interior_wall",
    PART_ROOF: "roof",
    PART_FLOOR: "floor",
    PART_OTHER: "other",
}

NAME_TO_PART = {v: k for k, v in PART_NAMES.items()}


def _compute_bounds(
    points: np.ndarray,
    padding_fraction: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute padded bounding box for points."""
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    extent = max_corner - min_corner
    padding = extent * padding_fraction
    return min_corner - padding, max_corner + padding


def _voxelize_points(
    points: np.ndarray,
    resolution: int,
    bounds: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Voxelize point cloud by binning into grid.
    Returns occupancy grid (D,H,W) uint8.
    """
    min_corner, max_corner = bounds
    extent = max_corner - min_corner
    voxel_size = extent / resolution
    
    # Normalize points to [0, resolution-1]
    normalized = (points - min_corner) / extent * (resolution - 1)
    indices = np.clip(normalized.astype(np.int32), 0, resolution - 1)
    
    # Create occupancy grid
    occ = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    occ[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
    
    return occ


def mesh_to_voxels(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int = 64,
    padding: float = 0.05,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], float]:
    """
    Convert mesh to voxel occupancy grid.
    
    Args:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle indices
        resolution: Grid resolution (e.g., 64 or 128)
        padding: Padding fraction around geometry
        
    Returns:
        (occ, bounds, voxel_size) where occ is (D,H,W) uint8 occupancy
    """
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        bounds = _compute_bounds(vertices, padding)
        min_corner, max_corner = bounds
        extent = max_corner - min_corner
        voxel_size = float(np.max(extent) / resolution)
        
        # Use trimesh's voxelization for accuracy
        voxels = mesh.voxelized(pitch=voxel_size)
        occ = voxels.matrix.astype(np.uint8)
        
        # Pad or crop to exact resolution
        occ = _resize_grid(occ, resolution)
        
        return occ, bounds, voxel_size
        
    except ImportError:
        # Fallback: simple point-based voxelization
        bounds = _compute_bounds(vertices, padding)
        min_corner, max_corner = bounds
        extent = max_corner - min_corner
        voxel_size = float(np.max(extent) / resolution)
        
        occ = _voxelize_points(vertices, resolution, bounds)
        return occ, bounds, voxel_size


def stl_to_voxels(
    stl_path: Union[str, Path],
    resolution: int = 64,
    padding: float = 0.05,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], float]:
    """
    Load STL file and convert to voxel occupancy grid.
    
    Args:
        stl_path: Path to STL file
        resolution: Grid resolution
        padding: Padding fraction
        
    Returns:
        (occ, bounds, voxel_size)
    """
    try:
        import trimesh
        mesh = trimesh.load(str(stl_path))
        return mesh_to_voxels(mesh.vertices, mesh.faces, resolution, padding)
    except ImportError:
        import meshio
        mesh = meshio.read(str(stl_path))
        vertices = mesh.points
        # Get triangles from cells
        faces = None
        for cells in mesh.cells:
            if cells.type == "triangle":
                faces = cells.data
                break
        if faces is None:
            raise ValueError(f"No triangles found in {stl_path}")
        return mesh_to_voxels(vertices, faces, resolution, padding)


def _resize_grid(grid: np.ndarray, target_size: int) -> np.ndarray:
    """Resize 3D grid to target size using padding or cropping."""
    current_shape = np.array(grid.shape)
    target_shape = np.array([target_size, target_size, target_size])
    
    if np.all(current_shape == target_shape):
        return grid
    
    # Create output grid
    result = np.zeros(target_shape, dtype=grid.dtype)
    
    # Compute overlap region
    min_shape = np.minimum(current_shape, target_shape)
    
    # Center the smaller in the larger
    offset_src = np.maximum(0, (current_shape - target_shape) // 2)
    offset_dst = np.maximum(0, (target_shape - current_shape) // 2)
    
    src_slices = tuple(slice(o, o + s) for o, s in zip(offset_src, min_shape))
    dst_slices = tuple(slice(o, o + s) for o, s in zip(offset_dst, min_shape))
    
    result[dst_slices] = grid[src_slices]
    return result


def compute_sdf(occ: np.ndarray) -> np.ndarray:
    """
    Compute signed distance field from occupancy grid.
    
    Positive values: outside the geometry
    Negative values: inside the geometry
    
    Args:
        occ: (D,H,W) occupancy grid {0,1}
        
    Returns:
        (D,H,W) float32 signed distance field
    """
    # Distance transform on occupied voxels (inside)
    dist_inside = ndimage.distance_transform_edt(occ)
    
    # Distance transform on empty voxels (outside)
    dist_outside = ndimage.distance_transform_edt(1 - occ)
    
    # SDF: negative inside, positive outside
    sdf = dist_outside - dist_inside
    
    return sdf.astype(np.float32)


def compute_distance_to_outside(occ: np.ndarray) -> np.ndarray:
    """
    Compute distance from each voxel to the nearest outside (empty) voxel.
    Used for determining protected exterior skin band.
    
    Args:
        occ: (D,H,W) occupancy grid {0,1}
        
    Returns:
        (D,H,W) float32 distance to outside (0 for empty voxels)
    """
    # Pad with zeros to ensure exterior is "outside"
    padded = np.pad(occ, pad_width=1, mode='constant', constant_values=0)
    
    # Invert: find distance from occupied voxels to empty
    dist = ndimage.distance_transform_edt(padded)
    
    # Remove padding
    dist = dist[1:-1, 1:-1, 1:-1]
    
    # Zero out empty voxels
    dist = dist * occ
    
    return dist.astype(np.float32)


def generate_masks(
    occ: np.ndarray,
    part: np.ndarray,
    config: VoxelizationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate edit_mask and protected_mask based on part labels and rules.
    
    Rules:
    - Exterior walls: only interior-side editing (protected skin band)
    - Interior walls: fully editable
    - Roof/floor: editable within limits
    - Protected regions: never editable
    
    Args:
        occ: (D,H,W) occupancy grid
        part: (D,H,W) part labels
        config: Voxelization configuration
        
    Returns:
        (edit_mask, protected_mask) both (D,H,W) uint8
    """
    resolution = occ.shape[0]
    edit_mask = np.zeros_like(occ)
    protected_mask = np.zeros_like(occ)
    
    # Compute distance to outside for exterior wall protection
    dist_to_outside = compute_distance_to_outside(occ)
    
    # Exterior walls: protect shell, allow interior editing
    ext_wall_mask = part == PART_EXTERIOR_WALL
    ext_protected = ext_wall_mask & (dist_to_outside <= config.shell_thickness_voxels)
    ext_editable = ext_wall_mask & (dist_to_outside > config.shell_thickness_voxels)
    
    protected_mask = protected_mask | ext_protected.astype(np.uint8)
    edit_mask = edit_mask | ext_editable.astype(np.uint8)
    
    # Interior walls: fully editable
    int_wall_mask = part == PART_INTERIOR_WALL
    edit_mask = edit_mask | int_wall_mask.astype(np.uint8)
    
    # Roof: editable (with min thickness enforced during optimization)
    roof_mask = part == PART_ROOF
    edit_mask = edit_mask | roof_mask.astype(np.uint8)
    
    # Floor: limited editing, protect key zones (edges)
    floor_mask = part == PART_FLOOR
    # Protect floor edges (2 voxels from boundary)
    floor_dist = ndimage.distance_transform_edt(floor_mask)
    floor_protected = floor_mask & (floor_dist <= 2)
    floor_editable = floor_mask & (floor_dist > 2)
    
    protected_mask = protected_mask | floor_protected.astype(np.uint8)
    edit_mask = edit_mask | floor_editable.astype(np.uint8)
    
    # Any other parts: conservatively protect
    other_mask = (part == PART_OTHER) & (occ == 1)
    protected_mask = protected_mask | other_mask.astype(np.uint8)
    
    # Ensure edit_mask and protected_mask don't overlap
    edit_mask = edit_mask & (~protected_mask.astype(bool)).astype(np.uint8)
    
    return edit_mask, protected_mask


def voxelize_parts_separately(
    part_stls: Dict[str, Path],
    resolution: int = 64,
    padding: float = 0.05,
) -> VoxelGrids:
    """
    Voxelize multiple part STLs separately and merge with priority.
    
    This is the fallback approach when labels are lost in the mesh pipeline.
    Each part is voxelized independently and combined with priority rules.
    
    Args:
        part_stls: Dict mapping part names to STL file paths
        resolution: Grid resolution
        padding: Padding fraction
        
    Returns:
        VoxelGrids with combined occupancy, part labels, and masks
    """
    config = VoxelizationConfig(resolution=resolution, padding=padding)
    
    # Collect all vertices to compute global bounds
    all_vertices = []
    for stl_path in part_stls.values():
        try:
            import trimesh
            mesh = trimesh.load(str(stl_path))
            all_vertices.append(mesh.vertices)
        except ImportError:
            import meshio
            mesh = meshio.read(str(stl_path))
            all_vertices.append(mesh.points)
    
    all_vertices = np.vstack(all_vertices)
    bounds = _compute_bounds(all_vertices, padding)
    min_corner, max_corner = bounds
    extent = max_corner - min_corner
    voxel_size = float(np.max(extent) / resolution)
    
    # Initialize grids
    occ = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    part = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    
    # Priority order for overlapping regions
    priority_order = [
        PART_EXTERIOR_WALL,  # Highest priority
        PART_FLOOR,
        PART_ROOF,
        PART_INTERIOR_WALL,
        PART_OTHER,  # Lowest priority
    ]
    
    # Voxelize each part
    for part_name, stl_path in part_stls.items():
        part_id = NAME_TO_PART.get(part_name, PART_OTHER)
        
        try:
            import trimesh
            mesh = trimesh.load(str(stl_path))
            vertices, faces = mesh.vertices, mesh.faces
        except ImportError:
            import meshio
            mesh = meshio.read(str(stl_path))
            vertices = mesh.points
            faces = None
            for cells in mesh.cells:
                if cells.type == "triangle":
                    faces = cells.data
                    break
        
        # Voxelize this part
        part_occ, _, _ = mesh_to_voxels(vertices, faces, resolution, padding)
        
        # Merge with priority
        new_voxels = (part_occ > 0) & (
            (part == PART_EMPTY) | 
            (priority_order.index(part_id) < priority_order.index(part.max()) if part.max() > 0 else True)
        )
        
        occ = occ | part_occ
        part[new_voxels] = part_id
    
    # Generate masks
    edit_mask, protected_mask = generate_masks(occ, part, config)
    
    # Compute SDF
    sdf = compute_sdf(occ)
    
    return VoxelGrids(
        occ=occ,
        sdf=sdf,
        part=part,
        edit_mask=edit_mask,
        protected_mask=protected_mask,
        bounds=bounds,
        voxel_size=voxel_size,
    )


def voxels_to_mesh(
    occ: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    simplify: bool = True,
    repair: bool = True,
) -> "trimesh.Trimesh":
    """
    Convert voxel occupancy grid to mesh using marching cubes.
    
    Args:
        occ: (D,H,W) occupancy grid
        bounds: (min_corner, max_corner) world coordinates
        simplify: Whether to simplify the mesh
        repair: Whether to repair for watertightness
        
    Returns:
        trimesh.Trimesh mesh object
    """
    import trimesh
    from skimage import measure
    
    # Pad to ensure closed surface
    padded = np.pad(occ, pad_width=1, mode='constant', constant_values=0)
    
    # Marching cubes
    verts, faces, normals, values = measure.marching_cubes(
        padded.astype(np.float32),
        level=0.5,
    )
    
    # Adjust for padding
    verts = verts - 1.0
    
    # Scale to world coordinates
    min_corner, max_corner = bounds
    extent = max_corner - min_corner
    resolution = occ.shape[0]
    
    verts = verts / resolution * extent + min_corner
    
    # Create mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    # Simplify if requested
    if simplify and len(mesh.faces) > 10000:
        target_faces = max(5000, len(mesh.faces) // 4)
        mesh = mesh.simplify_quadric_decimation(target_faces)
    
    # Repair for watertightness
    if repair:
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
    
    return mesh


def save_voxel_grids(grids: VoxelGrids, output_dir: Path) -> None:
    """Save voxel grids to npz files in the specified directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(output_dir / "occ.npz", data=grids.occ)
    
    if grids.sdf is not None:
        np.savez_compressed(output_dir / "sdf.npz", data=grids.sdf)
    
    np.savez_compressed(output_dir / "part.npz", data=grids.part)
    np.savez_compressed(output_dir / "edit_mask.npz", data=grids.edit_mask)
    np.savez_compressed(output_dir / "protected_mask.npz", data=grids.protected_mask)
    
    # Save bounds and voxel size as metadata
    import json
    meta = {
        "bounds_min": grids.bounds[0].tolist(),
        "bounds_max": grids.bounds[1].tolist(),
        "voxel_size": grids.voxel_size,
        "resolution": grids.occ.shape[0],
    }
    with open(output_dir / "voxel_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_voxel_grids(input_dir: Path) -> VoxelGrids:
    """Load voxel grids from npz files in the specified directory."""
    import json
    
    occ = np.load(input_dir / "occ.npz")["data"]
    part = np.load(input_dir / "part.npz")["data"]
    edit_mask = np.load(input_dir / "edit_mask.npz")["data"]
    protected_mask = np.load(input_dir / "protected_mask.npz")["data"]
    
    sdf = None
    sdf_path = input_dir / "sdf.npz"
    if sdf_path.exists():
        sdf = np.load(sdf_path)["data"]
    
    # Load metadata (support both voxel_meta.json and meta.json)
    meta_path = input_dir / "voxel_meta.json"
    if not meta_path.exists():
        meta_path = input_dir / "meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    bounds = (
        np.array(meta["bounds_min"]),
        np.array(meta["bounds_max"]),
    )
    voxel_size = meta["voxel_size"]
    
    return VoxelGrids(
        occ=occ,
        sdf=sdf,
        part=part,
        edit_mask=edit_mask,
        protected_mask=protected_mask,
        bounds=bounds,
        voxel_size=voxel_size,
    )
