"""
Generate synthetic data for testing the FEA surrogate pipeline.

Creates random voxel geometries with simulated FEA targets for testing
training, evaluation, and optimization scripts.

Usage:
    python -m fea_ml.scripts.generate_synthetic_data --output data/runs_test --n-samples 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fea_ml.geometry.voxelize import (
    PART_EXTERIOR_WALL,
    PART_INTERIOR_WALL,
    PART_ROOF,
    PART_FLOOR,
    compute_sdf,
    compute_distance_to_outside,
    generate_masks,
    VoxelizationConfig,
)


def generate_house_voxels(
    resolution: int = 64,
    seed: int = 0,
) -> dict:
    """
    Generate a simple house-like voxel structure.
    
    Returns dict with occ, part, edit_mask, protected_mask.
    """
    rng = np.random.default_rng(seed)
    
    occ = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    part = np.zeros_like(occ)
    
    # House dimensions (randomized)
    width = rng.integers(resolution // 3, 2 * resolution // 3)
    depth = rng.integers(resolution // 3, 2 * resolution // 3)
    height = rng.integers(resolution // 4, resolution // 2)
    
    # Center the house
    x0 = (resolution - width) // 2
    y0 = (resolution - depth) // 2
    z0 = 2  # Slightly above ground
    
    # Floor
    floor_thickness = 2
    occ[x0:x0+width, y0:y0+depth, z0:z0+floor_thickness] = 1
    part[x0:x0+width, y0:y0+depth, z0:z0+floor_thickness] = PART_FLOOR
    
    # Exterior walls
    wall_thickness = 2
    wall_bottom = z0 + floor_thickness
    wall_top = z0 + floor_thickness + height
    
    # Four walls
    occ[x0:x0+wall_thickness, y0:y0+depth, wall_bottom:wall_top] = 1
    occ[x0+width-wall_thickness:x0+width, y0:y0+depth, wall_bottom:wall_top] = 1
    occ[x0:x0+width, y0:y0+wall_thickness, wall_bottom:wall_top] = 1
    occ[x0:x0+width, y0+depth-wall_thickness:y0+depth, wall_bottom:wall_top] = 1
    
    part[x0:x0+width, y0:y0+depth, wall_bottom:wall_top] = PART_EXTERIOR_WALL
    
    # Interior wall (random position)
    if width > 10:
        interior_pos = x0 + rng.integers(4, width - 4)
        occ[interior_pos:interior_pos+1, y0+2:y0+depth-2, wall_bottom:wall_top] = 1
        part[interior_pos:interior_pos+1, y0+2:y0+depth-2, wall_bottom:wall_top] = PART_INTERIOR_WALL
    
    # Roof (simple flat roof)
    roof_thickness = 2
    occ[x0:x0+width, y0:y0+depth, wall_top:wall_top+roof_thickness] = 1
    part[x0:x0+width, y0:y0+depth, wall_top:wall_top+roof_thickness] = PART_ROOF
    
    # Generate masks
    config = VoxelizationConfig(resolution=resolution)
    edit_mask, protected_mask = generate_masks(occ, part, config)
    
    # Compute SDF
    sdf = compute_sdf(occ)
    
    # Compute bounds
    bounds = (
        np.array([0, 0, 0], dtype=np.float32),
        np.array([10, 10, 10], dtype=np.float32),
    )
    voxel_size = 10.0 / resolution
    
    return {
        "occ": occ,
        "sdf": sdf,
        "part": part,
        "edit_mask": edit_mask,
        "protected_mask": protected_mask,
        "bounds": bounds,
        "voxel_size": voxel_size,
    }


def generate_synthetic_targets(
    occ: np.ndarray,
    seed: int = 0,
) -> dict:
    """
    Generate synthetic FEA targets based on geometry.
    
    Uses heuristics to create plausible-looking values.
    """
    rng = np.random.default_rng(seed)
    
    volume = np.sum(occ)
    surface_area = compute_surface_area_simple(occ)
    
    # Base values with geometry-based modulation
    base_stress = 10e6 + volume * 100 + rng.normal() * 1e6
    base_displacement = 0.001 + (surface_area / volume) * 0.0001 + rng.normal() * 0.0002
    
    # Safety factor inversely related to stress
    base_sf = 3.0 - base_stress / 20e6 + rng.normal() * 0.3
    base_sf = max(0.5, base_sf)
    
    # Compliance related to volume/stiffness
    base_compliance = 0.01 + (1 / (volume + 1)) * 100 + rng.normal() * 0.002
    
    return {
        "max_von_mises": float(max(1e5, base_stress)),
        "max_displacement": float(max(1e-5, base_displacement)),
        "min_safety_factor": float(base_sf),
        "compliance": float(max(1e-4, base_compliance)),
    }


def compute_surface_area_simple(occ: np.ndarray) -> int:
    """Compute surface area (exposed faces)."""
    padded = np.pad(occ, pad_width=1, mode='constant', constant_values=0)
    
    area = 0
    area += np.sum(np.abs(np.diff(padded, axis=0)))
    area += np.sum(np.abs(np.diff(padded, axis=1)))
    area += np.sum(np.abs(np.diff(padded, axis=2)))
    
    return int(area)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic FEA data")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.n_samples} synthetic samples...")
    
    materials = ["concrete", "mortar", "polymer"]
    load_cases = ["gravity", "wind", "combined"]
    
    for i in range(args.n_samples):
        sample_seed = args.seed + i
        family_id = i // 5  # Group samples into families
        
        run_id = f"house_{family_id:03d}_v{i % 5}"
        run_dir = output_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Generate geometry
        data = generate_house_voxels(
            resolution=args.resolution,
            seed=sample_seed,
        )
        
        # Save voxel data
        np.savez_compressed(run_dir / "occ.npz", data=data["occ"])
        np.savez_compressed(run_dir / "sdf.npz", data=data["sdf"])
        np.savez_compressed(run_dir / "part.npz", data=data["part"])
        np.savez_compressed(run_dir / "edit_mask.npz", data=data["edit_mask"])
        np.savez_compressed(run_dir / "protected_mask.npz", data=data["protected_mask"])
        
        # Save voxel metadata
        voxel_meta = {
            "bounds_min": data["bounds"][0].tolist(),
            "bounds_max": data["bounds"][1].tolist(),
            "voxel_size": data["voxel_size"],
            "resolution": args.resolution,
        }
        with open(run_dir / "voxel_meta.json", "w") as f:
            json.dump(voxel_meta, f, indent=2)
        
        # Generate metadata
        rng = np.random.default_rng(sample_seed)
        meta = {
            "youngs_modulus": 2e11 + rng.normal() * 1e10,
            "poisson_ratio": 0.3 + rng.normal() * 0.02,
            "density": 2400.0 + rng.normal() * 100,
            "yield_stress": 30e6 + rng.normal() * 2e6,
            "material_type": materials[i % len(materials)],
            "material_label": materials[i % len(materials)],
            "load_case_id": load_cases[i % len(load_cases)],
            "length_unit": "meters",
            "baseline_run_id": f"house_{family_id:03d}_v0",
        }
        
        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        # Generate targets
        targets = generate_synthetic_targets(data["occ"], seed=sample_seed)
        
        with open(run_dir / "targets.json", "w") as f:
            json.dump(targets, f, indent=2)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{args.n_samples} samples")
    
    print(f"\nSynthetic data saved to {output_dir}")
    print(f"  - {args.n_samples} samples in {args.n_samples // 5} design families")


if __name__ == "__main__":
    main()
