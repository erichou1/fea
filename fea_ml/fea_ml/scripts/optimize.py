"""
Optimization script for voxel-based structural optimization.

Takes a baseline geometry and uses CMA-ES with surrogate model
to find a lower-volume design satisfying structural constraints.

Usage:
    python -m fea_ml.scripts.optimize \
        --config configs/voxel_config.yaml \
        --checkpoint runs/exp1/best.pt \
        --baseline data/runs/sample_001 \
        --output runs/exp1/optimization
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from fea_ml.data.voxel_dataset import VoxelNormalizationStats
from fea_ml.geometry.voxelize import load_voxel_grids, VoxelGrids
from fea_ml.models.cnn3d import create_surrogate_model
from fea_ml.models.ensemble import DeepEnsemble
from fea_ml.optim.voxel_optimizer import (
    OptimizationConfig,
    run_optimization,
)
from fea_ml.utils.config import load_config


def load_model_and_stats(
    config: Dict,
    checkpoint_path: Path,
    device: torch.device,
):
    """Load trained model and normalization stats."""
    in_channels = 1 + 6
    if config["data"].get("use_sdf", False):
        in_channels += 1
    
    feature_dim = 4 + len(config["materials"]) + len(config["load_cases"])
    target_dim = len(config["targets"])
    
    # Check for ensemble
    ensemble_paths_file = checkpoint_path.parent / "ensemble_paths.json"
    if ensemble_paths_file.exists():
        with open(ensemble_paths_file, "r") as f:
            paths = [Path(p) for p in json.load(f)]
        
        def model_factory():
            return create_surrogate_model(
                in_channels=in_channels,
                feature_dim=feature_dim,
                target_dim=target_dim,
                resolution=config["data"]["resolution"],
                dropout=config["model"].get("dropout", 0.15),
                drop_path=config["model"].get("drop_path", 0.1),
                backbone=config["model"].get("backbone", "cnn3d"),
                base_channels=config["model"].get("base_channels", None),
            )
        
        model = DeepEnsemble.from_checkpoints(paths, model_factory, device)
    else:
        model = create_surrogate_model(
            in_channels=in_channels,
            feature_dim=feature_dim,
            target_dim=target_dim,
            resolution=config["data"]["resolution"],
            dropout=config["model"].get("dropout", 0.15),
            drop_path=config["model"].get("drop_path", 0.1),
            backbone=config["model"].get("backbone", "cnn3d"),
            base_channels=config["model"].get("base_channels", None),
        )
        
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
    
    model = model.to(device)
    
    # Load normalization - check current dir and parent dir
    norm_path = checkpoint_path.parent / "normalization.json"
    if not norm_path.exists():
        norm_path = checkpoint_path.parent.parent / "normalization.json"
    if not norm_path.exists():
        print("Warning: normalization.json not found, using empty stats")
        return model, {}
    with open(norm_path, "r") as f:
        norm_dict = json.load(f)
    
    return model, norm_dict


def load_baseline(
    baseline_dir: Path,
    config: Dict,
) -> tuple:
    """Load baseline geometry and metadata."""
    # Load voxel grids
    grids = load_voxel_grids(baseline_dir)
    
    # Load metadata
    with open(baseline_dir / "meta.json", "r") as f:
        meta = json.load(f)
    
    # Load baseline targets if available
    baseline_targets = None
    targets_path = baseline_dir / "targets.json"
    if targets_path.exists():
        with open(targets_path, "r") as f:
            baseline_targets = json.load(f)
    
    # Build feature vector
    material_types = config["materials"]
    load_cases = config["load_cases"]
    
    # Material properties
    youngs = meta.get("youngs_modulus", meta.get("E", 2e11)) / 1e11
    poisson = meta.get("poisson_ratio", meta.get("nu", 0.3))
    density = meta.get("density", 2400.0) / 1000.0
    yield_stress = meta.get("yield_stress", 30e6) / 1e7
    
    material_props = [youngs, poisson, density, yield_stress]
    
    # Material one-hot
    material_label = meta.get("material_type", meta.get("material_label", "concrete"))
    material_onehot = [0.0] * len(material_types)
    if material_label in material_types:
        material_onehot[material_types.index(material_label)] = 1.0
    else:
        material_onehot[0] = 1.0
    
    # Load case one-hot
    load_case = str(meta.get("load_case_id", meta.get("load_case", "case_a")))
    load_onehot = [0.0] * len(load_cases)
    if load_case in load_cases:
        load_onehot[load_cases.index(load_case)] = 1.0
    else:
        load_onehot[0] = 1.0
    
    features = np.array(
        material_props + material_onehot + load_onehot,
        dtype=np.float32,
    )
    
    return grids, features, baseline_targets, meta


def main():
    parser = argparse.ArgumentParser(description="Run structural optimization")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True, help="Baseline run directory")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--population", type=int, default=None)
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    model, norm_stats = load_model_and_stats(config, checkpoint_path, device)
    print(f"Loaded model from {checkpoint_path}")
    
    # Load baseline
    baseline_dir = Path(args.baseline)
    grids, features, baseline_targets, meta = load_baseline(baseline_dir, config)
    
    print(f"Loaded baseline from {baseline_dir}")
    print(f"  Original volume: {np.sum(grids.occ)} voxels")
    if baseline_targets:
        print(f"  Baseline targets: {baseline_targets}")
    
    # Create optimization config
    opt_cfg = config["optimization"]["search"]
    opt_config = OptimizationConfig(
        iterations=args.iterations or opt_cfg.get("iterations", 50),
        population_size=args.population or opt_cfg.get("population_size", 16),
        sigma0=opt_cfg.get("sigma0", 0.3),
        min_safety_factor=config["constraints"]["min_safety_factor"],
        max_displacement=config["constraints"]["max_displacement"],
        max_compliance_ratio=config["constraints"]["max_compliance_ratio"],
        uncertainty_k=opt_cfg.get("uncertainty_k", 2.0),
        mc_samples=opt_cfg.get("mc_samples", 16),
        surface_area_weight=opt_cfg.get("surface_area_weight", 0.01),
        thin_feature_weight=opt_cfg.get("thin_feature_weight", 0.1),
        constraint_penalty=opt_cfg.get("constraint_penalty", 100.0),
        min_thickness_voxels=config["validation"].get("min_thickness_voxels", 2.0),
    )
    
    # Normalize features
    if "feature_mean" in norm_stats:
        feature_mean = np.array(norm_stats["feature_mean"], dtype=np.float32)
        feature_std = np.array(norm_stats["feature_std"], dtype=np.float32)
        features = (features - feature_mean) / (feature_std + 1e-8)
    
    # Run optimization
    print("\nStarting optimization...")
    print(f"  Iterations: {opt_config.iterations}")
    print(f"  Population: {opt_config.population_size}")
    
    result = run_optimization(
        model=model,
        grids=grids,
        features=features,
        config=opt_config,
        device=device,
        baseline_targets=baseline_targets,
        normalization_stats=norm_stats,
        output_dir=output_dir,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Success: {result.success}")
    print(f"Volume: {result.volume_original} -> {result.volume_optimized} voxels")
    print(f"Volume reduction: {result.volume_reduction:.1%}")
    print(f"Constraints satisfied: {result.constraints_satisfied}")
    print(f"Validity passed: {result.validity_passed}")
    
    if result.success:
        print(f"\nPredicted metrics (mean ± std):")
        target_names = config["targets"]
        for i, name in enumerate(target_names):
            mean = result.predicted_mean[i]
            std = result.predicted_std[i]
            print(f"  {name}: {mean:.4f} ± {std:.4f}")
        
        print(f"\nOptimization parameters: {result.best_params}")
        print(f"\nOutputs saved to: {output_dir}")
        print(f"  - optimized_occ.npz")
        print(f"  - candidate.stl / candidate.obj")
        print(f"  - optimization_summary.json")
    else:
        print(f"\nOptimization failed: {result.details}")


if __name__ == "__main__":
    main()
