"""
CMA-ES optimizer for voxel-based structural optimization.

Searches for geometry modifications that minimize volume while satisfying
structural constraints using uncertainty-aware conservative bounds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cma
import numpy as np
import torch

from fea_ml.geometry.validity_checks import (
    ValidityResult,
    check_watertight,
    check_min_thickness,
    check_connectivity,
    count_thin_features,
    compute_surface_area,
)
from fea_ml.geometry.voxelize import VoxelGrids, voxels_to_mesh
from fea_ml.models.uncertainty import predict_with_uncertainty
from fea_ml.optim.voxel_parameterization import (
    SurfaceErosionConfig,
    SurfaceErosionParam,
    VoxelMaskedErosion,         # backward-compatible alias
    VoxelMaskedErosionConfig,   # backward-compatible alias
)


def _json_safe(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@dataclass
class OptimizationConfig:
    """Configuration for CMA-ES optimization."""
    # CMA-ES settings
    iterations: int = 50
    population_size: int = 16
    sigma0: float = 0.3
    
    # Constraint thresholds (absolute values)
    max_displacement: float = 1.0
    max_compliance_ratio: float = 1.05  # relative to baseline
    
    # Uncertainty settings
    uncertainty_k: float = 2.0  # std multiplier for conservative bounds
    mc_samples: int = 16
    
    # Objective weights
    volume_weight: float = 1.0
    surface_area_weight: float = 0.01  # printability regularizer
    thin_feature_weight: float = 0.1   # penalty for thin features
    constraint_penalty: float = 100.0
    
    # Validity thresholds
    min_thickness_voxels: float = 2.0
    max_components: int = 1
    thin_threshold_voxels: float = 1.5
    
    # Target indices (in prediction output)
    von_mises_idx: int = 0
    displacement_idx: int = 1
    compliance_idx: int = 2
    
    # Max allowable von Mises stress (Pa) — conservative upper bound
    max_von_mises: float = 3.5e6  # default: yield stress for concrete
    
    # Skip expensive geometry validity checks (watertight, thickness, connectivity)
    # Useful for fast demos or when running at high resolution
    skip_validity_checks: bool = False


@dataclass
class OptimizationResult:
    """Result of optimization run."""
    success: bool
    best_params: np.ndarray
    best_occ: np.ndarray
    predicted_mean: np.ndarray
    predicted_std: np.ndarray
    volume_original: int
    volume_optimized: int
    volume_reduction: float
    constraints_satisfied: bool
    validity_passed: bool
    objective_history: List[float] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


class VoxelSurrogateOptimizer:
    """
    CMA-ES optimizer using surrogate model for objective/constraint evaluation.
    
    Objective: volume + λ_surf * surface_area + λ_thin * thin_features
    Constraints: conservative (mean ± k*std) bounds on SF, displacement, compliance
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        parameterization,  # SurfaceErosionParam (or legacy VoxelMaskedErosion)
        config: OptimizationConfig,
        device: torch.device,
        normalization_stats: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            model: Trained surrogate model (single or ensemble)
            parameterization: Voxel geometry parameterization
            config: Optimization configuration
            device: Torch device
            normalization_stats: Dict with target_mean, target_std for denormalization
        """
        self.model = model
        self.parameterization = parameterization
        self.config = config
        self.device = device
        self.normalization_stats = normalization_stats
        self.target_resolution = None  # set to e.g. 128 to upsample lower-res voxels
    
    def optimize(
        self,
        grids: VoxelGrids,
        features: np.ndarray,
        baseline_targets: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """
        Run optimization to find best geometry modification.
        
        Args:
            grids: Original voxel grids (occ, part, masks)
            features: Non-spatial feature vector
            baseline_targets: Optional baseline FEA results for relative constraints
            
        Returns:
            OptimizationResult with best solution
        """
        # Store baseline info
        baseline_compliance = None
        if baseline_targets and "compliance" in baseline_targets:
            baseline_compliance = baseline_targets["compliance"]
        
        original_volume = int(np.sum(grids.occ))
        
        # Set up CMA-ES
        dim = self.parameterization.parameter_dim()
        es = cma.CMAEvolutionStrategy(
            x0=dim * [0.5],
            sigma0=self.config.sigma0,
            inopts={
                "popsize": self.config.population_size,
                "maxiter": self.config.iterations,
                "bounds": [0, 1],
                "verbose": -9,  # Suppress output
            },
        )
        
        objective_history = []
        best_result = None
        best_objective = float("inf")
        
        # Use simple counter instead of tqdm to avoid stderr issues
        iteration = 0
        
        while not es.stop():
            solutions = es.ask()
            fitnesses = []
            
            for params in solutions:
                params_arr = np.array(params, dtype=np.float32)
                
                # Evaluate candidate
                result = self._evaluate_candidate(
                    grids=grids,
                    features=features,
                    params=params_arr,
                    baseline_compliance=baseline_compliance,
                )
                
                fitnesses.append(result["objective"])
                
                # Track best
                if result["objective"] < best_objective:
                    best_objective = result["objective"]
                    best_result = result
                    best_result["params"] = params_arr.copy()
            
            es.tell(solutions, fitnesses)
            objective_history.append(min(fitnesses))
            
            iteration += 1
            vol_red = 0.0
            if best_result is not None:
                vol_red = 1.0 - best_result["volume"] / original_volume
            print(f"  iter {iteration:3d}/{self.config.iterations} | "
                  f"best_obj={best_objective:.4f} | vol_red={vol_red:.1%}",
                  flush=True)
        
        # Done
        if best_result is None:
            # No valid solution found
            return OptimizationResult(
                success=False,
                best_params=np.zeros(dim, dtype=np.float32),
                best_occ=grids.occ,
                predicted_mean=np.zeros(4),
                predicted_std=np.zeros(4),
                volume_original=original_volume,
                volume_optimized=original_volume,
                volume_reduction=0.0,
                constraints_satisfied=False,
                validity_passed=False,
                objective_history=objective_history,
                details={"error": "No valid solution found"},
            )
        
        optimized_volume = best_result["volume"]
        volume_reduction = 1.0 - optimized_volume / original_volume
        
        return OptimizationResult(
            success=True,
            best_params=best_result["params"],
            best_occ=best_result["occ"],
            predicted_mean=best_result["pred_mean"],
            predicted_std=best_result["pred_std"],
            volume_original=original_volume,
            volume_optimized=optimized_volume,
            volume_reduction=volume_reduction,
            constraints_satisfied=best_result["constraints_ok"],
            validity_passed=best_result["validity_ok"],
            objective_history=objective_history,
            details=best_result.get("details", {}),
        )
    
    def _evaluate_candidate(
        self,
        grids: VoxelGrids,
        features: np.ndarray,
        params: np.ndarray,
        baseline_compliance: Optional[float],
    ) -> Dict:
        """Evaluate a single candidate geometry."""
        cfg = self.config
        
        # Apply parameterization
        modified_occ = self.parameterization.apply(
            occ=grids.occ,
            part=grids.part,
            edit_mask=grids.edit_mask,
            protected_mask=grids.protected_mask,
            params=params,
        )
        
        # Compute geometry metrics
        volume = int(np.sum(modified_occ))
        surface_area = compute_surface_area(modified_occ)
        
        if cfg.skip_validity_checks:
            thin_count, thin_frac = 0, 0.0
            validity_ok = True
            wt = ValidityResult(passed=True, value=0.0, message="skipped")
            th = ValidityResult(passed=True, value=0.0, message="skipped")
            cn = ValidityResult(passed=True, value=0.0, message="skipped")
        else:
            thin_count, thin_frac = count_thin_features(
                modified_occ, 
                thin_threshold_voxels=cfg.thin_threshold_voxels,
            )
            wt = check_watertight(modified_occ, grids.part)
            th = check_min_thickness(modified_occ, cfg.min_thickness_voxels)
            cn = check_connectivity(modified_occ, cfg.max_components)
            validity_ok = wt.passed and th.passed and cn.passed
        
        # Build voxel input for surrogate
        voxel_input = self._build_voxel_input(modified_occ, grids.part, grids.sdf)
        voxel_tensor = torch.from_numpy(voxel_input[None, ...]).float().to(self.device)
        features_tensor = torch.from_numpy(features[None, ...]).float().to(self.device)
        
        # Upsample if model expects a different resolution (e.g., 64³ data → 128³ model)
        if hasattr(self, 'target_resolution') and self.target_resolution is not None and voxel_tensor.shape[-1] != self.target_resolution:
            voxel_tensor = torch.nn.functional.interpolate(
                voxel_tensor, size=self.target_resolution, mode="nearest"
            )
        
        # Get predictions with uncertainty
        self.model.eval()
        with torch.no_grad():
            pred_mean, pred_std = predict_with_uncertainty(
                self.model,
                voxel_tensor,
                features_tensor,
                method="ensemble" if hasattr(self.model, "n_models") else "mc_dropout",
                n_samples=cfg.mc_samples,
            )
        
        pred_mean = pred_mean.cpu().numpy().squeeze(0)
        pred_std = pred_std.cpu().numpy().squeeze(0)
        
        # Denormalize predictions: undo z-score, then undo log1p
        if self.normalization_stats:
            target_mean = np.array(self.normalization_stats["target_mean"])
            target_std = np.array(self.normalization_stats["target_std"])
            
            # Undo z-score (in log1p space)
            pred_mean_log = pred_mean * target_std + target_mean
            pred_std_log = pred_std * target_std  # std scales linearly
            
            # Check if log transform was applied
            log_targets = self.normalization_stats.get("log_transform_targets", [])
            if log_targets:
                # Undo log1p: expm1(x) = exp(x) - 1
                # For the mean, use the median of the log-normal: expm1(mu)
                # For uncertainty, propagate through expm1 approximately
                pred_mean_raw = np.expm1(pred_mean_log)
                # First-order propagation: sigma_raw ≈ sigma_log * exp(mu_log)
                pred_std_raw = pred_std_log * np.exp(pred_mean_log)
                pred_mean = pred_mean_raw
                pred_std = pred_std_raw
            else:
                pred_mean = pred_mean_log
                pred_std = pred_std_log
        
        # Check constraints conservatively
        k = cfg.uncertainty_k
        
        # Von Mises stress: upper bound (mean + k*std <= threshold)
        vm_pred = pred_mean[cfg.von_mises_idx]
        vm_std = pred_std[cfg.von_mises_idx]
        vm_conservative = vm_pred + k * vm_std
        vm_ok = vm_conservative <= cfg.max_von_mises
        vm_violation = max(0, vm_conservative - cfg.max_von_mises) / max(cfg.max_von_mises, 1e-12)
        
        # Displacement: upper bound (mean + k*std <= threshold)
        disp_pred = pred_mean[cfg.displacement_idx]
        disp_std = pred_std[cfg.displacement_idx]
        disp_conservative = disp_pred + k * disp_std
        disp_ok = disp_conservative <= cfg.max_displacement
        disp_violation = max(0, disp_conservative - cfg.max_displacement)
        
        # Compliance: relative to baseline
        comp_pred = pred_mean[cfg.compliance_idx]
        comp_std = pred_std[cfg.compliance_idx]
        comp_conservative = comp_pred + k * comp_std
        
        if baseline_compliance and baseline_compliance > 0:
            comp_limit = cfg.max_compliance_ratio * baseline_compliance
            comp_ok = comp_conservative <= comp_limit
            comp_violation = max(0, comp_conservative - comp_limit) / comp_limit
        else:
            comp_ok = True
            comp_violation = 0.0
        
        constraints_ok = vm_ok and disp_ok and comp_ok
        
        # Compute objective
        # Normalize volume to [0, 1] range
        original_volume = np.sum(grids.occ)
        norm_volume = volume / max(original_volume, 1)
        norm_surface = surface_area / max(original_volume, 1)
        
        objective = (
            cfg.volume_weight * norm_volume +
            cfg.surface_area_weight * norm_surface +
            cfg.thin_feature_weight * thin_frac
        )
        
        # Add constraint penalties
        total_violation = vm_violation + disp_violation + comp_violation
        if not constraints_ok:
            objective += cfg.constraint_penalty * total_violation
        
        # Add validity penalty
        if not validity_ok:
            objective += cfg.constraint_penalty * 0.5
        
        return {
            "objective": objective,
            "occ": modified_occ,
            "volume": volume,
            "surface_area": surface_area,
            "thin_count": thin_count,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "constraints_ok": constraints_ok,
            "validity_ok": validity_ok,
            "details": {
                "vm_conservative": vm_conservative,
                "disp_conservative": disp_conservative,
                "comp_conservative": comp_conservative,
                "validity": {
                    "watertight": wt.passed,
                    "min_thickness": th.passed,
                    "connectivity": cn.passed,
                },
            },
        }
    
    def _build_voxel_input(
        self,
        occ: np.ndarray,
        part: np.ndarray,
        sdf: Optional[np.ndarray],
    ) -> np.ndarray:
        """Build multi-channel voxel input for model."""
        from fea_ml.data.voxel_dataset import NUM_PARTS
        
        channels = [occ[None, ...].astype(np.float32)]
        
        # Part one-hot
        for p in range(NUM_PARTS):
            part_channel = (part == p).astype(np.float32)[None, ...]
            channels.append(part_channel)
        
        # Optional SDF
        if sdf is not None:
            sdf_norm = np.tanh(sdf / 10.0)
            channels.append(sdf_norm[None, ...].astype(np.float32))
        
        return np.concatenate(channels, axis=0)


def run_optimization(
    model: torch.nn.Module,
    grids: VoxelGrids,
    features: np.ndarray,
    config: OptimizationConfig,
    device: torch.device,
    baseline_targets: Optional[Dict[str, float]] = None,
    normalization_stats: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
    target_resolution: Optional[int] = None,
) -> OptimizationResult:
    """
    Convenience function to run optimization.
    
    Args:
        model: Trained surrogate model
        grids: Input voxel grids
        features: Non-spatial features
        config: Optimization config
        device: Torch device
        baseline_targets: Baseline FEA results
        normalization_stats: Normalization info for denormalization
        output_dir: Optional directory to save results
        target_resolution: If set, upsample voxels to this resolution before model inference
        
    Returns:
        OptimizationResult
    """
    # Create parameterization (surface-erosion based)
    param_config = SurfaceErosionConfig(
        grid_res=3,
        max_filter_size=7,
    )
    parameterization = SurfaceErosionParam(
        param_config,
        occ_128=grids.occ,
        part_128=grids.part,
    )
    
    # Create optimizer
    optimizer = VoxelSurrogateOptimizer(
        model=model,
        parameterization=parameterization,
        config=config,
        device=device,
        normalization_stats=normalization_stats,
    )
    optimizer.target_resolution = target_resolution
    
    # Run optimization
    result = optimizer.optimize(
        grids=grids,
        features=features,
        baseline_targets=baseline_targets,
    )
    
    # Save results if output directory provided
    if output_dir and result.success:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save optimized voxels
        np.savez_compressed(output_dir / "optimized_occ.npz", data=result.best_occ)
        
        # Save parameters
        np.save(output_dir / "best_params.npy", result.best_params)
        
        # Save summary
        summary = {
            "success": bool(result.success),
            "volume_original": int(result.volume_original),
            "volume_optimized": int(result.volume_optimized),
            "volume_reduction": float(result.volume_reduction),
            "constraints_satisfied": bool(result.constraints_satisfied),
            "validity_passed": bool(result.validity_passed),
            "predicted_mean": result.predicted_mean.tolist(),
            "predicted_std": result.predicted_std.tolist(),
            "details": result.details,
        }
        with open(output_dir / "optimization_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=_json_safe)
        
        # Export mesh
        try:
            mesh = voxels_to_mesh(
                result.best_occ,
                grids.bounds,
                simplify=True,
                repair=True,
            )
            mesh.export(str(output_dir / "candidate.stl"))
            mesh.export(str(output_dir / "candidate.obj"))
        except Exception as e:
            print(f"Warning: mesh export failed: {e}")
    
    return result
