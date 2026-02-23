"""
Evaluation script for voxel-based FEA surrogate model.

Generates MAE/RMSE per target, constraint classification accuracy,
uncertainty calibration plots, and parity plots.

Usage:
    python -m fea_ml.scripts.evaluate \
        --config configs/voxel_config.yaml \
        --checkpoint runs/exp1/best.pt \
        --output runs/exp1/eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from fea_ml_hires.data.voxel_dataset import (
    VoxelFEADataset,
    VoxelNormalizationStats,
)
from fea_ml_hires.models.cnn3d import create_surrogate_model
from fea_ml_hires.models.ensemble import DeepEnsemble
from fea_ml_hires.models.uncertainty import predict_with_uncertainty
from fea_ml_hires.utils.config import load_config


def load_model(
    config: Dict,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    in_channels = 1 + 6  # occ + part one-hot
    if config["data"].get("use_sdf", False):
        in_channels += 1
    
    feature_dim = 4 + len(config["materials"]) + len(config["load_cases"])
    target_dim = len(config["targets"])
    
    # Check if ensemble
    if (checkpoint_path.parent / "ensemble_paths.json").exists():
        with open(checkpoint_path.parent / "ensemble_paths.json", "r") as f:
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
    
    return model.to(device)


def evaluate_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    stats: VoxelNormalizationStats,
    device: torch.device,
    mc_samples: int = 16,
) -> Dict[str, np.ndarray]:
    """Get predictions with uncertainty on a dataset."""
    model.eval()
    
    all_preds_mean = []
    all_preds_std = []
    all_targets = []
    
    for batch in loader:
        voxel = batch["voxel"].to(device)
        features = batch["features"].to(device)
        targets = batch["targets"]  # Keep on CPU
        
        with torch.no_grad():
            mean, std = predict_with_uncertainty(
                model, voxel, features,
                method="ensemble" if hasattr(model, "n_models") else "mc_dropout",
                n_samples=mc_samples,
            )
        
        all_preds_mean.append(mean.cpu().numpy())
        all_preds_std.append(std.cpu().numpy())
        all_targets.append(targets.numpy())
    
    preds_mean = np.concatenate(all_preds_mean, axis=0)
    preds_std = np.concatenate(all_preds_std, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Denormalize: undo z-score, then undo log1p
    target_mean = stats.target_mean
    target_std = stats.target_std
    
    preds_mean_log = preds_mean * target_std + target_mean
    preds_std_log = preds_std * target_std
    targets_log = targets * target_std + target_mean
    
    # If log1p was applied, undo it for physical-units metrics
    if stats.log_transform_targets:
        preds_mean_denorm = np.expm1(preds_mean_log)
        preds_std_denorm = preds_std_log * np.exp(preds_mean_log)
        targets_denorm = np.expm1(targets_log)
    else:
        preds_mean_denorm = preds_mean_log
        preds_std_denorm = preds_std_log
        targets_denorm = targets_log
    
    return {
        "preds_mean": preds_mean_denorm,
        "preds_std": preds_std_denorm,
        "targets": targets_denorm,
        "preds_norm": preds_mean,
        "targets_norm": targets,
    }


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    target_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute MAE, RMSE, and R² per target."""
    metrics = {}
    
    for i, name in enumerate(target_names):
        p = preds[:, i]
        t = targets[:, i]
        
        mae = np.abs(p - t).mean()
        rmse = np.sqrt(((p - t) ** 2).mean())
        
        # R²
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
        metrics[name] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        }
    
    return metrics


def compute_constraint_accuracy(
    preds_mean: np.ndarray,
    preds_std: np.ndarray,
    targets: np.ndarray,
    config: Dict,
    target_names: List[str],
    k: float = 2.0,
) -> Dict[str, float]:
    """
    Compute constraint classification accuracy.
    
    For safety factor: actual >= threshold vs predicted conservative >= threshold
    For displacement: actual <= threshold vs predicted conservative <= threshold
    """
    results = {}
    
    sf_threshold = config["constraints"]["min_safety_factor"]
    disp_threshold = config["constraints"]["max_displacement"]
    
    if "min_safety_factor" in target_names:
        idx = target_names.index("min_safety_factor")
        actual_safe = targets[:, idx] >= sf_threshold
        pred_conservative = preds_mean[:, idx] - k * preds_std[:, idx]
        pred_safe = pred_conservative >= sf_threshold
        
        # True positives: both say safe
        # False negatives: actual safe, predicted unsafe (conservative, OK)
        # False positives: actual unsafe, predicted safe (dangerous!)
        tp = np.sum(actual_safe & pred_safe)
        tn = np.sum(~actual_safe & ~pred_safe)
        fp = np.sum(~actual_safe & pred_safe)
        fn = np.sum(actual_safe & ~pred_safe)
        
        accuracy = (tp + tn) / len(targets)
        fp_rate = fp / len(targets)  # Dangerous errors
        
        results["safety_factor_accuracy"] = float(accuracy)
        results["safety_factor_fp_rate"] = float(fp_rate)
    
    if "max_displacement" in target_names:
        idx = target_names.index("max_displacement")
        actual_ok = targets[:, idx] <= disp_threshold
        pred_conservative = preds_mean[:, idx] + k * preds_std[:, idx]
        pred_ok = pred_conservative <= disp_threshold
        
        tp = np.sum(actual_ok & pred_ok)
        tn = np.sum(~actual_ok & ~pred_ok)
        fp = np.sum(~actual_ok & pred_ok)
        
        accuracy = (tp + tn) / len(targets)
        fp_rate = fp / len(targets)
        
        results["displacement_accuracy"] = float(accuracy)
        results["displacement_fp_rate"] = float(fp_rate)
    
    return results


def compute_calibration(
    preds_mean: np.ndarray,
    preds_std: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute calibration curve and ECE.
    
    For each confidence level, check if that fraction of predictions
    fall within that confidence interval.
    """
    # Standardized residuals
    residuals = (targets - preds_mean) / (preds_std + 1e-8)
    
    # Expected coverage at various confidence levels
    from scipy import stats
    confidence_levels = np.linspace(0.1, 0.99, n_bins)
    
    expected_coverage = []
    actual_coverage = []
    
    for conf in confidence_levels:
        # Z-value for this confidence
        z = stats.norm.ppf((1 + conf) / 2)
        
        # Fraction of residuals within [-z, z]
        within = np.abs(residuals) <= z
        
        expected_coverage.append(conf)
        actual_coverage.append(within.mean())
    
    expected_coverage = np.array(expected_coverage)
    actual_coverage = np.array(actual_coverage)
    
    # Expected Calibration Error
    ece = np.abs(expected_coverage - actual_coverage).mean()
    
    return expected_coverage, actual_coverage, float(ece)


def plot_parity(
    preds: np.ndarray,
    targets: np.ndarray,
    target_names: List[str],
    output_dir: Path,
) -> None:
    """Generate parity plots for each target."""
    n_targets = len(target_names)
    fig, axes = plt.subplots(1, n_targets, figsize=(4 * n_targets, 4))
    
    if n_targets == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, target_names)):
        p = preds[:, i]
        t = targets[:, i]
        
        ax.scatter(t, p, alpha=0.5, s=10)
        
        # Perfect prediction line
        lims = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lims, lims, 'r--', linewidth=1)
        
        ax.set_xlabel(f"Actual {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(name)
        ax.set_aspect('equal', 'box')
    
    plt.tight_layout()
    plt.savefig(output_dir / "parity_plots.png", dpi=150)
    plt.close()


def plot_calibration(
    expected: np.ndarray,
    actual: np.ndarray,
    ece: float,
    output_dir: Path,
) -> None:
    """Plot calibration curve."""
    plt.figure(figsize=(6, 6))
    
    plt.plot([0, 1], [0, 1], 'r--', label="Perfect calibration")
    plt.plot(expected, actual, 'b-o', label=f"Model (ECE={ece:.3f})")
    
    plt.fill_between(expected, expected, actual, alpha=0.2)
    
    plt.xlabel("Expected confidence level")
    plt.ylabel("Observed coverage")
    plt.title("Uncertainty Calibration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / "calibration.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate FEA voxel surrogate")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--mc-samples", type=int, default=16)
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    model = load_model(config, checkpoint_path, device)
    print(f"Loaded model from {checkpoint_path}")
    
    # Load normalization stats
    norm_path = checkpoint_path.parent / "normalization.json"
    with open(norm_path, "r") as f:
        stats = VoxelNormalizationStats.from_dict(json.load(f))
    
    # Load splits
    splits_path = checkpoint_path.parent / "splits.json"
    with open(splits_path, "r") as f:
        splits = json.load(f)
    
    run_dirs = [Path(d) for d in splits[args.split]]
    print(f"Evaluating on {len(run_dirs)} {args.split} samples")
    
    # Create dataset
    target_names = list(config["targets"])
    dataset = VoxelFEADataset(
        run_dirs=run_dirs,
        target_names=tuple(target_names),
        material_types=tuple(config["materials"]),
        load_cases=tuple(config["load_cases"]),
        resolution=config["data"]["resolution"],
        use_sdf=config["data"].get("use_sdf", False),
        stats=stats,
        augment=False,
    )
    
    import platform
    nw = 0 if platform.system() == "Windows" else 4
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=nw)
    
    # Evaluate
    results = evaluate_predictions(model, loader, stats, device, args.mc_samples)
    
    # Compute metrics
    metrics = compute_metrics(results["preds_mean"], results["targets"], target_names)
    print("\nPer-target metrics:")
    for name, m in metrics.items():
        print(f"  {name}: MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, R²={m['r2']:.4f}")
    
    # Constraint accuracy
    constraint_acc = compute_constraint_accuracy(
        results["preds_mean"],
        results["preds_std"],
        results["targets"],
        config,
        target_names,
        k=config["optimization"]["search"].get("uncertainty_k", 2.0),
    )
    print("\nConstraint classification:")
    for k, v in constraint_acc.items():
        print(f"  {k}: {v:.4f}")
    
    # Calibration
    expected, actual, ece = compute_calibration(
        results["preds_norm"],
        results["preds_std"] / stats.target_std,  # Normalized
        results["targets_norm"],
    )
    print(f"\nUncertainty calibration ECE: {ece:.4f}")
    
    # Generate plots
    plot_parity(results["preds_mean"], results["targets"], target_names, output_dir)
    plot_calibration(expected, actual, ece, output_dir)
    
    # Save results
    all_results = {
        "metrics": metrics,
        "constraint_accuracy": constraint_acc,
        "calibration": {
            "expected": expected.tolist(),
            "actual": actual.tolist(),
            "ece": ece,
        },
    }
    
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
