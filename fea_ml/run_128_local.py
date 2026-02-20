#!/usr/bin/env python
"""
Generate native 128³ data for test-set samples, evaluate, and optimize one house.

Steps:
  1. Read test split from runs/v2/splits.json
  2. Generate native 128³ voxelizations (skip if already exist)
  3. Evaluate the v2 ensemble on all 1,430 test samples at native 128³
  4. Pick a sample and run CMA-ES optimization
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent          # fea_ml/
RUNS_V2 = ROOT / "runs" / "v2"
SPLITS = RUNS_V2 / "splits.json"
NORM_JSON = RUNS_V2 / "normalization.json"
CONFIG_YAML = RUNS_V2 / "config.yaml"
ENSEMBLE_JSON = RUNS_V2 / "ensemble_paths.json"
DATA_128 = ROOT / "data" / "runs_real_128"
EVAL_OUT = RUNS_V2 / "eval_128"
OPT_OUT = RUNS_V2 / "optimization_128"

PARTS_DIR = ROOT.parent / "optimization" / "data" / "3dwire_parts_combined"
FEA_DIR = ROOT.parent / "optimization" / "fea_gmsh_run" / "fea_results"

TARGET_NAMES = ["max_von_mises", "max_displacement", "min_safety_factor", "compliance"]
RESOLUTION = 128
N_WORKERS = 12

# ---------------------------------------------------------------------------
# Step 1 & 2: Generate 128³ data for test samples
# ---------------------------------------------------------------------------
def generate_test_128():
    """Generate native 128³ voxels for test-set samples."""
    from fea_ml.scripts.prepare_real_data import process_single_sample

    splits = json.load(open(SPLITS))
    test_ids = [p.split("/")[-1] for p in splits["test"]]
    
    # Filter to only those not already generated
    missing = [sid for sid in test_ids if not (DATA_128 / sid / "occ.npz").exists()]
    print(f"\nTest samples: {len(test_ids)} total, {len(test_ids) - len(missing)} already exist, {len(missing)} to generate")
    
    if not missing:
        print("All test samples already at 128³ — skipping generation.")
        return test_ids
    
    DATA_128.mkdir(parents=True, exist_ok=True)
    
    success = 0
    failed = 0
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(
                process_single_sample,
                sid, PARTS_DIR, FEA_DIR, DATA_128, RESOLUTION, 30e6,
            ): sid
            for sid in missing
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating 128³"):
            sid = futures[fut]
            try:
                ok, reason = fut.result()
                if ok:
                    success += 1
                else:
                    failed += 1
                    print(f"  SKIP {sid}: {reason}")
            except Exception:
                failed += 1
                print(f"  ERROR {sid}: {traceback.format_exc()}")
    
    elapsed = time.time() - t0
    print(f"Generation complete: {success} OK, {failed} failed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return test_ids


# ---------------------------------------------------------------------------
# Step 3: Evaluate
# ---------------------------------------------------------------------------
def _build_feature_vector(norm_dict: dict) -> np.ndarray:
    """Build the normalized 10-dim feature vector matching training pipeline.
    
    Raw features: [E/1e11, nu, rho/1000, sigma_y/1e7, mat_onehot(3), load_onehot(3)]
    All samples use: concrete, combined load, E=25GPa, nu=0.2, rho=2400, sigma_y=30MPa
    """
    raw_features = np.array([
        25e9 / 1e11,    # youngs_modulus / 1e11 = 0.25
        0.20,            # poisson_ratio
        2400.0 / 1000,   # density / 1000 = 2.4
        30e6 / 1e7,      # yield_stress / 1e7 = 3.0
        1.0, 0.0, 0.0,  # material one-hot: concrete
        0.0, 0.0, 1.0,  # load case one-hot: combined
    ], dtype=np.float32)
    
    # Normalize features using training stats
    feat_mean = np.array(norm_dict["feature_mean"], dtype=np.float32)
    feat_std = np.array(norm_dict["feature_std"], dtype=np.float32)
    features = (raw_features - feat_mean) / (feat_std + 1e-8)
    return features


def _build_voxel_input(occ: np.ndarray, part: np.ndarray) -> np.ndarray:
    """Build multi-channel voxel input: [occ, part_0, ..., part_5] = 7 channels."""
    NUM_PARTS = 6
    channels = [occ[None, ...].astype(np.float32)]
    for p in range(NUM_PARTS):
        channels.append((part == p).astype(np.float32)[None, ...])
    return np.concatenate(channels, axis=0)  # (7, D, H, W)


def _predict_ensemble_sequential(models, voxel_batch, feat_batch, device):
    """Run ensemble inference by moving one model at a time to GPU.
    
    This avoids having all 5 models (~875MB) on GPU simultaneously.
    Returns (mean, std) as numpy arrays of shape (B, T).
    """
    predictions = []
    for model in models:
        model.to(device)
        with torch.no_grad():
            pred = model(voxel_batch, feat_batch)
            predictions.append(pred.cpu().numpy())
        model.cpu()
        torch.cuda.empty_cache()
    
    stacked = np.stack(predictions, axis=0)  # (N, B, T)
    return stacked.mean(axis=0), stacked.std(axis=0)


def evaluate(test_ids: list[str], models: list, device, norm_dict: dict):
    """Evaluate on native 128³ test data."""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    # Build normalized 10-dim feature vector
    features = _build_feature_vector(norm_dict)
    
    # Collect samples that actually exist at 128³
    valid_ids = [sid for sid in test_ids if (DATA_128 / sid / "occ.npz").exists()]
    print(f"\nEvaluating on {len(valid_ids)}/{len(test_ids)} test samples at native 128³")
    
    all_preds = []
    all_trues = []
    all_uncs = []
    
    # Process one sample at a time — move each model to GPU sequentially
    for i in tqdm(range(len(valid_ids)), desc="Evaluating"):
        sid = valid_ids[i]
        sample_dir = DATA_128 / sid
        occ = np.load(sample_dir / "occ.npz")["data"].astype(np.float32)
        part = np.load(sample_dir / "part.npz")["data"]
        targets = json.load(open(sample_dir / "targets.json"))
        
        # Build 7-channel input
        voxel_input = _build_voxel_input(occ, part)
        voxel_batch = torch.from_numpy(voxel_input).unsqueeze(0).to(device)  # (1, 7, 128, 128, 128)
        feat_batch = torch.from_numpy(features[None, :]).to(device)  # (1, 10)
        target_vals = [targets[k] for k in TARGET_NAMES]
        
        preds_mean, preds_std = _predict_ensemble_sequential(models, voxel_batch, feat_batch, device)
        
        del voxel_batch, feat_batch
        torch.cuda.empty_cache()
        
        all_preds.append(preds_mean)
        all_trues.append([target_vals])
        all_uncs.append(preds_std)
    
    preds = np.concatenate(all_preds)   # (N, 4) — normalized space
    trues = np.concatenate(all_trues)   # (N, 4) — physical space
    uncs  = np.concatenate(all_uncs)
    
    # Model outputs are in normalized (log1p + z-score) space
    # Ground truth is in physical space
    # To compare, denormalize predictions to physical space
    t_mean = np.array(norm_dict["target_mean"])
    t_std  = np.array(norm_dict["target_std"])
    
    # Also normalize ground truth for normalized-space metrics
    trues_log = np.log1p(trues)
    trues_norm = (trues_log - t_mean) / t_std
    
    # Denormalize predictions to physical space
    preds_log = preds * t_std + t_mean
    preds_phys = np.expm1(preds_log)
    
    # Denormalize uncertainties
    uncs_phys = uncs * t_std
    uncs_phys = np.expm1(preds_log + uncs_phys) - preds_phys  # approx
    
    print("\n--- Per-target metrics (physical units, native 128³) ---")
    print(f"{'Target':<25s} {'MAE':>14s} {'RMSE':>14s} {'R²':>8s} {'Mean Unc':>12s}")
    print("-" * 75)
    
    results = {"n_test_samples": len(valid_ids), "note": "Native 128³ evaluation", "metrics": {}}
    
    for i, name in enumerate(TARGET_NAMES):
        t = trues[:, i]
        p = preds_phys[:, i]
        u = np.abs(uncs_phys[:, i])
        
        mae = mean_absolute_error(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        r2 = r2_score(t, p)
        mean_unc = float(np.mean(u))
        
        # Normalized-space R²
        r2_norm = r2_score(trues_norm[:, i], preds[:, i])
        
        print(f"{name:<25s} {mae:>14.4f} {rmse:>14.4f} {r2:>8.4f} {mean_unc:>12.4f}")
        
        results["metrics"][name] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "r2_normalized": float(r2_norm),
            "mean_uncertainty": float(mean_unc),
        }
    
    print(f"\n--- Per-target R² in normalized (log1p + z-score) space ---\n")
    for name in TARGET_NAMES:
        print(f"  {name:<25s} R²(norm) = {results['metrics'][name]['r2_normalized']:.4f}")
    
    # Save
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    with open(EVAL_OUT / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Parity plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for i, (ax, name) in enumerate(zip(axes.flat, TARGET_NAMES)):
            t = trues[:, i]
            p = preds_phys[:, i]
            r2 = results["metrics"][name]["r2"]
            
            ax.scatter(t, p, alpha=0.15, s=8)
            lo = min(t.min(), p.min())
            hi = max(t.max(), p.max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            ax.set_title(f"{name}\nR²={r2:.4f}")
        
        fig.suptitle("Parity Plots — Native 128³", fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(EVAL_OUT / "parity_plots_128.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nParity plots saved to {EVAL_OUT / 'parity_plots_128.png'}")
    except Exception as e:
        print(f"Parity plot error: {e}")
    
    return results, valid_ids


# ---------------------------------------------------------------------------
# Step 4: Optimize one house
# ---------------------------------------------------------------------------
def optimize_one_house(sample_id: str, models: list, device, norm_dict: dict):
    """Run CMA-ES optimization on one house at native 128³."""
    import yaml
    from fea_ml.optim.voxel_optimizer import OptimizationConfig, run_optimization
    from fea_ml.geometry.voxelize import VoxelGrids
    
    # Create a lightweight wrapper that the optimizer recognizes as an ensemble
    class SequentialEnsemble:
        """Wrapper that runs models one at a time on GPU to avoid OOM."""
        def __init__(self, models, device):
            self.models = models
            self.device = device
            self.n_models = len(models)
        
        def eval(self):
            for m in self.models:
                m.eval()
            return self
        
        def predict_with_uncertainty(self, voxel, features):
            from fea_ml.models.ensemble import EnsemblePrediction
            predictions = []
            for model in self.models:
                model.to(self.device)
                with torch.no_grad():
                    pred = model(voxel, features)
                    predictions.append(pred.cpu().numpy())
                model.cpu()
                torch.cuda.empty_cache()
            stacked = np.stack(predictions, axis=0)
            return EnsemblePrediction(
                mean=stacked.mean(axis=0),
                std=stacked.std(axis=0),
                predictions=np.empty((0,)),
            )
    
    seq_ensemble = SequentialEnsemble(models, device)
    
    config = yaml.safe_load(open(CONFIG_YAML))
    
    sample_dir = DATA_128 / sample_id
    occ = np.load(sample_dir / "occ.npz")["data"]
    part = np.load(sample_dir / "part.npz")["data"]
    edit_mask = np.load(sample_dir / "edit_mask.npz")["data"]
    protected_mask = np.load(sample_dir / "protected_mask.npz")["data"]
    targets = json.load(open(sample_dir / "targets.json"))
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION — Sample {sample_id} (native 128³)")
    print(f"{'='*60}")
    print(f"  Volume: {int(occ.sum())} voxels")
    for k in TARGET_NAMES:
        print(f"  {k}: {targets[k]:.4f}")
    
    grids = VoxelGrids(
        occ=occ.astype(np.uint8),
        sdf=None,
        part=part.astype(np.uint8),
        edit_mask=edit_mask.astype(np.uint8),
        protected_mask=protected_mask.astype(np.uint8),
        bounds=(np.zeros(3), np.ones(3) * 128),
        voxel_size=1.0,
    )
    
    baseline_targets = {k: targets[k] for k in TARGET_NAMES}
    features = _build_feature_vector(norm_dict)
    
    opt_cfg = config.get("optimization", {})
    opt_config = OptimizationConfig(
        iterations=50,
        population_size=16,
        sigma0=opt_cfg.get("sigma0", 0.3),
        volume_weight=opt_cfg.get("volume_weight", 1.0),
        uncertainty_k=opt_cfg.get("uncertainty_k", 2.0),
        mc_samples=opt_cfg.get("mc_samples", 16),
        surface_area_weight=opt_cfg.get("surface_area_weight", 0.01),
        thin_feature_weight=opt_cfg.get("thin_feature_weight", 0.1),
        constraint_penalty=opt_cfg.get("constraint_penalty", 100.0),
        min_thickness_voxels=config.get("validation", {}).get("min_thickness_voxels", 2.0),
    )
    
    OPT_OUT.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRunning CMA-ES optimization (iters={opt_config.iterations}, pop={opt_config.population_size})...")
    
    # No target_resolution needed — data is already 128³ matching model
    result = run_optimization(
        model=seq_ensemble,
        grids=grids,
        features=features,
        config=opt_config,
        device=device,
        baseline_targets=baseline_targets,
        normalization_stats=norm_dict,
        output_dir=OPT_OUT,
    )
    
    print(f"\n{'='*50}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*50}")
    print(f"Success: {result.success}")
    print(f"Volume: {result.volume_original} -> {result.volume_optimized} voxels")
    print(f"Volume reduction: {result.volume_reduction:.1%}")
    print(f"Constraints satisfied: {result.constraints_satisfied}")
    
    if result.success:
        print(f"\nPredicted metrics (mean ± std):")
        for i, name in enumerate(TARGET_NAMES):
            mean = result.predicted_mean[i]
            std = result.predicted_std[i]
            print(f"  {name}: {mean:.4f} ± {std:.4f}")
    
    print(f"\nOptimization outputs saved to: {OPT_OUT}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ---- Step 1 & 2: Generate 128³ test data ----
    test_ids = generate_test_128()
    
    # ---- Load ensemble ----
    import yaml
    config = yaml.safe_load(open(CONFIG_YAML))
    norm_dict = json.load(open(NORM_JSON))
    ensemble_paths = json.load(open(ENSEMBLE_JSON))
    
    from fea_ml.models.cnn3d import Surrogate3DResNet
    from fea_ml.models.ensemble import DeepEnsemble
    
    model_cfg = config.get("model", {})
    
    # Load models on CPU to avoid GPU OOM (5 × 175MB = 875MB just for weights)
    models = []
    for ckpt_path in ensemble_paths:
        full_path = ROOT / ckpt_path if not os.path.isabs(ckpt_path) else Path(ckpt_path)
        if not full_path.exists():
            full_path = RUNS_V2 / Path(ckpt_path).name
        
        model = Surrogate3DResNet(
            in_channels=7,
            feature_dim=10,
            target_dim=4,
            base_channels=model_cfg.get("base_channels", 64),
            dropout=model_cfg.get("dropout", 0.15),
            drop_path=model_cfg.get("drop_path", 0.1),
        )
        ckpt = torch.load(full_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", ckpt))
        model.load_state_dict(state)
        model.eval()
        models.append(model)  # keep on CPU
    
    n_params = sum(p.numel() for p in models[0].parameters())
    print(f"Ensemble loaded: {len(models)} models, {n_params:,} params each (on CPU)\n")
    
    # ---- Step 3: Evaluate ----
    eval_results_file = EVAL_OUT / "evaluation_results.json"
    if eval_results_file.exists():
        print(f"Evaluation results already exist at {eval_results_file} — skipping eval.")
        valid_ids = test_ids  # assume all valid
    else:
        results, valid_ids = evaluate(test_ids, models, device, norm_dict)
    
    # ---- Step 4: Optimize one house ----
    # Pick a sample with decent volume (not too small/big)
    sample_volumes = []
    for sid in valid_ids[:100]:  # check first 100
        occ = np.load(DATA_128 / sid / "occ.npz")["data"]
        vol = int(occ.sum())
        sample_volumes.append((sid, vol))
    
    # Pick one near the median volume
    sample_volumes.sort(key=lambda x: x[1])
    median_idx = len(sample_volumes) // 2
    opt_sid, opt_vol = sample_volumes[median_idx]
    print(f"\nSelected sample {opt_sid} for optimization (volume={opt_vol} voxels, median of checked)")
    
    optimize_one_house(opt_sid, models, device, norm_dict)
    
    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
