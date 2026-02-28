#!/usr/bin/env python3
"""
Compute surrogate performance metrics (MAE, MAPE, R²) on the held-out test split.
Outputs both normalized-space and denormalized (physical) metrics per target.

Usage:
    cd fea_ml
    python compute_surrogate_metrics.py
"""
import sys, os, json, gc
import numpy as np
import torch
from pathlib import Path
import yaml

# ── Paths ─────────────────────────────────────────────────────
RUNS_V3 = Path("runs/v3")
CHECKPOINT_DIR = RUNS_V3 / "ensemble"
DATA_DIR = Path("data/runs_real_128")

config = yaml.safe_load(open(RUNS_V3 / "config.yaml"))
norm_dict = json.load(open(RUNS_V3 / "normalization.json"))
splits = json.load(open(RUNS_V3 / "splits.json"))

target_names = config["targets"]  # ['max_von_mises', 'max_displacement', 'compliance']
NUM_PARTS = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load ensemble ─────────────────────────────────────────────
from fea_ml.models.cnn3d import Surrogate3DResNet

model_cfg = config.get("model", {})
members = sorted(CHECKPOINT_DIR.glob("ensemble_member_*.pt"))
models = []
for mp in members:
    ckpt = torch.load(mp, map_location="cpu", weights_only=False)
    m = Surrogate3DResNet(
        in_channels=7, feature_dim=10,
        target_dim=len(target_names),
        base_channels=model_cfg.get("base_channels", 64),
        dropout=model_cfg.get("dropout", 0.15),
        drop_path=model_cfg.get("drop_path", 0.1),
    )
    state = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", ckpt))
    m.load_state_dict(state)
    m.eval()
    m.to(device)
    models.append(m)
    del ckpt, state
    gc.collect()
print(f"Loaded {len(models)} ensemble members")

# ── Normalization ────────────────────────────────────────────
feat_mean = np.array(norm_dict["feature_mean"], dtype=np.float32)
feat_std = np.array(norm_dict["feature_std"], dtype=np.float32)
target_mean = np.array(norm_dict["target_mean"], dtype=np.float32)
target_std = np.array(norm_dict["target_std"], dtype=np.float32)
log_targets = norm_dict.get("log_transform_targets", [])

raw_features = np.array([
    25e9/1e11, 0.20, 2400.0/1000, 30e6/1e7,
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
], dtype=np.float32)
features_norm = (raw_features - feat_mean) / (feat_std + 1e-8)


def build_voxel_input(occ_arr, part_arr):
    channels = [occ_arr[None].astype(np.float32)]
    for p in range(NUM_PARTS):
        channels.append((part_arr == p).astype(np.float32)[None])
    return np.concatenate(channels, axis=0)


def ensemble_predict(voxel_np, feat_np):
    """Returns ensemble mean prediction in NORMALIZED space."""
    vt = torch.from_numpy(voxel_np[None]).float().to(device)
    ft = torch.from_numpy(feat_np[None]).float().to(device)
    preds = []
    for model in models:
        with torch.no_grad():
            p = model(vt, ft).cpu().numpy()
        preds.append(p)
    del vt, ft
    if device.type == "cuda":
        torch.cuda.empty_cache()
    stacked = np.stack(preds, axis=0)
    return stacked.mean(axis=0).squeeze(0), stacked.std(axis=0).squeeze(0)


def denorm_targets(normalized):
    """Denormalize from z-score + log space to physical units."""
    val = normalized * target_std + target_mean
    # Undo log1p: log1p(|x|) stored, so expm1 to get |x|
    result = np.expm1(val)
    return result


# ── Get test samples ─────────────────────────────────────────
test_dirs = [Path(d) for d in splits["test"]]
print(f"Test split: {len(test_dirs)} samples")

# ── Evaluate ──────────────────────────────────────────────────
all_true = []
all_pred_mean = []
all_pred_std = []
skipped = 0

for i, run_dir in enumerate(test_dirs):
    if not run_dir.exists():
        skipped += 1
        continue
    
    occ_path = run_dir / "occ.npz"
    part_path = run_dir / "part.npz"
    targets_path = run_dir / "targets.json"
    
    if not occ_path.exists() or not targets_path.exists():
        skipped += 1
        continue
    
    try:
        occ = np.load(occ_path)["data"].astype(np.uint8)
        part = np.load(part_path)["data"].astype(np.uint8) if part_path.exists() else np.zeros_like(occ)
        with open(targets_path) as f:
            tgt = json.load(f)
        
        # Extract ground truth in physical units
        true_vals = np.array([tgt[n] for n in target_names], dtype=np.float32)
        
        # Skip invalid data (same filter as training)
        if true_vals[1] > 1.0:  # displacement > 1.0m = diverged solver
            skipped += 1
            continue
        if true_vals[2] < 1e-6:  # compliance < 1e-6 = degenerate
            skipped += 1
            continue
        if true_vals[0] <= 0:  # VM stress <= 0 = invalid
            skipped += 1
            continue
        
        vi = build_voxel_input(occ, part)
        pmean, pstd = ensemble_predict(vi, features_norm)
        
        # Denormalize predictions
        pred_physical = denorm_targets(pmean)
        pred_std_physical = pstd * target_std  # approximate
        
        all_true.append(true_vals)
        all_pred_mean.append(pred_physical)
        all_pred_std.append(pred_std_physical)
        
    except Exception as e:
        print(f"  Error on {run_dir.name}: {e}")
        skipped += 1
        continue
    
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(test_dirs)} ({skipped} skipped)")

print(f"\nEvaluated {len(all_true)} test samples ({skipped} skipped)")

all_true = np.array(all_true)       # (N, 3)
all_pred_mean = np.array(all_pred_mean)  # (N, 3)

# ── Compute metrics ──────────────────────────────────────────
print("\n" + "="*70)
print("SURROGATE PERFORMANCE METRICS (Test Set, Physical Units)")
print("="*70)

target_units = ["Pa", "m", "J"]
target_labels = ["Peak Von Mises Stress", "Max Displacement", "Compliance"]

results = {}
for i, (name, label, unit) in enumerate(zip(target_names, target_labels, target_units)):
    true = all_true[:, i]
    pred = all_pred_mean[:, i]
    
    # MAE
    mae = np.mean(np.abs(true - pred))
    
    # MAPE (avoid division by zero)
    mask = np.abs(true) > 1e-12
    mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100
    
    # RMSE
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    
    # R²
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    
    # Median absolute error
    medae = np.median(np.abs(true - pred))
    
    # Spearman rank correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(true, pred)
    
    results[name] = {
        "mae": mae, "mape": mape, "rmse": rmse, "r2": r2,
        "medae": medae, "spearman": rho, "spearman_pval": pval,
        "mean_true": np.mean(true), "std_true": np.std(true),
        "mean_pred": np.mean(pred), "std_pred": np.std(pred),
    }
    
    print(f"\n{label} ({unit}):")
    print(f"  MAE       = {mae:.4g} {unit}")
    print(f"  MedAE     = {medae:.4g} {unit}")
    print(f"  MAPE      = {mape:.1f}%")
    print(f"  RMSE      = {rmse:.4g} {unit}")
    print(f"  R²        = {r2:.4f}")
    print(f"  Spearman  = {rho:.4f} (p={pval:.2e})")
    print(f"  True mean = {np.mean(true):.4g} ± {np.std(true):.4g}")
    print(f"  Pred mean = {np.mean(pred):.4g} ± {np.std(pred):.4g}")

# ── Save results ──────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(RUNS_V3 / "surrogate_metrics.json", "w") as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)

# Save raw predictions for plotting
np.savez(
    RUNS_V3 / "test_predictions.npz",
    true=all_true,
    pred_mean=all_pred_mean,
    target_names=target_names,
)

print(f"\nSaved metrics to {RUNS_V3 / 'surrogate_metrics.json'}")
print(f"Saved predictions to {RUNS_V3 / 'test_predictions.npz'}")

# ── Print LaTeX table ─────────────────────────────────────────
print("\n" + "="*70)
print("LATEX TABLE:")
print("="*70)
print(r"""
\begin{table}[t]
\centering
\caption{Surrogate model performance on 1,114 held-out test samples (physical units). Metrics are computed on denormalized predictions.}
\label{tab:surrogate_metrics}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Target} & \textbf{MAE} & \textbf{MAPE (\%)} & \textbf{R²} & \textbf{Spearman $\rho$} \\
\midrule""")

for i, (name, label, unit) in enumerate(zip(target_names, target_labels, target_units)):
    r = results[name]
    mae_str = f"{r['mae']:.3g} {unit}" if r['mae'] > 0.01 else f"{r['mae']:.3e} {unit}"
    print(f"{label} & {mae_str} & {r['mape']:.1f} & {r['r2']:.3f} & {r['spearman']:.3f} \\\\")

print(r"""\bottomrule
\end{tabular}
\end{table}""")
