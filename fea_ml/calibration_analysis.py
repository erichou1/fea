#!/usr/bin/env python3
"""
Phase 1 & 2: Isotonic calibration, k-factor ablation, and ΓD safety-trigger analysis.

Produces:
  - Isotonic calibration mapping (test set: pred_mean → true)
  - Calibrated batch-result feasibility rates
  - k-factor ablation sweep
  - ΓD (ensemble disagreement) trigger statistics
  - 100-design stratified sample detailed report

Usage:
    cd fea_ml
    python calibration_analysis.py
"""
import json, os, sys, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
RUNS_V3 = Path("runs/v3")
BATCH_DIR = RUNS_V3 / "batch_results_all"
TEST_PRED = RUNS_V3 / "test_predictions.npz"
NORM_JSON = RUNS_V3 / "normalization.json"

# Constraints (same as run_batch_all.py)
MAX_VON_MISES = 5.0e6
MAX_DISPLACEMENT = 1.0
MAX_COMPLIANCE_RATIO = 1.15
UNCERTAINTY_K_DEFAULT = 1.0

TARGET_NAMES = ["max_von_mises", "max_displacement", "compliance"]
TARGET_LABELS = ["Von Mises Stress (Pa)", "Max Displacement (m)", "Compliance (J)"]

# ──────────────────────────────────────────────────────────────
# 1. Load test-set predictions
# ──────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 1: ISOTONIC CALIBRATION")
print("=" * 70)

tp = np.load(TEST_PRED)
true_test = tp["true"]        # (1114, 3)  physical units
pred_test = tp["pred_mean"]   # (1114, 3)  physical units
N_test = true_test.shape[0]
print(f"Test set: {N_test} samples")

# Split test set: first 50% for calibration, second 50% for validation
np.random.seed(42)
idx = np.random.permutation(N_test)
n_cal = N_test // 2
cal_idx = idx[:n_cal]
val_idx = idx[n_cal:]
print(f"  Calibration subset: {len(cal_idx)}, Validation subset: {len(val_idx)}")

# ──────────────────────────────────────────────────────────────
# 2. Fit isotonic regression for each target
# ──────────────────────────────────────────────────────────────
from sklearn.isotonic import IsotonicRegression

iso_models = []
print("\nIsotonic calibration (pred → true mapping):")
for t in range(3):
    cal_pred = pred_test[cal_idx, t]
    cal_true = true_test[cal_idx, t]

    iso = IsotonicRegression(y_min=0.0, out_of_bounds="clip")
    iso.fit(cal_pred, cal_true)
    iso_models.append(iso)

    # Validate on held-out half
    val_pred_raw = pred_test[val_idx, t]
    val_true = true_test[val_idx, t]
    val_pred_cal = iso.predict(val_pred_raw)

    # Metrics before calibration
    mae_before = np.mean(np.abs(val_true - val_pred_raw))
    mape_before = np.mean(np.abs((val_true - val_pred_raw) / (val_true + 1e-12))) * 100
    ss_res_b = np.sum((val_true - val_pred_raw) ** 2)
    ss_tot = np.sum((val_true - np.mean(val_true)) ** 2)
    r2_before = 1 - ss_res_b / (ss_tot + 1e-8)
    bias_before = np.mean(val_pred_raw - val_true)

    # Metrics after calibration
    mae_after = np.mean(np.abs(val_true - val_pred_cal))
    mape_after = np.mean(np.abs((val_true - val_pred_cal) / (val_true + 1e-12))) * 100
    ss_res_a = np.sum((val_true - val_pred_cal) ** 2)
    r2_after = 1 - ss_res_a / (ss_tot + 1e-8)
    bias_after = np.mean(val_pred_cal - val_true)

    print(f"\n  {TARGET_LABELS[t]}:")
    print(f"    BEFORE: MAE={mae_before:.4g}, MAPE={mape_before:.1f}%, R²={r2_before:.4f}, Bias={bias_before:.4g}")
    print(f"    AFTER:  MAE={mae_after:.4g}, MAPE={mape_after:.1f}%, R²={r2_after:.4f}, Bias={bias_after:.4g}")
    print(f"    MAE reduction: {(1 - mae_after/mae_before)*100:.1f}%")

# Refit on FULL test set for final calibration (more data)
iso_models_full = []
for t in range(3):
    iso = IsotonicRegression(y_min=0.0, out_of_bounds="clip")
    iso.fit(pred_test[:, t], true_test[:, t])
    iso_models_full.append(iso)
print("\nRefitted isotonic models on full test set for production use.")

# ──────────────────────────────────────────────────────────────
# 3. Load all batch results
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("LOADING BATCH RESULTS")
print("=" * 70)

batch_results = []
for d in sorted(BATCH_DIR.iterdir()):
    summary_path = d / "optimization_summary.json"
    if not summary_path.exists():
        continue
    with open(summary_path) as f:
        r = json.load(f)
    if not r.get("success", False):
        continue
    batch_results.append(r)

print(f"Loaded {len(batch_results)} successful batch results")

# ──────────────────────────────────────────────────────────────
# 4. Apply isotonic calibration to batch results
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("APPLYING ISOTONIC CALIBRATION TO BATCH RESULTS")
print("=" * 70)

def check_constraints(vm_val, disp_val, comp_val, comp_limit):
    """Check if values satisfy all constraints."""
    return (vm_val <= MAX_VON_MISES and
            disp_val <= MAX_DISPLACEMENT and
            comp_val <= comp_limit)

original_satisfied = 0
calibrated_satisfied = 0
calibrated_relaxed_satisfied = 0  # Using calibrated mean (no k padding)
results_detailed = []

for r in batch_results:
    pm = np.array(r["pred_mean"])   # [VM, disp, compliance] physical
    ps = np.array(r["pred_std"])    # [VM, disp, compliance] physical
    comp_limit = r["comp_limit"]
    baseline = r["baseline_targets"]

    # Original conservative values (mean + k*std)
    vm_orig = pm[0] + UNCERTAINTY_K_DEFAULT * ps[0]
    disp_orig = pm[1] + UNCERTAINTY_K_DEFAULT * ps[1]
    comp_orig = pm[2] + UNCERTAINTY_K_DEFAULT * ps[2]
    orig_ok = check_constraints(vm_orig, disp_orig, comp_orig, comp_limit)

    # Calibrated predictions (apply isotonic to pred_mean)
    pm_cal = np.array([
        iso_models_full[0].predict([pm[0]])[0],
        iso_models_full[1].predict([pm[1]])[0],
        iso_models_full[2].predict([pm[2]])[0],
    ])

    # Calibrated conservative values
    vm_cal = pm_cal[0] + UNCERTAINTY_K_DEFAULT * ps[0]
    disp_cal = pm_cal[1] + UNCERTAINTY_K_DEFAULT * ps[1]
    comp_cal = pm_cal[2] + UNCERTAINTY_K_DEFAULT * ps[2]
    cal_ok = check_constraints(vm_cal, disp_cal, comp_cal, comp_limit)

    # Also check calibrated mean without k-padding (for reference)
    cal_mean_ok = check_constraints(pm_cal[0], pm_cal[1], pm_cal[2], comp_limit)

    original_satisfied += int(orig_ok)
    calibrated_satisfied += int(cal_ok)
    calibrated_relaxed_satisfied += int(cal_mean_ok)

    results_detailed.append({
        "sample_id": r["sample_id"],
        "volume_reduction_pct": r["volume_reduction_pct"],
        "orig_constraints_ok": bool(orig_ok),
        "cal_constraints_ok": bool(cal_ok),
        "cal_mean_ok": bool(cal_mean_ok),
        "pred_mean": pm.tolist(),
        "pred_std": ps.tolist(),
        "pred_mean_cal": pm_cal.tolist(),
        "vm_orig": float(vm_orig),
        "vm_cal": float(vm_cal),
        "comp_orig": float(comp_orig),
        "comp_cal": float(comp_cal),
        "comp_limit": float(comp_limit),
        "vm_binding": bool(vm_orig > MAX_VON_MISES * 0.9),
        "comp_binding": bool(comp_orig > comp_limit * 0.9),
    })

N = len(batch_results)
print(f"\nConstraint satisfaction (N={N}):")
print(f"  Original (k=1.0):         {original_satisfied}/{N} = {original_satisfied/N*100:.1f}%")
print(f"  Calibrated (k=1.0):       {calibrated_satisfied}/{N} = {calibrated_satisfied/N*100:.1f}%")
print(f"  Calibrated (mean only):   {calibrated_relaxed_satisfied}/{N} = {calibrated_relaxed_satisfied/N*100:.1f}%")

# How many flipped from fail → pass?
flipped_to_pass = sum(1 for r in results_detailed if not r["orig_constraints_ok"] and r["cal_constraints_ok"])
flipped_to_fail = sum(1 for r in results_detailed if r["orig_constraints_ok"] and not r["cal_constraints_ok"])
print(f"\n  Flipped fail→pass: {flipped_to_pass}")
print(f"  Flipped pass→fail: {flipped_to_fail}")
print(f"  Net gain: {flipped_to_pass - flipped_to_fail}")

# ──────────────────────────────────────────────────────────────
# 5. k-factor ablation
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 2a: k-FACTOR ABLATION")
print("=" * 70)

k_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
k_results = {}

for k in k_values:
    n_ok = 0
    mean_reduction = []
    for r in batch_results:
        pm = np.array(r["pred_mean"])
        ps = np.array(r["pred_std"])
        comp_limit = r["comp_limit"]
        vm_c = pm[0] + k * ps[0]
        disp_c = pm[1] + k * ps[1]
        comp_c = pm[2] + k * ps[2]
        ok = check_constraints(vm_c, disp_c, comp_c, comp_limit)
        n_ok += int(ok)
        if ok:
            mean_reduction.append(r["volume_reduction_pct"])
    avg_red = np.mean(mean_reduction) if mean_reduction else 0
    k_results[k] = {"n_ok": n_ok, "pct": n_ok / N * 100, "avg_reduction": avg_red}
    print(f"  k={k:.2f}: {n_ok:4d}/{N} ({n_ok/N*100:5.1f}%), avg reduction = {avg_red:.1f}%")

# Also with calibration
print("\nWith isotonic calibration:")
for k in k_values:
    n_ok = 0
    mean_reduction = []
    for r in batch_results:
        pm = np.array(r["pred_mean"])
        ps = np.array(r["pred_std"])
        comp_limit = r["comp_limit"]
        pm_cal = np.array([
            iso_models_full[t].predict([pm[t]])[0] for t in range(3)
        ])
        vm_c = pm_cal[0] + k * ps[0]
        disp_c = pm_cal[1] + k * ps[1]
        comp_c = pm_cal[2] + k * ps[2]
        ok = check_constraints(vm_c, disp_c, comp_c, comp_limit)
        n_ok += int(ok)
        if ok:
            mean_reduction.append(r["volume_reduction_pct"])
    avg_red = np.mean(mean_reduction) if mean_reduction else 0
    print(f"  k={k:.2f}: {n_ok:4d}/{N} ({n_ok/N*100:5.1f}%), avg reduction = {avg_red:.1f}%")

# ──────────────────────────────────────────────────────────────
# 6. ΓD safety-trigger analysis
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 2b: ΓD ENSEMBLE DISAGREEMENT ANALYSIS")
print("=" * 70)

# ΓD = max over targets of (pred_std / pred_mean)
# This is the coefficient of variation — higher = more disagreement
gamma_d_values = []
for r in batch_results:
    pm = np.array(r["pred_mean"])
    ps = np.array(r["pred_std"])
    # Coefficient of variation per target, take max
    cv = ps / (np.abs(pm) + 1e-12)
    gamma_d = np.max(cv)
    gamma_d_values.append(gamma_d)

gamma_d_values = np.array(gamma_d_values)
print(f"\nΓD (max CV) distribution:")
print(f"  Mean:   {np.mean(gamma_d_values):.4f}")
print(f"  Median: {np.median(gamma_d_values):.4f}")
print(f"  Std:    {np.std(gamma_d_values):.4f}")
print(f"  P5:     {np.percentile(gamma_d_values, 5):.4f}")
print(f"  P25:    {np.percentile(gamma_d_values, 25):.4f}")
print(f"  P75:    {np.percentile(gamma_d_values, 75):.4f}")
print(f"  P95:    {np.percentile(gamma_d_values, 95):.4f}")
print(f"  Max:    {np.max(gamma_d_values):.4f}")

# Per-target CV breakdown
print("\nPer-target CV breakdown:")
for t in range(3):
    cvs = []
    for r in batch_results:
        pm = r["pred_mean"][t]
        ps = r["pred_std"][t]
        cvs.append(ps / (abs(pm) + 1e-12))
    cvs = np.array(cvs)
    print(f"  {TARGET_NAMES[t]:20s}: mean={np.mean(cvs):.4f}, median={np.median(cvs):.4f}, P95={np.percentile(cvs, 95):.4f}")

# ΓD threshold sweep: at what ΓD threshold do we trigger "reject / flag"?
print("\nΓD threshold ablation (reject designs with ΓD > threshold):")
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
for thresh in thresholds:
    accepted_mask = gamma_d_values <= thresh
    n_accepted = accepted_mask.sum()
    # Among accepted, how many satisfy constraints?
    accepted_ok = sum(1 for i, r in enumerate(batch_results)
                      if accepted_mask[i] and r["constraints_satisfied"])
    accepted_total = n_accepted
    feas_rate = accepted_ok / max(accepted_total, 1) * 100
    # Among rejected, how many were originally OK?
    rejected_mask = ~accepted_mask
    rejected_ok = sum(1 for i, r in enumerate(batch_results)
                      if rejected_mask[i] and r["constraints_satisfied"])
    print(f"  ΓD≤{thresh:.2f}: accept {n_accepted:4d}/{N} ({n_accepted/N*100:.1f}%), "
          f"feasibility among accepted = {feas_rate:.1f}%, "
          f"rejected-but-OK = {rejected_ok}")

# ──────────────────────────────────────────────────────────────
# 7. Stratified 100-design sample for detailed reporting
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("100-DESIGN STRATIFIED SAMPLE")
print("=" * 70)

# Sort by volume reduction for constraint-satisfied designs
satisfied = [r for r in results_detailed if r["orig_constraints_ok"]]
not_satisfied = [r for r in results_detailed if not r["orig_constraints_ok"]]



# Among satisfied: top 30 by reduction
satisfied_sorted = sorted(satisfied, key=lambda r: -r["volume_reduction_pct"])
top_30 = satisfied_sorted[:30]

# Near-boundary: designs where at least one constraint utilization > 90%
near_boundary = [r for r in results_detailed if r["vm_binding"] or r["comp_binding"]]
# Exclude those already in top_30
top_ids = {r["sample_id"] for r in top_30}
near_boundary = [r for r in near_boundary if r["sample_id"] not in top_ids]
np.random.seed(42)
if len(near_boundary) > 40:
    nb_idx = np.random.choice(len(near_boundary), 40, replace=False)
    near_40 = [near_boundary[i] for i in nb_idx]
else:
    near_40 = near_boundary[:40]

# Mid-range: random from remainder
sample_ids_used = top_ids | {r["sample_id"] for r in near_40}
remaining = [r for r in results_detailed if r["sample_id"] not in sample_ids_used]
if len(remaining) > 30:
    rem_idx = np.random.choice(len(remaining), 30, replace=False)
    mid_30 = [remaining[i] for i in rem_idx]
else:
    mid_30 = remaining[:30]

sample_100 = top_30 + near_40 + mid_30
print(f"  Top-30 highest reduction: {len(top_30)} designs")
print(f"  Near-boundary (40):       {len(near_40)} designs")
print(f"  Mid-range (30):           {len(mid_30)} designs")
print(f"  Total sample:             {len(sample_100)} designs\n")

# Summary statistics for the 100-design sample
orig_ok_100 = sum(1 for r in sample_100 if r["orig_constraints_ok"])
cal_ok_100 = sum(1 for r in sample_100 if r["cal_constraints_ok"])
cal_mean_100 = sum(1 for r in sample_100 if r["cal_mean_ok"])
reductions = [r["volume_reduction_pct"] for r in sample_100]

print(f"  Original constraints OK: {orig_ok_100}/{len(sample_100)} ({orig_ok_100/len(sample_100)*100:.1f}%)")
print(f"  Calibrated (k=1.0) OK:   {cal_ok_100}/{len(sample_100)} ({cal_ok_100/len(sample_100)*100:.1f}%)")
print(f"  Calibrated (mean) OK:    {cal_mean_100}/{len(sample_100)} ({cal_mean_100/len(sample_100)*100:.1f}%)")
print(f"  Avg volume reduction:    {np.mean(reductions):.1f}% (range: {np.min(reductions):.1f}% – {np.max(reductions):.1f}%)")

# Binding constraint analysis for the 100 designs
vm_binding_count = sum(1 for r in sample_100 if r["vm_binding"])
comp_binding_count = sum(1 for r in sample_100 if r["comp_binding"])
print(f"\n  VM binding (>90% util):   {vm_binding_count}")
print(f"  Comp binding (>90% util): {comp_binding_count}")

# How do calibrated predictions shift individual targets?
print("\n  Calibration shift analysis (100-design sample):")
for t in range(3):
    orig_vals = [r["pred_mean"][t] for r in sample_100]
    cal_vals = [r["pred_mean_cal"][t] for r in sample_100]
    shifts = [(c - o) / (abs(o) + 1e-12) * 100 for o, c in zip(orig_vals, cal_vals)]
    print(f"    {TARGET_NAMES[t]:20s}: mean shift = {np.mean(shifts):+.2f}%, median = {np.median(shifts):+.2f}%")

# ──────────────────────────────────────────────────────────────
# 8. Projected FEA validation table
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PROJECTED FEA VALIDATION (100-design sample)")
print("=" * 70)

# Use test-set residual statistics to estimate FEA outcomes
# Compute residual distribution on test set
residuals_pct = {}
for t in range(3):
    res = (true_test[:, t] - pred_test[:, t]) / (true_test[:, t] + 1e-12)
    residuals_pct[t] = {"mean": np.mean(res), "std": np.std(res),
                         "p5": np.percentile(res, 5), "p95": np.percentile(res, 95)}
    print(f"  Test residuals for {TARGET_NAMES[t]:20s}: "
          f"mean={residuals_pct[t]['mean']:+.3f}, std={residuals_pct[t]['std']:.3f}, "
          f"P5={residuals_pct[t]['p5']:+.3f}, P95={residuals_pct[t]['p95']:+.3f}")

# For each of the 100 designs, simulate worst-case FEA using P95 residual
print("\nProjected FEA outcomes using P95 residual:")
proj_ok_count = 0
for r in sample_100:
    pm = np.array(r["pred_mean"])
    comp_limit = r["comp_limit"]
    # Worst case: actual value is pred_mean * (1 + P95_residual) for upper-bound constraints
    vm_worst = pm[0] * (1 + residuals_pct[0]["p95"])
    disp_worst = pm[1] * (1 + residuals_pct[1]["p95"])
    comp_worst = pm[2] * (1 + residuals_pct[2]["p95"])
    ok = check_constraints(abs(vm_worst), abs(disp_worst), abs(comp_worst), comp_limit)
    proj_ok_count += int(ok)

print(f"  Projected FEA pass rate (P95 worst-case): {proj_ok_count}/{len(sample_100)} ({proj_ok_count/len(sample_100)*100:.1f}%)")

# Also with calibrated predictions
proj_cal_ok_count = 0
for r in sample_100:
    pm_cal = np.array(r["pred_mean_cal"])
    comp_limit = r["comp_limit"]
    vm_worst = pm_cal[0] * (1 + residuals_pct[0]["p95"])
    disp_worst = pm_cal[1] * (1 + residuals_pct[1]["p95"])
    comp_worst = pm_cal[2] * (1 + residuals_pct[2]["p95"])
    ok = check_constraints(abs(vm_worst), abs(disp_worst), abs(comp_worst), comp_limit)
    proj_cal_ok_count += int(ok)

print(f"  Projected FEA pass rate (calibrated, P95): {proj_cal_ok_count}/{len(sample_100)} ({proj_cal_ok_count/len(sample_100)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────
# 9. Save all results
# ──────────────────────────────────────────────────────────────
out = {
    "n_batch": N,
    "original_satisfied": original_satisfied,
    "original_pct": round(original_satisfied / N * 100, 1),
    "calibrated_satisfied": calibrated_satisfied,
    "calibrated_pct": round(calibrated_satisfied / N * 100, 1),
    "calibrated_mean_satisfied": calibrated_relaxed_satisfied,
    "calibrated_mean_pct": round(calibrated_relaxed_satisfied / N * 100, 1),
    "flipped_fail_to_pass": flipped_to_pass,
    "flipped_pass_to_fail": flipped_to_fail,
    "k_ablation": {str(k): v for k, v in k_results.items()},
    "gamma_d_stats": {
        "mean": float(np.mean(gamma_d_values)),
        "median": float(np.median(gamma_d_values)),
        "std": float(np.std(gamma_d_values)),
        "p5": float(np.percentile(gamma_d_values, 5)),
        "p95": float(np.percentile(gamma_d_values, 95)),
    },
    "sample_100": {
        "n_total": len(sample_100),
        "n_orig_ok": int(orig_ok_100),
        "n_cal_ok": int(cal_ok_100),
        "n_cal_mean_ok": int(cal_mean_100),
        "avg_reduction": float(np.mean(reductions)),
        "projected_fea_pass_p95": int(proj_ok_count),
        "projected_fea_pass_cal_p95": int(proj_cal_ok_count),
    },
    "test_residual_stats": {name: {k: float(v) for k, v in stats.items()}
                            for name, stats in zip(TARGET_NAMES, [residuals_pct[i] for i in range(3)])},
}

out_path = RUNS_V3 / "calibration_results.json"
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

with open(out_path, "w") as f:
    json.dump(out, f, indent=2, cls=NpEncoder)
print(f"\nSaved calibration results to {out_path}")

# Save 100-design sample details
sample_out_path = RUNS_V3 / "sample_100_details.json"
with open(sample_out_path, "w") as f:
    json.dump(sample_100, f, indent=2, cls=NpEncoder)
print(f"Saved 100-design sample details to {sample_out_path}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
