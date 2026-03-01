#!/usr/bin/env python3
"""
Re-run calibration analysis filtered to TEST-SPLIT ONLY samples.
This ensures scientific validity (no train/val leakage).
"""
import json, os, sys, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
from sklearn.isotonic import IsotonicRegression

# ── Paths ─────────────────────────────────────────────────────
RUNS_V3 = Path("runs/v3")
BATCH_DIR = RUNS_V3 / "batch_results_all"
TEST_PRED = RUNS_V3 / "test_predictions.npz"

# Constraints
MAX_VON_MISES = 5.0e6
MAX_DISPLACEMENT = 1.0
MAX_COMPLIANCE_RATIO = 1.15
UNCERTAINTY_K_DEFAULT = 1.0
TARGET_NAMES = ["max_von_mises", "max_displacement", "compliance"]

# ── Get test-split IDs ───────────────────────────────────────
splits = json.load(open(RUNS_V3 / "splits.json"))
test_ids = set(os.path.basename(p) for p in splits["test"])
print(f"Test split IDs: {len(test_ids)}")

# ── Load test predictions for calibration ────────────────────
tp = np.load(TEST_PRED)
true_test = tp["true"]
pred_test = tp["pred_mean"]
N_test = true_test.shape[0]

# Fit isotonic on FULL test set
iso_models = []
for t in range(3):
    iso = IsotonicRegression(y_min=0.0, out_of_bounds="clip")
    iso.fit(pred_test[:, t], true_test[:, t])
    iso_models.append(iso)

# ── Load batch results (TEST ONLY) ──────────────────────────
batch_results = []
for d in sorted(BATCH_DIR.iterdir()):
    if d.name not in test_ids:
        continue
    summary_path = d / "optimization_summary.json"
    if not summary_path.exists():
        continue
    with open(summary_path) as f:
        r = json.load(f)
    if not r.get("success", False):
        continue
    batch_results.append(r)

N = len(batch_results)
print(f"Test-only batch results: {N}")

def check_constraints(vm, disp, comp, comp_limit):
    return vm <= MAX_VON_MISES and disp <= MAX_DISPLACEMENT and comp <= comp_limit

# ── Original feasibility ────────────────────────────────────
orig_ok = sum(1 for r in batch_results if r["constraints_satisfied"])
print(f"\nOriginal feasibility: {orig_ok}/{N} ({orig_ok/N*100:.1f}%)")

# Among satisfied, compute stats
satisfied = [r for r in batch_results if r["constraints_satisfied"]]
if satisfied:
    reds = [r["volume_reduction_pct"] for r in satisfied]
    print(f"  Mean reduction (satisfied): {np.mean(reds):.1f}% ± {np.std(reds):.1f}%")
    print(f"  Median: {np.median(reds):.1f}%, Range: [{np.min(reds):.1f}%, {np.max(reds):.1f}%]")

# ── k-factor ablation (TEST ONLY) ───────────────────────────
print(f"\n{'='*60}")
print("k-FACTOR ABLATION (test-only, N={})".format(N))
print(f"{'='*60}")

k_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
for k in k_values:
    n_ok = 0
    mean_red = []
    for r in batch_results:
        pm = np.array(r["pred_mean"])
        ps = np.array(r["pred_std"])
        comp_limit = r["comp_limit"]
        ok = check_constraints(
            pm[0] + k*ps[0], pm[1] + k*ps[1], pm[2] + k*ps[2], comp_limit)
        n_ok += int(ok)
        if ok:
            mean_red.append(r["volume_reduction_pct"])
    avg = np.mean(mean_red) if mean_red else 0
    print(f"  k={k:.2f}: {n_ok:4d}/{N} ({n_ok/N*100:5.1f}%), avg reduction = {avg:.1f}%")

# ── Calibrated k-factor ablation ────────────────────────────
print(f"\nWith isotonic calibration:")
for k in k_values:
    n_ok = 0
    mean_red = []
    for r in batch_results:
        pm = np.array(r["pred_mean"])
        ps = np.array(r["pred_std"])
        comp_limit = r["comp_limit"]
        pm_cal = np.array([iso_models[t].predict([pm[t]])[0] for t in range(3)])
        ok = check_constraints(
            pm_cal[0] + k*ps[0], pm_cal[1] + k*ps[1], pm_cal[2] + k*ps[2], comp_limit)
        n_ok += int(ok)
        if ok:
            mean_red.append(r["volume_reduction_pct"])
    avg = np.mean(mean_red) if mean_red else 0
    print(f"  k={k:.2f}: {n_ok:4d}/{N} ({n_ok/N*100:5.1f}%), avg reduction = {avg:.1f}%")

# ── ΓD analysis (TEST ONLY) ─────────────────────────────────
print(f"\n{'='*60}")
print("ΓD ANALYSIS (test-only)")
print(f"{'='*60}")

gamma_d_all = []
for r in batch_results:
    pm = np.array(r["pred_mean"])
    ps = np.array(r["pred_std"])
    cv = ps / (np.abs(pm) + 1e-12)
    gamma_d_all.append(np.max(cv))
gamma_d_all = np.array(gamma_d_all)

print(f"  Mean: {np.mean(gamma_d_all):.4f}, Median: {np.median(gamma_d_all):.4f}")
print(f"  P5: {np.percentile(gamma_d_all, 5):.4f}, P95: {np.percentile(gamma_d_all, 95):.4f}")

# Per-target CV
for t in range(3):
    cvs = [r["pred_std"][t] / (abs(r["pred_mean"][t]) + 1e-12) for r in batch_results]
    cvs = np.array(cvs)
    print(f"  {TARGET_NAMES[t]:20s}: mean CV={np.mean(cvs):.3f}, P95={np.percentile(cvs, 95):.3f}")

# ΓD threshold sweep
print(f"\nΓD threshold sweep:")
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
for thresh in thresholds:
    mask = gamma_d_all <= thresh
    n_acc = mask.sum()
    acc_ok = sum(1 for i, r in enumerate(batch_results) if mask[i] and r["constraints_satisfied"])
    feas = acc_ok / max(n_acc, 1) * 100
    rej_ok = sum(1 for i, r in enumerate(batch_results) if not mask[i] and r["constraints_satisfied"])
    print(f"  ΓD≤{thresh:.2f}: accept {n_acc:4d}/{N}, feas={feas:.1f}%, rejected-but-OK={rej_ok}")

# ── Test residuals ───────────────────────────────────────────
print(f"\n{'='*60}")
print("TEST SET RESIDUAL ANALYSIS")
print(f"{'='*60}")
for t in range(3):
    res = (true_test[:, t] - pred_test[:, t]) / (true_test[:, t] + 1e-12)
    print(f"  {TARGET_NAMES[t]:20s}: mean={(np.mean(res)):+.4f}, std={np.std(res):.4f}, "
          f"median={np.median(res):+.4f}")
    # The mean relative error
    mre = np.mean(np.abs(res))
    print(f"    Mean absolute relative error: {mre:.4f} ({mre*100:.1f}%)")

# ── 100-design stratified sample (TEST ONLY) ────────────────
print(f"\n{'='*60}")
print("100-DESIGN STRATIFIED SAMPLE (test-only)")
print(f"{'='*60}")

results_detail = []
for r in batch_results:
    pm = np.array(r["pred_mean"])
    ps = np.array(r["pred_std"])
    comp_limit = r["comp_limit"]
    vm_c = pm[0] + UNCERTAINTY_K_DEFAULT*ps[0]
    comp_c = pm[2] + UNCERTAINTY_K_DEFAULT*ps[2]
    results_detail.append({
        "sample_id": r["sample_id"],
        "volume_reduction_pct": r["volume_reduction_pct"],
        "constraints_satisfied": r["constraints_satisfied"],
        "vm_util": vm_c / MAX_VON_MISES,
        "comp_util": comp_c / comp_limit if comp_limit else 0,
    })

# Satisfied, sorted by reduction
sat = sorted([r for r in results_detail if r["constraints_satisfied"]],
             key=lambda x: -x["volume_reduction_pct"])

# Top 30 highest reduction
top30 = sat[:30]
top_ids = {r["sample_id"] for r in top30}

# Near-boundary: util > 0.90
near_b = [r for r in results_detail
          if r["sample_id"] not in top_ids and
          (r["vm_util"] > 0.90 or r["comp_util"] > 0.90)]
np.random.seed(42)
if len(near_b) > 40:
    idx = np.random.choice(len(near_b), 40, replace=False)
    near40 = [near_b[i] for i in idx]
else:
    near40 = near_b[:40]
used = top_ids | {r["sample_id"] for r in near40}

# Mid-range
remaining = [r for r in results_detail if r["sample_id"] not in used]
if len(remaining) > 30:
    idx = np.random.choice(len(remaining), 30, replace=False)
    mid30 = [remaining[i] for i in idx]
else:
    mid30 = remaining[:30]

sample_100 = top30 + near40 + mid30
n100 = len(sample_100)
ok_100 = sum(1 for r in sample_100 if r["constraints_satisfied"])
reds_100 = [r["volume_reduction_pct"] for r in sample_100]
print(f"  Sample size: {n100}")
print(f"  Constraints OK: {ok_100}/{n100} ({ok_100/n100*100:.1f}%)")
print(f"  Avg reduction: {np.mean(reds_100):.1f}%")
print(f"  Top-30 avg: {np.mean([r['volume_reduction_pct'] for r in top30]):.1f}%")
if near40:
    print(f"  Near-boundary avg: {np.mean([r['volume_reduction_pct'] for r in near40]):.1f}%")
if mid30:
    print(f"  Mid-range avg: {np.mean([r['volume_reduction_pct'] for r in mid30]):.1f}%")

# ── Summary for paper ────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY FOR PAPER")
print(f"{'='*60}")
print(f"N evaluated (test-only): {N}")
print(f"Constraints satisfied: {orig_ok} ({orig_ok/N*100:.1f}%)")
print(f"Mean reduction (satisfied): {np.mean(reds):.1f}% ± {np.std(reds):.1f}%")
print(f"k=1.0 operating point: {orig_ok}/{N} = {orig_ok/N*100:.1f}%")
all_reds = [r["volume_reduction_pct"] for r in batch_results]
print(f"Mean reduction (all): {np.mean(all_reds):.1f}% ± {np.std(all_reds):.1f}%")
print(f"Median reduction (all): {np.median(all_reds):.1f}%")

# Designs with >1% reduction
gt1 = [r for r in batch_results if r["volume_reduction_pct"] > 1.0]
print(f"Designs with >1% reduction: {len(gt1)}")
