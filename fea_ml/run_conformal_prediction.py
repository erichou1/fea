#!/usr/bin/env python3
"""Conformal Prediction Analysis for SASTO Pipeline

Provides distribution-free coverage guarantees, replacing the heuristic μ+kσ bound.

Analysis 1: Constraint-satisfaction conformal certification (n FEA-validated CS designs)
  - 0 violations in n trials → P(violation) ≤ 1/(n+1)  [distribution-free]
  - Conformal upper bound on compliance ratio
  - Clopper-Pearson exact binomial CIs

Analysis 2: Split conformal prediction on test set (n=1114)
  - Calibrates surrogate prediction accuracy within SfePy domain
  - Reports conformal prediction intervals per target

Analysis 3: Calibrated k-factor via ensemble re-inference on test set
  - Loads all 5 ensemble members, computes per-sample std
  - Finds k such that μ+k·σ achieves desired conformal coverage
  - Directly replaces the heuristic k=1.0

References:
  Vovk, Gammerman, Shafer (2005). Algorithmic Learning in a Random World.
  Angelopoulos & Bates (2023). Conformal Prediction: A Gentle Introduction.
"""

import json
import numpy as np
from pathlib import Path
import sys
import os
import time


def conformal_quantile(scores, alpha):
    """Compute the split conformal quantile for coverage level 1-alpha.
    
    Returns q such that P(new_score ≤ q) ≥ 1-alpha.
    Formula: q = the ⌈(n+1)(1-α)⌉/n-th empirical quantile.
    """
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return np.quantile(scores, q_level)


def clopper_pearson_upper(n_trials, n_successes, confidence):
    """Upper bound of Clopper-Pearson exact binomial CI.
    
    For 0 failures in n trials at confidence level:
    p_upper = 1 - (1 - confidence)^(1/n)
    """
    n_failures = n_trials - n_successes
    if n_failures == 0:
        return 1.0 - (1.0 - confidence) ** (1.0 / n_trials)
    else:
        from scipy.stats import beta as beta_dist
        return beta_dist.ppf(confidence, n_failures + 1, n_successes)


def analysis1_fea_certification(runs_v3):
    """Conformal certification from FEA validation results."""
    print("=" * 70)
    print("ANALYSIS 1: Constraint-Satisfaction Conformal Certification")
    print("=" * 70)

    # Load all FEA validation results
    all_results = []
    for src in [runs_v3 / "fea_validation_100.json", runs_v3 / "fea_validation_full.json"]:
        if src.exists():
            with open(src) as f:
                data = json.load(f)
            for r in data:
                if 'error' not in r and 'comp_ratio' in r:
                    all_results.append(r)

    # Deduplicate by sample_id
    seen = set()
    results = []
    for r in all_results:
        sid = r['sample_id']
        if sid not in seen:
            results.append(r)
            seen.add(sid)

    # Separate feasible (CS) and rejected groups
    feasible = [r for r in results if r.get('group', 'feasible') in
                ('feasible', 'high_reduction', 'near_boundary', 'random')]
    rejected = [r for r in results if r.get('group', '').startswith('rejected')]

    n = len(feasible)
    if n == 0:
        print("No feasible FEA results found. Skipping Analysis 1.")
        return {}

    comp_ratios = np.array([r['comp_ratio'] for r in feasible])
    violations = int(np.sum(comp_ratios > 1.15))

    print(f"\nFEA-validated constraint-satisfying designs: n = {n}")
    print(f"Compliance ratio (C_opt / C_base) statistics:")
    print(f"  Mean ± Std:  {np.mean(comp_ratios):.4f} ± {np.std(comp_ratios):.4f}")
    print(f"  Median:      {np.median(comp_ratios):.4f}")
    print(f"  Max:         {np.max(comp_ratios):.4f}")
    print(f"  Min:         {np.min(comp_ratios):.4f}")
    print(f"  Violations (> 1.15):  {violations}/{n}")

    # --- Distribution-free conformal bound on P(violation) ---
    print(f"\n--- Distribution-Free Conformal Bound ---")
    p_conformal = 1.0 / (n + 1)
    print(f"  P(violation for new CS design) ≤ 1/(n+1) = 1/{n+1} = {p_conformal:.4%}")

    print(f"\n--- Clopper-Pearson Exact Binomial CIs ---")
    for conf in [0.90, 0.95, 0.99]:
        ci_upper = clopper_pearson_upper(n, n - violations, conf)
        print(f"  {conf:.0%} CI on P(violation): [0, {ci_upper:.4%}]")

    # --- Conformal upper bound on compliance ratio ---
    print(f"\n--- Conformal Upper Bound on Compliance Ratio ---")
    for alpha in [0.01, 0.05, 0.10]:
        q_hat = conformal_quantile(comp_ratios, alpha)
        margin = 1.15 - q_hat
        print(f"  α={alpha:.2f} ({1-alpha:.0%} coverage): C ≤ {q_hat:.4f}  "
              f"(margin to 1.15: {margin:.4f})")

    # --- VM stress ratio analysis ---
    if all('vm_ratio' in r for r in feasible):
        vm_ratios = np.array([r['vm_ratio'] for r in feasible])
        print(f"\nVon Mises stress ratio (σ_opt / σ_base):")
        print(f"  Mean ± Std: {np.mean(vm_ratios):.4f} ± {np.std(vm_ratios):.4f}")
        print(f"  Max: {np.max(vm_ratios):.4f}")

    # --- Rejected designs (false negative audit) ---
    if rejected:
        print(f"\n--- False Negative Audit ({len(rejected)} rejected designs) ---")
        rej_ratios = np.array([r['comp_ratio'] for r in rejected])
        rej_pass = int(np.sum(rej_ratios <= 1.15))
        print(f"  Would pass FEA: {rej_pass}/{len(rejected)} ({100*rej_pass/len(rejected):.1f}%)")
        if rej_pass > 0:
            passing = rej_ratios[rej_ratios <= 1.15]
            print(f"  False negatives: comp_ratio mean={np.mean(passing):.4f}, max={np.max(passing):.4f}")

    return {
        'n': n,
        'violations': violations,
        'comp_ratio_mean': float(np.mean(comp_ratios)),
        'comp_ratio_std': float(np.std(comp_ratios)),
        'comp_ratio_max': float(np.max(comp_ratios)),
        'p_conformal': p_conformal,
        'conformal_99_upper': float(conformal_quantile(comp_ratios, 0.01)),
        'conformal_95_upper': float(conformal_quantile(comp_ratios, 0.05)),
    }


def analysis2_test_set(runs_v3):
    """Split conformal prediction on the 1114-sample test set."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Test-Set Split Conformal Prediction (n=1114)")
    print("=" * 70)

    tp_path = runs_v3 / "test_predictions.npz"
    if not tp_path.exists():
        print("test_predictions.npz not found. Skipping Analysis 2.")
        return {}

    tp = np.load(tp_path)
    true_vals = tp['true']       # (1114, 3)
    pred_mean = tp['pred_mean']  # (1114, 3)
    target_names = list(tp['target_names'])

    n_total = len(true_vals)

    # Random split: 50% calibration, 50% evaluation
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_total)
    n_cal = n_total // 2
    cal_idx, eval_idx = perm[:n_cal], perm[n_cal:]

    results = {}
    for t, tname in enumerate(target_names):
        print(f"\n--- Target: {tname} (index {t}) ---")

        true_cal = true_vals[cal_idx, t]
        pred_cal = pred_mean[cal_idx, t]
        true_eval = true_vals[eval_idx, t]
        pred_eval = pred_mean[eval_idx, t]

        # Signed residuals (positive = model underestimates)
        cal_residuals = true_cal - pred_cal

        print(f"  Calibration residuals (n={n_cal}):")
        print(f"    Mean: {np.mean(cal_residuals):.6f}  Std: {np.std(cal_residuals):.6f}")
        print(f"    Median: {np.median(cal_residuals):.6f}")

        # Conformal upper bounds at various coverage levels
        for alpha in [0.01, 0.05, 0.10, 0.20]:
            q_hat = conformal_quantile(cal_residuals, alpha)

            # Evaluate coverage on held-out set
            eval_upper = pred_eval + q_hat
            coverage = np.mean(true_eval <= eval_upper)

            print(f"    α={alpha:.2f} ({1 - alpha:.0%}): q̂={q_hat:+.6f}  "
                  f"eval coverage={coverage:.1%}  (n_eval={len(eval_idx)})")

        # Relative error statistics
        rel_err = (pred_mean[:, t] - true_vals[:, t]) / (np.abs(true_vals[:, t]) + 1e-12)
        print(f"  Relative error: mean={np.mean(rel_err):.3%}  |mean|={np.mean(np.abs(rel_err)):.3%}")

        results[tname] = {
            'residual_mean': float(np.mean(cal_residuals)),
            'residual_std': float(np.std(cal_residuals)),
        }

    return results


def analysis3_calibrated_k(runs_v3):
    """Calibrate the k multiplier using ensemble re-inference on test set.
    
    Loads 5 ensemble members, computes per-sample (mean, std), and finds
    the k such that μ+k·σ covers (1-α)% of true values.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Calibrated k-Factor via Ensemble Predictions")
    print("=" * 70)

    tp_path = runs_v3 / "test_predictions.npz"
    if not tp_path.exists():
        print("test_predictions.npz not found. Skipping.")
        return {}

    tp = np.load(tp_path)
    true_vals = tp['true']       # (1114, 3)
    pred_mean = tp['pred_mean']  # (1114, 3)

    # Check if per-member predictions exist (may have been pre-computed)
    member_preds_path = runs_v3 / "test_predictions_per_member.npz"
    
    if member_preds_path.exists():
        mp = np.load(member_preds_path)
        all_preds = mp['member_preds']  # (5, 1114, 3)
        pred_std = np.std(all_preds, axis=0)  # (1114, 3)
        print(f"Loaded pre-computed per-member predictions.")
    else:
        # Try to compute from ensemble
        print("Per-member predictions not found. Attempting ensemble re-inference...")
        pred_std = _run_ensemble_inference(runs_v3, true_vals.shape[0])
        if pred_std is None:
            print("Could not load ensemble. Using proxy from optimization summaries.")
            pred_std = _proxy_std_from_opt_summaries(runs_v3, pred_mean)
            if pred_std is None:
                print("No optimization summaries available. Skipping Analysis 3.")
                return {}

    target_names = list(tp['target_names'])
    n = len(true_vals)

    # Random split
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    n_cal = n // 2
    cal_idx, eval_idx = perm[:n_cal], perm[n_cal:]

    results = {}
    for t, tname in enumerate(target_names):
        print(f"\n--- Target: {tname} ---")

        std_cal = pred_std[cal_idx, t]
        # Avoid division by zero
        std_cal = np.maximum(std_cal, 1e-12)

        # Normalized residuals: z_i = (true_i - pred_mean_i) / pred_std_i
        z_cal = (true_vals[cal_idx, t] - pred_mean[cal_idx, t]) / std_cal

        print(f"  Normalized residuals z = (true - μ) / σ:")
        print(f"    Mean: {np.mean(z_cal):.3f}  Std: {np.std(z_cal):.3f}")
        print(f"    Percentiles: 50th={np.percentile(z_cal, 50):.3f}, "
              f"90th={np.percentile(z_cal, 90):.3f}, "
              f"95th={np.percentile(z_cal, 95):.3f}, "
              f"99th={np.percentile(z_cal, 99):.3f}")

        # Calibrated k at various coverage levels
        print(f"  Calibrated k (conformal):")
        target_results = {}
        for alpha in [0.01, 0.05, 0.10, 0.159, 0.20]:
            k_cal = conformal_quantile(z_cal, alpha)

            # Evaluate on held-out set
            std_eval = np.maximum(pred_std[eval_idx, t], 1e-12)
            upper_eval = pred_mean[eval_idx, t] + k_cal * std_eval
            coverage = np.mean(true_vals[eval_idx, t] <= upper_eval)

            label = f"α={alpha:.3f} ({1 - alpha:.1%})"
            if abs(alpha - 0.159) < 0.001:
                label += " [Gaussian k=1.0 equiv]"
            print(f"    {label}: k_conformal = {k_cal:.3f}  "
                  f"(eval coverage: {coverage:.1%})")
            target_results[f'k_{1-alpha:.2f}'] = float(k_cal)

        results[tname] = target_results

    return results


def _run_ensemble_inference(runs_v3, n_samples):
    """Try to load ensemble and run inference on test set."""
    try:
        # Add parent directory to path for imports
        sys.path.insert(0, str(Path(__file__).parent))
        
        import torch
        from fea_ml.models.ensemble import DeepEnsemble
        from fea_ml.data.dataset import VoxelDataset
        
        ensemble_dir = runs_v3 / "ensemble"
        if not ensemble_dir.exists():
            return None

        members = sorted(ensemble_dir.glob("ensemble_member_*.pt"))
        if len(members) == 0:
            return None

        print(f"  Found {len(members)} ensemble members")

        # Load normalization
        norm_path = runs_v3 / "normalization.json"
        if not norm_path.exists():
            return None
        with open(norm_path) as f:
            norm = json.load(f)

        # Load splits
        with open(runs_v3 / "splits.json") as f:
            splits = json.load(f)
        test_ids = splits['test']

        # Load test data
        data_dir = Path("data/runs_real_128")
        device = torch.device('cpu')

        all_member_preds = []
        for m_path in members:
            print(f"  Loading {m_path.name}...")
            state = torch.load(m_path, map_location=device, weights_only=True)
            
            # Create model (infer architecture from state dict)
            from fea_ml.models.cnn3d import Surrogate3DResNet
            model = Surrogate3DResNet()
            model.load_state_dict(state)
            model.eval()

            preds = []
            with torch.no_grad():
                for sid in test_ids:
                    occ_path = data_dir / sid / "occupancy.npz"
                    if not occ_path.exists():
                        continue
                    occ = np.load(occ_path)['occupancy'].astype(np.float32)
                    x = torch.from_numpy(occ).unsqueeze(0).unsqueeze(0)  # (1,1,128,128,128)
                    y = model(x).squeeze().numpy()
                    # Denormalize
                    y_denorm = np.exp(y * np.array(norm['std']) + np.array(norm['mean']))
                    preds.append(y_denorm)

            all_member_preds.append(np.array(preds))
            print(f"    → {len(preds)} predictions")

        all_member_preds = np.array(all_member_preds)  # (5, n, 3)
        pred_std = np.std(all_member_preds, axis=0)     # (n, 3)

        # Save for future use
        np.savez(runs_v3 / "test_predictions_per_member.npz",
                 member_preds=all_member_preds)
        print(f"  Saved per-member predictions.")

        return pred_std

    except Exception as e:
        print(f"  Ensemble inference failed: {e}")
        return None


def _proxy_std_from_opt_summaries(runs_v3, pred_mean_test):
    """Estimate test-set pred_std using coefficient of variation from optimization summaries."""
    batch_dir = runs_v3 / "batch_results_all"
    if not batch_dir.exists():
        return None

    # Sample some optimization summaries to estimate CV = std/mean
    cvs = {0: [], 1: [], 2: []}
    dirs = list(batch_dir.iterdir())[:200]
    for d in dirs:
        summ_path = d / "optimization_summary.json"
        if not summ_path.exists():
            continue
        with open(summ_path) as f:
            s = json.load(f)
        if 'pred_std' not in s:
            continue
        for t in range(3):
            if s['pred_mean'][t] > 0 and s['pred_std'][t] > 0:
                cvs[t].append(s['pred_std'][t] / s['pred_mean'][t])

    if not all(len(v) > 10 for v in cvs.values()):
        return None

    # Compute median CV per target
    median_cvs = [np.median(cvs[t]) for t in range(3)]
    print(f"  Proxy CV from opt summaries: {[f'{c:.3f}' for c in median_cvs]}")

    # Apply CV to test set pred_mean
    pred_std_proxy = np.zeros_like(pred_mean_test)
    for t in range(3):
        pred_std_proxy[:, t] = np.abs(pred_mean_test[:, t]) * median_cvs[t]

    return pred_std_proxy


def main():
    os.chdir(Path(__file__).parent)
    RUNS_V3 = Path("runs/v3")

    t0 = time.time()

    # Analysis 1: FEA certification
    res1 = analysis1_fea_certification(RUNS_V3)

    # Analysis 2: Test-set conformal
    res2 = analysis2_test_set(RUNS_V3)

    # Analysis 3: Calibrated k
    res3 = analysis3_calibrated_k(RUNS_V3)

    # Save summary
    summary = {
        'fea_certification': res1,
        'test_set_conformal': res2,
        'calibrated_k': res3,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'runtime_s': time.time() - t0,
    }

    out_path = RUNS_V3 / "conformal_prediction_results.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
