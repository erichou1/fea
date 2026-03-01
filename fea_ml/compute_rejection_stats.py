#!/usr/bin/env python3
"""
Compute batch statistics and optimization stability metrics from batch results.
Filters to test-only samples using splits.json.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "runs" / "v3" / "batch_results_all"
SPLITS_FILE = Path(__file__).parent / "runs" / "v3" / "splits.json"


def load_test_summaries():
    """Load optimization_summary.json files for test-only samples."""
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    test_ids = set(os.path.basename(p) for p in splits["test"])
    print(f"Test split: {len(test_ids)} sample IDs")
    
    summaries = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.name not in test_ids:
            continue
        summary_path = d / "optimization_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
                summaries.append(s)
    print(f"Loaded: {len(summaries)} test summaries")
    return summaries


def compute_population_stats(summaries):
    """Compute population-level batch statistics."""
    n_batches_list = []
    total_removed_list = []
    vol_reduction_list = []
    constraint_sat = []
    runtimes = []
    vol_originals = []
    
    for s in summaries:
        nb = s.get("n_batches", 0)
        tr = s.get("total_removed", 0)
        vr = s.get("volume_reduction_pct", 0)
        cs = s.get("constraints_satisfied", False)
        rt = s.get("total_time_seconds", None)
        vo = s.get("volume_original", 0)
        
        n_batches_list.append(nb)
        total_removed_list.append(tr)
        vol_reduction_list.append(vr)
        constraint_sat.append(cs)
        vol_originals.append(vo)
        if rt is not None:
            runtimes.append(rt)

    n_batches = np.array(n_batches_list)
    total_removed = np.array(total_removed_list)
    vol_red = np.array(vol_reduction_list)
    cs_mask = np.array(constraint_sat)
    
    print("=" * 60)
    print("TEST-ONLY POPULATION BATCH STATISTICS")
    print(f"Total test samples: {len(summaries)}")
    print(f"Constraint-satisfying: {cs_mask.sum()}")
    print()
    
    # ---- All geometries ----
    print("--- All Test Geometries ---")
    print(f"  n_batches  mean={n_batches.mean():.1f}, median={np.median(n_batches):.0f}, "
          f"std={n_batches.std():.1f}, range=[{n_batches.min()}, {n_batches.max()}]")
    print(f"  removed    mean={total_removed.mean():.0f}, median={np.median(total_removed):.0f}")
    print(f"  vol_red    mean={vol_red.mean():.1f}%, median={np.median(vol_red):.1f}%")
    print()
    
    # ---- Constraint-satisfying subset ----
    cs_nb = n_batches[cs_mask]
    cs_tr = total_removed[cs_mask]
    cs_vr = vol_red[cs_mask]
    
    print(f"--- Constraint-Satisfying ({cs_mask.sum()} geometries) ---")
    print(f"  n_batches  mean={cs_nb.mean():.1f}, median={np.median(cs_nb):.0f}, "
          f"std={cs_nb.std():.1f}, range=[{cs_nb.min()}, {cs_nb.max()}]")
    print(f"  removed    mean={cs_tr.mean():.0f}, median={np.median(cs_tr):.0f}")
    print(f"  vol_red    mean={cs_vr.mean():.1f}%, median={np.median(cs_vr):.1f}%")
    print()
    
    # ---- Non-satisfying subset ----
    nc_nb = n_batches[~cs_mask]
    nc_tr = total_removed[~cs_mask]
    nc_vr = vol_red[~cs_mask]
    
    print(f"--- Non-Satisfying ({(~cs_mask).sum()} geometries) ---")
    print(f"  n_batches  mean={nc_nb.mean():.1f}, median={np.median(nc_nb):.0f}, "
          f"std={nc_nb.std():.1f}, range=[{nc_nb.min()}, {nc_nb.max()}]")
    print(f"  removed    mean={nc_tr.mean():.0f}, median={np.median(nc_tr):.0f}")
    print(f"  vol_red    mean={nc_vr.mean():.1f}%, median={np.median(nc_vr):.1f}%")
    print()
    
    # ---- Batch efficiency (approx) ----
    # Minimum accepted batches = ceil(total_removed / 200)
    # This gives upper bound on rejection rate
    BATCH_SIZE = 200
    valid = n_batches > 0
    min_accepted = np.ceil(total_removed[valid] / BATCH_SIZE).astype(int)
    max_rejected = n_batches[valid] - min_accepted
    max_rejection_rate = max_rejected / n_batches[valid]
    
    print("--- Batch Efficiency (approximate, upper-bound rejection rate) ---")
    print(f"  All (n_batches>0): mean={max_rejection_rate.mean():.3f}, "
          f"median={np.median(max_rejection_rate):.3f}")
    
    cs_valid = cs_mask & valid
    cs_min_acc = np.ceil(total_removed[cs_valid] / BATCH_SIZE).astype(int)
    cs_max_rej = n_batches[cs_valid] - cs_min_acc
    cs_max_rr = cs_max_rej / n_batches[cs_valid]
    print(f"  Constraint-sat:    mean={cs_max_rr.mean():.3f}, "
          f"median={np.median(cs_max_rr):.3f}")
    print()
    
    # ---- Runtime ----
    rt = np.array(runtimes)
    print(f"--- Runtime ({len(rt)} samples) ---")
    print(f"  Mean={rt.mean():.1f}s, Median={np.median(rt):.1f}s, "
          f"Std={rt.std():.1f}s, Range=[{rt.min():.1f}, {rt.max():.1f}]")
    
    # Constraint-satisfying runtimes
    cs_rts = []
    for s, c in zip(summaries, constraint_sat):
        if c and "total_time_seconds" in s:
            cs_rts.append(s["total_time_seconds"])
    cs_rt = np.array(cs_rts)
    print(f"  CS subset: Mean={cs_rt.mean():.1f}s, Median={np.median(cs_rt):.1f}s")
    print()
    
    # ---- Convergence termination reasons ----
    zero_removed = (total_removed == 0).sum()
    tiny_removed = (total_removed <= 10).sum()
    print(f"--- Termination Patterns ---")
    print(f"  Zero voxels removed: {zero_removed} ({100*zero_removed/len(summaries):.1f}%)")
    print(f"  ≤10 voxels removed: {tiny_removed} ({100*tiny_removed/len(summaries):.1f}%)")
    
    # Success rate (all optimization completed without error)
    success = sum(1 for s in summaries if s.get("success", False))
    print(f"  Successful completion: {success}/{len(summaries)} ({100*success/len(summaries):.1f}%)")
    print()
    
    # ---- Reference case ----
    print("--- Reference Case (00472) ---")
    for s in summaries:
        if s.get("sample_id") == "00472":
            nb_r = s["n_batches"]
            tr_r = s["total_removed"]
            min_acc_r = int(np.ceil(tr_r / BATCH_SIZE))
            max_rej_r = nb_r - min_acc_r
            print(f"  n_batches: {nb_r}")
            print(f"  total_removed: {tr_r}")
            print(f"  min_accepted_batches: {min_acc_r}")
            print(f"  max_rejected_batches: {max_rej_r}")
            print(f"  upper_bound_rejection_rate: {max_rej_r/nb_r:.3f}")
            print(f"  vol_reduction: {s['volume_reduction_pct']:.1f}%")
            print(f"  runtime: {s['total_time_seconds']:.1f}s")
            break


if __name__ == "__main__":
    summaries = load_test_summaries()
    compute_population_stats(summaries)
