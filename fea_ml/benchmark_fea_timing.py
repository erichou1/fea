#!/usr/bin/env python3
"""
Benchmark: Time a single building-scale voxel FEA solve
to provide a concrete baseline for speedup comparison.

Runs voxel FEA on 3 different geometry sizes and reports wall-clock time.
"""
import sys, os, json, time, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_fea_validation import voxel_fea_fast

# Paths
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "runs_real_128")
BATCH_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "v3", "batch_results_all")
SPLITS     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "v3", "splits.json")

def load_test_ids():
    with open(SPLITS) as f:
        sp = json.load(f)
    return [os.path.basename(p) for p in sp["test"]]

def get_element_count(sid, which="baseline"):
    """Return element count for a sample."""
    if which == "baseline":
        occ_path = os.path.join(DATA_DIR, sid, "occ.npz")
    else:
        occ_path = os.path.join(BATCH_DIR, sid, "optimized_occ.npz")
    if not os.path.exists(occ_path):
        return 0
    d = np.load(occ_path)
    key = list(d.keys())[0]
    occ = d[key].astype(np.uint8)
    return int(occ.sum())

def run_timed_fea(sid, which="baseline"):
    """Time a single FEA solve and return results."""
    if which == "baseline":
        occ_path = os.path.join(DATA_DIR, sid, "occ.npz")
    else:
        occ_path = os.path.join(BATCH_DIR, sid, "optimized_occ.npz")
    
    meta_path = os.path.join(DATA_DIR, sid, "meta.json")
    
    d = np.load(occ_path)
    key = list(d.keys())[0]
    occ = d[key].astype(np.uint8)
    with open(meta_path) as f:
        meta = json.load(f)
    
    phys = meta["physical_size"]
    shape = occ.shape
    voxel_size = max(phys[k] / shape[i] for i, k in enumerate(["x", "y", "z"]))
    
    n_elements = int(occ.sum())
    print(f"  Geometry: {sid} ({which}), {n_elements:,} elements, voxel_size={voxel_size:.5f} m")
    
    t0 = time.time()
    result = voxel_fea_fast(occ, voxel_size)
    t1 = time.time()
    
    if result is None:
        print(f"  FEA failed (too few elements)")
        return None
    
    elapsed = t1 - t0
    print(f"  Wall-clock: {elapsed:.1f}s, CG iters: {result['solve_iters']}")
    print(f"  VM={result['max_von_mises']:.3e}, disp={result['max_displacement']:.4e}, comp={result['compliance']:.3e}")
    return {
        "sid": sid, "which": which,
        "n_elements": n_elements,
        "wall_clock_s": elapsed,
        "solve_iters": result["solve_iters"],
    }

def main():
    test_ids = load_test_ids()

    # Find samples of different sizes
    # We want small (~20k), medium (~40k), large (~70k) baseline geometries
    print("Finding samples of different sizes...")
    sizes = []
    for sid in test_ids:
        n = get_element_count(sid, "baseline")
        if n > 0 and os.path.exists(os.path.join(BATCH_DIR, sid, "optimized_occ.npz")):
            sizes.append((sid, n))
    
    sizes.sort(key=lambda x: x[1])
    print(f"  {len(sizes)} samples with both baseline and optimized data")
    
    # Pick 3 representative sizes: small (25th pct), medium (50th), large (75th)
    n = len(sizes)
    targets = [
        ("small",  sizes[n//4]),
        ("medium", sizes[n//2]),
        ("large",  sizes[3*n//4]),
    ]
    
    results = []
    print("\n" + "="*60)
    print("SINGLE FEA SOLVE TIMING BENCHMARK")
    print("="*60)
    
    for label, (sid, est_n) in targets:
        print(f"\n--- {label.upper()} geometry ({est_n:,} est. elements) ---")
        
        # Time baseline FEA
        r = run_timed_fea(sid, "baseline")
        if r:
            r["label"] = label
            results.append(r)
        
        gc.collect()
        
        # Time optimized FEA
        r = run_timed_fea(sid, "optimized")
        if r:
            r["label"] = label
            results.append(r)
        
        gc.collect()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Single FEA solve times")
    print("="*60)
    print(f"{'Type':<20} {'Elements':>10} {'Time (s)':>10} {'CG iters':>10}")
    print("-"*52)
    for r in results:
        print(f"{r['label']+' '+r['which']:<20} {r['n_elements']:>10,} {r['wall_clock_s']:>10.1f} {r['solve_iters']:>10,}")
    
    # Compute SIMP comparison
    baseline_times = [r["wall_clock_s"] for r in results if r["which"] == "baseline"]
    if baseline_times:
        avg_fea_time = sum(baseline_times) / len(baseline_times)
        print(f"\nAverage single FEA solve time: {avg_fea_time:.1f}s")
        print(f"SIMP with 200 iterations: {200 * avg_fea_time / 60:.0f} min = {200 * avg_fea_time / 3600:.1f} hours")
        print(f"SIMP with 500 iterations: {500 * avg_fea_time / 60:.0f} min = {500 * avg_fea_time / 3600:.1f} hours")
        print(f"SASTO median: 50s")
        print(f"Speedup (200 iters): {200 * avg_fea_time / 50:.0f}x")
        print(f"Speedup (500 iters): {500 * avg_fea_time / 50:.0f}x")
    
    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "v3", "fea_timing_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
