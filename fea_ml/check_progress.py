#!/usr/bin/env python3
"""Quick check of batch optimization progress."""
import os, json, numpy as np

base = 'runs/v3/batch_results_all'
results = []
for d in sorted(os.listdir(base)):
    p = os.path.join(base, d, 'optimization_summary.json')
    if os.path.exists(p):
        results.append(json.load(open(p)))

if not results:
    print("No results yet")
    exit()

reds = [r['volume_reduction_pct'] for r in results]
ok = [r for r in results if r['constraints_satisfied']]
ok_reds = [r['volume_reduction_pct'] for r in ok]
meaningful = [r for r in results if r['volume_reduction_pct'] > 1.0]

print(f"Total completed: {len(results)}")
print(f"Constraint-OK: {len(ok)} ({len(ok)/len(results)*100:.1f}%)")
print(f"Meaningful (>1%): {len(meaningful)} ({len(meaningful)/len(results)*100:.1f}%)")
print(f"\nAll samples: {np.mean(reds):.1f}% ± {np.std(reds):.1f}% (median {np.median(reds):.1f}%)")
if ok_reds:
    print(f"Constraint-OK: {np.mean(ok_reds):.1f}% ± {np.std(ok_reds):.1f}% (median {np.median(ok_reds):.1f}%)")
if meaningful:
    m_reds = [r['volume_reduction_pct'] for r in meaningful]
    print(f"Meaningful: {np.mean(m_reds):.1f}% ± {np.std(m_reds):.1f}%")

times = [r['total_time_seconds'] for r in results]
print(f"\nRuntime: {np.mean(times):.0f}s ± {np.std(times):.0f}s (median {np.median(times):.0f}s)")
print(f"Latest sample: {results[-1]['sample_id']}")
