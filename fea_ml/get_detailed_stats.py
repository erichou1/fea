#!/usr/bin/env python3
"""Get detailed per-part retention stats for paper."""
import json, numpy as np
from pathlib import Path

results_dir = Path("runs/v3/batch_results_all")
results = []
for d in sorted(results_dir.iterdir()):
    s = d / "optimization_summary.json"
    if s.exists():
        try:
            results.append(json.load(open(s)))
        except:
            pass

print(f"Total: {len(results)}")
ok = [r for r in results if r.get("constraints_satisfied")]
print(f"Constraint-OK: {len(ok)}")

# Check what keys exist
if ok:
    print(f"\nSample keys: {list(ok[0].keys())}")
    # Look for per-part data
    for key in ok[0]:
        if "part" in key.lower():
            print(f"  {key}: {ok[0][key]}")

# Per-part retention
parts = {"exterior_wall": [], "interior_wall": [], "roof": [], "floor": []}
for r in ok:
    pr = r.get("per_part_retention_pct", {})
    for p in parts:
        v = pr.get(p)
        if v is not None:
            parts[p].append(v)

print(f"\nPer-Part Retention (constraint-OK):")
for p, vals in parts.items():
    if vals:
        arr = np.array(vals)
        print(f"  {p:18s}: {arr.mean():.1f}% +/- {arr.std():.1f}%  n={len(vals)}")
    else:
        print(f"  {p:18s}: NO DATA")

# Try alternate key names
if not any(parts.values()):
    print("\nChecking first OK result for part data...")
    r = ok[0]
    for k, v in r.items():
        print(f"  {k}: {type(v).__name__} = {str(v)[:200]}")
