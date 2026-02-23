import json
import numpy as np
from pathlib import Path

base = Path("runs/v3/batch_results")
results = []
for d in sorted(base.iterdir()):
    if d.is_dir():
        jf = d / "optimization_summary.json"
        if jf.exists():
            r = json.load(open(jf))
            results.append(r)
            sid = r["sample_id"]
            red = r["volume_reduction_pct"]
            ok = r["constraints_satisfied"]
            vm_u = r["vm_utilization"]
            c_u = r.get("comp_utilization", "N/A")
            print(f"  {sid}: red={red:>5.1f}% ok={ok} vm_util={vm_u:.3f} comp_util={c_u}")

print(f"\n--- AGGREGATE ---")
all_reds = [r["volume_reduction_pct"] for r in results]
ok_models = [r for r in results if r["constraints_satisfied"]]
ok_reds = [r["volume_reduction_pct"] for r in ok_models]
pos = [r for r in results if r["volume_reduction_pct"] > 1.0]
pos_reds = [r["volume_reduction_pct"] for r in pos]

print(f"ALL {len(results)}: mean={np.mean(all_reds):.1f}% +/- {np.std(all_reds):.1f}%  med={np.median(all_reds):.1f}%")
print(f"  range: [{np.min(all_reds):.1f}%, {np.max(all_reds):.1f}%]")
print(f"\n{len(ok_models)} constraint-OK: mean={np.mean(ok_reds):.1f}% +/- {np.std(ok_reds):.1f}%  med={np.median(ok_reds):.1f}%")
print(f"  IDs: {[r['sample_id'] for r in ok_models]}")
print(f"  Reds: {[r['volume_reduction_pct'] for r in ok_models]}")

print(f"\n{len(pos)} with >1% reduction: mean={np.mean(pos_reds):.1f}% +/- {np.std(pos_reds):.1f}%")
print(f"  non-optimizable (<= 1%): {len(results) - len(pos)}")

# Per-part stats for constraint-OK models
if ok_models:
    print(f"\n--- PER-PART RETENTION (constraint-OK models) ---")
    for part in ["exterior_wall", "interior_wall", "roof", "floor"]:
        vals = [r["part_breakdown"][part]["retained_pct"] for r in ok_models]
        print(f"  {part:16s}: {np.mean(vals):.1f}% +/- {np.std(vals):.1f}%")

