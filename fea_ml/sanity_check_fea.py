#!/usr/bin/env python3
"""Quick sanity check: run hex8 FEA on a baseline training sample
and compare with the known SfePy/tet-mesh ground truth."""
import json, numpy as np, time, sys
sys.path.insert(0, ".")
from validate_fea_ground_truth import voxel_fea

sample = "00000"
p = f"data/runs_real_128/{sample}"

with open(f"{p}/meta.json") as f:
    meta = json.load(f)
with open(f"{p}/targets.json") as f:
    targets = json.load(f)

occ = np.load(f"{p}/occ.npz")["data"].astype(np.uint8)

print(f"Sample {sample}")
print(f"  Known targets: VM={targets['max_von_mises']:.4g}, "
      f"disp={targets['max_displacement']:.4g}, comp={targets['compliance']:.4g}")
print(f"  Voxel size: {meta['voxel_size']:.5f} m")
print(f"  Elements: {occ.sum()}")

t0 = time.time()
result = voxel_fea(occ, voxel_size=meta["voxel_size"],
                   E=meta["E"], nu=meta["nu"], rho=meta["density"])
dt = time.time() - t0

if result:
    print(f"\n  Hex8 FEA results:")
    print(f"    VM:   {result['max_von_mises']:.4g} Pa  (known: {targets['max_von_mises']:.4g}, ratio: {result['max_von_mises']/targets['max_von_mises']:.3f})")
    print(f"    Disp: {result['max_displacement']:.4g} m  (known: {targets['max_displacement']:.4g}, ratio: {result['max_displacement']/targets['max_displacement']:.3f})")
    print(f"    Comp: {result['compliance']:.4g} J  (known: {targets['compliance']:.4g}, ratio: {result['compliance']/targets['compliance']:.3f})")
    print(f"    Time: {dt:.1f}s")
else:
    print("  FEA FAILED")
