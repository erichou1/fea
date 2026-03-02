"""Quick 1-sample baseline+optimized FEA test."""
import numpy as np, json, time, sys, gc
sys.path.insert(0, '.')

# Use a smaller sample with fewer elements
sid = '12705'  # Low reduction (0.55%) should be small/quick

data_dir = f'data/runs_real_128/{sid}'
batch_dir = f'runs/v3/batch_results_all/{sid}'

import os
if not os.path.exists(data_dir):
    print(f"Data dir missing: {data_dir}")
    sys.exit(1)

with open(f'{data_dir}/meta.json') as f:
    meta = json.load(f)
voxel_size = meta['voxel_size']

orig = np.load(f'{data_dir}/occ.npz')['data'].astype(np.uint8)
opt = np.load(f'{batch_dir}/optimized_occ.npz')['data'].astype(np.uint8)

with open(f'{batch_dir}/optimization_summary.json') as f:
    summary = json.load(f)

print(f"Sample {sid}: {summary['volume_reduction_pct']:.1f}% reduction")
print(f"Original: {orig.sum():,} voxels, Optimized: {opt.sum():,} voxels")
print(f"Voxel size: {voxel_size:.5f} m")

from validate_fea_ground_truth import voxel_fea

# Baseline FEA
print("\n=== Baseline Voxel FEA ===")
t0 = time.time()
res_base = voxel_fea(orig, voxel_size, E=meta.get('E',25e9), nu=meta.get('nu',0.2), rho=meta.get('density',2400))
dt_base = time.time() - t0
print(f"Time: {dt_base:.1f}s")
gc.collect()

# Optimized FEA
print("\n=== Optimized Voxel FEA ===")
t0 = time.time()
res_opt = voxel_fea(opt, voxel_size, E=meta.get('E',25e9), nu=meta.get('nu',0.2), rho=meta.get('density',2400))
dt_opt = time.time() - t0
print(f"Time: {dt_opt:.1f}s")

# Comparison
bl = summary.get('baseline_targets', {})
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"{'Metric':<15} {'Tet Baseline':>15} {'Vox Baseline':>15} {'Vox Optimized':>15} {'Surr Mean':>15} {'Surr Cons':>15}")
for name, tk, vbk, vok, sk, ck in [
    ('VM (Pa)', 'max_von_mises', 'max_von_mises', 'max_von_mises', 0, 'vm_conservative'),
    ('Disp (m)', 'max_displacement', 'max_displacement', 'max_displacement', 1, 'disp_conservative'),
    ('Comp (J)', 'compliance', 'compliance', 'compliance', 2, 'comp_conservative'),
]:
    tv = bl.get(tk, 0)
    vb = res_base[vbk] if res_base else 0
    vo = res_opt[vok] if res_opt else 0
    sv = summary['pred_mean'][sk]
    cv = summary.get(ck, 0)
    print(f"{name:<15} {tv:>15.4g} {vb:>15.4g} {vo:>15.4g} {sv:>15.4g} {cv:>15.4g}")

print(f"\nVoxel FEA ratio (opt/base):")
if res_base and res_opt:
    for name, k in [('VM', 'max_von_mises'), ('Disp', 'max_displacement'), ('Comp', 'compliance')]:
        ratio = res_opt[k] / res_base[k] if res_base[k] > 0 else float('inf')
        print(f"  {name}: {ratio:.4f} ({(ratio-1)*100:+.1f}%)")

print(f"\nConstraint check (comp ≤ 1.15 × baseline):")
if res_base and res_opt:
    comp_limit = 1.15 * res_base['compliance']
    ok = res_opt['compliance'] <= comp_limit
    print(f"  Voxel comp limit: {comp_limit:.4g} J, Opt: {res_opt['compliance']:.4g} J → {'PASS' if ok else 'FAIL'}")
