"""Quick FEA test on a SMALL sample - find one with ~20k elements."""
import numpy as np, json, os, time, sys, gc
sys.path.insert(0, '.')

# Find CS samples sorted by element count (smallest first)
results_dir = 'runs/v3/batch_results_all'
with open('runs/v3/splits.json') as f:
    splits = json.load(f)
test_ids = set(p.split('/')[-1] for p in splits['test'])

candidates = []
for d in sorted(os.listdir(results_dir)):
    sp = os.path.join(results_dir, d, 'optimization_summary.json')
    if not os.path.isfile(sp):
        continue
    with open(sp) as f:
        s = json.load(f)
    if s['sample_id'] not in test_ids or not s.get('constraints_satisfied'):
        continue
    occ = np.load(os.path.join(results_dir, d, 'optimized_occ.npz'))['data']
    orig_path = os.path.join('data/runs_real_128', s['sample_id'], 'occ.npz')
    if not os.path.exists(orig_path):
        continue
    orig = np.load(orig_path)['data']
    candidates.append({
        'id': s['sample_id'],
        'opt_vox': int(occ.sum()),
        'orig_vox': int(orig.sum()),
        'reduction': s['volume_reduction_pct'],
    })

candidates.sort(key=lambda x: x['orig_vox'])
print("Smallest baseline samples:")
for c in candidates[:10]:
    print(f"  {c['id']}: orig={c['orig_vox']:,}, opt={c['opt_vox']:,}, red={c['reduction']:.1f}%")

print("\nPicking smallest...")
pick = candidates[0]
sid = pick['id']
print(f"Selected: {sid} (orig={pick['orig_vox']:,}, opt={pick['opt_vox']:,})")

# Run FEA
from validate_fea_ground_truth import voxel_fea

with open(f'data/runs_real_128/{sid}/meta.json') as f:
    meta = json.load(f)
voxel_size = meta['voxel_size']

orig = np.load(f'data/runs_real_128/{sid}/occ.npz')['data'].astype(np.uint8)
opt = np.load(f'runs/v3/batch_results_all/{sid}/optimized_occ.npz')['data'].astype(np.uint8)
with open(f'runs/v3/batch_results_all/{sid}/optimization_summary.json') as f:
    summary = json.load(f)

print(f"\n=== Baseline FEA ({orig.sum():,} elems) ===")
t0 = time.time()
res_base = voxel_fea(orig, voxel_size, E=meta.get('E',25e9), nu=meta.get('nu',0.2), rho=meta.get('density',2400))
dt_base = time.time() - t0
print(f"Time: {dt_base:.1f}s")
gc.collect()

print(f"\n=== Optimized FEA ({opt.sum():,} elems) ===")
t0 = time.time()
res_opt = voxel_fea(opt, voxel_size, E=meta.get('E',25e9), nu=meta.get('nu',0.2), rho=meta.get('density',2400))
dt_opt = time.time() - t0
print(f"Time: {dt_opt:.1f}s")

bl = summary.get('baseline_targets', {})
print(f"\n{'='*60}")
print(f"{'Metric':<12} {'Tet Base':>12} {'Vox Base':>12} {'Vox Opt':>12} {'Surr Mean':>12}")
for name, k, sk in [('VM', 'max_von_mises', 0), ('Disp', 'max_displacement', 1), ('Comp', 'compliance', 2)]:
    tv = bl.get(k, 0)
    vb = res_base[k] if res_base else 0
    vo = res_opt[k] if res_opt else 0
    sv = summary['pred_mean'][sk]
    print(f"{name:<12} {tv:>12.4g} {vb:>12.4g} {vo:>12.4g} {sv:>12.4g}")

if res_base and res_opt:
    print(f"\nOpt/Base ratio (voxel FEA):")
    for name, k in [('VM', 'max_von_mises'), ('Disp', 'max_displacement'), ('Comp', 'compliance')]:
        r = res_opt[k] / res_base[k] if res_base[k] > 0 else float('inf')
        print(f"  {name}: {r:.4f} ({(r-1)*100:+.1f}%)")
    print(f"\nComp constraint: opt={res_opt['compliance']:.4g} vs 1.15*base={1.15*res_base['compliance']:.4g} → {'PASS' if res_opt['compliance'] <= 1.15*res_base['compliance'] else 'FAIL'}")
