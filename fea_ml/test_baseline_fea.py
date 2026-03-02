"""Quick check: voxel extents and baseline FEA timing test."""
import numpy as np, json, time, sys, gc
sys.path.insert(0, '.')
from run_fea_validation import voxel_fea_fast

sid = '04203'
orig = np.load(f'data/runs_real_128/{sid}/occ.npz')['data'].astype(np.uint8)
opt = np.load(f'runs/v3/batch_results_all/{sid}/optimized_occ.npz')['data'].astype(np.uint8)

with open(f'data/runs_real_128/{sid}/meta.json') as f:
    meta = json.load(f)
voxel_size = meta['voxel_size']

a0, a1, a2 = np.nonzero(orig)
print(f'Original: filled={orig.sum():,}')
print(f'  a0 range: {a0.min()}-{a0.max()} ({a0.max()-a0.min()+1} voxels)')
print(f'  a1 range: {a1.min()}-{a1.max()} ({a1.max()-a1.min()+1} voxels)')
print(f'  a2 range: {a2.min()}-{a2.max()} ({a2.max()-a2.min()+1} voxels)')
print(f'  Phys dims: x={float(a0.max()-a0.min())*voxel_size:.2f}m, y={float(a1.max()-a1.min())*voxel_size:.2f}m, z(H)={float(a2.max()-a2.min())*voxel_size:.2f}m')

a0o, a1o, a2o = np.nonzero(opt)
print(f'Optimized: filled={opt.sum():,}')

# Run baseline FEA
print('\n--- Baseline FEA ---')
t0 = time.time()
res_base = voxel_fea_fast(orig, voxel_size, E=meta.get('E',25e9), nu=meta.get('nu',0.2), rho=meta.get('density',2400))
dt_base = time.time() - t0
print(f'  Time: {dt_base:.1f}s, iters: {res_base["solve_iters"]}')
print(f'  VM={res_base["max_von_mises"]:.4g}, disp={res_base["max_displacement"]:.4g}, comp={res_base["compliance"]:.4g}')
gc.collect()

# Run optimized FEA
print('\n--- Optimized FEA ---')
t0 = time.time()
res_opt = voxel_fea_fast(opt, voxel_size, E=meta.get('E',25e9), nu=meta.get('nu',0.2), rho=meta.get('density',2400))
dt_opt = time.time() - t0
print(f'  Time: {dt_opt:.1f}s, iters: {res_opt["solve_iters"]}')
print(f'  VM={res_opt["max_von_mises"]:.4g}, disp={res_opt["max_displacement"]:.4g}, comp={res_opt["compliance"]:.4g}')

# Compare
print('\n--- Comparison ---')
with open(f'runs/v3/batch_results_all/{sid}/optimization_summary.json') as f:
    summary = json.load(f)
bl = summary.get('baseline_targets', {})
print(f'Training FEA baseline: VM={bl.get("max_von_mises",0):.4g}, disp={bl.get("max_displacement",0):.4g}, comp={bl.get("compliance",0):.4g}')
print(f'Voxel FEA baseline:    VM={res_base["max_von_mises"]:.4g}, disp={res_base["max_displacement"]:.4g}, comp={res_base["compliance"]:.4g}')
print(f'Voxel FEA optimized:   VM={res_opt["max_von_mises"]:.4g}, disp={res_opt["max_displacement"]:.4g}, comp={res_opt["compliance"]:.4g}')
print(f'\nRatio (voxel_base / tet_base):')
for name, vk, tk in [('VM', 'max_von_mises', 'max_von_mises'), ('disp', 'max_displacement', 'max_displacement'), ('comp', 'compliance', 'compliance')]:
    vv = res_base[vk]
    tv = bl.get(tk, 1e-30)
    print(f'  {name}: {vv/tv:.2f}x')
print(f'\nRatio (voxel_opt / voxel_base):')
for name, k in [('VM', 'max_von_mises'), ('disp', 'max_displacement'), ('comp', 'compliance')]:
    print(f'  {name}: {res_opt[k]/res_base[k]:.3f} ({(res_opt[k]/res_base[k]-1)*100:+.1f}%)')
