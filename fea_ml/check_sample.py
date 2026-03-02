import json, os, numpy as np

sid = '04203'
occ = np.load(f'runs/v3/batch_results_all/{sid}/optimized_occ.npz')['data'].astype(np.uint8)
print(f'Occupancy shape: {occ.shape}, filled: {occ.sum():,}')

with open(f'data/runs_real_128/{sid}/meta.json') as f:
    meta = json.load(f)
print(f'Voxel size: {meta["voxel_size"]}')
print(f'E={meta.get("E",25e9)}, nu={meta.get("nu",0.2)}, rho={meta.get("density",2400)}')

with open(f'runs/v3/batch_results_all/{sid}/optimization_summary.json') as f:
    s = json.load(f)
print(f'Reduction: {s["volume_reduction_pct"]:.1f}%')
print(f'Surrogate mean: VM={s["pred_mean"][0]:.4g}, disp={s["pred_mean"][1]:.4g}, comp={s["pred_mean"][2]:.4g}')
print(f'Surrogate conservative: VM={s["vm_conservative"]:.4g}, disp={s["disp_conservative"]:.4g}, comp={s["comp_conservative"]:.4g}')
print(f'Baseline: {s.get("baseline_targets",{})}')
print(f'Comp limit: {s.get("comp_limit",0):.4g}')
