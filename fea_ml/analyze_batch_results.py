"""Analyze batch results for stratified sampling and validation."""
import json, os, numpy as np

results = []
batch_dir = 'runs/v3/batch_results_all'
dirs = sorted([d for d in os.listdir(batch_dir) if os.path.isdir(os.path.join(batch_dir, d))])
print(f'Total dirs: {len(dirs)}')

for d in dirs:
    sfile = os.path.join(batch_dir, d, 'optimization_summary.json')
    if os.path.exists(sfile):
        with open(sfile) as f:
            s = json.load(f)
        if s.get('success', False):
            results.append({
                'id': d,
                'reduction': s['volume_reduction_pct'],
                'satisfied': s['constraints_satisfied'],
                'vm_conservative': s.get('vm_conservative', 0),
                'comp_conservative': s.get('comp_conservative', 0),
                'comp_limit': s.get('comp_limit', 0),
                'baseline_vm': s.get('baseline_targets', {}).get('max_von_mises', 0),
                'baseline_comp': s.get('baseline_targets', {}).get('compliance', 0),
                'pred_mean': s.get('pred_mean', [0,0,0]),
                'pred_std': s.get('pred_std', [0,0,0]),
            })

print(f'Loaded {len(results)} successful results')
satisfied = [r for r in results if r['satisfied']]
unsatisfied = [r for r in results if not r['satisfied']]
print(f'Constraints OK: {len(satisfied)}')
print(f'Constraints failed: {len(unsatisfied)}')

# Sort by reduction for stratification
satisfied_sorted = sorted(satisfied, key=lambda x: x['reduction'], reverse=True)
print(f'\nTop 5 reductions (satisfied):')
for r in satisfied_sorted[:5]:
    print(f'  {r["id"]}: {r["reduction"]:.1f}%')
print(f'Bottom 5 reductions (satisfied):')
for r in satisfied_sorted[-5:]:
    print(f'  {r["id"]}: {r["reduction"]:.1f}%')

# Check near-boundary (comp within 5% of limit)
near_boundary = [r for r in satisfied if r['comp_limit'] > 0 and abs(r['comp_conservative'] / r['comp_limit'] - 1.0) < 0.05]
print(f'\nNear-boundary (within 5% of comp limit): {len(near_boundary)}')

# Check stalled (<=1% reduction)
stalled = [r for r in results if r['reduction'] <= 1.0]
print(f'Stalled (<=1% reduction): {len(stalled)}')

# Check data availability for FEA re-analysis
# We need: baseline FEA targets, optimized voxel grids, surrogate predictions
has_occ = 0
has_baseline = 0
for r in results[:20]:
    occ_path = os.path.join(batch_dir, r['id'], 'optimized_occ.npz')
    if os.path.exists(occ_path):
        has_occ += 1
    if r['baseline_comp'] > 0:
        has_baseline += 1
print(f'\nData availability (first 20): has_occ={has_occ}, has_baseline={has_baseline}')

# Load test predictions to check what we have
test_pred_path = 'runs/v3/test_predictions.npz'
if os.path.exists(test_pred_path):
    tp = np.load(test_pred_path)
    print(f'\nTest predictions: keys={list(tp.keys())}')
    for k in tp.keys():
        print(f'  {k}: shape={tp[k].shape}, dtype={tp[k].dtype}')

# Check if ensemble checkpoints exist
ens_dir = 'runs/v3/ensemble'
if os.path.exists(ens_dir):
    ckpts = [f for f in os.listdir(ens_dir) if f.endswith('.pt')]
    print(f'\nEnsemble checkpoints: {len(ckpts)} files')
    for c in sorted(ckpts):
        size_mb = os.path.getsize(os.path.join(ens_dir, c)) / 1048576
        print(f'  {c}: {size_mb:.1f} MB')

# Save the analysis for later use
print('\n--- Reduction distribution ---')
reductions = [r['reduction'] for r in results]
print(f'Mean: {np.mean(reductions):.1f}%')
print(f'Median: {np.median(reductions):.1f}%')
print(f'Std: {np.std(reductions):.1f}%')
print(f'Min: {np.min(reductions):.1f}%')
print(f'Max: {np.max(reductions):.1f}%')
print(f'P25: {np.percentile(reductions, 25):.1f}%')
print(f'P75: {np.percentile(reductions, 75):.1f}%')
