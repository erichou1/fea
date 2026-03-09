import json, numpy as np
from scipy import stats
fea_data = json.load(open('fea_ml/runs/v3/fea_validation_full.json'))
surr_sub = [(r["surrogate_comp_mean"], r["voxel_opt_comp"])
             for r in fea_data
             if r.get("surrogate_comp_mean") and r.get("voxel_opt_comp")]
s_pred = np.array([x[0] for x in surr_sub])
s_true = np.array([x[1] for x in surr_sub])
print(f"n={len(s_pred)}")
print(f"surrogate: min={s_pred.min():.4f} max={s_pred.max():.4f} mean={s_pred.mean():.4f} std={s_pred.std():.4f}")
print(f"FEA:       min={s_true.min():.4f} max={s_true.max():.4f} mean={s_true.mean():.4f} std={s_true.std():.4f}")
rho, p = stats.spearmanr(s_pred, s_true)
print(f"Spearman rho={rho:.4f}, p={p:.2e}")
# Check if the range difference is the problem
print(f"Ratio of ranges: FEA_range/surr_range = {(s_true.max()-s_true.min())/(s_pred.max()-s_pred.min()):.1f}")
