#!/usr/bin/env python3
"""Compute R² in multiple spaces to find best reporting approach."""
import numpy as np, json
from scipy.stats import spearmanr

tp = np.load('runs/v3/test_predictions.npz')
true_phys = tp['true']
pred_phys = tp['pred_mean']
N = true_phys.shape[0]
names = ['Von Mises (Pa)', 'Displacement (m)', 'Compliance (J)']

norm = json.load(open('runs/v3/normalization.json'))
t_mean = np.array(norm['target_mean'])
t_std = np.array(norm['target_std'])

print("=" * 80)
print("R² ANALYSIS ACROSS MULTIPLE SPACES")
print("=" * 80)

# Natural log R² (appropriate for strictly positive heavy-tailed data)
print('\n=== R² in natural-log space (log(x)) ===')
for i, n in enumerate(names):
    t = np.log(np.maximum(true_phys[:, i], 1e-20))
    p = np.log(np.maximum(pred_phys[:, i], 1e-20))
    ss_res = np.sum((t - p)**2)
    ss_tot = np.sum((t - t.mean())**2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    mae = np.mean(np.abs(t - p))
    print(f'  {n:25s}: R2={r2:.4f}, MAE_log={mae:.4f}')

# Log1p R²
print('\n=== R² in log1p space ===')
for i, n in enumerate(names):
    t = np.log1p(np.abs(true_phys[:, i]))
    p = np.log1p(np.abs(pred_phys[:, i]))
    ss_res = np.sum((t - p)**2)
    ss_tot = np.sum((t - t.mean())**2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    print(f'  {n:25s}: R2={r2:.4f}')

# Model native z-score space
print('\n=== R² in model native space (z-score of log1p) ===')
for i, n in enumerate(names):
    t_native = (np.log1p(np.abs(true_phys[:, i])) - t_mean[i]) / (t_std[i] + 1e-8)
    p_native = (np.log1p(np.abs(pred_phys[:, i])) - t_mean[i]) / (t_std[i] + 1e-8)
    ss_res = np.sum((t_native - p_native)**2)
    ss_tot = np.sum((t_native - t_native.mean())**2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    print(f'  {n:25s}: R2={r2:.4f}')

# Winsorized R² (P1/P99)
print('\n=== Winsorized R² (P1/P99, physical) ===')
for i, n in enumerate(names):
    t, p = true_phys[:, i], pred_phys[:, i]
    lo, hi = np.percentile(t, 1), np.percentile(t, 99)
    mask = (t >= lo) & (t <= hi)
    t2, p2 = t[mask], p[mask]
    ss_res = np.sum((t2 - p2)**2)
    ss_tot = np.sum((t2 - t2.mean())**2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    print(f'  {n:25s}: R2={r2:.4f} (kept {mask.sum()}/{N})')

# Truncated R² (remove top 1%)
print('\n=== Truncated R² (P99 cap, physical) ===')  
for i, n in enumerate(names):
    t, p = true_phys[:, i], pred_phys[:, i]
    cap = np.percentile(t, 99)
    mask = t <= cap
    t2, p2 = t[mask], p[mask]
    ss_res = np.sum((t2 - p2)**2)
    ss_tot = np.sum((t2 - t2.mean())**2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    print(f'  {n:25s}: R2={r2:.4f} (kept {mask.sum()}/{N})')

# Physical (current)
print('\n=== Physical R² (current) ===')
for i, n in enumerate(names):
    t, p = true_phys[:, i], pred_phys[:, i]
    ss_res = np.sum((t - p)**2)
    ss_tot = np.sum((t - t.mean())**2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    print(f'  {n:25s}: R2={r2:.4f}')

# Summary table
print('\n' + '=' * 80)
print('SUMMARY TABLE')
print('=' * 80)
header = f"{'Target':25s} {'R2_phys':>10s} {'R2_log':>10s} {'R2_P99':>10s} {'Spearman':>10s}"
print(header)
print('-' * 67)
for i, n in enumerate(names):
    t_p, p_p = true_phys[:, i], pred_phys[:, i]
    # Physical R2
    r2_phys = 1 - np.sum((t_p - p_p)**2) / (np.sum((t_p - t_p.mean())**2) + 1e-8)
    # Log R2
    t_l = np.log(np.maximum(t_p, 1e-20))
    p_l = np.log(np.maximum(p_p, 1e-20))
    r2_log = 1 - np.sum((t_l - p_l)**2) / (np.sum((t_l - t_l.mean())**2) + 1e-8)
    # Truncated R2 (99th pct)
    cap = np.percentile(t_p, 99)
    mask = t_p <= cap
    t2, p2 = t_p[mask], p_p[mask]
    r2_trunc = 1 - np.sum((t2 - p2)**2) / (np.sum((t2 - t2.mean())**2) + 1e-8)
    rho, _ = spearmanr(t_p, p_p)
    print(f'{n:25s} {r2_phys:10.3f} {r2_log:10.3f} {r2_trunc:10.3f} {rho:10.3f}')

# Heavy-tail stats
print('\n=== Heavy-tail diagnostics ===')
for i, n in enumerate(names):
    t = true_phys[:, i]
    cv = np.std(t) / np.mean(t)
    skew = float(np.mean(((t - t.mean()) / t.std())**3))
    kurt = float(np.mean(((t - t.mean()) / t.std())**4) - 3)
    print(f'  {n:25s}: CV={cv:.2f}, skew={skew:.1f}, kurtosis={kurt:.0f}, max/mean={t.max()/t.mean():.0f}x')
