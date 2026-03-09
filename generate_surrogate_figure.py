#!/usr/bin/env python3
"""
Surrogate model figure – 5 panels using real project data.

(a)  Architecture schematic  – 7-ch voxel input → 3D ResNet → ×5 ensemble → 3 outputs
(b)  Performance metrics bar – Spearman ρ, R²_log, MAPE for all 3 targets
(c)  Compliance scatter      – surrogate_comp_mean  vs  FEA ground-truth (n=100)
(d)  Von Mises scatter       – surrogate_vm_mean    vs  FEA ground-truth (n=100)
(e)  Optimization convergence– volume fraction + compliance vs batch (SASTO-PA)

Data sources
  fea_ml/runs/v3/fea_validation_100.json            – pred vs FEA for 100 designs
  fea_ml/runs/v3/optimization_128/
        optimization_summary_v11.json              – SASTO-PA batch history
  hard-coded metrics from poster (compute_surrogate_metrics.py output)

Output: figures/fig_surrogate_model.png  (300 dpi)
"""

import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import spearmanr

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE  = Path(__file__).parent
FEA   = BASE / "fea_ml" / "runs" / "v3"
OPT   = FEA / "optimization_128"
OUT   = BASE / "figures" / "fig_surrogate_model.png"
(BASE / "figures").mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":  "serif",
    "font.serif":   ["Times New Roman", "DejaVu Serif"],
    "font.size":    11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi":   130,
    "savefig.dpi":  300,
})

NAVY   = "#062B7A"
BLUE   = "#1A4FAA"
LBLUE  = "#D9E5FB"
TEAL   = "#00897B"
RED    = "#C0392B"
GOLD   = "#E67E22"
LGRAY  = "#F0F2F5"
DGRAY  = "#37474F"
GREEN  = "#2E7D32"

# ── Load data ─────────────────────────────────────────────────────────────────
val   = json.load(open(FEA / "fea_validation_100.json"))
opt11 = json.load(open(OPT / "optimization_summary_v11.json"))

# Scatter data (filter out zeros / diverged)
surr_comp = np.array([r["surrogate_comp_mean"] for r in val], dtype=float)
fea_comp  = np.array([r["voxel_opt_comp"]        for r in val], dtype=float)
surr_vm   = np.array([r["surrogate_vm_mean"]     for r in val], dtype=float)
fea_vm    = np.array([r["voxel_opt_vm"]          for r in val], dtype=float)

valid_c = (fea_comp > 0) & (surr_comp > 0) & np.isfinite(fea_comp) & np.isfinite(surr_comp)
valid_v = (fea_vm   > 0) & (surr_vm   > 0) & np.isfinite(fea_vm)   & np.isfinite(surr_vm)
surr_comp, fea_comp = surr_comp[valid_c], fea_comp[valid_c]
surr_vm,   fea_vm   = surr_vm[valid_v],   fea_vm[valid_v]

# Convergence history
hist = opt11["history"]
batch_num  = np.array([h["batch"]         for h in hist])
vol_red    = np.array([h["vol_reduction"]  for h in hist]) * 100   # %
comp_hist  = np.array([h["comp"]          for h in hist])
vm_hist    = np.array([h["vm"]            for h in hist])

# ── Known performance metrics (from compute_surrogate_metrics.py) ──────────────
metrics = {
    "Von Mises":    dict(spearman=0.737, r2=0.419, mape=37.4),
    "Displacement": dict(spearman=0.970, r2=0.842, mape=10.9),
    "Compliance":   dict(spearman=0.948, r2=0.814, mape=18.5),
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure layout
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 17), facecolor=LGRAY)
fig.suptitle("Deep Ensemble Surrogate Model — Architecture & Performance",
             fontsize=16, fontweight='bold', color=NAVY, y=0.985)

gs_top = gridspec.GridSpec(1, 2, figure=fig,
                           left=0.04, right=0.97, top=0.93, bottom=0.54,
                           wspace=0.08, width_ratios=[1.7, 1.0])
gs_bot = gridspec.GridSpec(1, 3, figure=fig,
                           left=0.05, right=0.97, top=0.50, bottom=0.07,
                           wspace=0.30)

ax_arch  = fig.add_subplot(gs_top[0])   # (a) architecture
ax_perf  = fig.add_subplot(gs_top[1])   # (b) performance bars
ax_comp  = fig.add_subplot(gs_bot[0])   # (c) compliance scatter
ax_vm    = fig.add_subplot(gs_bot[1])   # (d) VM scatter
ax_conv  = fig.add_subplot(gs_bot[2])   # (e) convergence

# ─────────────────────────────────────────────────────────────────────────────
# Panel (a): Architecture schematic
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_arch
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.set_facecolor(LGRAY)
ax.set_title('(a)  Deep Ensemble Architecture (×5 Members)',
             fontsize=12, fontweight='bold', color=NAVY, pad=7)

def rbox(ax, x, y, w, h, fc, ec=NAVY, lw=1.3, radius=0.012, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad={radius}",
                 facecolor=fc, edgecolor=ec, linewidth=lw, **kw))

def arrow(ax, x0, y0, x1, y1, color=NAVY, lw=1.4, ms=10):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=ms))

# --- Ensemble wrapper (light background) ---
rbox(ax, 0.17, 0.04, 0.66, 0.92, fc='#E8EDF8', ec=BLUE, lw=1.8,
     radius=0.018, zorder=0)
ax.text(0.50, 0.975, '× 5  Independent  Members', ha='center', va='top',
        fontsize=9, color=BLUE, style='italic', zorder=5)

# --- Input block ---
rbox(ax, 0.015, 0.38, 0.14, 0.24, fc=TEAL + 'DD', ec=TEAL)
ax.text(0.085, 0.51, "Input\nVoxel Grid\n128³", ha='center', va='center',
        fontsize=9.5, color='white', fontweight='bold')
ax.text(0.085, 0.38, "7 channels:\n1 occupancy\n6 part one-hot",
        ha='center', va='top', fontsize=7.8, color=TEAL)

# 3D cube icon
for z in [0.02, 0.01, 0.0]:
    ax.add_patch(plt.Polygon(
        [[0.04+z, 0.60+z],[0.13+z, 0.60+z],[0.13+z, 0.69+z],[0.04+z, 0.69+z]],
        closed=True, fc='white', ec=TEAL, lw=0.8, alpha=0.5-z*10, zorder=4))

arrow(ax, 0.155, 0.50, 0.185, 0.50)

# --- 3D ResNet stages ---
stage_colors = ['#1565C0','#1976D2','#1E88E5','#42A5F5']
stage_labels = ['Stage 1\n64ch\n32³', 'Stage 2\n128ch\n16³',
                'Stage 3\n256ch\n8³',  'Stage 4\n512ch\n4³']
xs = [0.19, 0.28, 0.37, 0.46]
for i, (x, lbl, fc) in enumerate(zip(xs, stage_labels, stage_colors)):
    rbox(ax, x, 0.34, 0.078, 0.32, fc=fc + 'EE', ec=fc)
    ax.text(x + 0.039, 0.50, lbl, ha='center', va='center',
            fontsize=8, color='white', fontweight='bold')
    # ResBlock stack lines
    for j in range(2):
        yy = 0.39 + j * 0.09
        ax.plot([x+0.01, x+0.068], [yy, yy], color='white', lw=0.7, alpha=0.6)
    if i < 3:
        arrow(ax, x + 0.078, 0.50, xs[i+1], 0.50, color='white', lw=1.2)

# SE-Attention label
ax.text(0.345, 0.285, 'Squeeze-and-Excitation\n(SE blocks at each stage)',
        ha='center', va='center', fontsize=7.5, color=BLUE, style='italic')

# --- Pooling + feature concat ---
rbox(ax, 0.555, 0.40, 0.095, 0.20, fc='#7B1FA2' + 'CC', ec='#7B1FA2')
ax.text(0.6025, 0.50, "AvgPool\n+\nFeat. Vec.\n(10-dim)",
        ha='center', va='center', fontsize=8, color='white', fontweight='bold')
arrow(ax, 0.538, 0.50, 0.555, 0.50, color='#7B1FA2')
ax.text(0.545, 0.405, 'Global\npooling', fontsize=7, ha='center',
        va='top', color='#7B1FA2')

# Feature vector annotation
rbox(ax, 0.558, 0.225, 0.090, 0.12, fc='#F3E5F5', ec='#7B1FA2', lw=1.0)
ax.text(0.603, 0.285, 'Feature\nvector\n(10-dim)', ha='center', va='center',
        fontsize=7.5, color='#7B1FA2')
arrow(ax, 0.603, 0.345, 0.603, 0.40, color='#7B1FA2', lw=1.2, ms=8)
ax.text(0.522, 0.283, 'Material\n+ Load\nparams', ha='right', va='center',
        fontsize=7, color='#555')
arrow(ax, 0.527, 0.283, 0.555, 0.283, color='#999', lw=1.0, ms=7)

# --- MLP head ---
rbox(ax, 0.668, 0.40, 0.090, 0.20, fc=GOLD + 'EE', ec=GOLD)
ax.text(0.713, 0.50, "MLP\nHead\n512→256\n→3",
        ha='center', va='center', fontsize=8, color='white', fontweight='bold')
arrow(ax, 0.650, 0.50, 0.668, 0.50, color=GOLD)

# Skip connection
ax.annotate("", xy=(0.713, 0.40), xytext=(0.713, 0.345),
            arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=0.9,
                            mutation_scale=7, connectionstyle="arc3,rad=0"))
rbox(ax, 0.668, 0.29, 0.090, 0.055, fc='#FFF8E1', ec=GOLD, lw=0.9)
ax.text(0.713, 0.317, 'Skip conn.', ha='center', va='center',
        fontsize=7, color=GOLD)

# --- Ensemble aggregation ---
rbox(ax, 0.78, 0.395, 0.08, 0.21, fc='#37474F' + 'CC', ec='#37474F')
ax.text(0.82, 0.50, "Ensemble\nAggregate\nμ ± σ",
        ha='center', va='center', fontsize=8, color='white', fontweight='bold')
arrow(ax, 0.758, 0.50, 0.78, 0.50, color='#37474F')

# Member lines (show 5 parallel paths symbolically)
for j, dy in enumerate([-0.14, -0.07, 0.0, 0.07, 0.14]):
    ax.plot([0.77, 0.78], [0.50+dy, 0.50], color='#90A4AE', lw=0.9, alpha=0.7)

# --- Outputs ---
out_labels = ['Von Mises\nStress (Pa)', 'Displacement\n(m)', 'Compliance\n(J)']
out_colors = [RED, BLUE, GREEN]
for i, (lbl, fc) in enumerate(zip(out_labels, out_colors)):
    y = 0.64 - i * 0.18
    rbox(ax, 0.876, y - 0.06, 0.105, 0.11, fc=fc + 'DD', ec=fc)
    ax.text(0.929, y, lbl, ha='center', va='center',
            fontsize=8, color='white', fontweight='bold')
    arrow(ax, 0.860, y, 0.876, y, color=fc)

arrow(ax, 0.860, 0.50, 0.860, 0.64, color=RED,   lw=0.9, ms=7)
arrow(ax, 0.860, 0.50, 0.860, 0.50, color=BLUE,  lw=0.9, ms=7)
arrow(ax, 0.860, 0.50, 0.860, 0.28, color=GREEN, lw=0.9, ms=7)

# --- Uncertainty bands annotation ---
ax.annotate("Conservative bound:\nμ + k·σ  (k=1.0)",
            xy=(0.820, 0.605), xytext=(0.700, 0.72),
            fontsize=7.5, color='#37474F', style='italic',
            arrowprops=dict(arrowstyle='->', color='#aaa', lw=0.9))

# --- Info boxes ---
rbox(ax, 0.015, 0.06, 0.155, 0.28, fc='#E3F2FD', ec=BLUE, lw=0.8)
for i, txt in enumerate([
        "Architecture:",
        "Surrogate3DResNet",
        "4 stages × 2 ResBlocks",
        "GELU + BatchNorm3d",
        "SE-attention (r=4)",
        "",
        "Training:",
        "11,178 FEA samples",
        "80/10/10 split",
        "AdamW, cosine LR",
        "Log + z-score targets",
]):
    bold = i in [0, 6]
    ax.text(0.022, 0.32 - i * 0.026, txt,
            fontsize=7.5, color=NAVY if bold else DGRAY,
            fontweight='bold' if bold else 'normal', va='top')

# ─────────────────────────────────────────────────────────────────────────────
# Panel (b): Performance bar chart
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_perf
ax.set_facecolor(LGRAY)
ax.set_title('(b)  Surrogate Performance Metrics', fontsize=12,
             fontweight='bold', color=NAVY, pad=7)

targets  = list(metrics.keys())
n_t      = len(targets)
bar_w    = 0.22
x        = np.arange(n_t)

# Three metric groups
spearman = [metrics[t]['spearman'] for t in targets]
r2       = [metrics[t]['r2']       for t in targets]
mape     = [metrics[t]['mape']     for t in targets]

ax2_perf = ax.twinx()

b1 = ax.bar(x - bar_w, spearman, bar_w, color=TEAL,   label='Spearman ρ', zorder=3)
b2 = ax.bar(x,          r2,      bar_w, color=BLUE,   label='R² (log)',   zorder=3)
b3 = ax2_perf.bar(x + bar_w, mape, bar_w, color=RED,  label='MAPE (%)', alpha=0.85, zorder=3)

# Value labels
for bar, val in zip(b1, spearman):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=TEAL)
for bar, val in zip(b2, r2):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=BLUE)
for bar, val in zip(b3, mape):
    ax2_perf.text(bar.get_x() + bar.get_width()/2, val + 0.6,
                  f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5,
                  fontweight='bold', color=RED)

ax.set_xticks(x)
ax.set_xticklabels(targets, fontsize=10)
ax.set_ylim(0, 1.18)
ax.set_ylabel('Spearman ρ  /  R² (log-space)', fontsize=10, color=NAVY)
ax2_perf.set_ylim(0, 55)
ax2_perf.set_ylabel('MAPE (%)', fontsize=10, color=RED)
ax2_perf.tick_params(axis='y', colors=RED)
ax.axhline(y=0.9, color=TEAL, lw=1.0, ls='--', alpha=0.6, zorder=2)
ax.text(n_t - 0.05, 0.91, 'ρ = 0.9', fontsize=7.5, color=TEAL, ha='right', alpha=0.8)

# Legend
h1 = mpatches.Patch(color=TEAL,  label='Spearman ρ')
h2 = mpatches.Patch(color=BLUE,  label='R² (log-space)')
h3 = mpatches.Patch(color=RED,   label='MAPE (%)', alpha=0.85)
ax.legend(handles=[h1, h2, h3], fontsize=9, loc='upper left',
          framealpha=0.9, edgecolor='#ccc')
ax.grid(axis='y', lw=0.5, alpha=0.5, zorder=1)
ax.set_facecolor('#F7F9FF')

# ─────────────────────────────────────────────────────────────────────────────
# Panel (c): Compliance scatter – surrogate vs FEA
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_comp
ax.set_facecolor('#F7F9FF')
ax.set_title('(c)  Compliance: Surrogate vs FEA', fontsize=12,
             fontweight='bold', color=NAVY, pad=6)

# Log-space for compliance
lsc = np.log10(surr_comp + 1e-12)
lfc = np.log10(fea_comp  + 1e-12)
rho_c, _ = spearmanr(lsc, lfc)

sc = ax.scatter(lfc, lsc, c=fea_comp, cmap='viridis', s=28, alpha=0.75,
                edgecolors='none', zorder=3)
lim_min = min(lfc.min(), lsc.min()) - 0.3
lim_max = max(lfc.max(), lsc.max()) + 0.3
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1.2,
        alpha=0.6, label='y = x (perfect)', zorder=2)

# ±20% band in log space
ax.fill_between([lim_min, lim_max],
                [lim_min - np.log10(1.2), lim_max - np.log10(1.2)],
                [lim_min + np.log10(1.2), lim_max + np.log10(1.2)],
                alpha=0.12, color=BLUE, label='±20% band', zorder=1)

cbar_c = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.040)
cbar_c.set_label('FEA Compliance (J)', fontsize=8)
cbar_c.ax.tick_params(labelsize=7)

ax.set_xlabel('FEA Ground Truth  (log₁₀ J)', fontsize=10)
ax.set_ylabel('Surrogate Prediction  (log₁₀ J)', fontsize=10)
stats_txt = f'n = {len(fea_comp)}\nSpearman ρ = {rho_c:.3f}\nR² = {metrics["Compliance"]["r2"]:.3f}\nMAPE = {metrics["Compliance"]["mape"]:.1f}%'
ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes,
        fontsize=9, va='top', ha='left', color=NAVY,
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=NAVY, lw=0.8, alpha=0.9))
ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax.grid(lw=0.4, alpha=0.45)

# ─────────────────────────────────────────────────────────────────────────────
# Panel (d): Von Mises scatter
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_vm
ax.set_facecolor('#F7F9FF')
ax.set_title('(d)  Von Mises Stress: Surrogate vs FEA', fontsize=12,
             fontweight='bold', color=NAVY, pad=6)

lsv = np.log10(surr_vm + 1e-1)
lfv = np.log10(fea_vm  + 1e-1)
rho_v, _ = spearmanr(lsv, lfv)

sv = ax.scatter(lfv, lsv, c=fea_vm, cmap='plasma', s=28, alpha=0.75,
                edgecolors='none', zorder=3,
                norm=mcolors.LogNorm(vmin=fea_vm.min(), vmax=fea_vm.max()))
lv_min = min(lfv.min(), lsv.min()) - 0.3
lv_max = max(lfv.max(), lsv.max()) + 0.3
ax.plot([lv_min, lv_max], [lv_min, lv_max], 'k--', lw=1.2, alpha=0.6,
        label='y = x')
ax.fill_between([lv_min, lv_max],
                [lv_min - np.log10(1.5), lv_max - np.log10(1.5)],
                [lv_min + np.log10(1.5), lv_max + np.log10(1.5)],
                alpha=0.12, color=RED, label='±50% band', zorder=1)

cbar_v = plt.colorbar(sv, ax=ax, pad=0.02, fraction=0.040)
cbar_v.set_label('FEA Von Mises (Pa)', fontsize=8)
cbar_v.ax.tick_params(labelsize=7)

ax.set_xlabel('FEA Ground Truth  (log₁₀ Pa)', fontsize=10)
ax.set_ylabel('Surrogate Prediction  (log₁₀ Pa)', fontsize=10)
sv_txt = f'n = {len(fea_vm)}\nSpearman ρ = {rho_v:.3f}\nR² = {metrics["Von Mises"]["r2"]:.3f}\nMAPE = {metrics["Von Mises"]["mape"]:.1f}%'
ax.text(0.03, 0.97, sv_txt, transform=ax.transAxes,
        fontsize=9, va='top', ha='left', color=NAVY,
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=NAVY, lw=0.8, alpha=0.9))
ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax.grid(lw=0.4, alpha=0.45)

# ─────────────────────────────────────────────────────────────────────────────
# Panel (e): Optimization convergence
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_conv
ax.set_facecolor('#F7F9FF')
ax.set_title('(e)  SASTO-PA Optimization Convergence (Sample 00472)',
             fontsize=12, fontweight='bold', color=NAVY, pad=6)

# Smooth slightly for readability
from scipy.ndimage import uniform_filter1d
vol_sm  = uniform_filter1d(vol_red,  size=5)
comp_sm = uniform_filter1d(comp_hist, size=5)
vm_sm   = uniform_filter1d(vm_hist,   size=5)

ax2_conv = ax.twinx()

ax.plot(batch_num, vol_sm, color=BLUE, lw=2.0, label='Volume removed (%)', zorder=3)
ax.fill_between(batch_num, 0, vol_sm, alpha=0.12, color=BLUE)

ax2_conv.plot(batch_num, comp_sm, color=RED,   lw=1.8, ls='-',
              label='Compliance (J)', zorder=3)
ax2_conv.plot(batch_num, vm_sm / 1e6, color=GOLD, lw=1.5, ls='--',
              label='VM Stress (MPa)', zorder=3, alpha=0.85)

# Phase annotations
phases = [(1, 90, 'Phase 1\n(Coarse)'), (91, 180, 'Phase 2\n(Medium)'),
          (181, 270, 'Phase 3\n(Fine)')]
pal = ['#E3F2FD', '#E8F5E9', '#FFF8E1']
for (s, e, lbl), c in zip(phases, pal):
    ax.axvspan(s, e, alpha=0.35, color=c, zorder=0)
    ax.text((s+e)/2, vol_sm.max()*0.88, lbl, ha='center', va='top',
            fontsize=8, color=DGRAY, style='italic')

# Final result annotation
final_vol = vol_red[-1]
ax.annotate(f'Final removal:\n{final_vol:.1f}%',
            xy=(batch_num[-1], vol_sm[-1]),
            xytext=(batch_num[-1] - 50, vol_sm[-1] + 3),
            fontsize=8.5, color=BLUE, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.0))

ax.set_xlabel('Batch Number', fontsize=10)
ax.set_ylabel('Volume Removed (%)', color=BLUE, fontsize=10)
ax2_conv.set_ylabel('Compliance (J)  /  VM Stress (MPa)', color=RED, fontsize=10)
ax2_conv.tick_params(axis='y', labelcolor=RED)
ax.tick_params(axis='y', labelcolor=BLUE)
ax.set_xlim(batch_num.min(), batch_num.max())
ax.grid(lw=0.4, alpha=0.45)

# Combined legend
lines_a, labels_a = ax.get_legend_handles_labels()
lines_b, labels_b = ax2_conv.get_legend_handles_labels()
ax.legend(lines_a + lines_b, labels_a + labels_b,
          fontsize=8.5, loc='lower right', framealpha=0.92, edgecolor='#ccc')

# ── Footer stats bar ──────────────────────────────────────────────────────────
banner = fig.add_axes([0.18, 0.012, 0.64, 0.048])
banner.set_xlim(0, 1); banner.set_ylim(0, 1); banner.axis('off')
banner.add_patch(FancyBboxPatch((0,0), 1, 1, boxstyle="round,pad=0",
                facecolor='#DAE3F3', edgecolor=NAVY, lw=1.2))
params = [
    ("43.8 M",       "Total parameters (×5)"),
    ("8.76 M",       "Params per member"),
    ("11,178",       "Training FEA runs"),
    ("7 channels",   "Input (occ + parts)"),
    ("3 outputs",    "VM · Disp · Compliance"),
    ("50 ms",        "Inference per sample"),
]
for i, (v, l) in enumerate(params):
    xi = (i + 0.5) / len(params)
    banner.text(xi, 0.74, v, ha='center', va='center',
                fontsize=10, fontweight='bold', color=NAVY)
    banner.text(xi, 0.20, l, ha='center', va='center',
                fontsize=8, color=DGRAY)
    if i < len(params)-1:
        banner.axvline(x=(i+1)/len(params), color='#90A4AE', lw=0.8,
                       ymin=0.1, ymax=0.9)

# ── Save ─────────────────────────────────────────────────────────────────────
fig.savefig(str(OUT), dpi=300, bbox_inches='tight', facecolor=LGRAY)
plt.close(fig)
print(f"Saved → {OUT}")
