"""Fix layout overlaps in 6 figures."""
import json, pathlib, warnings
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde, rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from matplotlib.image import imread as mpimread

warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": "#b0b8c8", "axes.linewidth": 0.9,
    "xtick.direction": "out", "ytick.direction": "out",
    "figure.dpi": 100, "savefig.dpi": 200, "mathtext.fontset": "dejavusans",
})

ROOT    = pathlib.Path(".")
DATA_V3 = ROOT / "fea_ml" / "runs" / "v3"
BATCH   = DATA_V3 / "batch_results_all"
OPT128  = DATA_V3 / "optimization_128"
OUT     = ROOT / "results_figures"
RENDERS = ROOT / "poster_final" / "renders_hq"

BG="#ffffff"; PANEL="#f7f8fb"; CARD="#f0f3f8"; GRAY="#1c2233"; MID="#4a5568"
DIM="#8896b0"; LGRID="#dde3ef"; SPINE="#b0b8c8"; ACC="#1a6fd4"; TEAL="#0e9a76"
RED="#d63031"; GOLD="#d18f00"; C_PA="#c95a10"; PURPLE="#7e3acb"
LIMIT = 1.15

def save(fig, name, dpi=200):
    p = OUT / name
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  OK  {p.name:<42}  {p.stat().st_size//1024:>5} KB")

def style_ax(ax, xlabel=None, ylabel=None, title=None, grid=True):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    ax.tick_params(colors=GRAY, labelsize=9.5, labelcolor=GRAY, length=4)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10.5, color=GRAY, labelpad=6)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10.5, color=GRAY, labelpad=6)
    if title:  ax.set_title(title, fontsize=12, color=GRAY, pad=8, fontweight="bold")
    if grid:   ax.grid(color=LGRID, linewidth=0.7, zorder=0); ax.set_axisbelow(True)

def eq_box(ax, text, x=0.97, y=0.97, ha="right", va="top", fs=10):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fs, va=va, ha=ha, color=MID,
            math_fontfamily="dejavusans",
            bbox=dict(boxstyle="round,pad=0.35", fc="#edf2fc", ec=ACC, lw=1.0, alpha=0.92))

def stat_box(ax, text, x=0.03, y=0.97, ha="left", va="top", fs=8.5):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fs, va=va, ha=ha, color=GRAY,
            bbox=dict(boxstyle="round,pad=0.35", fc=PANEL, ec=SPINE, lw=0.8))

def legend_styled(ax, **kw):
    kw.setdefault("frameon", True); kw.setdefault("framealpha", 0.92)
    kw.setdefault("edgecolor", SPINE); kw.setdefault("facecolor", PANEL)
    kw.setdefault("labelcolor", GRAY); kw.setdefault("fontsize", 9)
    return ax.legend(**kw)

# ── load data ──────────────────────────────────────────────────────
print("Loading data...")
reductions, times_s = [], []
for d in sorted(BATCH.iterdir()):
    p = d / "optimization_summary.json"
    try:
        s = json.load(open(p))
        if s.get("success"):
            reductions.append(s["volume_reduction_pct"])
            times_s.append(s["total_time_seconds"])
    except: pass
reductions = np.array(reductions); times_arr = np.array(times_s); n_pop = len(reductions)

fea_data   = json.load(open(DATA_V3 / "fea_validation_full.json"))
comp_ratio = np.array([r["comp_ratio"] for r in fea_data if r.get("comp_ratio") is not None])
vr_paired  = np.array([r["volume_reduction_pct"] for r in fea_data if r.get("comp_ratio") is not None])
n_fea      = len(comp_ratio)

cal_data   = json.load(open(DATA_V3 / "calibration_results.json"))
conn_data  = json.load(open(DATA_V3 / "connectivity_analysis.json"))
simp_data  = json.load(open(DATA_V3 / "simp_benchmark.json"))
opt_v11    = json.load(open(OPT128 / "optimization_summary_v11.json"))

surr_sub = [(r["surrogate_comp_mean"], r["voxel_opt_comp"])
             for r in fea_data if r.get("surrogate_comp_mean") and r.get("voxel_opt_comp")]
s_pred = np.array([x[0] for x in surr_sub]); s_true = np.array([x[1] for x in surr_sub])
n_reg  = len(s_pred); spearman_r, _ = stats.spearmanr(s_pred, s_true)
rank_pred = rankdata(s_pred) / n_reg * 100; rank_true = rankdata(s_true) / n_reg * 100
base_comp = np.array([r["voxel_base_comp"] for r in fea_data if r.get("voxel_base_comp") and r.get("voxel_opt_comp")])
opt_comp  = np.array([r["voxel_opt_comp"]  for r in fea_data if r.get("voxel_base_comp") and r.get("voxel_opt_comp")])
cr_paired = opt_comp / base_comp
print(f"  n_pop={n_pop}  n_fea={n_fea}  n_surr={n_reg}")


# ═══════════════════════════════════════════════════════════════════
# 1. VOLUME REDUCTION — eq_box moved to bottom-right; legend upper-left
# ═══════════════════════════════════════════════════════════════════
print("fig_volume_reduction.png")
mu, sig = reductions.mean(), reductions.std()
fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
style_ax(ax, xlabel="Volume reduction  (%)", ylabel="Number of designs",
         title=rf"Population Volume Reduction  ($n = {n_pop:,}$)")
bins = np.linspace(0, max(reductions) + 2, 40)
ax.hist(reductions, bins=bins, color=C_PA, alpha=0.55, edgecolor="white", lw=0.5, zorder=3)
kde = gaussian_kde(reductions, bw_method=0.18)
xkde = np.linspace(0, bins[-1], 300); ykde = kde(xkde) * n_pop * (bins[1] - bins[0])
ax.plot(xkde, ykde, color=C_PA, lw=2.4, zorder=4)
ax.axvline(mu, color=TEAL, lw=2.0, ls="-.", zorder=5, label=rf"Mean  {mu:.1f}%")

# stat_box TOP-LEFT, eq_box BOTTOM-RIGHT — no overlap
stat_box(ax, f"$n$ = {n_pop:,}\n"rf"$\mu$ = {mu:.1f}%,  $\sigma$ = {sig:.1f}%",
         x=0.03, y=0.97, fs=9)
eq_box(ax, r"$\Delta V = \frac{V_{base} - V_{opt}}{V_{base}} \times 100$",
       x=0.97, y=0.12, ha="right", va="bottom", fs=10.5)
legend_styled(ax, loc="upper right")
ax.set_xlim(0, bins[-1])
ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.tight_layout()
save(fig, "fig_volume_reduction.png")


# ═══════════════════════════════════════════════════════════════════
# 2. FEA COMPLIANCE — stat_box moved to LEFT panel left side
# ═══════════════════════════════════════════════════════════════════
print("fig_fea_compliance.png")
mu_cr = comp_ratio.mean(); mx_cr = comp_ratio.max(); violations = (comp_ratio > LIMIT).sum()

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG, gridspec_kw=dict(wspace=0.32))
ax = axes[0]
style_ax(ax, xlabel=r"Compliance ratio  $C_{ratio}$", ylabel="Designs",
         title=rf"Compliance Ratio  ($n = {n_fea}$)")
bins2 = np.linspace(comp_ratio.min() - 0.02, LIMIT + 0.05, 36)
ax.hist(comp_ratio, bins=bins2, color=TEAL, alpha=0.55, edgecolor="white", lw=0.5, zorder=3)
kde2 = gaussian_kde(comp_ratio, bw_method=0.20)
xk2 = np.linspace(bins2[0], bins2[-1], 300); yk2 = kde2(xk2) * n_fea * (bins2[1] - bins2[0])
ax.plot(xk2, yk2, color=TEAL, lw=2.2, zorder=4)
ax.axvline(LIMIT, color=RED, lw=2.0, ls="--", zorder=5, label=rf"Safety limit  {LIMIT}")
legend_styled(ax, fontsize=9, loc="upper left")
# Both boxes left of sparse zone, clear of red line at axes x≈0.94 — right edge at 0.84
eq_box(ax, r"$C_{ratio} = \frac{C_{opt}}{C_{base}} \leq 1.15$",
       x=0.84, y=0.97, ha="right", va="top", fs=10)
stat_box(ax, f"$\\mu$ = {mu_cr:.3f},  max = {mx_cr:.3f}\nviolations = {violations}",
         x=0.84, y=0.62, ha="right", va="top", fs=8.5)

ax2 = axes[1]
style_ax(ax2, xlabel="Volume reduction (%)", ylabel=r"$C_{opt} / C_{base}$",
         title="Structural Safety vs Aggressiveness")
sc = ax2.scatter(vr_paired, comp_ratio, c=comp_ratio, cmap="RdYlGn_r",
                 vmin=0.2, vmax=LIMIT, s=24, alpha=0.65, zorder=3, edgecolors="none")
ax2.axhline(LIMIT, color=RED, lw=1.8, ls="--", zorder=4)
ax2.axhline(1.0, color=MID, lw=1.0, ls=":", alpha=0.5, zorder=3)
cb = plt.colorbar(sc, ax=ax2, pad=0.03, shrink=0.88)
cb.ax.tick_params(colors=GRAY, labelsize=8); cb.set_label(r"$C_{ratio}$", color=GRAY, fontsize=9.5)
ax2.text(0.50, 0.05, rf"Zero violations  ({n_fea:,} designs)",
         ha="center", va="bottom", color=TEAL, fontsize=11, fontweight="bold",
         transform=ax2.transAxes,
         bbox=dict(boxstyle="round,pad=0.35", fc="#ecfaf4", ec=TEAL, lw=1.6))
plt.tight_layout()
save(fig, "fig_fea_compliance.png")


# ═══════════════════════════════════════════════════════════════════
# 3. K-CALIBRATION — n= box moved to lower-left
# ═══════════════════════════════════════════════════════════════════
print("fig_k_calibration.png")
k_abl = cal_data["k_ablation"]
k_vals, pct_ok, avg_red = [], [], []
for k_str in sorted(k_abl.keys(), key=float):
    k_vals.append(float(k_str)); pct_ok.append(k_abl[k_str]["pct"]); avg_red.append(k_abl[k_str]["avg_reduction"])
k_vals = np.array(k_vals); pct_ok = np.array(pct_ok); avg_red = np.array(avg_red)

fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
style_ax(ax, xlabel=r"Safety factor  $k$", ylabel="FEA pass rate  (%)",
         title=r"Safety Factor Calibration  —  $k$-Ablation")
line1, = ax.plot(k_vals, pct_ok, "o-", color=TEAL, lw=2.4, ms=8, zorder=4, label="FEA pass rate (%)")
ax.fill_between(k_vals, 0, pct_ok, color=TEAL, alpha=0.06, zorder=1)
ax2 = ax.twinx()
ax2.set_ylabel("Avg. volume reduction (%)", fontsize=10.5, color=C_PA)
ax2.tick_params(axis="y", colors=C_PA, labelsize=9.5)
line2, = ax2.plot(k_vals, avg_red, "s--", color=C_PA, lw=2.0, ms=7, zorder=4, label="Avg. volume reduction (%)")
ax2.set_ylim(15, 30); ax2.spines["right"].set_color(C_PA)

chosen_k = 1.0
idx_k = list(k_vals).index(chosen_k) if chosen_k in k_vals else None
if idx_k is not None:
    ax.axvline(chosen_k, color=GOLD, lw=2.0, ls="--", zorder=3, alpha=0.7)
    ax.annotate(rf"$k = {chosen_k}$" + f"\n{pct_ok[idx_k]:.0f}% pass",
                xy=(chosen_k, pct_ok[idx_k]),
                xytext=(chosen_k + 0.6, pct_ok[idx_k] + 8),
                color=GOLD, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.3))

eq_box(ax, r"$C_{limit} = C_{base} \cdot (1 + k \cdot \sigma)$",
       x=0.97, y=0.12, ha="right", va="bottom", fs=10)
lines = [line1, line2]; labels = [l.get_label() for l in lines]
ax.legend(lines, labels, frameon=True, framealpha=0.92, edgecolor=SPINE,
          facecolor=PANEL, labelcolor=GRAY, fontsize=9.5, loc="upper left")
# n= box moved to LOWER-LEFT where fill is thin
stat_box(ax, f"$n$ = {cal_data['n_batch']:,}", x=0.03, y=0.12, fs=9)
ax.set_xlim(-0.1, 3.2); ax.set_ylim(0, 75)
plt.tight_layout()
save(fig, "fig_k_calibration.png")


# ═══════════════════════════════════════════════════════════════════
# 4. SIMP COMPARISON — more bottom margin for x-labels + legend
# ═══════════════════════════════════════════════════════════════════
print("fig_simp_comparison.png")
s_ids    = [s["sample_id"] for s in simp_data]
s_group  = [s["group"] for s in simp_data]
simp_red = np.array([s["volume_reduction_pct"] for s in simp_data])
sasto_red= np.array([s["sasto_reduction_pct"] for s in simp_data])
simp_t   = np.array([s["total_time_s"] for s in simp_data])
simp_cr  = np.array([s["comp_ratio"] for s in simp_data])
sasto_cr = np.array([s["sasto_comp_ratio"] for s in simp_data])
group_colors = {"high_reduction": RED, "near_boundary": GOLD, "easy": TEAL}
gc = [group_colors.get(g, DIM) for g in s_group]

fig, axes = plt.subplots(1, 3, figsize=(18, 7.0), facecolor=BG, gridspec_kw=dict(wspace=0.30))
x = np.arange(len(s_ids)); w = 0.35

ax = axes[0]
style_ax(ax, ylabel="Volume reduction  (%)", title="(a) Reduction: SIMP vs SASTO")
ax.bar(x - w/2, simp_red, w, color=PURPLE, alpha=0.60, edgecolor="white", label="SIMP (64\u00b3)", zorder=3)
ax.bar(x + w/2, sasto_red, w, color=C_PA, alpha=0.60, edgecolor="white", label="SASTO (128\u00b3)", zorder=3)
ax.set_xticks(x); ax.set_xticklabels(s_ids, rotation=45, ha="right", fontsize=8)
for tick, col in zip(ax.get_xticklabels(), gc): tick.set_color(col)
legend_styled(ax, loc="upper left", fontsize=8.5)

ax = axes[1]
sasto_med_sp = float(np.median(times_arr))
style_ax(ax, ylabel="Runtime  (seconds)", title="(b) Runtime: SIMP vs SASTO")
for xi, st, col in zip(x, simp_t, gc):
    ax.bar(xi, st, 0.50, color=col, alpha=0.72, edgecolor="white", zorder=3)
ax.axhline(sasto_med_sp, color=RED, lw=2.2, ls="--", zorder=4)
min_speedup = simp_t.min() / sasto_med_sp
max_speedup = simp_t.max() / sasto_med_sp
# Nx labels on top of each bar
for xi, st in zip(x, simp_t):
    sp = st / sasto_med_sp
    ax.text(xi, st * 1.12, f"{sp:.0f}\u00d7",
            ha="center", va="bottom", fontsize=8, color=GRAY, fontweight="bold")
# Speedup range box (top-right)
stat_box(ax, f"{min_speedup:.0f}\u2013{max_speedup:.0f}\u00d7 slower than SASTO",
         x=0.97, y=0.97, ha="right", fs=8.5)
# SASTO median box just below, styled like eq_box to distinguish it
ax.text(0.97, 0.80, f"SASTO median:  {sasto_med_sp:.0f} s  (\u2014 \u2014)",
        transform=ax.transAxes, fontsize=8.5, va="top", ha="right", color=RED,
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff0f0", ec=RED, lw=1.2, alpha=0.92))
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(s_ids, rotation=45, ha="right", fontsize=8)
for tick, col in zip(ax.get_xticklabels(), gc): tick.set_color(col)

ax = axes[2]
style_ax(ax, ylabel=r"Compliance ratio  $C_{opt}/C_{base}$", title="(c) Compliance Ratio")
ax.bar(x - w/2, simp_cr, w, color=PURPLE, alpha=0.60, edgecolor="white", label="SIMP", zorder=3)
ax.bar(x + w/2, sasto_cr, w, color=C_PA, alpha=0.60, edgecolor="white", label="SASTO", zorder=3)
ax.axhline(LIMIT, color=RED, lw=1.8, ls="--", zorder=4, alpha=0.7, label=f"Safety limit {LIMIT}")
ax.set_xticks(x); ax.set_xticklabels(s_ids, rotation=45, ha="right", fontsize=8)
for tick, col in zip(ax.get_xticklabels(), gc): tick.set_color(col)
legend_styled(ax, loc="upper left", fontsize=8.5)

grp_handles = [mpatches.Patch(fc=RED, alpha=0.6, label="High reduction"),
               mpatches.Patch(fc=GOLD, alpha=0.6, label="Near boundary"),
               mpatches.Patch(fc=TEAL, alpha=0.6, label="Easy")]
fig.legend(handles=grp_handles, loc="upper center", ncol=3, fontsize=10,
           frameon=True, framealpha=0.92, edgecolor=SPINE, facecolor=PANEL,
           labelcolor=GRAY, bbox_to_anchor=(0.5, 0.99))
plt.tight_layout(rect=[0, 0.0, 1, 0.91])
save(fig, "fig_simp_comparison.png")


# ═══════════════════════════════════════════════════════════════════
# 5. SURROGATE ACCURACY — stat_box on right panel moved to bottom
# ═══════════════════════════════════════════════════════════════════
print("fig_surrogate_accuracy.png")
fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=BG, gridspec_kw=dict(wspace=0.30))

ax = axes[0]
style_ax(ax, xlabel="Surrogate rank  (percentile)", ylabel="FEA compliance rank  (percentile)",
         title=rf"Surrogate Ranking Accuracy  ($n = {n_reg}$)")
rank_err = np.abs(rank_pred - rank_true)
sc = ax.scatter(rank_pred, rank_true, c=rank_err, cmap="RdYlGn_r",
                vmin=0, vmax=30, s=40, alpha=0.75, zorder=3, edgecolors="white", linewidths=0.4)
ax.plot([0, 100], [0, 100], color=MID, lw=1.3, ls="--", alpha=0.6, label=r"$y=x$ (perfect ranking)")
m, b = stats.linregress(rank_pred, rank_true)[:2]
ax.plot([0, 100], [b, m*100+b], color=TEAL, lw=2.2, zorder=4, label=rf"Fit  ($\rho = {spearman_r:.3f}$)")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
cb = plt.colorbar(sc, ax=ax, pad=0.03, shrink=0.88)
cb.ax.tick_params(colors=GRAY, labelsize=8); cb.set_label("Rank error (pctile)", color=GRAY, fontsize=9)
legend_styled(ax, fontsize=9, loc="upper left")
eq_box(ax, rf"$\rho_{{Spearman}} = {spearman_r:.3f}$", x=0.97, y=0.15, ha="right", va="bottom", fs=10.5)

ax2 = axes[1]
style_ax(ax2, xlabel=r"$C_{opt} / C_{base}$", ylabel="Designs", title="Compliance After Optimisation")
bins4 = np.linspace(0, 1.25, 36)
ax2.hist(cr_paired, bins=bins4, color=TEAL, alpha=0.55, edgecolor="white", lw=0.5, zorder=3)
ax2.axvline(1.0, color=ACC, lw=1.8, ls="-.", zorder=4, label=r"Baseline $C = 1$")
ax2.axvline(LIMIT, color=RED, lw=2.0, ls="--", zorder=5, label=f"Safety limit {LIMIT}")
pct_below = (cr_paired < 1.0).mean() * 100
# stat_box BOTTOM-CENTER (clear of legend and vertical lines)
stat_box(ax2, rf"{pct_below:.0f}% improve compliance", x=0.50, y=0.06, ha="center", va="bottom", fs=10)
legend_styled(ax2, loc="upper left")
plt.tight_layout()
save(fig, "fig_surrogate_accuracy.png")


# ═══════════════════════════════════════════════════════════════════
# 6. CONNECTIVITY — annotation box moved to upper-right
# ═══════════════════════════════════════════════════════════════════
print("fig_connectivity.png")
cs = conn_data["summary"]; ps_c = conn_data["per_sample"]; n_conn = cs["n_samples"]
comp_m = [r["mesh_components"] for r in ps_c]
all_comps = np.array(comp_m); pct_single_mesh = (all_comps == 1).mean() * 100

fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
style_ax(ax, ylabel="Percentage  (%)",
         title=rf"Structural Connectivity Verification  ($n = {n_conn}$)")
cats = ["Voxel\n6-connected", "Voxel\n26-connected", "Mesh\nwatertight", "Mesh\nsingle body"]
vals = [cs["voxel_6conn_all_single"]/n_conn*100,
        cs["voxel_26conn_all_single"]/n_conn*100,
        cs["mesh_all_single"]/n_conn*100,
        pct_single_mesh]
cols_bar = [ACC, ACC, TEAL, TEAL]
bars = ax.bar(cats, vals, color=cols_bar, alpha=0.65, edgecolor="white", lw=1.0, width=0.52, zorder=3)
for bar, v, col in zip(bars, vals, cols_bar):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{v:.0f}%", ha="center", va="bottom", fontsize=14, fontweight="bold", color=col)
ax.set_ylim(0, 118)
ax.axhline(100, color=DIM, lw=0.8, ls=":", alpha=0.5)
# box moved to UPPER-RIGHT (clear of bars)
ax.text(0.97, 0.97,
        "100% of voxel models are fully connected\n"
        f"{pct_single_mesh:.0f}% of exported meshes are single-body (printable)",
        ha="right", va="top", fontsize=9.5, color=GRAY,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.45", fc=PANEL, ec=SPINE, lw=0.8))
leg_h = [mpatches.Patch(fc=ACC, alpha=0.65, label="Voxel domain"),
         mpatches.Patch(fc=TEAL, alpha=0.65, label="Exported mesh (STL)")]
legend_styled(ax, handles=leg_h, loc="lower right", fontsize=10)
plt.tight_layout()
save(fig, "fig_connectivity.png")


# ═══════════════════════════════════════════════════════════════════
# 7. UNCERTAINTY — stronger shading + labelled zones
# ═══════════════════════════════════════════════════════════════════
print("fig_uncertainty.png")
h11 = opt_v11["history"]
vm0 = h11[0]["vm"]; c0 = h11[0]["comp"]; d0 = h11[0]["disp"]
vfrac   = np.array([1 - h["vol_reduction"] for h in h11])
vm_norm = np.array([h["vm"]   / vm0 for h in h11])
c_norm  = np.array([h["comp"] / c0  for h in h11])
d_norm  = np.array([h["disp"] / d0  for h in h11])
vm_allow_ratio = 5e6 / vm0

fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
style_ax(ax, xlabel=r"Volume fraction  $\phi = V/V_0$",
         ylabel="Normalised response  (ratio to baseline)",
         title=r"Safety Margin Evolution During Optimisation")

ax.plot(vfrac, vm_norm, color=RED,  lw=2.0, alpha=0.85, label="VM stress",    zorder=3)
ax.plot(vfrac, c_norm,  color=ACC,  lw=2.0, alpha=0.85, label="Compliance",   zorder=3)
ax.plot(vfrac, d_norm,  color=TEAL, lw=2.0, alpha=0.85, label="Displacement", zorder=3)
ax.axhline(LIMIT,           color=RED,  lw=1.5, ls="--", alpha=0.65, label=rf"$C_{{allow}}/C_0 = {LIMIT}$")
ax.axhline(vm_allow_ratio,  color=GOLD, lw=1.5, ls="--", alpha=0.65, label=rf"$\sigma_{{allow}}/\sigma_0 = {vm_allow_ratio:.2f}$")
ax.axhline(1.0,             color=DIM,  lw=0.8, ls=":",  alpha=0.5)

ax.invert_xaxis()
ax.set_xlim(1.0, vfrac.min() - 0.02)
ax.set_ylim(0.7, 2.05)   # explicit so zone shading is predictable

x_fill = [vfrac.min() - 0.02, 1.0]
# Safe zone: green fill below LIMIT
ax.fill_between(x_fill, 0.7, LIMIT, alpha=0.14, color=TEAL, zorder=0)
# Danger zone: red fill above LIMIT
ax.fill_between(x_fill, LIMIT, 2.05, alpha=0.10, color=RED, zorder=0)

# Zone labels with opaque white background
ax.text(0.50, 0.22, "\u2713  Safe zone  (pass constraint)",
        color=TEAL, fontsize=11, fontweight="bold",
        va="center", ha="center", transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=TEAL, lw=1.4, alpha=0.90))
ax.text(0.50, 0.78, "\u26A0  Constraint exceeded",
        color=RED, fontsize=10, fontweight="bold",
        va="center", ha="center", transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RED, lw=1.2, alpha=0.85))

legend_styled(ax, loc="upper right", fontsize=9)
eq_box(ax, r"$\phi = V_{opt} / V_{base}$", x=0.03, y=0.03, ha="left", va="bottom", fs=9.5)
plt.tight_layout()
save(fig, "fig_uncertainty.png")


# ═══════════════════════════════════════════════════════════════════
# 8. DIVERSE GALLERY — bottom row of STL gallery + SASTO-U panel
# ═══════════════════════════════════════════════════════════════════
print("fig_diverse_gallery.png")
stl_path   = RENDERS / "fig_diverse_stl_gallery.png"
sasto_u_path = RENDERS / "sasto_u_solid.png"

if stl_path.exists() and sasto_u_path.exists():
    stl_full = mpimread(str(stl_path))   # shape (H, W, C)
    sasto_u  = mpimread(str(sasto_u_path))

    gh, gw = stl_full.shape[:2]
    N_ROWS = 5   # 1 ref row + 4 gallery rows (matches generate_additional_figures)
    row_h  = gh // N_ROWS
    # Crop bottom gallery row (last of the 5 rows)
    bottom = stl_full[4 * row_h : gh, :, :]

    # Split horizontally into 3 equal column panels
    cw = gw // 3
    orig_panel = bottom[:, :cw, :]
    sapa_panel = bottom[:, cw : 2 * cw, :]
    isom_panel = bottom[:, 2 * cw :, :]

    # Build 1×4 figure: Original | SASTO-U | SASTO-PA | Isometric
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), facecolor=BG,
                             gridspec_kw=dict(wspace=0.04))
    panels = [
        (orig_panel,  "Original",          GRAY),
        (sasto_u,     "Optimized  (SASTO-U)",  TEAL),
        (sapa_panel,  "Optimized  (SASTO-PA)", C_PA),
        (isom_panel,  "Isometric view",     MID),
    ]
    for ax, (img, title, col) in zip(axes, panels):
        ax.imshow(img, interpolation="lanczos")
        ax.set_axis_off()
        ax.set_title(title, fontsize=13, fontweight="bold", color=col, pad=7)

    # Divider lines between panels
    for spine_ax in axes[1:]:
        for sp in spine_ax.spines.values():
            sp.set_visible(True); sp.set_color(SPINE); sp.set_linewidth(1.2)

    fig.suptitle("Diverse Optimisation Gallery  —  Original → SASTO-U → SASTO-PA",
                 fontsize=15, fontweight="bold", color=GRAY, y=1.02)
    plt.tight_layout()
    save(fig, "fig_diverse_gallery.png", dpi=150)
else:
    print("  SKIP — source images not found")


print("\nAll figures updated.")
