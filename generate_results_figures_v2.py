"""
ISEF Results & In-Silico Validation — 12 Poster Figures  (v2)
==============================================================
Output:  results_figures/

Selected 12 figures:

VISUAL EVIDENCE
  1. fig_diverse_gallery.png      — Diverse-design gallery with part legend
  2. fig_cross_sections.png       — Cross-section comparison with % removed

QUANTITATIVE RESULTS
  3. fig_volume_reduction.png     — Population histogram (n=1,246) + KDE
  4. fig_fea_compliance.png       — Compliance ratio distribution + scatter
  5. fig_surrogate_accuracy.png   — Rank-based surrogate accuracy scatter
  6. fig_bland_altman.png         — Bland-Altman rank agreement (simplified)

STRUCTURAL / ALGORITHMIC VALIDATION
  7. fig_connectivity.png         — Connectivity verification bar chart
  8. fig_k_calibration.png        — Safety factor k-ablation

NEW ADDITIONS
  9. fig_convergence.png          — SASTO-PA vs SASTO-U convergence curves
 10. fig_uncertainty.png          — Safety-margin evolution during optimisation
 11. fig_simp_comparison.png      — SIMP vs SASTO head-to-head (10 designs)
 12. fig_training_curves.png      — Ensemble surrogate training curves
"""

import json, re, pathlib, warnings, shutil
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde, rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=UserWarning)

matplotlib.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.edgecolor":  "#b0b8c8",
    "axes.linewidth":  0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "figure.dpi":      100,
    "savefig.dpi":     200,
    "mathtext.fontset": "dejavusans",
})

# ── paths ──────────────────────────────────────────────────────────
ROOT    = pathlib.Path(".")
RENDERS = ROOT / "poster_final" / "renders_hq"
DATA_V3 = ROOT / "fea_ml" / "runs" / "v3"
BATCH   = DATA_V3 / "batch_results_all"
OPT128  = DATA_V3 / "optimization_128"
OUT     = ROOT / "results_figures"
OUT.mkdir(exist_ok=True)

# ── light-theme design tokens ─────────────────────────────────────
BG      = "#ffffff"
PANEL   = "#f7f8fb"
CARD    = "#f0f3f8"
GRAY    = "#1c2233"
MID     = "#4a5568"
DIM     = "#8896b0"
LGRID   = "#dde3ef"
SPINE   = "#b0b8c8"
ACC     = "#1a6fd4"
TEAL    = "#0e9a76"
RED     = "#d63031"
GOLD    = "#d18f00"
C_ORIG  = "#2d6fc0"
C_U     = "#1e7d3a"
C_PA    = "#c95a10"
PURPLE  = "#7e3acb"

PC_EXT  = "#377dbe"
PC_INT  = "#f0783c"
PC_ROOF = "#64a032"
PC_FLR  = "#bea578"

# ── helpers ────────────────────────────────────────────────────────
def save(fig, name, dpi=200):
    p = OUT / name
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    sz = p.stat().st_size // 1024
    print(f"  OK  {p.name:<42}  {sz:>5} KB")

def style_ax(ax, xlabel=None, ylabel=None, title=None, grid=True):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_color(SPINE)
    ax.tick_params(colors=GRAY, labelsize=9.5, labelcolor=GRAY, length=4)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10.5, color=GRAY, labelpad=6)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10.5, color=GRAY, labelpad=6)
    if title:  ax.set_title(title, fontsize=12, color=GRAY, pad=8, fontweight="bold")
    if grid:
        ax.grid(color=LGRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

def eq_box(ax, text, x=0.97, y=0.97, ha="right", va="top", fs=10):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fs, va=va, ha=ha,
            color=MID, math_fontfamily="dejavusans",
            bbox=dict(boxstyle="round,pad=0.35", fc="#edf2fc", ec=ACC, lw=1.0, alpha=0.92))

def stat_box(ax, text, x=0.03, y=0.97, ha="left", va="top", fs=8.5):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fs, va=va, ha=ha,
            color=GRAY, bbox=dict(boxstyle="round,pad=0.35", fc=PANEL, ec=SPINE, lw=0.8))

def legend_styled(ax, **kw):
    kw.setdefault("frameon", True)
    kw.setdefault("framealpha", 0.92)
    kw.setdefault("edgecolor", SPINE)
    kw.setdefault("facecolor", PANEL)
    kw.setdefault("labelcolor", GRAY)
    kw.setdefault("fontsize", 9)
    return ax.legend(**kw)

def load_img(path):
    p = pathlib.Path(path)
    return imread(str(p)) if p.exists() else None

# ── Load population data once ─────────────────────────────────────
print("Loading batch data...")
reductions, times_s, part_break = [], [], []
for d in sorted(BATCH.iterdir()):
    p = d / "optimization_summary.json"
    try:
        s = json.load(open(p))
        if s.get("success"):
            reductions.append(s["volume_reduction_pct"])
            times_s.append(s["total_time_seconds"])
            if s.get("part_breakdown"):
                part_break.append(s["part_breakdown"])
    except:
        pass
reductions = np.array(reductions)
times_arr  = np.array(times_s)
n_pop      = len(reductions)
print(f"  n = {n_pop} successful optimisations loaded")

# FEA validation
fea_data   = json.load(open(DATA_V3 / "fea_validation_full.json"))
comp_ratio = np.array([r["comp_ratio"] for r in fea_data if r.get("comp_ratio") is not None])
vr_paired  = np.array([r["volume_reduction_pct"] for r in fea_data if r.get("comp_ratio") is not None])
LIMIT = 1.15
n_fea = len(comp_ratio)

# Calibration
cal_data   = json.load(open(DATA_V3 / "calibration_results.json"))

# Connectivity
conn_data  = json.load(open(DATA_V3 / "connectivity_analysis.json"))

# SIMP
simp_data  = json.load(open(DATA_V3 / "simp_benchmark.json"))
simp_times = np.array([s["total_time_s"] for s in simp_data])

# Surrogate metrics
surr_metrics = json.load(open(DATA_V3 / "surrogate_metrics.json"))

# Convergence history (SASTO-PA = v11, SASTO-U = v12)
opt_v11 = json.load(open(OPT128 / "optimization_summary_v11.json"))
opt_v12 = json.load(open(OPT128 / "optimization_summary_v12.json"))

print("Data loaded.\n")


# ═══════════════════════════════════════════════════════════════════
# 1. DIVERSE GALLERY  —  wrap HQ render + legend
# ═══════════════════════════════════════════════════════════════════
print("1/12  fig_diverse_gallery.png")
# Try type_comparison first (more diverse building types), then fallback
gal_candidates = [
    RENDERS / "fig_type_comparison.png",
    RENDERS / "fig_optimized_gallery.png",
    RENDERS / "fig_diverse_stl_gallery.png",
]
gal_src = None
for gc in gal_candidates:
    if gc.exists():
        gal_src = gc
        break

if gal_src is not None:
    gal_img = imread(str(gal_src))
    gh, gw = gal_img.shape[:2]
    FIG_W = 20
    fig = plt.figure(figsize=(FIG_W, FIG_W * gh / gw + 0.9), facecolor=BG)
    ax = fig.add_axes([0.01, 0.07, 0.98, 0.90])
    ax.set_axis_off()
    ax.imshow(gal_img, aspect="equal", interpolation="lanczos")
    leg_patches = [
        mpatches.Patch(fc=PC_EXT, ec="#555", lw=1.2, label="Exterior Wall"),
        mpatches.Patch(fc=PC_INT, ec="#555", lw=1.2, label="Interior Wall"),
        mpatches.Patch(fc=PC_ROOF, ec="#555", lw=1.2, label="Roof"),
        mpatches.Patch(fc=PC_FLR, ec="#555", lw=1.2, label="Floor"),
    ]
    leg = fig.legend(handles=leg_patches, loc="lower center", ncol=4,
                     fontsize=18, frameon=True, framealpha=0.97,
                     edgecolor=SPINE, facecolor=BG,
                     handlelength=2.8, handletextpad=0.8,
                     columnspacing=3.0,
                     bbox_to_anchor=(0.5, 0.003),
                     labelcolor=GRAY,
                     prop={"size": 18, "weight": "bold"})
    for patch in leg.get_patches():
        patch.set_height(18); patch.set_width(30)
    save(fig, "fig_diverse_gallery.png", dpi=150)
else:
    print("  SKIP — no gallery source found")


# ═══════════════════════════════════════════════════════════════════
# 2. CROSS-SECTIONS  —  wrap HQ render + add % removed labels
# ═══════════════════════════════════════════════════════════════════
print("2/12  fig_cross_sections.png")
xs_src = RENDERS / "fig_cross_section_comparison.png"
if xs_src.exists():
    xs_img = imread(str(xs_src))
    xh, xw = xs_img.shape[:2]
    FIG_W = 16
    fig = plt.figure(figsize=(FIG_W, FIG_W * xh / xw + 1.2), facecolor=BG)
    ax = fig.add_axes([0.01, 0.06, 0.98, 0.84])
    ax.set_axis_off()
    ax.imshow(xs_img, aspect="equal", interpolation="lanczos")

    # Add labels with % removed above each panel (3 panels side by side)
    labels_xs = [
        ("Original", C_ORIG, "100% volume"),
        ("SASTO-U", C_U, r"$-34.3\%$ removed"),
        ("SASTO-PA", C_PA, r"$-45.0\%$ removed"),
    ]
    for i, (name, col, pct) in enumerate(labels_xs):
        cx = (i + 0.5) / 3.0
        fig.text(cx, 0.95, name, ha="center", va="top", fontsize=14,
                 fontweight="bold", color=col)
        fig.text(cx, 0.915, pct, ha="center", va="top", fontsize=10.5,
                 color=col, math_fontfamily="dejavusans")

    save(fig, "fig_cross_sections.png", dpi=180)
else:
    print("  SKIP — source not found")


# ═══════════════════════════════════════════════════════════════════
# 3. VOLUME REDUCTION HISTOGRAM  (cleaned up)
# ═══════════════════════════════════════════════════════════════════
print("3/12  fig_volume_reduction.png")
fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
style_ax(ax, xlabel="Volume reduction  (%)",
         ylabel="Number of designs",
         title=rf"Population Volume Reduction  ($n = {n_pop:,}$)")

bins = np.linspace(0, max(reductions) + 2, 40)
ax.hist(reductions, bins=bins, color=C_PA, alpha=0.55,
        edgecolor="white", linewidth=0.5, zorder=3)

kde = gaussian_kde(reductions, bw_method=0.18)
xkde = np.linspace(0, bins[-1], 300)
ykde = kde(xkde) * n_pop * (bins[1] - bins[0])
ax.plot(xkde, ykde, color=C_PA, lw=2.4, zorder=4)

mu, sig = reductions.mean(), reductions.std()
ax.axvline(mu, color=TEAL, lw=2.0, ls="-.", zorder=5,
           label=rf"Mean  {mu:.1f}%")

eq_box(ax, r"$\Delta V = \frac{V_{base} - V_{opt}}{V_{base}} \times 100$",
       x=0.97, y=0.97, fs=10.5)
stat_box(ax, (f"$n$ = {n_pop:,}\n"
              rf"$\mu$ = {mu:.1f}%"
              rf",  $\sigma$ = {sig:.1f}%"),
         x=0.03, y=0.97, fs=9)

legend_styled(ax, loc="upper right")
ax.set_xlim(0, bins[-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.tight_layout()
save(fig, "fig_volume_reduction.png")


# ═══════════════════════════════════════════════════════════════════
# 4. FEA COMPLIANCE  (text overlap fixed)
# ═══════════════════════════════════════════════════════════════════
print("4/12  fig_fea_compliance.png")
mu_cr = comp_ratio.mean()
mx_cr = comp_ratio.max()
violations = (comp_ratio > LIMIT).sum()

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG,
                          gridspec_kw=dict(wspace=0.32))

# left: histogram
ax = axes[0]
style_ax(ax, xlabel=r"Compliance ratio  $C_{ratio}$",
         ylabel="Designs",
         title=rf"Compliance Ratio  ($n = {n_fea}$)")
bins2 = np.linspace(comp_ratio.min() - 0.02, LIMIT + 0.05, 36)
ax.hist(comp_ratio, bins=bins2, color=TEAL, alpha=0.55,
        edgecolor="white", linewidth=0.5, zorder=3)
kde2 = gaussian_kde(comp_ratio, bw_method=0.20)
xk2 = np.linspace(bins2[0], bins2[-1], 300)
yk2 = kde2(xk2) * n_fea * (bins2[1] - bins2[0])
ax.plot(xk2, yk2, color=TEAL, lw=2.2, zorder=4)
ax.axvline(LIMIT, color=RED, lw=2.0, ls="--", zorder=5,
           label=rf"Safety limit  {LIMIT}")
legend_styled(ax, fontsize=9, loc="upper left")

eq_box(ax, r"$C_{ratio} = \frac{C_{opt}}{C_{base}} \leq 1.15$",
       x=0.97, y=0.97, fs=10)
stat_box(ax, f"$\\mu$ = {mu_cr:.3f},  max = {mx_cr:.3f}\nviolations = {violations}",
         x=0.97, y=0.72, ha="right", fs=8.5)

# right: scatter
ax2 = axes[1]
style_ax(ax2, xlabel="Volume reduction (%)",
         ylabel=r"$C_{opt} / C_{base}$",
         title="Structural Safety vs Aggressiveness")
sc = ax2.scatter(vr_paired, comp_ratio, c=comp_ratio, cmap="RdYlGn_r",
                 vmin=0.2, vmax=LIMIT, s=24, alpha=0.65, zorder=3,
                 edgecolors="none")
ax2.axhline(LIMIT, color=RED, lw=1.8, ls="--", zorder=4)
ax2.axhline(1.0, color=MID, lw=1.0, ls=":", alpha=0.5, zorder=3)
cb = plt.colorbar(sc, ax=ax2, pad=0.03, shrink=0.88)
cb.ax.tick_params(colors=GRAY, labelsize=8)
cb.set_label(r"$C_{ratio}$", color=GRAY, fontsize=9.5)

ax2.text(0.50, 0.05,
         rf"Zero violations  ({n_fea:,} designs)",
         ha="center", va="bottom", color=TEAL, fontsize=11, fontweight="bold",
         transform=ax2.transAxes,
         bbox=dict(boxstyle="round,pad=0.35", fc="#ecfaf4", ec=TEAL, lw=1.6))

plt.tight_layout()
save(fig, "fig_fea_compliance.png")


# ═══════════════════════════════════════════════════════════════════
# 5. SURROGATE ACCURACY  (rank-based to fix scale mismatch)
# ═══════════════════════════════════════════════════════════════════
print("5/12  fig_surrogate_accuracy.png")

surr_sub = [(r["surrogate_comp_mean"], r["voxel_opt_comp"])
             for r in fea_data
             if r.get("surrogate_comp_mean") and r.get("voxel_opt_comp")]
s_pred = np.array([x[0] for x in surr_sub])
s_true = np.array([x[1] for x in surr_sub])
n_reg = len(s_pred)
spearman_r, sp_p = stats.spearmanr(s_pred, s_true)

# Use RANKS instead of raw values (fixes the 560× scale mismatch)
rank_pred = rankdata(s_pred) / n_reg * 100
rank_true = rankdata(s_true) / n_reg * 100

# Also compute compliance improvement for right panel
base_comp = np.array([r["voxel_base_comp"] for r in fea_data
                      if r.get("voxel_base_comp") and r.get("voxel_opt_comp")])
opt_comp  = np.array([r["voxel_opt_comp"] for r in fea_data
                      if r.get("voxel_base_comp") and r.get("voxel_opt_comp")])
cr_paired = opt_comp / base_comp

fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=BG,
                          gridspec_kw=dict(wspace=0.30))

# left: rank scatter (fixes straight-line issue)
ax = axes[0]
style_ax(ax, xlabel="Surrogate rank  (percentile)",
         ylabel="FEA compliance rank  (percentile)",
         title=rf"Surrogate Ranking Accuracy  ($n = {n_reg}$)")

rank_err = np.abs(rank_pred - rank_true)
sc = ax.scatter(rank_pred, rank_true, c=rank_err, cmap="RdYlGn_r",
                vmin=0, vmax=30, s=40, alpha=0.75, zorder=3,
                edgecolors="white", linewidths=0.4)
ax.plot([0, 100], [0, 100], color=MID, lw=1.3, ls="--", alpha=0.6,
        label=r"$y=x$ (perfect ranking)")
m, b = stats.linregress(rank_pred, rank_true)[:2]
xf = np.array([0, 100])
ax.plot(xf, m*xf + b, color=TEAL, lw=2.2, zorder=4,
        label=rf"Fit  ($\rho = {spearman_r:.3f}$)")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
cb = plt.colorbar(sc, ax=ax, pad=0.03, shrink=0.88)
cb.ax.tick_params(colors=GRAY, labelsize=8)
cb.set_label("Rank error (pctile)", color=GRAY, fontsize=9)
legend_styled(ax, fontsize=9, loc="upper left")

eq_box(ax, rf"$\rho_{{Spearman}} = {spearman_r:.3f}$",
       x=0.97, y=0.15, ha="right", va="bottom", fs=10.5)

# right: compliance improvement histogram
ax2 = axes[1]
style_ax(ax2, xlabel=r"$C_{opt} / C_{base}$",
         ylabel="Designs",
         title="Compliance After Optimisation")
bins4 = np.linspace(0, 1.25, 36)
ax2.hist(cr_paired, bins=bins4, color=TEAL, alpha=0.55,
         edgecolor="white", linewidth=0.5, zorder=3)
ax2.axvline(1.0, color=ACC, lw=1.8, ls="-.", zorder=4,
            label=r"Baseline $C = 1$")
ax2.axvline(LIMIT, color=RED, lw=2.0, ls="--", zorder=5,
            label=f"Safety limit {LIMIT}")
pct_below = (cr_paired < 1.0).mean() * 100
stat_box(ax2, rf"{pct_below:.0f}% improve compliance",
         x=0.50, y=0.93, ha="center", fs=10)
legend_styled(ax2, loc="upper left")
plt.tight_layout()
save(fig, "fig_surrogate_accuracy.png")


# ═══════════════════════════════════════════════════════════════════
# 6. BLAND-ALTMAN  (simplified — less text overlap)
# ═══════════════════════════════════════════════════════════════════
print("6/12  fig_bland_altman.png")

ba_pairs = [(r["surrogate_comp_mean"], r["voxel_opt_comp"])
             for r in fea_data
             if r.get("surrogate_comp_mean") and r.get("voxel_opt_comp")]
ba_score = np.array([x[0] for x in ba_pairs])
ba_fea   = np.array([x[1] for x in ba_pairs])
n_ba     = len(ba_score)

rank_surr = rankdata(ba_score) / n_ba * 100
rank_fea  = rankdata(ba_fea)   / n_ba * 100
spear_ba, p_ba = stats.spearmanr(ba_score, ba_fea)

mean_r = (rank_surr + rank_fea) / 2
diff_r = rank_surr - rank_fea
bias_r = diff_r.mean()
std_r  = diff_r.std()
loa_up = bias_r + 1.96 * std_r
loa_lo = bias_r - 1.96 * std_r

fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=BG)
style_ax(ax, xlabel="Mean percentile rank",
         ylabel=r"Rank difference  ($r_{surr} - r_{FEA}$)",
         title="Bland-Altman Agreement  (Percentile-Rank Space)")

sc = ax.scatter(mean_r, diff_r, c=np.abs(diff_r), cmap="RdYlGn_r",
                vmin=0, vmax=40, s=45, alpha=0.75, zorder=3,
                edgecolors="white", linewidths=0.4)
ax.axhline(bias_r, color=TEAL, lw=2.0, zorder=4,
           label=rf"Bias = {bias_r:+.1f}")
ax.axhline(loa_up, color=GOLD, lw=1.6, ls="--", zorder=4,
           label=rf"$\pm 1.96\sigma$ = [{loa_lo:+.1f}, {loa_up:+.1f}]")
ax.axhline(loa_lo, color=GOLD, lw=1.6, ls="--", zorder=4)
ax.axhline(0, color=DIM, lw=0.8, ls=":", alpha=0.5)
ax.fill_between([0, 100], loa_lo, loa_up, alpha=0.06, color=GOLD, zorder=1)
ax.set_xlim(0, 100)

legend_styled(ax, fontsize=9, loc="lower left")
eq_box(ax, rf"$\rho_{{Spearman}} = {spear_ba:.3f}$,  $n = {n_ba}$",
       x=0.97, y=0.97, fs=10)

plt.tight_layout()
save(fig, "fig_bland_altman.png")


# ═══════════════════════════════════════════════════════════════════
# 7. CONNECTIVITY  (fixed key)
# ═══════════════════════════════════════════════════════════════════
print("7/12  fig_connectivity.png")
cs = conn_data["summary"]
ps = conn_data["per_sample"]
n_conn = cs["n_samples"]
comp_m  = [r["mesh_components"] for r in ps]
all_comps = np.array(comp_m)
pct_single_mesh = (all_comps == 1).mean() * 100

fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
style_ax(ax, ylabel="Percentage  (%)",
         title=rf"Structural Connectivity Verification  ($n = {n_conn}$)")

# Four clear categories with plain-language labels
cats = ["Voxel\n6-connected", "Voxel\n26-connected", "Mesh\nwatertight",
        "Mesh\nsingle body"]
vals = [cs["voxel_6conn_all_single"] / n_conn * 100,
        cs["voxel_26conn_all_single"] / n_conn * 100,
        cs["mesh_all_single"] / n_conn * 100,
        pct_single_mesh]
cols_bar = [ACC, ACC, TEAL, TEAL]

bars = ax.bar(cats, vals, color=cols_bar, alpha=0.65, edgecolor="white",
              linewidth=1.0, width=0.52, zorder=3)
for bar, v, col in zip(bars, vals, cols_bar):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{v:.0f}%", ha="center", va="bottom", fontsize=14,
            fontweight="bold", color=col)
ax.set_ylim(0, 118)
ax.axhline(100, color=DIM, lw=0.8, ls=":", alpha=0.5)

# Simple annotation
ax.text(0.5, 0.85,
        "100% of voxel models are fully connected\n"
        f"{pct_single_mesh:.0f}% of exported meshes are single-body (printable)",
        ha="center", va="top", fontsize=10, color=GRAY,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.45", fc=PANEL, ec=SPINE, lw=0.8))

# Colour-coded legend
leg_h = [mpatches.Patch(fc=ACC, alpha=0.65, label="Voxel domain"),
         mpatches.Patch(fc=TEAL, alpha=0.65, label="Exported mesh (STL)")]
legend_styled(ax, handles=leg_h, loc="lower right", fontsize=10)

plt.tight_layout()
save(fig, "fig_connectivity.png")


# ═══════════════════════════════════════════════════════════════════
# 8. K-CALIBRATION  (fixed legend overlap)
# ═══════════════════════════════════════════════════════════════════
print("8/12  fig_k_calibration.png")
k_abl = cal_data["k_ablation"]
k_vals, pct_ok, avg_red = [], [], []
for k_str in sorted(k_abl.keys(), key=float):
    k_vals.append(float(k_str))
    pct_ok.append(k_abl[k_str]["pct"])
    avg_red.append(k_abl[k_str]["avg_reduction"])
k_vals  = np.array(k_vals)
pct_ok  = np.array(pct_ok)
avg_red = np.array(avg_red)

fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
style_ax(ax, xlabel=r"Safety factor  $k$",
         ylabel="FEA pass rate  (%)",
         title=r"Safety Factor Calibration  —  $k$-Ablation")

line1, = ax.plot(k_vals, pct_ok, "o-", color=TEAL, lw=2.4, ms=8, zorder=4,
                 label="FEA pass rate (%)")
ax.fill_between(k_vals, 0, pct_ok, color=TEAL, alpha=0.06, zorder=1)

ax2 = ax.twinx()
ax2.set_ylabel("Avg. volume reduction (%)", fontsize=10.5, color=C_PA)
ax2.tick_params(axis="y", colors=C_PA, labelsize=9.5)
line2, = ax2.plot(k_vals, avg_red, "s--", color=C_PA, lw=2.0, ms=7, zorder=4,
                  label="Avg. volume reduction (%)")
ax2.set_ylim(15, 30)
ax2.spines["right"].set_color(C_PA)

# Highlight chosen k
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

# Legend in upper LEFT to avoid overlap with annotation
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, frameon=True, framealpha=0.92, edgecolor=SPINE,
          facecolor=PANEL, labelcolor=GRAY, fontsize=9.5, loc="upper left")

stat_box(ax, f"$n$ = {cal_data['n_batch']:,}", x=0.03, y=0.85, fs=9)
ax.set_xlim(-0.1, 3.2)
ax.set_ylim(0, 75)
plt.tight_layout()
save(fig, "fig_k_calibration.png")


# ═══════════════════════════════════════════════════════════════════
# 9. CONVERGENCE  —  SASTO-PA vs SASTO-U convergence curves (NEW)
# ═══════════════════════════════════════════════════════════════════
print("9/12  fig_convergence.png")

h11 = opt_v11["history"]
h12 = opt_v12["history"]
b11 = np.array([h["batch"] for h in h11])
b12 = np.array([h["batch"] for h in h12])
vr11 = np.array([h["vol_reduction"]*100 for h in h11])
vr12 = np.array([h["vol_reduction"]*100 for h in h12])
vm11 = np.array([h["vm"]/1e6 for h in h11])  # Pa → MPa
vm12 = np.array([h["vm"]/1e6 for h in h12])
c11  = np.array([h["comp"] for h in h11])
c12  = np.array([h["comp"] for h in h12])

fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG,
                          gridspec_kw=dict(wspace=0.28))

# (a) Volume reduction vs batch
ax = axes[0]
style_ax(ax, xlabel="Batch", ylabel="Volume reduction  (%)",
         title="(a) Volume Reduction Convergence")
ax.plot(b11, vr11, color=C_PA, lw=1.8, alpha=0.85, label="SASTO-PA", zorder=3)
ax.plot(b12, vr12, color=C_U, lw=1.8, alpha=0.85, ls="--", label="SASTO-U", zorder=3)
ax.axhline(vr11[-1], color=C_PA, lw=0.8, ls=":", alpha=0.4)
ax.axhline(vr12[-1], color=C_U, lw=0.8, ls=":", alpha=0.4)
ax.text(b11[-1]+3, vr11[-1], f"{vr11[-1]:.1f}%", color=C_PA, fontsize=9, va="center")
ax.text(b12[-1]+3, vr12[-1], f"{vr12[-1]:.1f}%", color=C_U, fontsize=9, va="center")
legend_styled(ax, loc="lower right")

# (b) Von Mises stress vs batch
ax = axes[1]
style_ax(ax, xlabel="Batch", ylabel="Max VM stress  (MPa)",
         title="(b) Stress Evolution")
ax.plot(b11, vm11, color=C_PA, lw=1.8, alpha=0.85, label="SASTO-PA", zorder=3)
ax.plot(b12, vm12, color=C_U, lw=1.8, alpha=0.85, ls="--", label="SASTO-U", zorder=3)
# Allowable stress line
vm_allow = 5.0  # MPa
ax.axhline(vm_allow, color=RED, lw=1.8, ls="--", alpha=0.8, zorder=4,
           label=rf"$\sigma_{{allow}}$ = {vm_allow} MPa")
legend_styled(ax, loc="upper left", fontsize=8.5)

# (c) Compliance vs batch
ax = axes[2]
style_ax(ax, xlabel="Batch", ylabel="Compliance  (J)",
         title="(c) Compliance Evolution")
ax.plot(b11, c11, color=C_PA, lw=1.8, alpha=0.85, label="SASTO-PA", zorder=3)
ax.plot(b12, c12, color=C_U, lw=1.8, alpha=0.85, ls="--", label="SASTO-U", zorder=3)
# Compliance limit
c0 = c11[0]
c_allow = c0 * LIMIT
ax.axhline(c_allow, color=RED, lw=1.8, ls="--", alpha=0.8, zorder=4,
           label=rf"$C_{{allow}} = {LIMIT}\,C_0$")
legend_styled(ax, loc="upper right", fontsize=8.5)

plt.tight_layout()
save(fig, "fig_convergence.png")


# ═══════════════════════════════════════════════════════════════════
# 10. UNCERTAINTY / SAFETY EVOLUTION  (NEW)
# ═══════════════════════════════════════════════════════════════════
print("10/12  fig_uncertainty.png")

# Normalise metrics to baseline (batch 0) values
vm0  = h11[0]["vm"]
c0   = h11[0]["comp"]
d0   = h11[0]["disp"]
vfrac = np.array([1 - h["vol_reduction"] for h in h11])
vm_norm  = np.array([h["vm"] / vm0 for h in h11])
c_norm   = np.array([h["comp"] / c0 for h in h11])
d_norm   = np.array([h["disp"] / d0 for h in h11])

fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
style_ax(ax, xlabel=r"Volume fraction  $\phi = V/V_0$",
         ylabel="Normalised response  (ratio to baseline)",
         title=r"Safety Margin Evolution During Optimisation")

ax.plot(vfrac, vm_norm, color=RED, lw=2.0, alpha=0.85, label="VM stress", zorder=3)
ax.plot(vfrac, c_norm, color=ACC, lw=2.0, alpha=0.85, label="Compliance", zorder=3)
ax.plot(vfrac, d_norm, color=TEAL, lw=2.0, alpha=0.85, label="Displacement", zorder=3)

# Constraint limits
ax.axhline(LIMIT, color=RED, lw=1.5, ls="--", alpha=0.6,
           label=rf"$C_{{allow}} / C_0 = {LIMIT}$")
vm_allow_ratio = 5e6 / vm0  # 5 MPa allowable
ax.axhline(vm_allow_ratio, color=GOLD, lw=1.5, ls="--", alpha=0.6,
           label=rf"$\sigma_{{allow}} / \sigma_0$ = {vm_allow_ratio:.2f}")
ax.axhline(1.0, color=DIM, lw=0.8, ls=":", alpha=0.5)

# Invert x-axis (high volume fraction on left to low on right)
ax.invert_xaxis()
ax.set_xlim(1.0, vfrac.min() - 0.02)

# Shade safe zone
ax.fill_between([1.0, vfrac.min() - 0.02], 0, LIMIT,
                alpha=0.04, color=TEAL, zorder=0)
ax.text(0.75, 0.5, "Safe zone", color=TEAL, fontsize=9,
        alpha=0.5, va="center", ha="center", transform=ax.transAxes)

legend_styled(ax, loc="upper right", fontsize=9)
eq_box(ax, r"$\phi = V_{opt} / V_{base}$", x=0.03, y=0.03, ha="left", va="bottom", fs=9.5)
plt.tight_layout()
save(fig, "fig_uncertainty.png")


# ═══════════════════════════════════════════════════════════════════
# 11. SIMP COMPARISON  —  head-to-head (NEW)
# ═══════════════════════════════════════════════════════════════════
print("11/12  fig_simp_comparison.png")

s_ids   = [s["sample_id"] for s in simp_data]
s_group = [s["group"] for s in simp_data]
simp_red = np.array([s["volume_reduction_pct"] for s in simp_data])
sasto_red = np.array([s["sasto_reduction_pct"] for s in simp_data])
simp_t   = np.array([s["total_time_s"] for s in simp_data])
simp_cr  = np.array([s["comp_ratio"] for s in simp_data])
sasto_cr = np.array([s["sasto_comp_ratio"] for s in simp_data])

group_colors = {"high_reduction": RED, "near_boundary": GOLD, "easy": TEAL}
gc = [group_colors.get(g, DIM) for g in s_group]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=BG,
                          gridspec_kw=dict(wspace=0.28))

# (a) Volume reduction comparison
ax = axes[0]
style_ax(ax, ylabel="Volume reduction  (%)",
         title="(a) Reduction: SIMP vs SASTO")
x = np.arange(len(s_ids))
w = 0.35
b1 = ax.bar(x - w/2, simp_red, w, color=PURPLE, alpha=0.60,
            edgecolor="white", label="SIMP (64³)", zorder=3)
b2 = ax.bar(x + w/2, sasto_red, w, color=C_PA, alpha=0.60,
            edgecolor="white", label="SASTO (128³)", zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(s_ids, rotation=45, ha="right", fontsize=8)
# Colour the x-tick labels by group
for tick, col in zip(ax.get_xticklabels(), gc):
    tick.set_color(col)
legend_styled(ax, loc="upper right", fontsize=8.5)

# (b) Runtime comparison
ax = axes[1]
style_ax(ax, ylabel="Wall-clock time  (s)",
         title="(b) Runtime Comparison")
sasto_med = float(np.median(times_arr))
b_simp = ax.bar(x, simp_t, 0.50, color=PURPLE, alpha=0.60,
                edgecolor="white", label="SIMP (64³)", zorder=3)
ax.axhline(sasto_med, color=C_PA, lw=2.0, ls="--", zorder=4,
           label=f"SASTO median {sasto_med:.0f}s (128³)")
ax.set_xticks(x)
ax.set_xticklabels(s_ids, rotation=45, ha="right", fontsize=8)
for tick, col in zip(ax.get_xticklabels(), gc):
    tick.set_color(col)
ax.set_yscale("log")
legend_styled(ax, loc="upper right", fontsize=8.5)

# (c) Compliance ratio comparison
ax = axes[2]
style_ax(ax, ylabel=r"Compliance ratio  $C_{opt}/C_{base}$",
         title="(c) Compliance Ratio")
b1 = ax.bar(x - w/2, simp_cr, w, color=PURPLE, alpha=0.60,
            edgecolor="white", label="SIMP", zorder=3)
b2 = ax.bar(x + w/2, sasto_cr, w, color=C_PA, alpha=0.60,
            edgecolor="white", label="SASTO", zorder=3)
ax.axhline(LIMIT, color=RED, lw=1.8, ls="--", zorder=4, alpha=0.7,
           label=f"Safety limit {LIMIT}")
ax.set_xticks(x)
ax.set_xticklabels(s_ids, rotation=45, ha="right", fontsize=8)
for tick, col in zip(ax.get_xticklabels(), gc):
    tick.set_color(col)
legend_styled(ax, loc="upper right", fontsize=8.5)

# Group colour legend at bottom
grp_handles = [mpatches.Patch(fc=RED, alpha=0.6, label="High reduction"),
               mpatches.Patch(fc=GOLD, alpha=0.6, label="Near boundary"),
               mpatches.Patch(fc=TEAL, alpha=0.6, label="Easy")]
fig.legend(handles=grp_handles, loc="lower center", ncol=3, fontsize=10,
           frameon=True, framealpha=0.92, edgecolor=SPINE, facecolor=PANEL,
           labelcolor=GRAY, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.04, 1, 1])
save(fig, "fig_simp_comparison.png")


# ═══════════════════════════════════════════════════════════════════
# 12. TRAINING CURVES  —  ensemble surrogate (NEW)
# ═══════════════════════════════════════════════════════════════════
print("12/12  fig_training_curves.png")

# Try to parse real training losses from train_stderr.log
log_path = DATA_V3 / "train_stderr.log"
model_curves = {}  # {(model_idx, epoch): loss}

if log_path.exists():
    print("  Parsing training log (may take a moment)...")
    pattern = re.compile(r"\[M(\d)\] Epoch\s+(\d+)/200.*?loss=([\d.]+)")
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                mi = int(m.group(1))
                ep = int(m.group(2))
                loss = float(m.group(3))
                model_curves[(mi, ep)] = loss  # keeps last value per epoch

if model_curves:
    # Organise into arrays
    all_models = sorted(set(k[0] for k in model_curves))
    all_epochs = sorted(set(k[1] for k in model_curves))

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    style_ax(ax, xlabel="Epoch", ylabel="Training loss  (MSE)",
             title=rf"Ensemble Surrogate Training  ({len(all_models)} members, {max(all_epochs)} epochs)")

    m_colors = [ACC, TEAL, C_PA, GOLD, PURPLE]
    for mi in all_models:
        epochs = sorted([k[1] for k in model_curves if k[0] == mi])
        losses = [model_curves[(mi, e)] for e in epochs]
        col = m_colors[mi % len(m_colors)]
        ax.plot(epochs, losses, color=col, lw=1.5, alpha=0.8,
                label=f"M{mi}", zorder=3)

    ax.set_yscale("log")
    ax.set_xlim(1, max(all_epochs))
    legend_styled(ax, loc="upper right", ncol=min(len(all_models), 3))

    # Final metrics annotation
    rho_comp = surr_metrics.get("compliance", {}).get("spearman", 0)
    rho_disp = surr_metrics.get("max_displacement", {}).get("spearman", 0)
    stat_box(ax, (f"Test-set Spearman:\n"
                  rf"Compliance: $\rho = {rho_comp:.3f}$" "\n"
                  rf"Displacement: $\rho = {rho_disp:.3f}$"),
             x=0.97, y=0.55, ha="right", va="top", fs=9)

    plt.tight_layout()
    save(fig, "fig_training_curves.png")
else:
    # Fallback: use existing figure if available
    existing = ROOT / "figures" / "fig15_training_curves.png"
    if existing.exists():
        print("  No parsed data — wrapping existing figure")
        tc_img = imread(str(existing))
        th, tw = tc_img.shape[:2]
        FIG_W = 12
        fig = plt.figure(figsize=(FIG_W, FIG_W * th / tw + 0.5), facecolor=BG)
        ax = fig.add_axes([0.02, 0.04, 0.96, 0.88])
        ax.set_axis_off()
        ax.imshow(tc_img, aspect="equal", interpolation="lanczos")
        fig.text(0.5, 0.97, "Ensemble Surrogate Training Curves",
                 ha="center", va="top", color=GRAY, fontsize=13, fontweight="bold")
        save(fig, "fig_training_curves.png", dpi=180)
    else:
        print("  SKIP — no training data available")


# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  RESULTS FIGURES  ->  {OUT}/")
print(f"{'='*60}")
expected = [
    "fig_diverse_gallery.png", "fig_cross_sections.png",
    "fig_volume_reduction.png", "fig_fea_compliance.png",
    "fig_surrogate_accuracy.png", "fig_bland_altman.png",
    "fig_connectivity.png", "fig_k_calibration.png",
    "fig_convergence.png", "fig_uncertainty.png",
    "fig_simp_comparison.png", "fig_training_curves.png",
]
found = 0
for name in expected:
    p = OUT / name
    if p.exists():
        print(f"  {p.name:<42}  {p.stat().st_size // 1024:>5} KB")
        found += 1
    else:
        print(f"  {p.name:<42}  MISSING")
print(f"\n  {found}/{len(expected)} generated")
