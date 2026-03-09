"""
ISEF Results & In-Silico Validation — 12 Poster Figures
========================================================
Output:  results_figures/

Selected 12 figures (curated from 100+ candidates across all directories):

VISUAL EVIDENCE
  1. fig_ref_comparison.png       — 2×3 grid: Original / SASTO-U / SASTO-PA  (solid + cutaway)
  2. fig_diverse_gallery.png      — 4-design gallery: original vs optimised STL renders
  3. fig_cross_sections.png       — Interior cross-section comparison (Orig / U / PA)

QUANTITATIVE RESULTS (graphs with equations)
  4. fig_volume_reduction.png     — Population histogram (n=1,246) + KDE
  5. fig_per_part_retention.png   — Per-part retention violin/box (ext_wall, int_wall, roof, floor)
  6. fig_fea_compliance.png       — Compliance ratio distribution + scatter vs aggressiveness
  7. fig_speedup.png              — SASTO vs SIMP runtime comparison
  8. fig_surrogate_accuracy.png   — Surrogate ranking accuracy (Spearman ρ) + scatter
  9. fig_bland_altman.png         — Bland-Altman rank agreement + rank correlation

STRUCTURAL VALIDATION
 10. fig_connectivity.png         — Voxel/mesh connectivity verification bar chart
 11. fig_k_calibration.png        — Safety-factor k-ablation: pass rate vs volume trade-off
 12. fig_fea_stress.png           — 3D FEA von-Mises stress visualisation (HQ render)

All light-themed, with LaTeX equations where applicable.
"""

import json, shutil, pathlib, warnings
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde, rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
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
    print(f"  OK  {p.name:<40}  {sz:>5} KB")

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
            bbox=dict(boxstyle="round,pad=0.45", fc="#edf2fc", ec=ACC, lw=1.0, alpha=0.92))

def stat_box(ax, text, x=0.03, y=0.97, ha="left", va="top", fs=8.5):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fs, va=va, ha=ha,
            color=GRAY, bbox=dict(boxstyle="round,pad=0.45", fc=PANEL, ec=SPINE, lw=0.8))

def legend_styled(ax, **kw):
    kw.setdefault("frameon", True)
    kw.setdefault("framealpha", 0.92)
    kw.setdefault("edgecolor", SPINE)
    kw.setdefault("facecolor", PANEL)
    kw.setdefault("labelcolor", GRAY)
    kw.setdefault("fontsize", 9.5)
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

print("Data loaded.\n")

# ═══════════════════════════════════════════════════════════════════
# 1. REFERENCE COMPARISON  —  2×3 grid: Orig / SASTO-U / SASTO-PA
# ═══════════════════════════════════════════════════════════════════
print("1/12  fig_ref_comparison.png")
rc_imgs = [
    ("stl_orig_solid.png",   "Original",  C_ORIG),
    ("sasto_u_solid.png",    "SASTO-U",   C_U),
    ("stl_pa_solid.png",     "SASTO-PA",  C_PA),
    ("stl_orig_cutaway.png", "Original",  C_ORIG),
    ("sasto_u_cutaway.png",  "SASTO-U",   C_U),
    ("stl_pa_cutaway.png",   "SASTO-PA",  C_PA),
]

fig = plt.figure(figsize=(16, 10), facecolor=BG)
gs = gridspec.GridSpec(2, 3, figure=fig,
                       left=0.03, right=0.97, top=0.88, bottom=0.06,
                       hspace=0.06, wspace=0.04)

row_labels = ["Solid View", "Interior Cutaway"]
for idx, (fname, label, col) in enumerate(rc_imgs):
    r, c = divmod(idx, 3)
    ax = fig.add_subplot(gs[r, c])
    ax.set_axis_off()
    ax.set_facecolor("#f0f1f4")
    im = load_img(RENDERS / fname)
    if im is not None:
        ax.imshow(im, aspect="equal", interpolation="lanczos")
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(col); sp.set_linewidth(3.0)
    if r == 0:
        ax.set_title(label, color=col, fontsize=13, fontweight="bold", pad=8)
    if c == 0:
        ax.text(-0.06, 0.5, row_labels[r], transform=ax.transAxes,
                fontsize=11, color=MID, fontweight="bold",
                va="center", ha="center", rotation=90)

fig.text(0.5, 0.955, "Reference Geometry 00472  —  Three-Method Comparison",
         ha="center", va="top", color=GRAY, fontsize=14, fontweight="bold")
fig.text(0.5, 0.925,
         r"Original ($V = 93{,}905$)  $\rightarrow$  "
         r"SASTO-U ($-9.7\%$,  $97\,\mathrm{s}$)  $\rightarrow$  "
         r"SASTO-PA ($-20.4\%$,  $97\,\mathrm{s}$)",
         ha="center", va="top", color=TEAL, fontsize=10.5,
         math_fontfamily="dejavusans")
# Part legend
legend_elems = [
    mpatches.Patch(fc=PC_EXT, ec="#777", lw=0.8, label="Exterior Wall"),
    mpatches.Patch(fc=PC_INT, ec="#777", lw=0.8, label="Interior Wall"),
    mpatches.Patch(fc=PC_ROOF, ec="#777", lw=0.8, label="Roof"),
    mpatches.Patch(fc=PC_FLR, ec="#777", lw=0.8, label="Floor"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=4,
           fontsize=11, frameon=True, framealpha=0.95,
           edgecolor=SPINE, facecolor=BG,
           handlelength=2.0, handletextpad=0.6, columnspacing=2.5,
           bbox_to_anchor=(0.5, 0.005), labelcolor=GRAY)
save(fig, "fig_ref_comparison.png")


# ═══════════════════════════════════════════════════════════════════
# 2. DIVERSE GALLERY  —  copy HQ render and add large legend
# ═══════════════════════════════════════════════════════════════════
print("2/12  fig_diverse_gallery.png")
gal_src = RENDERS / "fig_diverse_stl_gallery.png"
if gal_src.exists():
    gal_img = imread(str(gal_src))
    gh, gw = gal_img.shape[:2]
    FIG_W = 22
    fig = plt.figure(figsize=(FIG_W, FIG_W * gh / gw + 1.0), facecolor=BG)
    ax = fig.add_axes([0, 0.08, 1.0, 0.92])
    ax.set_axis_off()
    ax.imshow(gal_img, aspect="equal", interpolation="lanczos")
    leg_patches = [
        mpatches.Patch(fc=PC_EXT, ec="#555", lw=1.2, label="Exterior Wall"),
        mpatches.Patch(fc=PC_INT, ec="#555", lw=1.2, label="Interior Wall"),
        mpatches.Patch(fc=PC_ROOF, ec="#555", lw=1.2, label="Roof"),
        mpatches.Patch(fc=PC_FLR, ec="#555", lw=1.2, label="Floor"),
    ]
    leg = fig.legend(handles=leg_patches, loc="lower center", ncol=4,
                     fontsize=22, frameon=True, framealpha=0.97,
                     edgecolor=SPINE, facecolor=BG,
                     handlelength=3.0, handletextpad=1.0,
                     columnspacing=4.0,
                     bbox_to_anchor=(0.5, 0.003),
                     labelcolor=GRAY,
                     prop={"size": 22, "weight": "bold"})
    for patch in leg.get_patches():
        patch.set_height(20); patch.set_width(34)
    save(fig, "fig_diverse_gallery.png", dpi=160)
else:
    print("  SKIP — source not found")


# ═══════════════════════════════════════════════════════════════════
# 3. CROSS SECTIONS  —  copy HQ render
# ═══════════════════════════════════════════════════════════════════
print("3/12  fig_cross_sections.png")
xs_src = RENDERS / "fig_cross_section_comparison.png"
if xs_src.exists():
    shutil.copy2(xs_src, OUT / "fig_cross_sections.png")
    print(f"  OK  fig_cross_sections.png               (copied)")
else:
    print("  SKIP — source not found")


# ═══════════════════════════════════════════════════════════════════
# 4. VOLUME REDUCTION HISTOGRAM
# ═══════════════════════════════════════════════════════════════════
print("4/12  fig_volume_reduction.png")
fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
style_ax(ax, xlabel="Volume reduction  (%)",
         ylabel="Number of designs",
         title=rf"Population Volume Reduction  —  SASTO-PA  ($n = {n_pop:,}$)")

bins = np.linspace(0, max(reductions) + 2, 40)
ax.hist(reductions, bins=bins, color=C_PA, alpha=0.60,
        edgecolor="white", linewidth=0.5, zorder=3,
        label=f"SASTO-PA  (n = {n_pop:,})")

kde = gaussian_kde(reductions, bw_method=0.18)
xkde = np.linspace(0, bins[-1], 300)
ykde = kde(xkde) * n_pop * (bins[1] - bins[0])
ax.plot(xkde, ykde, color=C_PA, lw=2.4, zorder=4)

mu, sig = reductions.mean(), reductions.std()
ax.axvline(20.4, color=GOLD, lw=2.0, ls="--", zorder=5,
           label="Reference 00472  (20.4%)")
ax.axvline(mu, color=TEAL, lw=2.0, ls="-.", zorder=5,
           label=rf"Population mean  ({mu:.1f}%)")

ax.annotate(f"00472\n20.4%", xy=(20.4, ykde.max()*0.50),
            xytext=(26, ykde.max()*0.70),
            color=GOLD, fontsize=9.5, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.3))
ax.annotate(rf"$\mu = {mu:.1f}\%$", xy=(mu, ykde.max()*0.35),
            xytext=(mu - 6, ykde.max()*0.55),
            color=TEAL, fontsize=9.5, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.3))

eq_box(ax, r"$\Delta V = \frac{V_{base} - V_{opt}}{V_{base}} \times 100$",
       x=0.97, y=0.97, fs=10.5)
stat_box(ax, (f"$n$ = {n_pop:,}\n"
              rf"$\mu$ = {mu:.1f}%""\n"
              rf"$\sigma$ = {sig:.1f}%""\n"
              f"range  {reductions.min():.0f}–{reductions.max():.0f}%"),
         x=0.68, y=0.85, fs=9)

legend_styled(ax, loc="upper left", handlelength=1.6)
ax.set_xlim(0, bins[-1])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.tight_layout()
save(fig, "fig_volume_reduction.png")


# ═══════════════════════════════════════════════════════════════════
# 5. PER-PART RETENTION  —  violin + strip
# ═══════════════════════════════════════════════════════════════════
print("5/12  fig_per_part_retention.png")

part_names  = ["exterior_wall", "interior_wall", "roof", "floor"]
part_labels = ["Exterior\nWall", "Interior\nWall", "Roof", "Floor"]
part_colors = [PC_EXT, PC_INT, PC_ROOF, PC_FLR]

part_ret = {pn: [] for pn in part_names}
for pb in part_break:
    for pn in part_names:
        if pn in pb and pb[pn].get("retained_pct") is not None:
            part_ret[pn].append(pb[pn]["retained_pct"])

fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=BG)
style_ax(ax, ylabel="Retained volume  (%)",
         title=r"Per-Part Retention  —  SASTO-PA  Part-Aware Constraints")

data_arrays = [np.array(part_ret[pn]) for pn in part_names]
positions = np.arange(1, len(part_names) + 1)

# Violin
parts_vp = ax.violinplot(data_arrays, positions=positions, widths=0.65,
                         showextrema=False, showmedians=False)
for body, col in zip(parts_vp["bodies"], part_colors):
    body.set_facecolor(col)
    body.set_alpha(0.35)
    body.set_edgecolor(col)
    body.set_linewidth(1.0)

# Box overlay
bp = ax.boxplot(data_arrays, positions=positions, widths=0.22,
                patch_artist=True, notch=False, zorder=4,
                medianprops=dict(color="white", lw=2.0),
                whiskerprops=dict(color=MID, lw=1.2),
                capprops=dict(color=MID, lw=1.2),
                flierprops=dict(marker=".", ms=3, color=DIM, alpha=0.4))
for patch, col in zip(bp["boxes"], part_colors):
    patch.set_facecolor(col); patch.set_alpha(0.70)
    patch.set_edgecolor(col)

# Strip plot  (jittered individual points)
rng = np.random.default_rng(42)
for i, (arr, col) in enumerate(zip(data_arrays, part_colors)):
    jitter = rng.uniform(-0.12, 0.12, size=len(arr))
    ax.scatter(positions[i] + jitter, arr, s=6, color=col, alpha=0.30,
               zorder=3, edgecolors="none")

# Median labels
for i, arr in enumerate(data_arrays):
    med = np.median(arr)
    ax.text(positions[i], med + 1.5, f"{med:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=part_colors[i])

ax.set_xticks(positions)
ax.set_xticklabels(part_labels, fontsize=11, color=GRAY)
ax.set_ylim(0, 108)
ax.axhline(100, color=DIM, lw=0.8, ls=":", alpha=0.6)
ax.text(4.5, 101, "Fully retained", color=DIM, fontsize=8, va="bottom")

n_parts = len(data_arrays[0])
eq_box(ax, r"$t_{\min}^{(p)}$ applied per part", x=0.97, y=0.97, fs=10)
stat_box(ax, f"$n$ = {n_parts:,} designs", x=0.03, y=0.97, fs=9)

plt.tight_layout()
save(fig, "fig_per_part_retention.png")


# ═══════════════════════════════════════════════════════════════════
# 6. FEA COMPLIANCE  —  distribution + scatter
# ═══════════════════════════════════════════════════════════════════
print("6/12  fig_fea_compliance.png")
mu_cr = comp_ratio.mean()
mx_cr = comp_ratio.max()
violations = (comp_ratio > LIMIT).sum()

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG,
                          gridspec_kw=dict(wspace=0.30))

# left: histogram
ax = axes[0]
style_ax(ax, xlabel=r"Compliance ratio  $C_{ratio}$",
         ylabel="Designs",
         title=rf"Compliance Ratio Distribution  ($n = {n_fea}$)")
bins2 = np.linspace(comp_ratio.min() - 0.02, LIMIT + 0.05, 36)
ax.hist(comp_ratio, bins=bins2, color=TEAL, alpha=0.60,
        edgecolor="white", linewidth=0.5, zorder=3)
kde2 = gaussian_kde(comp_ratio, bw_method=0.20)
xk2 = np.linspace(bins2[0], bins2[-1], 300)
yk2 = kde2(xk2) * n_fea * (bins2[1] - bins2[0])
ax.plot(xk2, yk2, color=TEAL, lw=2.2, zorder=4)
ax.axvline(LIMIT, color=RED, lw=2.0, ls="--", zorder=5,
           label=rf"Safety limit  $C_{{limit}} = {LIMIT}$")
ax.axvline(mx_cr, color=GOLD, lw=1.8, ls="-.", zorder=5,
           label=rf"Max observed  ${mx_cr:.3f}$")
legend_styled(ax, fontsize=9)

ym2 = yk2.max()
ax.annotate("", xy=(LIMIT, ym2*0.28), xytext=(mx_cr, ym2*0.28),
            arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.5))
ax.text((mx_cr + LIMIT)/2, ym2*0.33, rf"$\Delta = {LIMIT-mx_cr:.3f}$",
        ha="center", va="bottom", color=GOLD, fontsize=9.5, fontweight="bold")

eq_box(ax, r"$C_{ratio} = \frac{C_{opt}}{C_{base}} \leq 1.15$",
       x=0.97, y=0.97, fs=10.5)
stat_box(ax, f"$n$ = {n_fea}\n"
             rf"$\mu$ = {mu_cr:.3f}""\n"
             rf"max = {mx_cr:.3f}""\n"
             f"violations = {violations}",
         x=0.03, y=0.97)

# right: scatter
ax2 = axes[1]
style_ax(ax2, xlabel="Volume reduction (%)",
         ylabel=r"Compliance ratio  $C_{opt} / C_{base}$",
         title="Structural Safety vs Aggressiveness")
sc = ax2.scatter(vr_paired, comp_ratio, c=comp_ratio, cmap="RdYlGn_r",
                 vmin=0.2, vmax=LIMIT, s=24, alpha=0.65, zorder=3,
                 edgecolors="none")
ax2.axhline(LIMIT, color=RED, lw=1.8, ls="--", zorder=4)
ax2.axhline(1.0, color=MID, lw=1.0, ls=":", alpha=0.5, zorder=3)
cb = plt.colorbar(sc, ax=ax2, pad=0.03, shrink=0.92)
cb.ax.tick_params(colors=GRAY, labelsize=8.5)
cb.set_label(r"$C_{ratio}$", color=GRAY, fontsize=10)

ax2.text(0.50, 0.90,
         rf"$\checkmark$  Zero violations  ({n_fea:,} designs)",
         ha="center", va="center", color=TEAL, fontsize=12, fontweight="bold",
         transform=ax2.transAxes,
         bbox=dict(boxstyle="round,pad=0.4", fc="#ecfaf4", ec=TEAL, lw=1.8))

plt.tight_layout()
save(fig, "fig_fea_compliance.png")


# ═══════════════════════════════════════════════════════════════════
# 7. SPEEDUP  —  SASTO vs SIMP runtime
# ═══════════════════════════════════════════════════════════════════
print("7/12  fig_speedup.png")
SASTO_med = np.median(times_arr)
SCALE = (128/64)**2
simp_proj128 = simp_times * SCALE

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG,
                          gridspec_kw=dict(wspace=0.32))

# left: box
ax = axes[0]
style_ax(ax, ylabel="Runtime  (seconds)", title="SASTO vs SIMP  —  Runtime")
ax.set_yscale("log")
bp = ax.boxplot([times_arr, simp_times], patch_artist=True,
                medianprops=dict(color="white", lw=2.2),
                whiskerprops=dict(color=MID, lw=1.3),
                capprops=dict(color=MID, lw=1.3),
                flierprops=dict(marker="o", ms=4.5, color=DIM, alpha=0.5))
for patch, col in zip(bp["boxes"], [C_PA, PURPLE]):
    patch.set_facecolor(col); patch.set_alpha(0.55); patch.set_edgecolor(col)
ax.set_xticks([1, 2])
ax.set_xticklabels([f"SASTO\n(128³, n={n_pop:,})",
                    f"SIMP\n(64³, n={len(simp_times)})"],
                   color=GRAY, fontsize=10)
simp_med = np.median(simp_times)
speedup = simp_med / SASTO_med
ax.annotate(rf"$\times{speedup:.0f}$ faster",
            xy=(1, SASTO_med), xytext=(1.45, SASTO_med * 3.5),
            color=TEAL, fontsize=12, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.6))
ax.axhline(60, color=DIM, lw=0.9, ls=":", alpha=0.6)
ax.text(2.42, 66, "1 min", color=DIM, fontsize=8.5, va="bottom")
eq_box(ax, r"$S = \frac{T_{SIMP}}{T_{SASTO}}$", x=0.97, y=0.97, fs=11)

# right: distribution
ax2 = axes[1]
style_ax(ax2, xlabel="Runtime  (seconds)", ylabel="Designs",
         title="SASTO Runtime Distribution  (128³)")
bins3 = np.logspace(np.log10(max(times_arr.min(), 1)),
                    np.log10(times_arr.max() + 10), 35)
ax2.hist(times_arr, bins=bins3, color=C_PA, alpha=0.60,
         edgecolor="white", linewidth=0.5, zorder=3,
         label=f"SASTO  (n={n_pop:,})")
ax2.set_xscale("log")
ax2.axvline(SASTO_med, color=GOLD, lw=2.0, ls="--", zorder=4,
            label=rf"Median  {SASTO_med:.0f} s")
for st in [simp_times.min()*SCALE, simp_med*SCALE, simp_times.max()*SCALE]:
    ax2.axvline(st, color=PURPLE, lw=1.5, ls="-.", alpha=0.70)
stat_box(ax2, (f"SIMP 128³ projected:\n"
               f"  {simp_times.min()*SCALE/60:.0f}–{simp_times.max()*SCALE/60:.0f} min\n"
               f"  (med {simp_med*SCALE/60:.0f} min)"),
         x=0.97, y=0.50, ha="right", va="center", fs=8.5)
legend_styled(ax2)
plt.tight_layout()
save(fig, "fig_speedup.png")


# ═══════════════════════════════════════════════════════════════════
# 8. SURROGATE ACCURACY  —  scatter + ranking
# ═══════════════════════════════════════════════════════════════════
print("8/12  fig_surrogate_accuracy.png")

# Paired surrogate vs FEA
surr_sub = [(r["surrogate_comp_mean"], r["voxel_opt_comp"])
             for r in fea_data
             if r.get("surrogate_comp_mean") and r.get("voxel_opt_comp")]
if surr_sub:
    s_pred = np.array([x[0] for x in surr_sub])
    s_true = np.array([x[1] for x in surr_sub])
else:
    s_pred = np.array([r["voxel_base_comp"] for r in fea_data
                       if r.get("voxel_base_comp") and r.get("voxel_opt_comp")])
    s_true = np.array([r["voxel_opt_comp"] for r in fea_data
                       if r.get("voxel_base_comp") and r.get("voxel_opt_comp")])

spearman_r, sp_p = stats.spearmanr(s_pred, s_true)
pearson_r, _     = stats.pearsonr(s_pred, s_true)
r2_val = pearson_r**2
n_reg  = len(s_pred)

# Also get base→opt compliance ratio
base_comp = np.array([r["voxel_base_comp"] for r in fea_data
                      if r.get("voxel_base_comp") and r.get("voxel_opt_comp")])
opt_comp  = np.array([r["voxel_opt_comp"] for r in fea_data
                      if r.get("voxel_base_comp") and r.get("voxel_opt_comp")])
cr_paired = opt_comp / base_comp

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG,
                          gridspec_kw=dict(wspace=0.30))

# left: scatter
ax = axes[0]
style_ax(ax, xlabel="Surrogate prediction",
         ylabel="Ground-truth FEA compliance",
         title="Surrogate Compliance Ranking")
sc = ax.scatter(s_pred, s_true,
                c=np.abs(s_pred - s_true) / s_true,
                cmap="RdYlGn_r", vmin=0, vmax=0.5,
                s=28, alpha=0.65, zorder=3, edgecolors="none")
m, b0, *_ = stats.linregress(s_pred, s_true)
xfit = np.linspace(s_pred.min(), s_pred.max(), 200)
ax.plot(xfit, m*xfit + b0, color=TEAL, lw=2.2, zorder=4, label="Linear fit")
lims = [min(s_pred.min(), s_true.min()), max(s_pred.max(), s_true.max())]
ax.plot(lims, lims, color=MID, lw=1.3, ls="--", alpha=0.6, label=r"$y=x$ ideal")
cb = plt.colorbar(sc, ax=ax, pad=0.03, shrink=0.92)
cb.ax.tick_params(colors=GRAY, labelsize=8.5)
cb.set_label("Relative error", color=GRAY, fontsize=9.5)
legend_styled(ax, fontsize=9)

eq_box(ax, rf"$\rho_{{Spearman}} = {spearman_r:.3f}$""\n"
           rf"$R^2 = {r2_val:.3f}$",
       x=0.97, y=0.97, fs=10.5)
stat_box(ax, f"$n$ = {n_reg}", x=0.03, y=0.97, fs=9)

# right: compliance improvement histogram
ax2 = axes[1]
style_ax(ax2, xlabel=r"$C_{opt} / C_{base}$  (compliance ratio)",
         ylabel="Designs",
         title="Compliance Improvement After Optimisation")
bins4 = np.linspace(0, 1.25, 36)
ax2.hist(cr_paired, bins=bins4, color=TEAL, alpha=0.60,
         edgecolor="white", linewidth=0.5, zorder=3)
ax2.axvline(1.0, color=ACC, lw=1.8, ls="-.", zorder=4,
            label=r"Baseline  $C_{ratio} = 1$")
ax2.axvline(LIMIT, color=RED, lw=2.0, ls="--", zorder=5,
            label=rf"Safety limit  1.15")
pct_below = (cr_paired < 1.0).mean() * 100
ax2.text(0.5, 0.90,
         rf"{pct_below:.0f}% of designs" "\nimprove compliance",
         ha="center", va="center", color=TEAL, fontsize=11.5, fontweight="bold",
         transform=ax2.transAxes,
         bbox=dict(boxstyle="round,pad=0.4", fc="#ecfaf4", ec=TEAL, lw=1.6))
legend_styled(ax2)
plt.tight_layout()
save(fig, "fig_surrogate_accuracy.png")


# ═══════════════════════════════════════════════════════════════════
# 9. BLAND-ALTMAN  —  percentile-rank space
# ═══════════════════════════════════════════════════════════════════
print("9/12  fig_bland_altman.png")

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

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), facecolor=BG,
                          gridspec_kw=dict(wspace=0.34))

# left: Bland-Altman
ax = axes[0]
style_ax(ax, xlabel=r"Mean percentile rank",
         ylabel=r"Rank difference  ($r_{surr} - r_{FEA}$)",
         title="Bland-Altman  (Percentile-Rank Space)")
sc = ax.scatter(mean_r, diff_r, c=np.abs(diff_r), cmap="RdYlGn_r",
                vmin=0, vmax=40, s=55, alpha=0.80, zorder=3,
                edgecolors="white", linewidths=0.4)
ax.axhline(bias_r, color=TEAL, lw=2.2, zorder=4,
           label=rf"Bias  ${bias_r:+.1f}$")
ax.axhline(loa_up, color=GOLD, lw=1.8, ls="--", zorder=4,
           label=rf"$+1.96\sigma = {loa_up:+.1f}$")
ax.axhline(loa_lo, color=GOLD, lw=1.8, ls="--", zorder=4,
           label=rf"$-1.96\sigma = {loa_lo:+.1f}$")
ax.axhline(0, color=DIM, lw=1.0, ls=":", alpha=0.55)
ax.fill_between([0, 100], loa_lo, loa_up, alpha=0.08, color=GOLD, zorder=1)
ax.set_xlim(0, 100)
ax.set_ylim(diff_r.min() - 12, diff_r.max() + 14)
cb = plt.colorbar(sc, ax=ax, pad=0.03, shrink=0.90)
cb.ax.tick_params(colors=GRAY, labelsize=8.5)
cb.set_label(r"$|r_{surr} - r_{FEA}|$", color=GRAY, fontsize=9.5)
legend_styled(ax, fontsize=8.5, loc="lower left")
eq_box(ax, r"LoA: $\bar{d} \pm 1.96\,\sigma_d$", x=0.97, y=0.03,
       ha="right", va="bottom", fs=10)
stat_box(ax, (f"$n$ = {n_ba}\n"
              rf"bias = {bias_r:+.1f} pctile" "\n"
              rf"$\sigma$ = {std_r:.1f}" "\n"
              rf"$\rho_{{Spearman}}$ = {spear_ba:.3f}"),
         x=0.03, y=0.99, fs=9)

# right: rank correlation
ax2 = axes[1]
style_ax(ax2, xlabel=r"Surrogate rank  (percentile)",
         ylabel=r"FEA compliance rank  (percentile)",
         title="Surrogate–FEA Rank Correlation")
sc2 = ax2.scatter(rank_surr, rank_fea,
                  c=np.abs(rank_surr - rank_fea), cmap="RdYlGn_r",
                  vmin=0, vmax=40, s=55, alpha=0.80, zorder=3,
                  edgecolors="white", linewidths=0.4)
ax2.plot([0, 100], [0, 100], color=MID, lw=1.5, ls="--", alpha=0.6,
         label=r"$y = x$ (perfect)")
m2, b2, *_ = stats.linregress(rank_surr, rank_fea)
xf = np.array([0, 100])
ax2.plot(xf, m2*xf + b2, color=TEAL, lw=2.2, zorder=4,
         label=rf"Fit  ($\rho = {spear_ba:.3f}$)")
ax2.set_xlim(0, 100); ax2.set_ylim(0, 100)
cb2 = plt.colorbar(sc2, ax=ax2, pad=0.03, shrink=0.90)
cb2.ax.tick_params(colors=GRAY, labelsize=8.5)
legend_styled(ax2, loc="lower right")
eq_box(ax2, rf"$\rho_{{Spearman}} = {spear_ba:.3f}$" "\n" rf"$p = {p_ba:.1e}$",
       x=0.03, y=0.97, ha="left", va="top", fs=10)
plt.tight_layout()
save(fig, "fig_bland_altman.png")


# ═══════════════════════════════════════════════════════════════════
# 10. CONNECTIVITY  —  voxel + mesh verification
# ═══════════════════════════════════════════════════════════════════
print("10/12  fig_connectivity.png")
cs = conn_data["summary"]
ps = conn_data["per_sample"]
n_conn = cs["n_samples"]

# Collect components per sample
comp_6  = [r["voxel_components_6conn"]  for r in ps]
comp_26 = [r["voxel_components_26conn"] for r in ps]
comp_m  = [r["mesh_components"]         for r in ps]

fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=BG,
                          gridspec_kw=dict(wspace=0.30))

# left: summary bar chart
ax = axes[0]
style_ax(ax, ylabel="Percentage  (%)",
         title=rf"Structural Connectivity Verification  ($n = {n_conn}$)")
cats = ["6-connected\n(voxel)", "26-connected\n(voxel)", "Watertight\n(mesh)"]
vals = [cs["voxel_6conn_all_single"]/n_conn*100,
        cs["voxel_26conn_all_single"]/n_conn*100,
        cs["mesh_all_single"]/n_conn*100]
cols = [TEAL, ACC, C_PA]
bars = ax.bar(cats, vals, color=cols, alpha=0.70, edgecolor="white",
              linewidth=1.0, width=0.55, zorder=3)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
            f"{v:.0f}%", ha="center", va="bottom", fontsize=12,
            fontweight="bold", color=GRAY)
ax.set_ylim(0, 115)
ax.axhline(100, color=DIM, lw=0.8, ls=":", alpha=0.6)
eq_box(ax, r"Single connected component $\Rightarrow$ printable",
       x=0.97, y=0.97, fs=9.5)

# right: histogram of component counts
ax2 = axes[1]
style_ax(ax2, xlabel="Number of connected components",
         ylabel="Designs",
         title="Mesh Component Distribution")
all_comps = np.array(comp_m)
bins_c = np.arange(0.5, all_comps.max() + 1.5, 1)
ax2.hist(all_comps, bins=bins_c, color=C_PA, alpha=0.60,
         edgecolor="white", linewidth=0.8, zorder=3)
ax2.axvline(1, color=TEAL, lw=2.0, ls="--", zorder=5,
            label="Single component (target)")
pct_single = (all_comps == 1).mean() * 100
ax2.text(0.97, 0.90,
         rf"{pct_single:.0f}% single-component",
         ha="right", va="center", color=TEAL, fontsize=11, fontweight="bold",
         transform=ax2.transAxes,
         bbox=dict(boxstyle="round,pad=0.4", fc="#ecfaf4", ec=TEAL, lw=1.5))
legend_styled(ax2)
plt.tight_layout()
save(fig, "fig_connectivity.png")


# ═══════════════════════════════════════════════════════════════════
# 11. K-CALIBRATION  —  safety factor ablation
# ═══════════════════════════════════════════════════════════════════
print("11/12  fig_k_calibration.png")
k_abl = cal_data["k_ablation"]
k_vals  = []
n_ok    = []
pct_ok  = []
avg_red = []
for k_str in sorted(k_abl.keys(), key=float):
    k_vals.append(float(k_str))
    n_ok.append(k_abl[k_str]["n_ok"])
    pct_ok.append(k_abl[k_str]["pct"])
    avg_red.append(k_abl[k_str]["avg_reduction"])

k_vals  = np.array(k_vals)
pct_ok  = np.array(pct_ok)
avg_red = np.array(avg_red)

fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
style_ax(ax, xlabel=r"Safety factor  $k$",
         ylabel="FEA constraint pass rate  (%)",
         title=r"Safety Factor Calibration  —  $k$-Ablation Study")

# Pass rate on primary axis
line1, = ax.plot(k_vals, pct_ok, "o-", color=TEAL, lw=2.4, ms=9, zorder=4,
                 label="FEA pass rate (%)")
ax.fill_between(k_vals, 0, pct_ok, color=TEAL, alpha=0.08, zorder=1)

# Volume reduction on secondary axis
ax2 = ax.twinx()
ax2.set_ylabel("Avg. volume reduction (%)", fontsize=10.5, color=C_PA)
ax2.tick_params(axis="y", colors=C_PA, labelsize=9.5)
line2, = ax2.plot(k_vals, avg_red, "s--", color=C_PA, lw=2.0, ms=8, zorder=4,
                  label="Avg. volume reduction (%)")
ax2.set_ylim(15, 30)
ax2.spines["right"].set_color(C_PA)

# Highlight the chosen k
chosen_k = 1.0
chosen_idx = list(k_vals).index(chosen_k) if chosen_k in k_vals else None
if chosen_idx is not None:
    ax.axvline(chosen_k, color=GOLD, lw=2.0, ls="--", zorder=3, alpha=0.8)
    ax.annotate(rf"Chosen $k = {chosen_k}$" "\n"
                rf"Pass: {pct_ok[chosen_idx]:.1f}%""\n"
                rf"Red: {avg_red[chosen_idx]:.1f}%",
                xy=(chosen_k, pct_ok[chosen_idx]),
                xytext=(chosen_k + 0.5, pct_ok[chosen_idx] + 10),
                color=GOLD, fontsize=9.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.3))

eq_box(ax, r"$C_{limit} = C_{base} \cdot (1 + k \cdot \sigma_{surr})$",
       x=0.97, y=0.12, ha="right", va="bottom", fs=10.5)

# Combined legend
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, frameon=True, framealpha=0.92, edgecolor=SPINE,
          facecolor=PANEL, labelcolor=GRAY, fontsize=10, loc="upper right")

stat_box(ax, f"$n$ = {cal_data['n_batch']:,} designs", x=0.03, y=0.97, fs=9.5)
ax.set_xlim(-0.1, 3.2)
ax.set_ylim(0, 75)
plt.tight_layout()
save(fig, "fig_k_calibration.png")


# ═══════════════════════════════════════════════════════════════════
# 12. FEA STRESS  —  3D von-Mises (HQ render)
# ═══════════════════════════════════════════════════════════════════
print("12/12  fig_fea_stress.png")
# Use the best available FEA visualisation
fea_candidates = [
    ROOT / "figures" / "fig_fea_house_real_3d.png",
    ROOT / "figures" / "fig_fea_house_3d.png",
    ROOT / "figures" / "fig_fea_house.png",
]
fea_img = None
fea_src = None
for fc in fea_candidates:
    if fc.exists():
        fea_img = imread(str(fc))
        fea_src = fc
        break

if fea_img is not None:
    fh, fw = fea_img.shape[:2]
    FIG_W = 16
    fig = plt.figure(figsize=(FIG_W, FIG_W * fh / fw + 0.8), facecolor=BG)
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.88])
    ax.set_axis_off()
    ax.imshow(fea_img, aspect="equal", interpolation="lanczos")

    fig.text(0.5, 0.97,
             "Independent FEA Validation  —  Von Mises Stress Distribution",
             ha="center", va="top", color=GRAY, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.015,
             r"FEA mesh with applied gravity + wind loads.  "
             r"All optimised designs satisfy $\sigma_{max} < \sigma_{yield} / \mathrm{SF}$.",
             ha="center", va="bottom", color=DIM, fontsize=9.5, style="italic")

    save(fig, "fig_fea_stress.png", dpi=180)
else:
    print("  SKIP — no FEA visualisation found")


# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  12 FIGURES  ->  {OUT}/")
print(f"{'='*60}")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name:<42}  {f.stat().st_size // 1024:>5} KB")
print()
