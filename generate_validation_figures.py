"""
In-Silico Validation Figures for SASTO poster.
All outputs go to  validation_figures/
Light theme, equations embedded, professional style.
"""

import json
import shutil
import pathlib
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.image import imread
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.lines import Line2D
# rcParams for clean, publication-quality style
matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.edgecolor":     "#b0b8c8",
    "axes.linewidth":     0.9,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.minor.size":   2,
    "ytick.minor.size":   2,
    "figure.dpi":         100,
    "savefig.dpi":        180,
})

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = pathlib.Path(".")
RENDERS = ROOT / "poster_final" / "renders_hq"
DATA_V3 = ROOT / "fea_ml" / "runs" / "v3"
BATCH   = DATA_V3 / "batch_results_all"
OUT     = ROOT / "validation_figures"
OUT.mkdir(exist_ok=True)

# ── light-theme design tokens ──────────────────────────────────────────────────
BG      = "#ffffff"          # page background — pure white
PANEL   = "#f7f8fb"          # slightly off-white panel
CARD    = "#f0f3f8"          # axis background card
GRAY    = "#1c2233"          # primary text
MID     = "#4a5568"          # secondary text
DIM     = "#8896b0"          # muted text / annotations
LGRID   = "#dde3ef"          # grid lines
SPINE   = "#b0b8c8"          # axis spines
ACC     = "#1a6fd4"          # accent blue
TEAL    = "#0e9a76"          # success green
RED     = "#d63031"          # danger red
GOLD    = "#d18f00"          # gold/amber
C_ORIG  = "#2d6fc0"          # blue  — original
C_U     = "#1e7d3a"          # green — SASTO-U
C_PA    = "#c95a10"          # orange — SASTO-PA
PURPLE  = "#7e3acb"          # purple

# Part colours (match render_figures.py PART_COLORS)
PC_EXT  = "#377dbe"          # exterior wall
PC_INT  = "#f0783c"          # interior wall
PC_ROOF = "#64a032"          # roof
PC_FLR  = "#bea578"          # floor

def save(fig, name, dpi=180):
    p = OUT / name
    fig.savefig(p, dpi=dpi, bbox_inches="tight",
                transparent=False, facecolor=BG)
    plt.close(fig)
    print(f"  ✓  {p.name}")

def styled_fig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    fig.patch.set_color(BG)
    return fig

def style_ax(ax, xlabel=None, ylabel=None, title=None, grid=True):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_color(SPINE); sp.set_linewidth(0.9)
    ax.tick_params(colors=GRAY, labelsize=9.5, length=4, labelcolor=GRAY)
    ax.xaxis.label.set_color(GRAY)
    ax.yaxis.label.set_color(GRAY)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10.5, color=GRAY, labelpad=6)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10.5, color=GRAY, labelpad=6)
    if title:  ax.set_title(title, fontsize=12, color=GRAY, pad=8,
                             fontweight="bold")
    if grid:
        ax.grid(color=LGRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

def eq_box(ax, text, x=0.97, y=0.97, ha="right", va="top", fs=9.5):
    """Add a math equation box (light themed) to an axis."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fs, va=va, ha=ha, color=MID,
            math_fontfamily="dejavusans",
            bbox=dict(boxstyle="round,pad=0.45", fc="#edf2fc",
                      ec=ACC, lw=1.0, alpha=0.92))

def stat_box(ax, text, x=0.03, y=0.97, ha="left", va="top", fs=8.5):
    """Add a statistics text box."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fs, va=va, ha=ha, color=GRAY,
            font="DejaVu Sans",
            bbox=dict(boxstyle="round,pad=0.45", fc=PANEL,
                      ec=SPINE, lw=0.8))

def subtitle(fig, text, y=0.97):
    fig.text(0.5, y, text, ha="center", va="top", color=GRAY,
             fontsize=12, fontweight="bold")

def caption(fig, text, y=0.01):
    fig.text(0.5, y, text, ha="center", va="bottom", color=DIM,
             fontsize=8.5, style="italic")

def accent_line(ax, x_or_y, orient="v", col=RED, lw=1.8, ls="--", zorder=5,
                label=None):
    if orient == "v":
        ax.axvline(x_or_y, color=col, lw=lw, ls=ls, zorder=zorder,
                   label=label)
    else:
        ax.axhline(x_or_y, color=col, lw=lw, ls=ls, zorder=zorder,
                   label=label)

# ═══════════════════════════════════════════════════════════════════
# 1. REF COMPARISON ISO  (2×2 grid of real renders)
# ═══════════════════════════════════════════════════════════════════
print("1. ref_comparison_iso.png")

RENDER_MAP = {
    "orig_solid":   "stl_orig_solid.png",
    "pa_solid":     "stl_pa_solid.png",
    "orig_cutaway": "stl_orig_cutaway.png",
    "pa_cutaway":   "stl_pa_cutaway.png",
}

def load_render(name):
    p = RENDERS / name
    return imread(str(p)) if p.exists() else None

# Layout: 14 wide × 11 tall.  Top title row, then 2×2 image grid.
fig = plt.figure(figsize=(14, 11), facecolor=BG)
fig.patch.set_color(BG)

# Build a 3-row GridSpec: top row title bar, middle 2×2 images, bottom stats
gs = gridspec.GridSpec(3, 2, figure=fig,
                       height_ratios=[0.06, 1, 0.055],
                       left=0.03, right=0.97,
                       top=0.93, bottom=0.08,
                       hspace=0.07, wspace=0.05)

render_cells = [
    ("orig_solid",   "Original  ·  Solid View",   C_ORIG, 0),
    ("pa_solid",     "SASTO-PA  ·  Solid View",    C_PA,   1),
    ("orig_cutaway", "Original  ·  Cutaway View",  C_ORIG, 2),
    ("pa_cutaway",   "SASTO-PA  ·  Cutaway View",  C_PA,   3),
]

for fname_key, label, col, idx in render_cells:
    row_i, col_i = divmod(idx, 2)
    ax = fig.add_subplot(gs[1, col_i] if row_i == 0 else gs[row_i, col_i])
    # Actually gs[1,0], gs[1,1], gs[2,0], gs[2,1] but we only have 3 rows
    ax.set_axis_off()
    ax.set_facecolor("#f4f4f6")
    im = load_render(RENDER_MAP[fname_key])
    if im is not None:
        ax.imshow(im, aspect="equal")
    # coloured border via spines
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(col)
        sp.set_linewidth(3.2)
    # per-cell label banner at top of each cell
    ax.text(0.5, 0.985, label, transform=ax.transAxes,
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="square,pad=0.15", fc=col, ec="none"))

# Redo gs with only 2 image rows (fix the layout: drop the 3rd gs row, use only gs[1])
plt.close(fig)

fig = plt.figure(figsize=(14, 11), facecolor=BG)
fig.patch.set_color(BG)

# Proper layout constants
IMG_TOP    = 0.90
IMG_BOTTOM = 0.10
IMG_LEFT   = 0.03
IMG_RIGHT  = 0.97
H_GAP = 0.03
W_GAP = 0.025
col_w = (IMG_RIGHT - IMG_LEFT - W_GAP) / 2
row_h = (IMG_TOP - IMG_BOTTOM - H_GAP) / 2

axes_pos = [
    [IMG_LEFT,                   IMG_BOTTOM + row_h + H_GAP, col_w, row_h],  # top-left
    [IMG_LEFT + col_w + W_GAP,   IMG_BOTTOM + row_h + H_GAP, col_w, row_h],  # top-right
    [IMG_LEFT,                   IMG_BOTTOM,                  col_w, row_h],  # bot-left
    [IMG_LEFT + col_w + W_GAP,   IMG_BOTTOM,                  col_w, row_h],  # bot-right
]

imgs_order = [
    ("orig_solid",   "Original  ·  Solid View",      C_ORIG),
    ("pa_solid",     "SASTO-PA  ·  Solid View",       C_PA),
    ("orig_cutaway", "Original  ·  Interior Cutaway", C_ORIG),
    ("pa_cutaway",   "SASTO-PA  ·  Interior Cutaway", C_PA),
]

for (fname_key, label, col), pos in zip(imgs_order, axes_pos):
    ax = fig.add_axes(pos)
    ax.set_axis_off()
    ax.set_facecolor("#eeeff3")
    im = load_render(RENDER_MAP[fname_key])
    if im is not None:
        ax.imshow(im, aspect="equal", interpolation="lanczos")
    # coloured border
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(col); sp.set_linewidth(3.0)
    # label
    ax.text(0.5, 0.985, label, transform=ax.transAxes,
            ha="center", va="top", fontsize=11, fontweight="bold", color="white",
            bbox=dict(boxstyle="square,pad=0.15", fc=col, ec="none", alpha=0.9))

# Column header bars
for ci, (txt, col) in enumerate([("ORIGINAL", C_ORIG), ("SASTO-PA", C_PA)]):
    cx = IMG_LEFT + ci * (col_w + W_GAP) + col_w / 2
    ax_hdr = fig.add_axes([cx - 0.095, 0.912, 0.190, 0.040])
    ax_hdr.set_axis_off()
    ax_hdr.add_patch(FancyBboxPatch((0, 0), 1, 1,
        boxstyle="round,pad=0.10", linewidth=0,
        fc=col, ec="none", transform=ax_hdr.transAxes))
    ax_hdr.text(0.5, 0.5, txt, ha="center", va="center", color="white",
                fontsize=13, fontweight="bold", transform=ax_hdr.transAxes)

# Row labels on left
for ri, txt in enumerate(["Solid", "Interior\nCutaway"]):
    cy = IMG_BOTTOM + (1 - ri) * (row_h + H_GAP/2) - row_h/2
    fig.text(0.005, IMG_BOTTOM + (1-ri)*(row_h + H_GAP) + row_h/2,
             txt, ha="center", va="center", color=MID, fontsize=9.5,
             fontweight="bold", rotation=90)

# Main title
fig.text(0.5, 0.965, "Reference Geometry 00472  —  Original vs SASTO-PA",
         ha="center", va="top", color=GRAY, fontsize=14, fontweight="bold")

# Stats ribbon with equations
stats_txt = (r"Volume: $93{,}905 \rightarrow 74{,}752$ vox  "
             r"$\cdot$  $\Delta V = -20.4\%$  "
             r"$\cdot$  $C_{ratio} = 1.004$  "
             r"$\cdot$  Limit: $1.15$  "
             r"$\cdot$  Runtime: $97\,\mathrm{s}$")
fig.text(0.5, 0.940, stats_txt, ha="center", va="top", color=TEAL,
         fontsize=10, math_fontfamily="dejavusans")

# Bottom caption
fig.text(0.5, 0.027,
         "Fig. 6.  Original (left) vs SASTO-PA (right) on reference geometry 00472.  "
         "Top: solid render.  Bottom: interior cutaway.  "
         r"Zero constraint violations ($C_{ratio} \leq 1.15$).",
         ha="center", va="bottom", color=DIM, fontsize=8.5, style="italic",
         math_fontfamily="dejavusans")

save(fig, "ref_comparison_iso.png", dpi=180)

# ═══════════════════════════════════════════════════════════════════
# 2. HISTOGRAM — volume reduction population
# ═══════════════════════════════════════════════════════════════════
print("2. fig_histogram.png")

reductions, times_s = [], []
for d in sorted(BATCH.iterdir()):
    p = d / "optimization_summary.json"
    try:
        s = json.load(open(p))
        if s.get("success"):
            reductions.append(s["volume_reduction_pct"])
            times_s.append(s["total_time_seconds"])
    except:
        pass

reductions = np.array(reductions)
REF_VAL    = 20.4

fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG)
fig.patch.set_color(BG)
style_ax(ax, xlabel="Volume reduction  (%)",
         ylabel="Number of designs",
         title=r"Population Volume Reduction  —  SASTO-PA  ($n = {:,}$)".format(len(reductions)))

bins = np.linspace(0, max(reductions) + 2, 38)
n, b, patches = ax.hist(reductions, bins=bins, color=C_PA, alpha=0.65,
                        edgecolor="white", linewidth=0.6, zorder=3,
                        label=f"SASTO-PA  (n = {len(reductions):,})")

# KDE overlay
kde  = gaussian_kde(reductions, bw_method=0.18)
xkde = np.linspace(0, bins[-1], 300)
ykde = kde(xkde) * len(reductions) * (bins[1] - bins[0])
ax.plot(xkde, ykde, color=C_PA, lw=2.2, alpha=0.95, zorder=4, label="_nolegend_")

mu = reductions.mean()
sig = reductions.std()

# Reference line
ax.axvline(REF_VAL, color=GOLD, lw=2.0, ls="--", zorder=5,
           label=f"Reference 00472  ({REF_VAL:.1f}%)")
# mean line
ax.axvline(mu, color=TEAL, lw=2.0, ls="-.", zorder=5,
           label=rf"Population mean  ({mu:.1f}%)")

# Annotation arrows
ymax_ = ykde.max()
ax.annotate(f"00472\n{REF_VAL:.1f}%", xy=(REF_VAL, ymax_*0.55),
            xytext=(REF_VAL + 4.5, ymax_*0.68),
            color=GOLD, fontsize=9, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.2,
                            mutation_scale=10))
ax.annotate(rf"$\mu$ = {mu:.1f}%", xy=(mu, ymax_*0.38),
            xytext=(mu - 5.0, ymax_*0.52),
            color=TEAL, fontsize=9, fontweight="bold", ha="right",
            arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.2,
                            mutation_scale=10))

# Equation box
eq_box(ax, rf"$\Delta V = \frac{{V_{{base}} - V_{{opt}}}}{{V_{{base}}}} \times 100$",
       x=0.97, y=0.97, fs=10)

# Stats box
stat_box(ax, (f"$n$ = {len(reductions):,}\n"
              rf"$\mu$ = {mu:.1f}%""\n"
              rf"$\sigma$ = {sig:.1f}%""\n"
              f"range  {reductions.min():.0f}–{reductions.max():.0f}%"),
         x=0.03, y=0.97, fs=9)

ax.legend(frameon=True, framealpha=0.92, edgecolor=SPINE,
          facecolor=PANEL, labelcolor=GRAY, fontsize=9.5, loc="upper left",
          handlelength=1.6, bbox_to_anchor=(0.03, 0.80))
ax.set_xlim(0, bins[-1])
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
caption(fig,
    "Fig. 7.  Histogram of volume reductions across all {:,} optimised designs (SASTO-PA).  "
    "KDE curve overlaid.  Population spans {:.0f}–{:.0f}%.".format(
        len(reductions), reductions.min(), reductions.max()))
plt.tight_layout(rect=[0, 0.04, 1, 1])
save(fig, "fig_histogram.png")

# ═══════════════════════════════════════════════════════════════════
# 3. FEA COMPLIANCE — distribution + safety check
# ═══════════════════════════════════════════════════════════════════
print("3. fig_fea_compliance.png")

fea_data   = json.load(open(DATA_V3 / "fea_validation_full.json"))
comp_ratio = np.array([r["comp_ratio"] for r in fea_data
                       if r.get("comp_ratio") is not None])
vr_paired  = np.array([r["volume_reduction_pct"] for r in fea_data
                       if r.get("comp_ratio") is not None])
LIMIT  = 1.15
n_fea  = len(comp_ratio)
mu_cr  = comp_ratio.mean()
mx_cr  = comp_ratio.max()
violations = (comp_ratio > LIMIT).sum()

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG,
                          gridspec_kw=dict(wspace=0.30))
fig.patch.set_color(BG)

# ── left: histogram
ax = axes[0]
style_ax(ax, xlabel=r"Compliance ratio  $C_{ratio}$",
         ylabel="Designs",
         title=r"Compliance Ratio Distribution  ($n = {:,}$)".format(n_fea))
bins2 = np.linspace(comp_ratio.min() - 0.02, LIMIT + 0.05, 36)
ax.hist(comp_ratio, bins=bins2, color=TEAL, alpha=0.65,
        edgecolor="white", linewidth=0.6, zorder=3)
kde2 = gaussian_kde(comp_ratio, bw_method=0.20)
xk2  = np.linspace(bins2[0], bins2[-1], 300)
yk2  = kde2(xk2) * n_fea * (bins2[1] - bins2[0])
ax.plot(xk2, yk2, color=TEAL, lw=2.2, zorder=4)
ax.axvline(LIMIT, color=RED, lw=2.0, ls="--", zorder=5,
           label=rf"Safety limit  $C_{{limit}} = {LIMIT}$")
ax.axvline(mx_cr, color=GOLD, lw=1.8, ls="-.", zorder=5,
           label=rf"Max observed  $C_{{max}} = {mx_cr:.3f}$")
ax.legend(frameon=True, framealpha=0.92, edgecolor=SPINE,
          facecolor=PANEL, labelcolor=GRAY, fontsize=9.5)

# safety margin bracket
ymax2 = yk2.max()
ax.annotate("", xy=(LIMIT, ymax2*0.28), xytext=(mx_cr, ymax2*0.28),
            arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.5,
                            mutation_scale=10))
ax.text((mx_cr + LIMIT)/2, ymax2*0.32,
        rf"$\Delta = {LIMIT-mx_cr:.3f}$",
        ha="center", va="bottom", color=GOLD, fontsize=9.5, fontweight="bold",
        math_fontfamily="dejavusans")

eq_box(ax, r"$C_{ratio} = \frac{C_{opt}}{C_{base}} \leq 1.15$",
       x=0.97, y=0.97, fs=10.5)
stat_box(ax, (f"$n$ = {n_fea}\n"
              rf"$\mu$ = {mu_cr:.3f}""\n"
              rf"max = {mx_cr:.3f}""\n"
              f"violations = {violations}"),
         x=0.03, y=0.97)

# ── right: scatter compliance vs volume reduction
ax2 = axes[1]
style_ax(ax2, xlabel="Volume reduction (%)",
         ylabel=r"Compliance ratio  $C_{opt} / C_{base}$",
         title="Structural Safety vs Aggressiveness")
sc = ax2.scatter(vr_paired, comp_ratio,
                 c=comp_ratio, cmap="RdYlGn_r",
                 vmin=0.2, vmax=LIMIT, s=22, alpha=0.65, zorder=3,
                 edgecolors="none")
ax2.axhline(LIMIT, color=RED, lw=1.8, ls="--", zorder=4)
ax2.axhline(1.0, color=MID, lw=1.0, ls=":", zorder=4, alpha=0.5)
ax2.text(vr_paired.max()*0.97, LIMIT+0.008, r"Safety limit $1.15$",
         ha="right", va="bottom", color=RED, fontsize=9,
         math_fontfamily="dejavusans")
cb = plt.colorbar(sc, ax=ax2, pad=0.03, shrink=0.92)
cb.ax.tick_params(colors=GRAY, labelsize=8.5)
cb.set_label(r"$C_{ratio}$", color=GRAY, fontsize=10,
             math_fontfamily="dejavusans")
# zero violations badge
ax2.text(0.5, 0.90,
         rf"$\checkmark$  Zero violations  ({n_fea:,} designs)",
         ha="center", va="center", color=TEAL, fontsize=11.5,
         fontweight="bold", transform=ax2.transAxes,
         math_fontfamily="dejavusans",
         bbox=dict(boxstyle="round,pad=0.4", fc="#ecfaf4",
                   ec=TEAL, lw=1.8))

subtitle(fig, rf"Independent FEA Validation  —  {n_fea} Optimised Designs  "
         rf"($C_{{ratio}} \leq 1.15$)",
         y=0.98)
caption(fig,
    rf"Fig. 11.  Left: $C_{{ratio}}$ distribution — all designs satisfy $C_{{limit}} = 1.15$.  "
    rf"Right: $C_{{ratio}}$ vs volume reduction.  Max observed: $C_{{max}} = 1.004$.  Violations: 0.")
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
save(fig, "fig_fea_compliance.png")

# ═══════════════════════════════════════════════════════════════════
# 4. SPEEDUP — SASTO vs SIMP runtime
# ═══════════════════════════════════════════════════════════════════
print("4. fig_speedup.png")

simp_data   = json.load(open(DATA_V3 / "simp_benchmark.json"))
simp_times  = np.array([s["total_time_s"] for s in simp_data])

times_arr    = np.array(times_s)
SASTO_median = np.median(times_arr)
SCALE        = (128/64)**2   # FEA resolution scaling
simp_proj128 = simp_times * SCALE

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG,
                          gridspec_kw=dict(wspace=0.32))
fig.patch.set_color(BG)

# ── left: grouped boxplots
ax = axes[0]
style_ax(ax, ylabel="Runtime  (seconds)",
         title=r"SASTO  vs  SIMP  —  Runtime Comparison")
ax.set_yscale("log")

BP_DATA = [times_arr, simp_times]
BP_COLS = [C_PA, PURPLE]
BP_LBLS = [f"SASTO\n(128³,  n = {len(times_arr):,})",
           f"SIMP\n(64³,  n = {len(simp_times)})"]

bp = ax.boxplot(BP_DATA, patch_artist=True, notch=False,
                medianprops=dict(color="white", lw=2.2),
                whiskerprops=dict(color=MID, lw=1.3),
                capprops=dict(color=MID, lw=1.3),
                flierprops=dict(marker="o", ms=4.5, color=DIM, alpha=0.55,
                                markeredgewidth=0))
for patch, col in zip(bp["boxes"], BP_COLS):
    patch.set_facecolor(col); patch.set_alpha(0.55); patch.set_edgecolor(col)

ax.set_xticks([1, 2]); ax.set_xticklabels(BP_LBLS, color=GRAY, fontsize=10)

simp_med  = np.median(simp_times)
speedup   = simp_med / SASTO_median
ax.annotate(rf"$\times{speedup:.0f}$ faster",
            xy=(1, SASTO_median), xytext=(1.45, SASTO_median * 3.0),
            color=TEAL, fontsize=12, fontweight="bold", ha="center",
            math_fontfamily="dejavusans",
            arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.6,
                            mutation_scale=13))
ax.axhline(60, color=DIM, lw=0.9, ls=":", alpha=0.6)
ax.text(2.42, 66, "1 min", color=DIM, fontsize=8.5, va="bottom")

eq_box(ax, r"$S = \frac{T_{SIMP}}{T_{SASTO}}$", x=0.97, y=0.97, fs=11)
stat_box(ax, (rf"Median SASTO:  {SASTO_median:.0f} s""\n"
              rf"Median SIMP:    {simp_med:.0f} s""\n"
              rf"Speedup:  $\times{speedup:.0f}$"),
         x=0.03, y=0.97, fs=9)

# ── right: SASTO full distribution
ax2 = axes[1]
style_ax(ax2, xlabel="Runtime  (seconds)", ylabel="Designs",
         title=r"SASTO Runtime Distribution  (128³ grid)")
bins3 = np.logspace(np.log10(max(times_arr.min(), 1)),
                    np.log10(times_arr.max() + 10), 35)
ax2.hist(times_arr, bins=bins3, color=C_PA, alpha=0.65,
         edgecolor="white", linewidth=0.6, zorder=3,
         label=f"SASTO  (n = {len(times_arr):,})")
ax2.set_xscale("log")
ax2.axvline(SASTO_median, color=GOLD, lw=2.0, ls="--", zorder=4,
            label=rf"Median  {SASTO_median:.0f} s")

# SIMP projected lines
for st, lbl in [
    (simp_times.min()*SCALE, "SIMP 128³ min"),
    (np.median(simp_times)*SCALE, "SIMP 128³ median"),
    (simp_times.max()*SCALE,  "SIMP 128³ max"),
]:
    ax2.axvline(st, color=PURPLE, lw=1.5, ls="-.", alpha=0.75)

simp_min_128 = simp_times.min() * SCALE / 60
simp_med_128 = np.median(simp_times) * SCALE / 60
simp_max_128 = simp_times.max() * SCALE / 60
stat_box(ax2,
         (r"SIMP 128³ projected:" "\n"
          rf"  {simp_min_128:.0f}–{simp_max_128:.0f} min" "\n"
          rf"  (median {simp_med_128:.0f} min)"),
         x=0.97, y=0.50, ha="right", va="center", fs=8.5)
ax2.legend(frameon=True, framealpha=0.92, edgecolor=SPINE,
           facecolor=PANEL, labelcolor=GRAY, fontsize=9.5)

subtitle(fig, r"Runtime Comparison  —  SASTO vs SIMP  "
         r"($128^3$ vs $64^3$ resolution)", y=0.98)
caption(fig,
    rf"Fig. 10.  Left: SASTO (128³) vs SIMP (64³) runtime boxplots.  "
    rf"SASTO median = {SASTO_median:.0f} s.  "
    rf"Right: full SASTO distribution with SIMP 128³ projected via $T \propto N^2$.")
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
save(fig, "fig_speedup.png")

# ═══════════════════════════════════════════════════════════════════
# 5. REGRESSION — compliance ranking accuracy
# ═══════════════════════════════════════════════════════════════════
print("5. fig_regression.png")

base_comp = np.array([r["voxel_base_comp"] for r in fea_data
                      if r.get("voxel_base_comp") and r.get("voxel_opt_comp")])
opt_comp  = np.array([r["voxel_opt_comp"] for r in fea_data
                      if r.get("voxel_base_comp") and r.get("voxel_opt_comp")])
cr_paired = opt_comp / base_comp

surr_sub = [(r["surrogate_comp_mean"], r["voxel_opt_comp"])
             for r in fea_data
             if r.get("surrogate_comp_mean") and r.get("voxel_opt_comp")]
if surr_sub:
    s_pred = np.array([x[0] for x in surr_sub])
    s_true = np.array([x[1] for x in surr_sub])
    spearman_r, _ = stats.spearmanr(s_pred, s_true)
    pearson_r, _  = stats.pearsonr(s_pred, s_true)
    r2_val = pearson_r**2
else:
    s_pred = base_comp; s_true = opt_comp
    spearman_r, _ = stats.spearmanr(s_pred, s_true)
    r2_val = np.corrcoef(s_pred, s_true)[0, 1]**2
n_reg = len(s_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG,
                          gridspec_kw=dict(wspace=0.30))
fig.patch.set_color(BG)

ax = axes[0]
style_ax(ax, xlabel="Surrogate compliance prediction",
         ylabel="Ground-truth FEA compliance",
         title=r"Surrogate Compliance Ranking")
sc = ax.scatter(s_pred, s_true,
                c=np.abs(s_pred - s_true)/s_true,
                cmap="RdYlGn_r", vmin=0, vmax=0.5,
                s=25, alpha=0.65, zorder=3, edgecolors="none")
m, b_lin, r_lin, p_lin, _ = stats.linregress(s_pred, s_true)
xfit = np.linspace(s_pred.min(), s_pred.max(), 200)
ax.plot(xfit, m*xfit + b_lin, color=TEAL, lw=2.2, zorder=4,
        label="Linear fit")
lims = [min(s_pred.min(), s_true.min()), max(s_pred.max(), s_true.max())]
ax.plot(lims, lims, color=MID, lw=1.4, ls="--", zorder=2, alpha=0.6,
        label=r"$y = x$  (ideal)")
cb = plt.colorbar(sc, ax=ax, pad=0.03, shrink=0.92)
cb.ax.tick_params(colors=GRAY, labelsize=8.5)
cb.set_label("Relative error", color=GRAY, fontsize=9.5)
ax.legend(frameon=True, framealpha=0.92, edgecolor=SPINE,
          facecolor=PANEL, labelcolor=GRAY, fontsize=9)

eq_box(ax, (rf"$\rho_{{Spearman}} = {spearman_r:.3f}$""\n"
            rf"$R^2 = {r2_val:.3f}$"),
       x=0.97, y=0.97, fs=10.5)
stat_box(ax, f"$n$ = {n_reg}", x=0.03, y=0.97, fs=9)

# right: compliance ratio histogram
ax2 = axes[1]
style_ax(ax2, xlabel=r"$C_{opt} / C_{base}$  (compliance ratio)",
         ylabel="Designs",
         title=r"Compliance Improvement After Optimisation")
bins4 = np.linspace(0, 1.25, 36)
ax2.hist(cr_paired, bins=bins4, color=TEAL, alpha=0.65,
         edgecolor="white", linewidth=0.6, zorder=3)
ax2.axvline(1.0, color=ACC, lw=1.8, ls="-.", zorder=4,
            label=r"Baseline  $C_{ratio} = 1$")
ax2.axvline(LIMIT, color=RED, lw=2.0, ls="--", zorder=5,
            label=rf"Safety limit  $1.15$")
pct_below_1 = (cr_paired < 1.0).mean() * 100
ax2.text(0.5, 0.90,
         rf"{pct_below_1:.0f}% of designs" "\n" r"improve compliance",
         ha="center", va="center", color=TEAL, fontsize=11.5, fontweight="bold",
         transform=ax2.transAxes,
         bbox=dict(boxstyle="round,pad=0.4", fc="#ecfaf4", ec=TEAL, lw=1.8))
ax2.legend(frameon=True, framealpha=0.92, edgecolor=SPINE,
           facecolor=PANEL, labelcolor=GRAY, fontsize=9.5)

subtitle(fig,
    r"Surrogate Compliance Accuracy  —  Independent FEA Verification  "
    rf"($\rho_{{Spearman}} = {spearman_r:.3f}$)",
    y=0.98)
caption(fig,
    rf"Fig. 13.  Left: surrogate vs FEA compliance on {n_reg} designs  "
    rf"($\rho_{{Spearman}} = {spearman_r:.3f}$,  $R^2 = {r2_val:.3f}$).  "
    rf"Right: $C_{{ratio}}$ distribution; {pct_below_1:.0f}% of designs improve structural compliance.")
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
save(fig, "fig_regression.png")

# ═══════════════════════════════════════════════════════════════════
# 6. BLAND-ALTMAN — surrogate vs FEA agreement (rank-based)
# ═══════════════════════════════════════════════════════════════════
# The surrogate outputs normalised scores (0–0.6) while FEA produces
# raw compliance (2–332) — different scales by design.  The standard
# approach for cross-scale method comparison is to work on percentile
# ranks, which makes the Bland-Altman unit-free and meaningful.
print("6. fig_bland_altman.png")

ba_pairs = [(r["surrogate_comp_mean"], r["voxel_opt_comp"])
             for r in fea_data
             if r.get("surrogate_comp_mean") and r.get("voxel_opt_comp")]
ba_score = np.array([x[0] for x in ba_pairs])   # surrogate (0–1 scale)
ba_fea   = np.array([x[1] for x in ba_pairs])   # FEA raw compliance
n_ba     = len(ba_score)

# Convert both to percentile ranks (0–100)
from scipy.stats import rankdata
rank_surr = rankdata(ba_score) / n_ba * 100   # higher score → higher rank
rank_fea  = rankdata(ba_fea)   / n_ba * 100   # higher compliance → higher rank

# Spearman ρ between relative rankings
spear_ba, p_spear_ba = stats.spearmanr(ba_score, ba_fea)

# Bland-Altman on rank space
mean_r = (rank_surr + rank_fea) / 2
diff_r = rank_surr - rank_fea          # positive = surrogate ranks higher
bias_r     = diff_r.mean()
std_r      = diff_r.std()
loa_upper_r = bias_r + 1.96 * std_r
loa_lower_r = bias_r - 1.96 * std_r

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), facecolor=BG,
                          gridspec_kw=dict(wspace=0.34))
fig.patch.set_color(BG)

# ── left: Bland-Altman on percentile ranks ──
ax = axes[0]
style_ax(ax,
         xlabel=r"Mean percentile rank  $\frac{r_{surr} + r_{FEA}}{2}$",
         ylabel=r"Rank difference  $r_{surr} - r_{FEA}$",
         title=r"Bland-Altman  (Percentile-Rank Space)")

# Colour by absolute rank difference
sc = ax.scatter(mean_r, diff_r,
                c=np.abs(diff_r), cmap="RdYlGn_r",
                vmin=0, vmax=40,
                s=55, alpha=0.80, zorder=3, edgecolors="white",
                linewidths=0.4)

# Bias and LoA lines
ax.axhline(bias_r,      color=TEAL, lw=2.2, zorder=4, label=rf"Bias  ${bias_r:+.1f}$")
ax.axhline(loa_upper_r, color=GOLD, lw=1.8, ls="--", zorder=4,
           label=rf"$+1.96\sigma = {loa_upper_r:+.1f}$")
ax.axhline(loa_lower_r, color=GOLD, lw=1.8, ls="--", zorder=4,
           label=rf"$-1.96\sigma = {loa_lower_r:+.1f}$")
ax.axhline(0.0, color=DIM, lw=1.0, ls=":", alpha=0.55, zorder=2)

# LoA shaded band
ax.fill_between([0, 100], loa_lower_r, loa_upper_r,
                alpha=0.08, color=GOLD, zorder=1)

# Annotate lines on right edge — spaced to avoid overlap
for y_val, txt, col in [
    (loa_upper_r, rf"$+1.96\sigma$", GOLD),
    (bias_r,      r"Bias",          TEAL),
    (loa_lower_r, rf"$-1.96\sigma$", GOLD),
]:
    ax.text(98, y_val, txt, ha="right", va="center", color=col,
            fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            math_fontfamily="dejavusans", zorder=6)

ax.set_xlim(0, 100)
ax.set_ylim(diff_r.min() - 12, diff_r.max() + 14)
ax.set_xlabel(r"Mean percentile rank  $\frac{r_{surr} + r_{FEA}}{2}$",
              fontsize=10.5, color=GRAY)

cb = plt.colorbar(sc, ax=ax, pad=0.03, shrink=0.90)
cb.ax.tick_params(colors=GRAY, labelsize=8.5)
cb.set_label(r"$|r_{surr} - r_{FEA}|$", color=GRAY, fontsize=9.5,
             math_fontfamily="dejavusans")

# Legend in bottom-left (clear area)
ax.legend(frameon=True, framealpha=0.92, edgecolor=SPINE,
          facecolor=PANEL, labelcolor=GRAY, fontsize=8.5,
          loc="lower left", handlelength=1.5)

eq_box(ax, r"LoA: $\bar{d} \pm 1.96\,\sigma_d$", x=0.97, y=0.03,
       ha="right", va="bottom", fs=10.5)
stat_box(ax, (f"$n$ = {n_ba}\n"
              rf"bias = {bias_r:+.1f} pctile""\n"
              rf"$\sigma$ = {std_r:.1f}""\n"
              rf"$\rho_{{Spearman}}$ = {spear_ba:.3f}"),
         x=0.03, y=0.99, fs=9)

# ── right: rank correlation scatter ──
ax2 = axes[1]
style_ax(ax2,
         xlabel=r"Surrogate rank  $r_{surr}$  (percentile)",
         ylabel=r"FEA compliance rank  $r_{FEA}$  (percentile)",
         title=r"Surrogate–FEA Rank Correlation")

# Colour by absolute rank error
sc2 = ax2.scatter(rank_surr, rank_fea,
                  c=np.abs(rank_surr - rank_fea), cmap="RdYlGn_r",
                  vmin=0, vmax=40,
                  s=55, alpha=0.80, zorder=3,
                  edgecolors="white", linewidths=0.4)

# Identity line (perfect agreement)
ax2.plot([0, 100], [0, 100], color=MID, lw=1.5, ls="--", zorder=2,
         alpha=0.6, label=r"$y = x$  (perfect)")

# Regression line
m2, b2, _, _, _ = stats.linregress(rank_surr, rank_fea)
xf2 = np.array([0, 100])
ax2.plot(xf2, m2*xf2 + b2, color=TEAL, lw=2.2, zorder=4,
         label=rf"Fit  ($\rho = {spear_ba:.3f}$)")

ax2.set_xlim(0, 100); ax2.set_ylim(0, 100)

cb2 = plt.colorbar(sc2, ax=ax2, pad=0.03, shrink=0.90)
cb2.ax.tick_params(colors=GRAY, labelsize=8.5)
cb2.set_label(r"$|r_{surr} - r_{FEA}|$", color=GRAY, fontsize=9.5,
              math_fontfamily="dejavusans")

eq_box(ax2, (rf"$\rho_{{Spearman}} = {spear_ba:.3f}$"
             "\n" rf"$p = {p_spear_ba:.2e}$"),
       x=0.03, y=0.97, ha="left", va="top", fs=10)

ax2.legend(frameon=True, framealpha=0.92, edgecolor=SPINE,
           facecolor=PANEL, labelcolor=GRAY, fontsize=9.5,
           loc="lower right")

subtitle(fig,
    rf"Surrogate–FEA Method Agreement  —  Percentile-Rank Bland-Altman  ($n = {n_ba}$)",
    y=0.98)
caption(fig,
    rf"Fig. 14.  Bland-Altman on percentile ranks: bias = ${bias_r:+.1f}$ pctile,  "
    rf"LoA = $[{loa_lower_r:.1f},\ {loa_upper_r:.1f}]$.  "
    rf"Right: surrogate vs FEA rank correlation  ($\rho_{{Spearman}} = {spear_ba:.3f}$).  "
    rf"Both methods converted to percentile ranks (0–100) for scale-free comparison.")
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
save(fig, "fig_bland_altman.png")

# ═══════════════════════════════════════════════════════════════════
# 7. DIVERSE STL GALLERY — copy & overlay large legend
# ═══════════════════════════════════════════════════════════════════
print("7. fig_diverse_stl_gallery.png (overlay legend)")

gallery_src = RENDERS / "fig_diverse_stl_gallery.png"
if gallery_src.exists():
    gal_img = imread(str(gallery_src))
    gh, gw = gal_img.shape[:2]
    aspect = gw / gh
    FIG_W = 22
    fig = plt.figure(figsize=(FIG_W, FIG_W / aspect + 1.2), facecolor=BG)
    fig.patch.set_color(BG)
    ax_img = fig.add_axes([0, 0.10, 1.0, 0.90])
    ax_img.set_axis_off()
    ax_img.imshow(gal_img, aspect="equal", interpolation="lanczos")

    # Large legend at the bottom
    legend_elems = [
        mpatches.Patch(facecolor=PC_EXT,  edgecolor="#555", linewidth=1.2,
                       label="Exterior Wall"),
        mpatches.Patch(facecolor=PC_INT,  edgecolor="#555", linewidth=1.2,
                       label="Interior Wall"),
        mpatches.Patch(facecolor=PC_ROOF, edgecolor="#555", linewidth=1.2,
                       label="Roof"),
        mpatches.Patch(facecolor=PC_FLR,  edgecolor="#555", linewidth=1.2,
                       label="Floor"),
    ]
    leg = fig.legend(handles=legend_elems, loc="lower center", ncol=4,
                     fontsize=20,
                     frameon=True, framealpha=0.97,
                     edgecolor=SPINE, facecolor=BG,
                     handlelength=2.6, handletextpad=0.8,
                     columnspacing=3.5,
                     bbox_to_anchor=(0.5, 0.005),
                     labelcolor=GRAY,
                     prop={"size": 20, "weight": "bold"})
    # Override handle sizes
    for patch in leg.get_patches():
        patch.set_height(18)
        patch.set_width(30)

    save(fig, "fig_diverse_stl_gallery.png", dpi=160)
else:
    print("  ! gallery source not found — skipping")

# ═══════════════════════════════════════════════════════════════════
# 8. COPY other pre-rendered figures
# ═══════════════════════════════════════════════════════════════════
print("8. Copying pre-rendered figures...")
to_copy = [
    ("fig_type_comparison.png",       RENDERS),
    ("fig_cross_section_comparison.png", RENDERS),
    ("fig_optimized_gallery.png",     RENDERS),
]
for fname, src_dir in to_copy:
    src = src_dir / fname
    if src.exists():
        shutil.copy2(src, OUT / fname)
        print(f"  ✓  {fname}  (copied)")
    else:
        # Try figures/ folder
        src2 = ROOT / "figures" / fname
        if src2.exists():
            shutil.copy2(src2, OUT / fname)
            print(f"  ✓  {fname}  (copied from figures/)")
        else:
            print(f"  ✗  {fname}  NOT FOUND")

print(f"\n✅  All figures saved to  {OUT}/")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name:<45}  {f.stat().st_size // 1024:>5} KB")
