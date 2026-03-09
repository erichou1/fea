"""
Clean visual pipeline — just images + arrows, no graphs or numbers.

  fig_opt_pipeline.png

Layout:
  ┌──────────────────────────────────────────────────────────────────┐
  │  Header                                                          │
  ├──────────────────────────────────────────────────────────────────┤
  │  [Original solid]  ──►  [SASTO-U solid]  ──►  [SASTO-PA solid] │
  │                                                                  │
  │  [Diff overlay]  [Sensitivity]  [Removal seq]  [Floor plan]     │
  └──────────────────────────────────────────────────────────────────┘
All images, minimal text, arrows only.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.image import imread
from pathlib import Path

PANEL  = "#111820"
GRAY   = "#d8d8d8"
DIM    = "#778899"
ACC    = "#4fc3f7"
GOLD   = "#f7c948"
C_ORIG = "#1a3a6a"
C_U    = "#1a5a38"
C_PA   = "#6a3010"

RENDERS = Path("poster_final/renders_hq")
FIGS    = Path("figures")
OUT     = FIGS; OUT.mkdir(exist_ok=True)

def load(name, folder=RENDERS):
    p = folder / name
    return imread(str(p)) if p.exists() else None

def img_ax(fig, pos, image, border_col="#304060", bw=1.4):
    ax = fig.add_axes(pos)
    ax.set_axis_off()
    ax.set_facecolor("#06090f")
    if image is not None:
        ax.imshow(image, aspect="auto")
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(border_col)
        sp.set_linewidth(bw)
    return ax

def label_ax(fig, pos, text, sub, col, fontsize=11):
    ax = fig.add_axes(pos)
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1,
        boxstyle="round,pad=0.06", fc=col, ec="none",
        transform=ax.transAxes))
    ax.text(0.5, 0.65, text, ha="center", va="center",
            color="white", fontsize=fontsize, fontweight="bold",
            transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.18, sub, ha="center", va="center",
                color="#bbddff", fontsize=fontsize*0.72,
                transform=ax.transAxes)

def arrow(fig, x0, x1, y, col=ACC, lw=2.4, label=""):
    fig.add_artist(plt.annotate(
        "", xytext=(x0, y), xy=(x1, y),
        xycoords="figure fraction",
        arrowprops=dict(arrowstyle="-|>", color=col,
                        lw=lw, mutation_scale=22)))
    if label:
        fig.text((x0+x1)/2, y + 0.018, label,
                 ha="center", va="bottom", color=col,
                 fontsize=9, fontstyle="italic",
                 transform=fig.transFigure)

# ── figure ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11), facecolor="none")
fig.patch.set_facecolor("none")

bg = fig.add_axes([0, 0, 1, 1]); bg.set_axis_off()
bg.add_patch(FancyBboxPatch((0.005, 0.005), 0.990, 0.990,
    boxstyle="round,pad=0.010", fc=PANEL, ec="#253550", lw=2.0,
    transform=bg.transAxes, zorder=0))

# ── header ────────────────────────────────────────────────────────
hdr = fig.add_axes([0.010, 0.943, 0.980, 0.048])
hdr.set_axis_off()
hdr.add_patch(FancyBboxPatch((0, 0), 1, 1,
    boxstyle="round,pad=0.05", fc="#152040", ec="#304080", lw=1.5,
    transform=hdr.transAxes))
hdr.text(0.016, 0.52,
    "SASTO  —  Surrogate-Accelerated Sensitivity Topology Optimisation",
    ha="left", va="center", color="white",
    fontsize=13.5, fontweight="bold", transform=hdr.transAxes)
hdr.text(0.984, 0.52, "Sample 00472  ·  128³ voxel grid",
    ha="right", va="center", color=ACC,
    fontsize=10, transform=hdr.transAxes)

# ── ROW 1: three solid renders ────────────────────────────────────
IW = 0.268; IH = 0.490; IY = 0.410
IGAP = 0.043
IX = [0.014, 0.014 + IW + IGAP, 0.014 + 2*(IW + IGAP)]
LH = 0.048; LGAP = 0.004

METHODS = [
    ("original_solid.png",  "Original",   "Baseline geometry",            C_ORIG),
    ("sasto_u_solid.png",   "SASTO-U",    "Uniform t_min = 2 vx",         C_U),
    ("sasto_pa_solid.png",  "SASTO-PA",   "Part-aware  t_min = 1–2 vx",   C_PA),
]

for ci, (fname, title, sub, col) in enumerate(METHODS):
    label_ax(fig,
             [IX[ci], IY + IH + LGAP, IW, LH],
             title, sub, col)
    img_ax(fig, [IX[ci], IY, IW, IH], load(fname), col, bw=1.8)

# arrows between methods
for ci in range(2):
    x0 = IX[ci] + IW + 0.006
    x1 = IX[ci+1]  - 0.006
    arrow(fig, x0, x1, IY + IH*0.50, col=ACC, label="optimise →")

# ── divider line ───────────────────────────────────────────────────
fig.add_artist(plt.Line2D(
    [0.020, 0.980], [IY - 0.026, IY - 0.026],
    color="#253550", lw=1.2, transform=fig.transFigure))

fig.text(0.022, IY - 0.040,
    "Supporting analysis  —  real voxel data",
    ha="left", va="top", color=ACC,
    fontsize=9.5, fontstyle="italic",
    transform=fig.transFigure)

# ── ROW 2: four support figures ────────────────────────────────────
SIW = 0.222; SIH = 0.240; SIY = 0.070
SIGAP = 0.009
SIX = [0.012 + i*(SIW+SIGAP) for i in range(4)]
SLH = 0.040

SUPPORT = [
    ("fig_diff_overlay.png",     "Difference Overlay",
     "voxel fate: kept / U / PA removed",          "#1a304a"),
    ("fig_sensitivity_map.png",  "Sensitivity Map",
     "red = safe to remove  ·  blue = critical",   "#1a3a25"),
    ("fig_removal_sequence.png", "Removal Sequence",
     "0 → 7 → 14 → 20%  material removed",         "#2a2010"),
    ("fig_floor_plan.png",       "Floor Plan",
     "top-down view  ·  colour by part",            "#251535"),
]

for si, (fname, title, sub, col) in enumerate(SUPPORT):
    label_ax(fig,
             [SIX[si], SIY + SIH + LGAP, SIW, SLH],
             title, sub, col, fontsize=9)
    img_ax(fig, [SIX[si], SIY, SIW, SIH], load(fname, FIGS), col, bw=1.2)

# connector arrows from main renders down to support figs
# Original → diff overlay and floor plan
# SASTO-PA → diff overlay
DOWN_PAIRS = [
    # (from_x_centre, from_y_bottom, to_x_centre, to_y_top)
    (IX[0]+IW/2,           IY,           SIX[0]+SIW/2,       SIY+SIH+SLH+LGAP),
    (IX[2]+IW/2,           IY,           SIX[0]+SIW/2,       SIY+SIH+SLH+LGAP),
    ((IX[0]+IX[2])/2+IW/2, IY,           SIX[1]+SIW/2,       SIY+SIH+SLH+LGAP),
    (IX[2]+IW/2,           IY,           SIX[2]+SIW/2,       SIY+SIH+SLH+LGAP),
    (IX[0]+IW/2,           IY,           SIX[3]+SIW/2,       SIY+SIH+SLH+LGAP),
]
for fx, fy, tx, ty in DOWN_PAIRS:
    fig.add_artist(plt.annotate(
        "", xytext=(fx, fy - 0.004), xy=(tx, ty + 0.004),
        xycoords="figure fraction",
        arrowprops=dict(arrowstyle="-|>", color="#304a60",
                        lw=1.0, mutation_scale=10,
                        connectionstyle="arc3,rad=0.0")))

# ── save ──────────────────────────────────────────────────────────
out = OUT / "fig_opt_pipeline.png"
fig.savefig(out, dpi=180, bbox_inches="tight",
            transparent=True, facecolor="none")
plt.close(fig)
print(f"saved -> {out}")
