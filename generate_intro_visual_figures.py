"""
generate_intro_visual_figures.py
Two purely visual figures for the Introduction panel:

  fig_intro_co2.png         — Global CO2 emissions donut, cement highlighted
  fig_intro_crosssection.png — Floor-plan cross-section: uniform vs SASTO thickness heatmap
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Arc
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path

OUT_DIR = Path("poster_figures_v5")
OUT_DIR.mkdir(exist_ok=True)

CARD    = "#F7F9FC"
TEAL    = "#008C9E"
GOLD    = "#CFA535"
RED     = "#D7263D"
NAVY    = "#062B7A"
TXT     = "#0B1736"
ORANGE  = "#E07B30"
PURPLE  = "#7B3FA0"
GRAY    = "#8A9BB0"
GREEN   = "#2D8A6E"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Global CO₂ Emissions Donut
# ═══════════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(12, 9), facecolor=CARD)
ax.set_facecolor(CARD)
ax.set_aspect("equal")
ax.axis("off")

# IEA 2023-aligned global CO2 sector data (%)
sectors = [
    ("Cement &\nConcrete",    8.0,  RED,    True),
    ("Electricity &\nHeat",  25.0,  "#3A6BC8", False),
    ("Transport",            16.0,  ORANGE, False),
    ("Industry\n(other)",    19.0,  PURPLE, False),
    ("Buildings\n(ops)",      6.0,  GREEN,  False),
    ("Agriculture",          11.0,  "#8B8B00", False),
    ("Other",                15.0,  GRAY,   False),
]

labels  = [s[0] for s in sectors]
sizes   = [s[1] for s in sectors]
colors  = [s[2] for s in sectors]
explode = [0.12 if s[3] else 0.0 for s in sectors]

wedges, texts = ax.pie(
    sizes,
    labels=None,
    colors=colors,
    explode=explode,
    startangle=125,
    wedgeprops=dict(linewidth=2.5, edgecolor="white"),
    pctdistance=0.80,
    counterclock=False,
)

# Donut hole
hole = plt.Circle((0, 0), 0.52, color=CARD, zorder=10)
ax.add_patch(hole)

# Center text
ax.text(0, 0.10, "Global\nCO₂", ha="center", va="center",
        fontsize=16, fontweight="bold", color=TXT, zorder=11)
ax.text(0, -0.22, "Emissions", ha="center", va="center",
        fontsize=13, color=GRAY, zorder=11)

# Cement callout annotation
cement_wedge = wedges[0]
# Find midpoint angle of cement wedge
theta_start = cement_wedge.theta1
theta_end   = cement_wedge.theta2
theta_mid   = np.deg2rad((theta_start + theta_end) / 2)
r_mid = 0.80
cx_tip = r_mid * np.cos(theta_mid) * 1.05
cy_tip = r_mid * np.sin(theta_mid) * 1.05

ax.annotate(
    "8%\nCement &\nConcrete",
    xy=(cx_tip, cy_tip),
    xytext=(cx_tip * 1.62, cy_tip * 1.55),
    fontsize=14,
    fontweight="bold",
    color=RED,
    ha="center",
    va="center",
    arrowprops=dict(
        arrowstyle="->, head_width=0.25, head_length=0.15",
        color=RED, lw=2.0, connectionstyle="arc3,rad=0.15"
    ),
    zorder=12,
)

# Aviation comparison callout
ax.text(
    cx_tip * 1.62, cy_tip * 1.55 - 0.47,
    "≈ 5× all commercial\naviation combined",
    ha="center", va="top",
    fontsize=11, color=TXT,
    style="italic",
    zorder=12,
)

# Legend
legend_handles = [
    mpatches.Patch(facecolor=c, edgecolor="white", linewidth=1,
                   label=f"{l.replace(chr(10),' ')}  ({s:.0f}%)")
    for l, s, c, _ in sectors
]
ax.legend(
    handles=legend_handles,
    loc="lower left",
    bbox_to_anchor=(-0.55, -0.52),
    fontsize=11,
    frameon=True,
    framealpha=0.92,
    facecolor=CARD,
    edgecolor=GRAY,
    ncol=2,
    handlelength=1.2,
    handleheight=1.2,
    borderpad=0.7,
)

ax.set_title(
    "Global CO₂ Emissions by Sector  [IEA 2023]",
    fontsize=17, fontweight="bold", color=TXT, pad=18,
)

fig1.savefig(OUT_DIR / "fig_intro_co2.png",
             dpi=200, bbox_inches="tight", pad_inches=0.15,
             facecolor=CARD, edgecolor="none")
plt.close(fig1)
print("Saved → poster_figures_v5/fig_intro_co2.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Floor-plan cross-section thickness heatmap, Uniform vs SASTO
# ═══════════════════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 2, figsize=(16, 8),
                          facecolor=CARD,
                          gridspec_kw={"wspace": 0.05})
fig2.patch.set_facecolor(CARD)

# Color maps: red=thick, white=empty interior
THIN_COL  = "#5CB8C4"   # teal-ish: 1-voxel interior wall
THICK_COL = "#D7263D"   # red: 2-voxel exterior wall
ROOM_COL  = "#E8EEF8"   # light blue: open room space

# Grid resolution: 32 x 32 cells (voxels at 128³ → downscaled for display)
N = 48

def make_uniform_plan(n):
    """Returns an n×n array: 0=room, 1=thin_wall(unused), 2=thick_wall."""
    grid = np.zeros((n, n), dtype=int)
    THICK = 3   # 2-voxel wall → displayed as 3 cells thick
    # Outer walls
    grid[:THICK, :]  = 2
    grid[-THICK:, :] = 2
    grid[:, :THICK]  = 2
    grid[:, -THICK:] = 2
    # Interior horizontal partition — also thick (uniform case)
    mid = n // 2
    grid[mid - 1 : mid + 2, THICK:-THICK] = 2
    # Interior vertical partition — also thick
    vmid = n * 2 // 3
    grid[THICK : mid - 1,    vmid - 1 : vmid + 2] = 2
    grid[mid + 2 : -THICK,   vmid - 1 : vmid + 2] = 2
    return grid

def make_sasto_plan(n):
    """Returns an n×n array: 0=room, 1=thin_wall, 2=thick_wall."""
    grid = np.zeros((n, n), dtype=int)
    THICK = 3
    THIN  = 1
    # Outer walls — still thick
    grid[:THICK, :]  = 2
    grid[-THICK:, :] = 2
    grid[:, :THICK]  = 2
    grid[:, -THICK:] = 2
    # Interior partitions — now thin
    mid = n // 2
    grid[mid : mid + THIN, THICK:-THICK] = 1
    vmid = n * 2 // 3
    grid[THICK : mid, vmid : vmid + THIN] = 1
    grid[mid + THIN : -THICK, vmid : vmid + THIN] = 1
    return grid

# Build color images directly
def grid_to_rgb(grid):
    h, w = grid.shape
    img = np.ones((h, w, 3))
    for i in range(h):
        for j in range(w):
            if grid[i, j] == 0:
                img[i, j] = [232/255, 238/255, 248/255]  # room fill
            elif grid[i, j] == 1:
                img[i, j] = [0x5C/255, 0xB8/255, 0xC4/255]  # thin teal
            else:
                img[i, j] = [0xD7/255, 0x26/255, 0x3D/255]  # thick red

    return img

grids = [make_uniform_plan(N), make_sasto_plan(N)]
titles = ["Conventional  —  Uniform Walls", "SASTO  —  Part-Aware Walls"]
subtitles = ["All walls = 156 mm  (2 voxels)", "Exterior = 156 mm  ·  Interior = 78 mm  (1 voxel)"]
title_colors = [RED, TEAL]
removed_pct = [None, "−23.5% concrete"]

for i, (ax_i, grid) in enumerate(zip(axes, grids)):
    ax_i.set_facecolor(CARD)
    img = grid_to_rgb(grid)
    ax_i.imshow(img, origin="upper", interpolation="nearest",
                aspect="equal", extent=[0, N, 0, N])

    # Grid overlay (subtle)
    for x in range(N + 1):
        ax_i.axvline(x, color="white", lw=0.3, alpha=0.4)
    for y in range(N + 1):
        ax_i.axhline(y, color="white", lw=0.3, alpha=0.4)

    # Thickness annotation arrows
    THICK = 3
    if i == 0:
        # Annotate outer wall thickness
        ax_i.annotate("", xy=(THICK, N * 0.5), xytext=(0, N * 0.5),
                      arrowprops=dict(arrowstyle="<->", color="white", lw=2.0))
        ax_i.text(THICK / 2, N * 0.5 + 1.5, "156mm",
                  ha="center", va="bottom", fontsize=9.5,
                  color="white", fontweight="bold")
        # Interior partition
        mid = N // 2
        ax_i.annotate("", xy=(N * 0.40, mid + 3), xytext=(N * 0.40, mid - 1),
                      arrowprops=dict(arrowstyle="<->", color="white", lw=2.0))
        ax_i.text(N * 0.40 + 1.5, mid + 1.0, "156mm",
                  ha="left", va="center", fontsize=9.5,
                  color="white", fontweight="bold")
    else:
        # Exterior wall — still thick
        ax_i.annotate("", xy=(THICK, N * 0.5), xytext=(0, N * 0.5),
                      arrowprops=dict(arrowstyle="<->", color="white", lw=2.0))
        ax_i.text(THICK / 2, N * 0.5 + 1.5, "156mm",
                  ha="center", va="bottom", fontsize=9.5,
                  color="white", fontweight="bold")
        # Interior thin partition
        mid = N // 2
        ax_i.annotate("", xy=(N * 0.40, mid + 1), xytext=(N * 0.40, mid),
                      arrowprops=dict(arrowstyle="<->", color=GOLD, lw=2.5))
        ax_i.text(N * 0.40 + 1.5, mid + 0.5, "78mm",
                  ha="left", va="center", fontsize=9.5,
                  color=GOLD, fontweight="bold")

        # Savings badge
        ax_i.text(N * 0.98, N * 0.03, "−23.5% concrete",
                  ha="right", va="bottom",
                  fontsize=14, fontweight="bold", color=GOLD,
                  bbox=dict(boxstyle="round,pad=0.4",
                            facecolor=NAVY, edgecolor=GOLD,
                            linewidth=2.0, alpha=0.92))

    ax_i.set_xlim(0, N); ax_i.set_ylim(0, N)
    ax_i.set_xticks([]); ax_i.set_yticks([])
    for spine in ax_i.spines.values():
        spine.set_visible(False)

    ax_i.set_title(titles[i], fontsize=16, fontweight="bold",
                   color=title_colors[i], pad=10)
    ax_i.set_xlabel(subtitles[i], fontsize=12, color=TXT, labelpad=6)

# Legend
legend_items = [
    mpatches.Patch(facecolor=[0xD7/255, 0x26/255, 0x3D/255], edgecolor="none",
                   label="Thick wall — 156 mm  (load-bearing exterior)"),
    mpatches.Patch(facecolor=[0x5C/255, 0xB8/255, 0xC4/255], edgecolor="none",
                   label="Thin wall — 78 mm  (non-structural interior, SASTO only)"),
    mpatches.Patch(facecolor=[232/255, 238/255, 248/255],
                   edgecolor=[0.6, 0.6, 0.7], linewidth=0.8,
                   label="Open room space"),
]
fig2.legend(
    handles=legend_items,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.04),
    ncol=3,
    fontsize=12,
    frameon=True,
    framealpha=0.92,
    facecolor=CARD,
    edgecolor=GRAY,
    handlelength=1.4,
    handleheight=1.2,
)

fig2.suptitle(
    "Floor Plan Wall Thickness  —  Top View",
    fontsize=18, fontweight="bold", color=TXT, y=1.01,
)

fig2.savefig(OUT_DIR / "fig_intro_crosssection.png",
             dpi=200, bbox_inches="tight", pad_inches=0.15,
             facecolor=CARD, edgecolor="none")
plt.close(fig2)
print("Saved → poster_figures_v5/fig_intro_crosssection.png")
