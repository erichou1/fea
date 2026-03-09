"""
generate_intro_figures.py
Generates two Introduction-panel figures:
  1. fig_intro_problem_flow.png  — "Why SASTO?" 3-box problem flow (like the reference slide)
  2. fig_intro_gap.png           — Uniform vs SASTO wall-thickness schematic ("The Gap")

Output: poster_figures_v5/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT_DIR = Path("poster_figures_v5")
OUT_DIR.mkdir(exist_ok=True)

# ── Shared palette ──────────────────────────────────────────────────────────
BG          = "#F7F9FC"
CARD_BORDER = "#B7C5E3"
TXT_DARK    = "#0B1736"
TEAL        = "#008C9E"
GOLD        = "#CFA535"
RED         = "#D7263D"
NAVY        = "#062B7A"
SECTION_BAR = "#0A3D9A"
PILL_BG     = "#E8EEF8"
ORANGE      = "#E07B30"

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — "The Problem" 3-box flow diagram
# ─────────────────────────────────────────────────────────────────────────────
FW, FH = 16, 5.2
fig1, ax1 = plt.subplots(figsize=(FW, FH), facecolor=BG)
ax1.set_xlim(0, FW); ax1.set_ylim(0, FH)
ax1.axis("off")
fig1.patch.set_facecolor(BG)

def rbox(ax, x, y, w, h, fc, ec, radius=0.30, lw=1.5, zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=fc, edgecolor=ec,
                       linewidth=lw, zorder=zorder, clip_on=False)
    ax.add_patch(p)

def t(ax, x, y, s, sz=11, color=TXT_DARK, bold=False, ha="center", va="center",
      italic=False, zorder=5, wrap=False):
    w = "bold" if bold else "normal"
    st = "italic" if italic else "normal"
    ax.text(x, y, s, fontsize=sz, color=color, fontweight=w, fontstyle=st,
            ha=ha, va=va, zorder=zorder,
            wrap=wrap, clip_on=False)

BOXES = [
    {
        "title":    "Uniform Wall Sizing",
        "icon":     "WASTE",
        "lines":    ["23–45% of concrete is", "wasted — walls sized for", "worst-case load, not service"],
        "color":    RED,
        "bg":       "#FDF0F2",
    },
    {
        "title":    "SIMP Needs 100+ FEA Solves",
        "icon":     "SLOW",
        "lines":    ["19–77 min each at 128³ scale", "100s of solves per design", "Intractable at building scale"],
        "color":    ORANGE,
        "bg":       "#FDF5EE",
    },
    {
        "title":    "Voxel Removal Breaks Topology",
        "icon":     "FRAG",
        "lines":    ["Diagonal adjacency (26-conn)", "produces floating fragments", "Unprintable toolpath"],
        "color":    "#8B2FC9",
        "bg":       "#F5F0FC",
    },
]

BOX_W   = 4.20
BOX_H   = 3.40
GAP_X   = 0.65    # gap between boxes (for arrow)
START_X = (FW - 3 * BOX_W - 2 * GAP_X) / 2
TOP_Y   = (FH - BOX_H) / 2

for i, bx_def in enumerate(BOXES):
    bx = START_X + i * (BOX_W + GAP_X)
    by = TOP_Y

    # Shadow
    rbox(ax1, bx + 0.07, by - 0.07, BOX_W, BOX_H,
         fc="#DDDDDD", ec="#DDDDDD", radius=0.28, lw=0, zorder=1)

    # Main card
    rbox(ax1, bx, by, BOX_W, BOX_H,
         fc=bx_def["bg"], ec=bx_def["color"], radius=0.28, lw=2.2, zorder=2)

    # Colored header band
    rbox(ax1, bx, by + BOX_H - 0.85, BOX_W, 0.85,
         fc=bx_def["color"], ec=bx_def["color"], radius=0.28, lw=0, zorder=3)
    # Square off the bottom corners of the header band
    ax1.add_patch(plt.Rectangle(
        (bx, by + BOX_H - 0.85), BOX_W, 0.30,
        facecolor=bx_def["color"], linewidth=0, zorder=3))

    # Step number badge
    circ = plt.Circle((bx + 0.48, by + BOX_H - 0.42), 0.24,
                       color="white", zorder=5)
    ax1.add_patch(circ)
    t(ax1, bx + 0.48, by + BOX_H - 0.42, str(i + 1),
      sz=11, color=bx_def["color"], bold=True, zorder=6)

    # Header title
    t(ax1, bx + BOX_W / 2 + 0.15, by + BOX_H - 0.42,
      bx_def["title"], sz=11.5, color="white", bold=True, zorder=5)

    # Icon pill
    rbox(ax1, bx + BOX_W / 2 - 0.52, by + BOX_H - 1.48, 1.04, 0.38,
         fc=bx_def["color"], ec=bx_def["color"], radius=0.15, lw=0, zorder=4)
    t(ax1, bx + BOX_W / 2, by + BOX_H - 1.28,
      bx_def["icon"], sz=9, color="white", bold=True, zorder=5)

    # Body lines
    for j, line in enumerate(bx_def["lines"]):
        t(ax1, bx + BOX_W / 2, by + BOX_H - 1.90 - j * 0.42,
          line, sz=10.5, color=TXT_DARK, zorder=5)

    # Arrow to next box
    if i < len(BOXES) - 1:
        ax1.annotate("",
            xy=(bx + BOX_W + GAP_X, by + BOX_H / 2),
            xytext=(bx + BOX_W + 0.06, by + BOX_H / 2),
            arrowprops=dict(
                arrowstyle="->, head_width=0.35, head_length=0.20",
                color=GOLD, lw=3.0,
            ),
            zorder=6,
        )

# SASTO solution callout at the bottom
sol_y = 0.22
rbox(ax1, START_X, sol_y, 3 * BOX_W + 2 * GAP_X, 0.62,
     fc=NAVY, ec=GOLD, radius=0.18, lw=2.0, zorder=4)
t(ax1, FW / 2, sol_y + 0.42,
  "SASTO solves all three:  surrogate replaces FEA  ·  6-connectivity enforces printability  ·  part-aware floors save material",
  sz=11, color="white", bold=False, zorder=5)
t(ax1, FW / 2, sol_y + 0.18,
  "Result: median 50 s per optimization  ·  0 / 1,114 structural violations  ·  −23.5% mean concrete reduction",
  sz=11, color=GOLD, bold=True, zorder=5)

fig1.savefig(OUT_DIR / "fig_intro_problem_flow.png",
             dpi=200, bbox_inches="tight", pad_inches=0.05,
             facecolor=BG, edgecolor="none")
plt.close(fig1)
print("Saved → poster_figures_v5/fig_intro_problem_flow.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — "The Gap" Uniform vs SASTO wall schematic
# ─────────────────────────────────────────────────────────────────────────────
FW2, FH2 = 16, 6.0
fig2, ax2 = plt.subplots(figsize=(FW2, FH2), facecolor=BG)
ax2.set_xlim(0, FW2); ax2.set_ylim(0, FH2)
ax2.axis("off")
fig2.patch.set_facecolor(BG)

HALF = FW2 / 2

def draw_panel(ax, ox, title, title_color, walls, label_color, sub_labels, badge_text, badge_color):
    """
    Draw one floor-plan cross-section panel.
    walls: list of (x, y, w, h, thickness_label) in panel-local coords
    """
    PW = HALF - 0.40      # panel width
    PH = FH2 - 0.60
    by = 0.30

    # Panel card
    rbox(ax, ox + 0.15, by, PW, PH,
         fc="white", ec=title_color, radius=0.25, lw=2.0, zorder=2)

    # Title bar
    rbox(ax, ox + 0.15, by + PH - 0.68, PW, 0.68,
         fc=title_color, ec=title_color, radius=0.25, lw=0, zorder=3)
    ax.add_patch(plt.Rectangle(
        (ox + 0.15, by + PH - 0.68), PW, 0.22,
        facecolor=title_color, linewidth=0, zorder=3))
    t(ax, ox + 0.15 + PW / 2, by + PH - 0.32,
      title, sz=13, color="white", bold=True, zorder=4)

    # Floor plan room — outer footprint
    PLAN_X = ox + 0.55
    PLAN_Y = by + 0.75
    PLAN_W = PW - 0.80
    PLAN_H = PH - 1.55

    # Draw walls as filled rectangles (floor plan view)
    for wx, wy, ww, wh, wt, wc, wlabel in walls:
        # Convert relative (0-1) to absolute
        ax_x = PLAN_X + wx * PLAN_W
        ax_y = PLAN_Y + wy * PLAN_H
        ax_w = ww * PLAN_W
        ax_h = wh * PLAN_H
        rbox(ax, ax_x, ax_y, ax_w, ax_h,
             fc=wc, ec=wc, radius=0.04, lw=0, zorder=4)
        # Thickness annotation
        if wlabel:
            mid_x = ax_x + ax_w / 2
            mid_y = ax_y + ax_h / 2
            t(ax, mid_x, mid_y, wlabel, sz=7.5, color="white", bold=True, zorder=6)

    # Interior floor fill (light gray room)
    for rx, ry, rw, rh in sub_labels:
        ax_x = PLAN_X + rx * PLAN_W
        ax_y = PLAN_Y + ry * PLAN_H
        ax_w = rw * PLAN_W
        ax_h = rh * PLAN_H
        rbox(ax, ax_x, ax_y, ax_w, ax_h,
             fc="#E8EEF8", ec=CARD_BORDER, radius=0.05, lw=0.8, zorder=3)

    # Badge
    rbox(ax, ox + 0.15 + PW / 2 - 1.80, by + 0.12, 3.60, 0.52,
         fc=badge_color, ec=badge_color, radius=0.18, lw=0, zorder=5)
    t(ax, ox + 0.15 + PW / 2, by + 0.38,
      badge_text, sz=12, color="white", bold=True, zorder=6)


# Wall thickness multiplier
THIN  = 0.040    # interior wall fraction of PLAN dimension
THICK = 0.080    # exterior wall fraction

# ── Panel 1: UNIFORM (all thick) ─────────────────────────────────────────────
WALL_RED = "#C0393A"

# Walls described as (x, y, w, h, thickness_label, color, label)
# Simple L-shaped floor plan: outer walls + 1 interior partition

uniform_walls = [
    # Exterior walls (thick)
    (0.00, 0.00, 1.00, THICK, "t=156mm", WALL_RED, "156mm"),   # bottom
    (0.00, 1.0 - THICK, 1.00, THICK, "", WALL_RED, "156mm"),   # top
    (0.00, THICK, THICK, 1.0 - 2 * THICK, "", WALL_RED, ""),   # left
    (1.0 - THICK, THICK, THICK, 1.0 - 2 * THICK, "", WALL_RED, ""),  # right
    # Interior partition (also thick in uniform case)
    (THICK, 0.42, 1.0 - 2 * THICK, THICK * 0.9, "", WALL_RED, "156mm"),
    # Interior partition vertical
    (0.48, THICK + THICK * 0.9, THICK, 1.0 - 2 * THICK - THICK, "", WALL_RED, ""),
]
# Room fills (go behind walls)
uniform_rooms = [
    (THICK, THICK + THICK * 0.9, 0.48 - THICK, 1.0 - 2 * THICK - THICK),
    (0.48 + THICK, THICK + THICK * 0.9, 1.0 - 2 * THICK - 0.48 - THICK, 1.0 - 2 * THICK - THICK),
    (THICK, THICK, 1.0 - 2 * THICK, 0.42 - THICK),
]

draw_panel(ax2, 0, "Conventional: Uniform Thickness",
           WALL_RED, uniform_walls, "white", uniform_rooms,
           "All walls = 156 mm  |  No structural differentiation", "#B02A2A")

# "WASTE" arrows pointing at fat walls
for (px, py) in [(0.35, 0.80), (0.35, 0.60)]:
    abs_px = 0.15 + (HALF - 0.55) * px + 0.55
    abs_py = 0.30 + 0.75 + py * (FH2 - 0.60 - 1.55)
    ax2.annotate("", xy=(abs_px - 0.35, abs_py),
                 xytext=(abs_px - 0.90, abs_py),
                 arrowprops=dict(arrowstyle="->, head_width=0.15", color=WALL_RED, lw=1.5))

# ── Panel 2: SASTO (differentiated) ─────────────────────────────────────────
WALL_EXT  = TEAL
WALL_INT  = "#1A7A48"

sasto_walls = [
    # Exterior walls (thick)
    (0.00, 0.00, 1.00, THICK, "t=156mm", WALL_EXT, "156mm"),
    (0.00, 1.0 - THICK, 1.00, THICK, "", WALL_EXT, "156mm"),
    (0.00, THICK, THICK, 1.0 - 2 * THICK, "", WALL_EXT, ""),
    (1.0 - THICK, THICK, THICK, 1.0 - 2 * THICK, "", WALL_EXT, ""),
    # Interior partition (THIN in SASTO)
    (THICK, 0.42, 1.0 - 2 * THICK, THIN * 1.0, "", WALL_INT, "78mm"),
    (0.48, THICK + THIN * 1.0, THIN, 1.0 - 2 * THICK - THIN, "", WALL_INT, ""),
]
sasto_rooms = [
    (THICK, THICK + THIN, 0.48 - THICK, 1.0 - 2 * THICK - THIN),
    (0.48 + THIN, THICK + THIN, 1.0 - 2 * THICK - 0.48 - THIN, 1.0 - 2 * THICK - THIN),
    (THICK, THICK, 1.0 - 2 * THICK, 0.42 - THIN),
]

draw_panel(ax2, HALF, "SASTO: Part-Aware Thickness",
           TEAL, sasto_walls, "white", sasto_rooms,
           "Exterior = 156 mm  ·  Interior = 78 mm", TEAL)

# Legend labels on right panel
PLAN_X2 = HALF + 0.55
PLAN_Y2 = 0.30 + 0.75
PLAN_W2 = (HALF - 0.40) - 0.80
PLAN_H2 = FH2 - 0.60 - 1.55
for (lx, ly, lc, lb) in [
    (0.85, 0.90, WALL_EXT, "Exterior: 156mm"),
    (0.85, 0.70, WALL_INT, "Interior: 78mm"),
]:
    ax2.add_patch(plt.Rectangle(
        (PLAN_X2 + lx * PLAN_W2, PLAN_Y2 + ly * PLAN_H2), 0.28, 0.18,
        facecolor=lc, edgecolor=lc, zorder=6))
    t(ax2, PLAN_X2 + lx * PLAN_W2 + 0.34,
      PLAN_Y2 + ly * PLAN_H2 + 0.09,
      lb, sz=8.5, color=TXT_DARK, ha="left", zorder=6)

# ── VS divider ───────────────────────────────────────────────────────────────
vs_x = HALF
vs_y = FH2 / 2
rbox(ax2, vs_x - 0.38, vs_y - 0.32, 0.76, 0.64,
     fc=GOLD, ec=GOLD, radius=0.28, lw=0, zorder=7)
t(ax2, vs_x, vs_y, "vs.", sz=15, color="white", bold=True, zorder=8)

# ── Central savings callout ───────────────────────────────────────────────────
sav_y = 0.05
rbox(ax2, 2.80, sav_y, FW2 - 5.60, 0.50,
     fc=NAVY, ec=GOLD, radius=0.18, lw=2.0, zorder=5)
t(ax2, FW2 / 2, sav_y + 0.25,
  "− 23.5% mean concrete reduction  ·  10.7 pp more savings vs. uniform-floor baseline",
  sz=12.5, color=GOLD, bold=True, zorder=6)

fig2.savefig(OUT_DIR / "fig_intro_gap.png",
             dpi=200, bbox_inches="tight", pad_inches=0.05,
             facecolor=BG, edgecolor="none")
plt.close(fig2)
print("Saved → poster_figures_v5/fig_intro_gap.png")
