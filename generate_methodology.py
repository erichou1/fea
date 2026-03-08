"""Generate the Engineering Methodology overview figure for the poster.

Seven panels arranged in a 2-column grid:
  Row 1: Dataset Generation | Structural Meshing
  Row 2: FEA Simulation     | Voxelization & Preprocessing
  Row 3: Surrogate Model    | SASTO Optimization
  Row 4: Optimized Structures (full width)

preceded by a top pipeline banner.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import trimesh
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

# ── Colors ────────────────────────────────────────────────────────────────────
NAVY   = "#062B7A"
BLUE   = "#1A4FAA"
LBLUE  = "#D9E5FB"
TEAL   = "#0BA6B7"
GOLD   = "#CFA535"
RED    = "#D7263D"
DARK   = "#0B1736"
WHITE  = "#FFFFFF"
BLACK  = "#111111"
CARD   = "#F7F9FF"
PANEL_BG = "#F4F7FF"

WALL      = "#4477CC"
INTERIOR  = "#E88843"
ROOF      = "#54A24B"
SLAB      = "#D6B48A"
OPT_BLUE  = "#2176AE"

OUT = Path("poster_images_extracted/methodology.png")

# ── Figure layout ─────────────────────────────────────────────────────────────
FW, FH = 20.0, 24.0
fig = plt.figure(figsize=(FW, FH), facecolor=WHITE)

MARGIN  = 0.025
COL_GAP = 0.022
ROW_GAP = 0.018
BANNER_H = 0.080   # pipeline banner
PANEL7_H = 0.130   # full-width bottom panel

# Two equal columns
CW = (1.0 - 2 * MARGIN - COL_GAP) / 2   # ~0.466
# Row heights for panels 1-6 (3 rows of 2)
ROWS_H_total = 1.0 - 2 * MARGIN - BANNER_H - ROW_GAP - PANEL7_H - 3 * ROW_GAP
ROW_H = ROWS_H_total / 3   # each of 3 rows

# Column x positions
CX1 = MARGIN
CX2 = MARGIN + CW + COL_GAP

# Row y positions (bottom of each row, from bottom)
RY3 = MARGIN + PANEL7_H + ROW_GAP               # bottom row of 2-col section
RY2 = RY3 + ROW_H + ROW_GAP
RY1 = RY2 + ROW_H + ROW_GAP
BANNER_Y = RY1 + ROW_H + ROW_GAP

# Panel 7 y
P7_Y = MARGIN
P7_H = PANEL7_H


# ── Helper utilities ──────────────────────────────────────────────────────────

def card(fig, x, y, w, h, fc=PANEL_BG, ec=NAVY, lw=2.0, radius=0.012, zo=1):
    fig.add_artist(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        transform=fig.transFigure, clip_on=False, zorder=zo,
    ))


def header_bar(fig, x, y, w, label, icon_char="",
               fc=NAVY, tc=WHITE, fontsize=13.5, radius=0.012, zo=2):
    HEADER_FRAC = 0.20   # header is 20% of card height (in axes units — we compute in fig)
    BAR_H = 0.032
    # top-rounded bar
    fig.add_artist(FancyBboxPatch(
        (x, y - BAR_H), w, BAR_H,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=fc, linewidth=0,
        transform=fig.transFigure, clip_on=False, zorder=zo,
    ))
    # cover bottom corners of bar (square them off)
    fig.add_artist(mpatches.Rectangle(
        (x, y - BAR_H), w, BAR_H * 0.5,
        facecolor=fc, edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=zo,
    ))
    fig.text(x + w / 2, y - BAR_H * 0.50, label,
             ha="center", va="center", color=tc,
             fontsize=fontsize, fontweight="bold",
             transform=fig.transFigure, zorder=zo + 1)
    if icon_char:
        fig.text(x + w - 0.012, y - BAR_H * 0.50, icon_char,
                 ha="right", va="center", color=WHITE,
                 fontsize=fontsize + 1,
                 transform=fig.transFigure, zorder=zo + 1)


def fig_text(x, y, s, **kw):
    kw.setdefault("transform", fig.transFigure)
    return fig.text(x, y, s, **kw)


def fig_arrow(x0, y0, x1, y1, color=DARK, lw=2.0, scale=14, zo=10):
    fig.add_artist(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="Simple,head_length=0.7,head_width=0.7,tail_width=0.30",
        mutation_scale=scale,
        facecolor=color, edgecolor=color,
        linewidth=lw,
        transform=fig.transFigure, clip_on=False, zorder=zo,
    ))


def mini_tag(ax, x, y, w, h, label, fc="#E8F0FE", ec=BLUE, fontsize=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                facecolor=fc, edgecolor=ec, linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fontsize, color=DARK, fontweight="bold")


def bullet_block(ax, x, y, items, fontsize=8.5, step=0.13, color=DARK):
    for i, txt in enumerate(items):
        ax.text(x, y - i * step, f"• {txt}",
                ha="left", va="top", fontsize=fontsize, color=color)


def ax_in_card(card_x, card_y, card_w, card_h, bar_h_frac, inner_rect,
               projection=None):
    """Create an axes inside a figure-coord card.
    inner_rect = [left, bottom, width, height] as fractions of card interior
    (interior = card minus header bar).
    bar_h_frac: fraction of card height taken by header bar.
    """
    ix, iy, iw, ih = inner_rect
    # Interior region in figure coords
    int_x = card_x
    int_y = card_y
    int_w = card_w
    int_h = card_h * (1.0 - bar_h_frac)

    ax_x = int_x + ix * int_w
    ax_y = int_y + iy * int_h
    ax_w = iw * int_w
    ax_h = ih * int_h
    if projection:
        return fig.add_axes([ax_x, ax_y, ax_w, ax_h], projection=projection)
    return fig.add_axes([ax_x, ax_y, ax_w, ax_h])


BAR_FRAC = 0.032 / ROW_H   # bar height as fraction of card height

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE BANNER
# ══════════════════════════════════════════════════════════════════════════════
card(fig, MARGIN, BANNER_Y, 1.0 - 2 * MARGIN, BANNER_H - 0.005,
     fc=LBLUE, ec=NAVY, lw=2.5, radius=0.016)

fig_text(0.5, BANNER_Y + (BANNER_H - 0.005) * 0.82,
         "ENGINEERING METHODOLOGY",
         ha="center", va="center", fontsize=19, fontweight="bold", color=NAVY)

# Pipeline steps
steps = ["Dataset\nGeneration", "Geometric\nModeling", "FEA\nSimulation",
         "Voxelization &\nPreprocessing", "Surrogate\nTraining",
         "Topology\nOptimization", "Structural\nValidation"]
step_colors = [NAVY, BLUE, "#1565C0", TEAL, "#6A0DAD", "#A3111A", GOLD]
N = len(steps)
_bw = 0.090; _bh = 0.040; _gap = (1.0 - 2 * MARGIN - N * _bw) / (N - 1)
_by = BANNER_Y + (BANNER_H - 0.005) * 0.22
for i, (s, sc) in enumerate(zip(steps, step_colors)):
    _bx = MARGIN + i * (_bw + _gap)
    fig.add_artist(FancyBboxPatch(
        (_bx, _by - _bh / 2), _bw, _bh,
        boxstyle="round,pad=0,rounding_size=0.008",
        facecolor=sc, edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=3,
    ))
    fig_text(_bx + _bw / 2, _by, s,
             ha="center", va="center", fontsize=8.2,
             fontweight="bold", color=WHITE, zorder=4,
             linespacing=1.2)
    if i < N - 1:
        ax_left = _bx + _bw
        ax_right = _bx + _bw + _gap
        fig_arrow(ax_left + 0.004, _by, ax_right - 0.004, _by,
                  color=DARK, lw=1.6, scale=10)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL BUILDER helper
# ══════════════════════════════════════════════════════════════════════════════
BAR_H_FIG = 0.032

def make_panel(cx, cy, cw, ch, title, icon=""):
    card(fig, cx, cy, cw, ch)
    header_bar(fig, cx, cy + ch, cw, title, icon_char=icon)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — Dataset Generation
# ══════════════════════════════════════════════════════════════════════════════
P1X, P1Y, P1W, P1H = CX1, RY1, CW, ROW_H
make_panel(P1X, P1Y, P1W, P1H, "\u2460 Dataset Generation", "")

# Three sub-images inside
_sub_w = 0.24; _sub_h = 0.70; _sub_gap = 0.04
_inner_xpad = 0.045; _inner_ypad = 0.04
_inner_top = P1Y + P1H - 0.01  # just below header

def _ax_sub(col, row_frac_y=0.55, sub_h_frac=0.60):
    """Place a sub-axes for panel 1 content."""
    avail_w = P1W - 2 * _inner_xpad
    sw = avail_w * 0.28
    gap = (avail_w - 3 * sw) / 2
    ax_x = P1X + _inner_xpad + col * (sw + gap)
    inner_h = P1H - BAR_H_FIG - _inner_ypad * 2
    ax_y = P1Y + _inner_ypad + inner_h * (1 - row_frac_y - sub_h_frac / 2 + 0.12)
    ax_h = inner_h * sub_h_frac
    return fig.add_axes([ax_x, ax_y, sw, ax_h])


def draw_wireframe_house(ax):
    """Sketch-style 3-D wireframe of a house."""
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.3, 1.6); ax.set_aspect("equal")
    ax.axis("off")
    # floor rect
    fp = np.array([[-1, 0], [1, 0], [1.5, 0.5], [-0.5, 0.5], [-1, 0]])
    ax.plot(fp[:, 0], fp[:, 1], color=BLUE, lw=1.5)
    # walls
    wall_h = 0.9
    corners_bot = [(-1, 0), (1, 0), (1.5, 0.5), (-0.5, 0.5)]
    corners_top = [(x, y + wall_h) for x, y in corners_bot]
    for (x0, y0), (x1, y1) in zip(corners_bot, corners_top):
        ax.plot([x0, x1], [y0, y1], color=BLUE, lw=1.5)
    top_rect = corners_top + [corners_top[0]]
    xs = [p[0] for p in top_rect]; ys = [p[1] for p in top_rect]
    ax.plot(xs, ys, color=BLUE, lw=1.5)
    # roof
    cx = np.mean([p[0] for p in corners_top])
    cy = max(p[1] for p in corners_top) + 0.65
    for x, y in corners_top:
        ax.plot([x, cx], [y, cy], color=TEAL, lw=1.8)
    ax.set_title("Wireframe", fontsize=8.5, color=DARK, pad=2, fontweight="bold")


def draw_extruded_house(ax):
    """Simple filled solid house geometry."""
    ax.set_xlim(-0.2, 1.4); ax.set_ylim(-0.1, 1.4); ax.set_aspect("equal")
    ax.axis("off")
    # wall box
    b = mpatches.FancyBboxPatch((0.0, 0.05), 1.0, 0.75,
                                 boxstyle="square,pad=0",
                                 facecolor=WALL + "88", edgecolor=WALL, lw=1.5)
    ax.add_patch(b)
    # perspective top
    from matplotlib.patches import Polygon as MPoly
    side = MPoly([[1.0, 0.05], [1.3, 0.22], [1.3, 0.97], [1.0, 0.80]],
                 closed=True, facecolor=WALL + "55", edgecolor=WALL, lw=1.2)
    ax.add_patch(side)
    top = MPoly([[0.0, 0.80], [0.30, 0.97], [1.3, 0.97], [1.0, 0.80]],
                closed=True, facecolor=WALL + "33", edgecolor=WALL, lw=1.2)
    ax.add_patch(top)
    # roof
    roof = MPoly([[0.0, 0.80], [0.5, 1.30], [1.0, 0.80]],
                 closed=True, facecolor=ROOF + "99", edgecolor=ROOF, lw=1.5)
    ax.add_patch(roof)
    ax.set_title("Volumetric", fontsize=8.5, color=DARK, pad=2, fontweight="bold")


def draw_labeled_parts(ax):
    """Colored labeled structural parts."""
    ax.set_xlim(-0.1, 1.3); ax.set_ylim(-0.1, 1.35); ax.set_aspect("equal")
    ax.axis("off")
    from matplotlib.patches import Polygon as MPoly
    # slab
    ax.add_patch(mpatches.FancyBboxPatch((0, 0), 1.0, 0.15,
                 boxstyle="square,pad=0", facecolor=SLAB, edgecolor=DARK, lw=0.8))
    # walls
    ax.add_patch(mpatches.FancyBboxPatch((0, 0.15), 1.0, 0.60,
                 boxstyle="square,pad=0", facecolor=WALL, edgecolor=DARK, lw=0.8))
    # interior
    ax.add_patch(mpatches.FancyBboxPatch((0.35, 0.15), 0.30, 0.60,
                 boxstyle="square,pad=0", facecolor=INTERIOR, edgecolor=DARK, lw=0.8))
    # roof
    ax.add_patch(MPoly([[0, 0.75], [0.5, 1.25], [1.0, 0.75]],
                       closed=True, facecolor=ROOF, edgecolor=DARK, lw=0.8))
    # labels
    for txt, xy, c in [("Floor", (0.5, 0.075), WHITE),
                        ("Walls", (0.18, 0.45), WHITE),
                        ("Interior", (0.50, 0.45), WHITE),
                        ("Roof", (0.50, 0.95), WHITE)]:
        ax.text(xy[0], xy[1], txt, ha="center", va="center",
                fontsize=7.0, color=c, fontweight="bold")
    ax.set_title("Labeled Parts", fontsize=8.5, color=DARK, pad=2, fontweight="bold")


for col, fn in enumerate([draw_wireframe_house, draw_extruded_house, draw_labeled_parts]):
    ax = _ax_sub(col)
    fn(ax)
    # arrow between sub-panels
    if col < 2:
        _avail_w = P1W - 2 * _inner_xpad
        _sw = _avail_w * 0.28
        _gap = (_avail_w - 3 * _sw) / 2
        _ax_r = P1X + _inner_xpad + col * (_sw + _gap) + _sw
        _ax_l = _ax_r + _gap
        _ay = P1Y + P1H * 0.42
        fig_arrow(_ax_r + 0.004, _ay, _ax_l - 0.004, _ay, color=DARK, lw=1.6, scale=10)

# Caption
fig_text(P1X + P1W / 2, P1Y + 0.025,
         "Residential building wireframes are extruded into\nvolumetric structural solids with labeled components.",
         ha="center", va="center", fontsize=8.0, color=DARK, fontstyle="italic",
         linespacing=1.4)
# Bullet tags
_bty = P1Y + P1H * 0.085
for i, txt in enumerate(["Input: 3DWire wireframes", "Output: volumetric solids",
                          "Labeled: walls / roof / floor"]):
    fig_text(P1X + 0.014, _bty - i * 0.016, f"• {txt}",
             ha="left", va="center", fontsize=7.5, color=BLUE)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — Structural Meshing
# ══════════════════════════════════════════════════════════════════════════════
P2X, P2Y, P2W, P2H = CX2, RY1, CW, ROW_H
make_panel(P2X, P2Y, P2W, P2H, "\u2461 Structural Meshing", "")


def draw_tet_mesh_pair(card_x, card_y, card_w, card_h):
    """Left: simple house silhouette. Right: triangulated mesh."""
    avail_w = card_w - 0.04
    ax_h_fig = card_h * 0.52
    ax_w = avail_w * 0.42
    ay = card_y + card_h * 0.28

    # Left: solid house
    axL = fig.add_axes([card_x + 0.012, ay, ax_w, ax_h_fig])
    axL.set_xlim(0, 1); axL.set_ylim(0, 1.2); axL.set_aspect("equal"); axL.axis("off")
    axL.add_patch(mpatches.Rectangle((0.05, 0.05), 0.90, 0.65,
                                      facecolor=WALL + "88", edgecolor=WALL, lw=1.5))
    from matplotlib.patches import Polygon as MPoly
    axL.add_patch(MPoly([[0.05, 0.70], [0.50, 1.15], [0.95, 0.70]],
                        closed=True, facecolor=ROOF + "99", edgecolor=ROOF, lw=1.5))
    axL.set_title("House Geometry", fontsize=8.5, color=DARK, pad=2, fontweight="bold")

    # Arrow
    arr_x = card_x + 0.012 + ax_w + 0.008
    fig_arrow(arr_x, ay + ax_h_fig * 0.5,
              arr_x + avail_w * 0.08, ay + ax_h_fig * 0.5,
              color=DARK, lw=1.6, scale=10)

    # Right: mesh (random triangle network)
    axR = fig.add_axes([card_x + 0.012 + ax_w + avail_w * 0.12,
                        ay, ax_w, ax_h_fig])
    axR.set_xlim(0, 1); axR.set_ylim(0, 1); axR.axis("off")
    rng = np.random.default_rng(42)
    # house-shaped region: triangulate it
    pts_x = np.concatenate([rng.uniform(0.05, 0.95, 55), [0.5]])
    pts_y = np.concatenate([rng.uniform(0.05, 0.75, 55), [0.95]])
    # mask to rough house shape
    mask = (pts_y < 0.75) | ((np.abs(pts_x - 0.5) < (0.95 - pts_y) * 0.55))
    pts_x, pts_y = pts_x[mask], pts_y[mask]
    from scipy.spatial import Delaunay
    pts = np.column_stack([pts_x, pts_y])
    try:
        tri = Delaunay(pts)
        colors = plt.get_cmap("Blues")(np.linspace(0.3, 0.85, len(tri.simplices)))
        from matplotlib.collections import PolyCollection
        polys = pts[tri.simplices]
        pc = PolyCollection(polys, facecolors=colors, edgecolors=DARK, linewidths=0.5)
        axR.add_collection(pc)
    except Exception:
        pass
    axR.set_title("Tet Mesh", fontsize=8.5, color=DARK, pad=2, fontweight="bold")


draw_tet_mesh_pair(P2X, P2Y, P2W, P2H)

fig_text(P2X + P2W / 2, P2Y + 0.025,
         "Volumetric geometries converted to tetrahedral\nmeshes for finite element simulation.",
         ha="center", va="center", fontsize=8.0, color=DARK, fontstyle="italic",
         linespacing=1.4)
for i, txt in enumerate(["Meshing: Gmsh", "50k–200k elements per building",
                          "Refinement near wall junctions"]):
    fig_text(P2X + 0.014, P2Y + P2H * 0.085 - i * 0.016, f"• {txt}",
             ha="left", va="center", fontsize=7.5, color=BLUE)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — FEA Simulation
# ══════════════════════════════════════════════════════════════════════════════
P3X, P3Y, P3W, P3H = CX1, RY2, CW, ROW_H
make_panel(P3X, P3Y, P3W, P3H, "\u2462 Finite Element Simulation", "")


def render_stress_bar(ax, x0, y0, w, h):
    """Horizontal stress colorbar annotation."""
    from matplotlib.colorbar import ColorbarBase
    cax = fig.add_axes([x0, y0, w, h])
    cb = plt.colorbar(
        plt.cm.ScalarMappable(mcolors.Normalize(0, 1), cmap="jet"),
        cax=cax, orientation="horizontal",
    )
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Low", "Med", "High"])
    cb.ax.tick_params(labelsize=7.5, size=0, colors=DARK)
    cb.outline.set_edgecolor(DARK); cb.outline.set_linewidth(0.7)
    cax.set_title("Von Mises Stress", fontsize=7.5, color=DARK, pad=2)


def draw_fea_panel(card_x, card_y, card_w, card_h):
    avail_w = card_w - 0.04
    ax_h_fig = card_h * 0.44
    ax_w = avail_w * 0.38
    ay = card_y + card_h * 0.35

    # Mesh panel
    axM = fig.add_axes([card_x + 0.012, ay, ax_w, ax_h_fig])
    axM.set_xlim(0, 1); axM.set_ylim(0, 1); axM.axis("off")
    rng = np.random.default_rng(7)
    pts = rng.uniform(0.05, 0.95, (50, 2))
    from scipy.spatial import Delaunay
    from matplotlib.collections import PolyCollection
    try:
        tri = Delaunay(pts)
        polys = pts[tri.simplices]
        pc = PolyCollection(polys, facecolors="#C8D8F5", edgecolors=BLUE, linewidths=0.6)
        axM.add_collection(pc)
    except Exception:
        pass
    axM.set_title("FE Mesh", fontsize=8.0, color=DARK, pad=2, fontweight="bold")

    # Arrow + Ku=f
    mid_x = card_x + 0.012 + ax_w + avail_w * 0.06
    mid_y = ay + ax_h_fig * 0.50
    fig_arrow(card_x + 0.012 + ax_w + 0.004, mid_y,
              mid_x - 0.002, mid_y, color=DARK, lw=1.6, scale=10)
    fig.text(mid_x + 0.002, mid_y + 0.008, r"$\mathbf{K}\mathbf{u}=\mathbf{f}$",
             ha="left", va="center", fontsize=13, color=NAVY, fontweight="bold",
             transform=fig.transFigure)
    fig_arrow(mid_x + avail_w * 0.16, mid_y,
              mid_x + avail_w * 0.20, mid_y, color=DARK, lw=1.6, scale=10)

    # Stress heatmap
    axS = fig.add_axes([card_x + 0.012 + ax_w + avail_w * 0.24,
                        ay, ax_w, ax_h_fig])
    rng2 = np.random.default_rng(9)
    Z = rng2.random((20, 20))
    # smooth it a bit
    from scipy.ndimage import gaussian_filter
    Z = gaussian_filter(Z, sigma=2.5)
    axS.imshow(Z, cmap="jet", origin="lower", aspect="auto",
               extent=[0, 1, 0, 1], interpolation="bilinear")
    axS.axis("off")
    axS.set_title("Stress Field", fontsize=8.0, color=DARK, pad=2, fontweight="bold")

    # Colorbar
    render_stress_bar(ax=None,
                      x0=card_x + 0.012 + ax_w + avail_w * 0.24,
                      y0=card_y + card_h * 0.20,
                      w=ax_w, h=0.018)


draw_fea_panel(P3X, P3Y, P3W, P3H)

fig_text(P3X + P3W / 2, P3Y + 0.025,
         "Linear elastic FEA computes stress, displacement,\nand compliance for each structural design.",
         ha="center", va="center", fontsize=8.0, color=DARK, fontstyle="italic",
         linespacing=1.4)
for i, txt in enumerate(["Peak von Mises stress", "Maximum displacement",
                          "Structural compliance"]):
    fig_text(P3X + P3W / 2 - 0.02, P3Y + P3H * 0.085 - i * 0.016, f"• {txt}",
             ha="left", va="center", fontsize=7.5, color=BLUE)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 4 — Voxelization & Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
P4X, P4Y, P4W, P4H = CX2, RY2, CW, ROW_H
make_panel(P4X, P4Y, P4W, P4H, "\u2463 Voxelization & Preprocessing", "")


def draw_vox_pipeline(card_x, card_y, card_w, card_h):
    avail_w = card_w - 0.04
    ax_h_fig = card_h * 0.48
    ax_w = avail_w * 0.28
    ay = card_y + card_h * 0.33

    stages = ["Tet Mesh", "Voxel Grid", "Labeled\nChannels"]
    for col, lbl in enumerate(stages):
        axV = fig.add_axes([card_x + 0.012 + col * (ax_w + avail_w * 0.07),
                            ay, ax_w, ax_h_fig])
        axV.set_xlim(0, 1); axV.set_ylim(0, 1); axV.axis("off")
        if col == 0:
            # rough triangulated shape
            rng = np.random.default_rng(3)
            pts = rng.uniform(0.1, 0.9, (30, 2))
            from scipy.spatial import Delaunay
            from matplotlib.collections import PolyCollection
            try:
                tri = Delaunay(pts)
                polys = pts[tri.simplices]
                pc = PolyCollection(polys, facecolors="#D0DCF5",
                                    edgecolors=BLUE, linewidths=0.5)
                axV.add_collection(pc)
            except Exception:
                pass
        elif col == 1:
            # plain voxel grid
            n = 8
            for ix in range(n):
                for iz in range(n):
                    fc = "#BBCFEE" if (ix + iz) % 2 == 0 else "#8BAEE0"
                    axV.add_patch(mpatches.Rectangle(
                        (ix / n, iz / n), 1 / n - 0.01, 1 / n - 0.01,
                        facecolor=fc, edgecolor=DARK, lw=0.4))
        else:
            # colored labeled voxels
            cmap_data = {0: WALL, 1: INTERIOR, 2: ROOF, 3: SLAB}
            rng = np.random.default_rng(8)
            n = 8
            labels = rng.integers(0, 4, (n, n))
            for ix in range(n):
                for iz in range(n):
                    fc = cmap_data[labels[ix, iz]] + "CC"
                    axV.add_patch(mpatches.Rectangle(
                        (ix / n, iz / n), 1 / n - 0.01, 1 / n - 0.01,
                        facecolor=fc, edgecolor=DARK, lw=0.3))
        axV.set_title(lbl, fontsize=7.5, color=DARK, pad=2, fontweight="bold",
                      multialignment="center")
        if col < 2:
            arr_x = card_x + 0.012 + col * (ax_w + avail_w * 0.07) + ax_w + 0.003
            fig_arrow(arr_x, ay + ax_h_fig * 0.50,
                      arr_x + avail_w * 0.055, ay + ax_h_fig * 0.50,
                      color=DARK, lw=1.5, scale=9)


draw_vox_pipeline(P4X, P4Y, P4W, P4H)

fig_text(P4X + P4W / 2, P4Y + 0.025,
         "Structural meshes voxelized onto 3-D grids;\npart labels encoded as input channels.",
         ha="center", va="center", fontsize=8.0, color=DARK, fontstyle="italic",
         linespacing=1.4)
for i, txt in enumerate(["128³ voxel resolution", "7 input channels",
                          "Part labels one-hot encoded"]):
    fig_text(P4X + 0.014, P4Y + P4H * 0.085 - i * 0.016, f"• {txt}",
             ha="left", va="center", fontsize=7.5, color=BLUE)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 5 — Surrogate Model Training
# ══════════════════════════════════════════════════════════════════════════════
P5X, P5Y, P5W, P5H = CX1, RY3, CW, ROW_H
make_panel(P5X, P5Y, P5W, P5H, "\u2464 Surrogate Model Training", "")


def draw_nn_arch(card_x, card_y, card_w, card_h):
    ax = fig.add_axes([card_x + 0.012, card_y + card_h * 0.14,
                       card_w - 0.024, card_h * 0.70])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    layers = [
        ("128³\nVoxel Input", "#2176AE", 0.08),
        ("3D Conv\nx3", "#1A5276", 0.08),
        ("Residual\nBlocks", "#6A0DAD", 0.08),
        ("Global\nAvg Pool", "#0E6655", 0.07),
        ("MLP\nHead", DARK, 0.065),
        ("σ / δ / C\nOutput", RED, 0.065),
    ]
    n = len(layers)
    xs = np.linspace(0.05, 0.95, n)
    yw = 0.44; yc = 0.50

    for i, (lbl, fc, wr) in enumerate(layers):
        x = xs[i]
        w = wr; h = 0.45
        ax.add_patch(FancyBboxPatch((x - w / 2, yc - h / 2), w, h,
                                    boxstyle="round,pad=0.01,rounding_size=0.03",
                                    facecolor=fc, edgecolor=WHITE, lw=1.4))
        ax.text(x, yc, lbl, ha="center", va="center",
                fontsize=7.5, color=WHITE, fontweight="bold", linespacing=1.2,
                multialignment="center")
        if i < n - 1:
            ax.annotate("", xy=(xs[i + 1] - layers[i + 1][2] / 2 - 0.005, yc),
                        xytext=(x + w / 2 + 0.005, yc),
                        arrowprops=dict(arrowstyle="-|>", color=DARK,
                                        lw=1.4, mutation_scale=10))
    # Ensemble label
    ax.text(0.50, 0.06, "× 5 independent models  (deep ensemble)",
            ha="center", va="center", fontsize=8.0, color=DARK, fontstyle="italic")


draw_nn_arch(P5X, P5Y, P5W, P5H)

fig_text(P5X + P5W / 2, P5Y + 0.025,
         "Deep ensemble of 3D CNNs trained to predict\nstress, displacement, and compliance.",
         ha="center", va="center", fontsize=8.0, color=DARK, fontstyle="italic",
         linespacing=1.4)
for i, txt in enumerate(["5 independent CNN models", "Trained on 11,178 FEA simulations",
                          "Uncertainty via ensemble variance"]):
    fig_text(P5X + 0.014, P5Y + P5H * 0.085 - i * 0.016, f"• {txt}",
             ha="left", va="center", fontsize=7.5, color=BLUE)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 6 — SASTO Optimization
# ══════════════════════════════════════════════════════════════════════════════
P6X, P6Y, P6W, P6H = CX2, RY3, CW, ROW_H
make_panel(P6X, P6Y, P6W, P6H, "\u2465 SASTO Optimization", "")


def draw_opt_flow(card_x, card_y, card_w, card_h):
    ax = fig.add_axes([card_x + 0.012, card_y + card_h * 0.13,
                       card_w - 0.024, card_h * 0.74])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    steps_flow = [
        ("Initial\nStructure",     "#2176AE"),
        ("Surrogate\nEvaluation",  "#6A0DAD"),
        ("Sensitivity\nAnalysis",  "#1565C0"),
        ("Voxel\nRemoval",         TEAL),
        ("Constraint\nCheck",      GOLD),
        ("Converged?",             RED),
    ]
    n = len(steps_flow)
    bw = 0.130; bh = 0.30; gap = (1.0 - n * bw) / (n - 1)
    yc = 0.62

    for i, (lbl, fc) in enumerate(steps_flow):
        x = i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, yc - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.01,rounding_size=0.04",
                                    facecolor=fc, edgecolor=WHITE, lw=1.2))
        ax.text(x + bw / 2, yc, lbl, ha="center", va="center",
                fontsize=7.0, color=WHITE, fontweight="bold",
                linespacing=1.2, multialignment="center")
        if i < n - 1:
            ax.annotate("", xy=(x + bw + gap - 0.005, yc),
                        xytext=(x + bw + 0.005, yc),
                        arrowprops=dict(arrowstyle="-|>", color=DARK,
                                        lw=1.2, mutation_scale=9))

    # Loop-back arrow from "Converged?" → "Sensitivity Analysis" if not converged
    ax.annotate("",
                xy=(2 * (bw + gap) + bw / 2, yc - bh / 2),
                xytext=((n - 1) * (bw + gap) + bw / 2, yc - bh / 2),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4,
                                mutation_scale=9,
                                connectionstyle="arc3,rad=-0.35"))
    ax.text(0.50, 0.15, "iterate until convergence",
            ha="center", va="center", fontsize=7.5, color=RED, fontstyle="italic")

    # Objective
    ax.text(0.50, 0.90, r"$J(\rho)=w_V\!\frac{V}{V_0}+w_S\!\frac{S}{V_0}+P_{\mathrm{constraint}}$",
            ha="center", va="center", fontsize=11.0, color=NAVY, fontweight="bold")


draw_opt_flow(P6X, P6Y, P6W, P6H)

fig_text(P6X + P6W / 2, P6Y + 0.025,
         "Surrogate-guided topology optimization iteratively\nremoves material while enforcing structural constraints.",
         ha="center", va="center", fontsize=8.0, color=DARK, fontstyle="italic",
         linespacing=1.4)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 7 — Optimized Structures (full width)
# ══════════════════════════════════════════════════════════════════════════════
make_panel(MARGIN, P7_Y, 1.0 - 2 * MARGIN, P7_H, "\u2466 Optimized Structures", "")


def render_ply_mesh(ply_path, ax3d, color_mode="part", elev=22, azim=-55):
    mesh = trimesh.load(str(ply_path), force="mesh", process=False)
    verts = np.array(mesh.vertices, dtype=float)
    faces = np.array(mesh.faces)
    lo, hi = verts.min(0), verts.max(0)
    span_real = hi - lo
    span = span_real.max()
    verts = (verts - (lo + hi) / 2.0) / span
    poly_v = verts[faces]

    n0, n1, n2 = poly_v[:, 0], poly_v[:, 1], poly_v[:, 2]
    normals = np.cross(n1 - n0, n2 - n0)
    mag = np.linalg.norm(normals, axis=1, keepdims=True)
    mag[mag == 0] = 1.0
    normals /= mag
    light = np.array([np.cos(np.radians(35)) * np.cos(np.radians(-35)),
                      np.cos(np.radians(35)) * np.sin(np.radians(-35)),
                      np.sin(np.radians(35))])
    lambert = np.clip(normals @ light, 0, 1)
    centers = poly_v.mean(axis=1)
    z_norm = (centers[:, 2] - centers[:, 2].min()) / (np.ptp(centers[:, 2]) + 1e-9)

    if color_mode == "stress":
        radial = np.sqrt(centers[:, 0] ** 2 + centers[:, 1] ** 2)
        radial /= radial.max() + 1e-9
        stress = np.clip(0.58 * (1 - z_norm) + 0.42 * radial, 0, 1)
        base = plt.get_cmap("jet")(stress)[:, :3]
    else:
        base = np.column_stack([0.25 + 0.2 * z_norm,
                                0.45 + 0.25 * z_norm,
                                0.85 + 0.10 * z_norm])

    ambient = 0.82
    shaded = np.clip(base * (ambient + (1 - ambient) * lambert[:, None]), 0, 1)
    poly = Poly3DCollection(poly_v, zsort="average")
    poly.set_facecolor(shaded); poly.set_edgecolor("none")
    ax3d.add_collection3d(poly)
    pad = 0.06
    ax3d.set_xlim(-0.5 - pad, 0.5 + pad)
    ax3d.set_ylim(-0.5 - pad, 0.5 + pad)
    ax3d.set_zlim(-0.5 - pad, 0.5 + pad)
    ax3d.set_box_aspect([span_real[0], span_real[1], span_real[2]])
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_facecolor(WHITE)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
    ax3d.set_axis_off()


P7_card_x = MARGIN
P7_card_y = P7_Y
P7_card_w = 1.0 - 2 * MARGIN

# Three 3D panels side by side
models = [
    ("figures/screenshot_stls/REF_original_colored.ply",   "part",   "Baseline House"),
    ("figures/screenshot_stls/REF_SASTO_PA_colored.ply",   "stress", "SASTO-PA Optimized"),
    ("figures/screenshot_stls/REF_SASTO_U_colored.ply",    "part",   "SASTO-U Optimized"),
]
_3w = 0.18; _3h_fig = P7_H * 0.72; _3gap = 0.015
_3tot = 3 * _3w + 2 * _3gap
_3x0 = P7_card_x + (P7_card_w - _3tot) / 2
_3y = P7_card_y + P7_H * 0.10

for i, (ply, cmode, lbl) in enumerate(models):
    ax3 = fig.add_axes([_3x0 + i * (_3w + _3gap), _3y, _3w, _3h_fig],
                        projection="3d")
    try:
        render_ply_mesh(ply, ax3, color_mode=cmode)
    except Exception:
        ax3.text(0.5, 0.5, "mesh\nnot found", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=9, color=RED)
    fig_text(_3x0 + i * (_3w + _3gap) + _3w / 2,
             _3y - 0.014, lbl,
             ha="center", va="top", fontsize=9.5, color=DARK,
             fontweight="bold")
    if i < 2:
        arr_x = _3x0 + i * (_3w + _3gap) + _3w + 0.003
        fig_arrow(arr_x, _3y + _3h_fig * 0.50,
                  arr_x + _3gap - 0.006, _3y + _3h_fig * 0.50,
                  color=DARK, lw=1.8, scale=12)

# Stats row
stats = ["Up to 45% material reduction",
         "23–92× faster than classical SIMP",
         "0 structural constraint violations",
         "Maintains printable connectivity"]
_sw_each = (P7_card_w - 0.06) / len(stats)
for i, st in enumerate(stats):
    sx = P7_card_x + 0.03 + i * _sw_each
    fig.add_artist(FancyBboxPatch(
        (sx, P7_card_y + 0.008), _sw_each - 0.010, 0.022,
        boxstyle="round,pad=0,rounding_size=0.005",
        facecolor=NAVY, edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=3,
    ))
    fig_text(sx + (_sw_each - 0.010) / 2, P7_card_y + 0.019,
             f"✓  {st}",
             ha="center", va="center", fontsize=8.2,
             color=WHITE, fontweight="bold", zorder=4)


# ══════════════════════════════════════════════════════════════════════════════
plt.savefig(str(OUT), dpi=200, bbox_inches="tight",
            facecolor=WHITE, edgecolor="none")
plt.close()
print(f"Saved → {OUT}")
