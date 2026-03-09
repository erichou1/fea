"""
generate_intro_house_comparison.py
Renders a 3-column × 2-row before/after house comparison figure for the Introduction.

Row 1: Original uniform-wall houses (all same gray)
Row 2: SASTO part-aware optimized (color-coded, interior walls visibly thinner)

Output: poster_figures_v5/fig_intro_house_comparison.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import numpy as np
from pathlib import Path

OUT_DIR = Path("poster_figures_v5")
OUT_DIR.mkdir(exist_ok=True)
SRC_DIR = Path("figures/screenshot_stls")

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY    = "#062B7A"
GOLD    = "#CFA535"
TEAL    = "#008C9E"
RED     = "#D7263D"
CARD    = "#F7F9FC"
WHITE   = "#FFFFFF"
TXT     = "#0B1736"
SECTION = "#0A3D9A"

# Part colors (matching export_colored_stl.py exactly)
COL_EXT   = np.array([69,  130, 181]) / 255   # steel blue  — exterior wall
COL_INT   = np.array([255, 128,  79]) / 255   # coral       — interior wall
COL_ROOF  = np.array([107, 143,  36]) / 255   # olive green — roof
COL_FLOOR = np.array([112, 128, 143]) / 255   # slate gray  — floor

# View angles: isometric, slightly front-left
ELEV =  22
AZIM = -50

HOUSES = [
    ("REF",   "House A"),
    ("01440", "House B"),
    ("05728", "House C"),
]

# Map → optimized cutaway PLY (cutaway shows interior wall thinning)
OPT_PA = {
    "REF":   "REF_SASTO_PA_cutaway",
    "01440": "01440_optimized_cutaway",
    "05728": "05728_optimized_cutaway",
}
# Use original colored PLY for "before" (cutaway PLY only exists for REF)
ORI_CUT = {
    "REF":   "REF_original_cutaway",
    "01440": "01440_original_colored",
    "05728": "05728_original_colored",
}

# Reductions
REDUCTIONS = {"REF": 27.4, "01440": 23.1, "05728": 29.8}

# ── Rendering helper ──────────────────────────────────────────────────────────

def load_ply(path):
    mesh = trimesh.load(str(path), force="mesh", process=False)
    verts = np.array(mesh.vertices, dtype=float)
    faces = np.array(mesh.faces,    dtype=int)
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        vc = np.array(mesh.visual.vertex_colors, dtype=float)[:, :3] / 255.0
        fc = vc[faces].mean(axis=1)
    else:
        fc = np.tile(np.array([0.55, 0.65, 0.75]), (len(faces), 1))
    return verts, faces, fc


def shade(verts, faces, fc, light=np.array([0.6, 0.3, 0.8])):
    """Lambertian shading applied to per-face colors."""
    poly = verts[faces]
    v0, v1, v2 = poly[:, 0], poly[:, 1], poly[:, 2]
    n = np.cross(v1 - v0, v2 - v0)
    nl = np.linalg.norm(n, axis=1, keepdims=True)
    nl[nl == 0] = 1.0
    n /= nl
    light = light / np.linalg.norm(light)
    intensity = np.clip(n @ light, 0, 1)[:, np.newaxis]
    return np.clip(fc * (0.35 + 0.65 * intensity), 0, 1)


def draw_house(ax, verts, faces, fc_shaded, bbox_orig):
    """Draw mesh into a 3D Axes."""
    # Normalise to unit cube
    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    span = bbox_orig.max() if bbox_orig.max() > 0 else 1.0
    center = (bmin + bmax) / 2
    v = (verts - center) / span

    poly = v[faces]
    col = Poly3DCollection(poly, zsort="average")
    col.set_facecolor(fc_shaded)
    col.set_edgecolor("none")
    col.set_alpha(1.0)
    ax.add_collection3d(col)

    pad = 0.10
    ax.set_xlim(-0.5 - pad, 0.5 + pad)
    ax.set_ylim(-0.5 - pad, 0.5 + pad)
    ax.set_zlim(-0.5 - pad, 0.5 + pad)
    ax.set_axis_off()
    ax.view_init(elev=ELEV, azim=AZIM)
    r = bbox_orig
    ax.set_box_aspect([r[0], r[1], r[2]])


# ── Build figure ──────────────────────────────────────────────────────────────
NCOLS = 3
# Layout: header row + 2 render rows + footer
FIG_W = 18
FIG_H = 13

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=CARD)

# Grid:  row0 = before label strip, row1-2 = before renders
#        row3 = after label strip,  row4-5 = after renders
#        row6 = legend strip
HEADER_H  = 0.07   # fraction of fig height for row/label strips
RENDER_H  = 0.40
STRIP_H   = 0.055
LEG_H     = 0.065

# We'll use add_axes with explicit [left, bottom, width, height] in figure fractions
COL_W  = 1.0 / NCOLS
ROW_H_RENDER = (1.0 - 2 * HEADER_H - 2 * STRIP_H - LEG_H) / 2

# ── Row label backgrounds ─────────────────────────────────────────────────────
# "BEFORE" bar
before_ax = fig.add_axes([0.0, 1.0 - HEADER_H, 1.0, HEADER_H])
before_ax.set_facecolor(RED)
before_ax.set_axis_off()
before_ax.text(0.5, 0.52, "BEFORE OPTIMIZATION  —  Uniform concrete walls, no material efficiency",
               ha="center", va="center", transform=before_ax.transAxes,
               fontsize=14, color=WHITE, fontweight="bold")

# "AFTER" bar
after_top = 1.0 - HEADER_H - ROW_H_RENDER - STRIP_H
after_ax = fig.add_axes([0.0, after_top, 1.0, HEADER_H])
after_ax.set_facecolor(SECTION)
after_ax.set_axis_off()
after_ax.text(0.5, 0.52, "AFTER  —  SASTO Part-Aware Topology Optimization  ·  Walls thinned where structurally safe",
              ha="center", va="center", transform=after_ax.transAxes,
              fontsize=14, color=WHITE, fontweight="bold")

# ── Render all 6 panels ───────────────────────────────────────────────────────
before_bottom = 1.0 - HEADER_H - ROW_H_RENDER
after_bottom  = after_top - HEADER_H - ROW_H_RENDER

for col_i, (hid, hname) in enumerate(HOUSES):
    left = col_i * COL_W

    # ── BEFORE ────────────────────────────────────────────────────────────────
    ply_path = SRC_DIR / f"{ORI_CUT[hid]}.ply"
    verts, faces, fc = load_ply(ply_path)
    bbox_span = verts.max(axis=0) - verts.min(axis=0)
    fc_shaded = shade(verts, faces, fc)

    ax_b = fig.add_axes(
        [left + 0.01, before_bottom, COL_W - 0.02, ROW_H_RENDER],
        projection="3d", facecolor=CARD
    )
    draw_house(ax_b, verts, faces, fc_shaded, bbox_span)

    # House label below
    lbl_b = fig.add_axes([left, before_bottom - STRIP_H, COL_W, STRIP_H])
    lbl_b.set_facecolor("#FDF0F2")
    lbl_b.set_axis_off()
    lbl_b.text(0.5, 0.55, f"{hname}  —  All walls = 156 mm",
               ha="center", va="center", transform=lbl_b.transAxes,
               fontsize=11, color=RED, fontweight="bold")

    # ── AFTER ─────────────────────────────────────────────────────────────────
    ply_path_opt = SRC_DIR / f"{OPT_PA[hid]}.ply"
    verts_o, faces_o, fc_o = load_ply(ply_path_opt)
    bbox_span_o = verts_o.max(axis=0) - verts_o.min(axis=0)
    fc_shaded_o = shade(verts_o, faces_o, fc_o)

    ax_a = fig.add_axes(
        [left + 0.01, after_bottom, COL_W - 0.02, ROW_H_RENDER],
        projection="3d", facecolor=CARD
    )
    draw_house(ax_a, verts_o, faces_o, fc_shaded_o, bbox_span_o)

    # Reduction badge label
    red_pct = REDUCTIONS.get(hid, 0)
    lbl_a = fig.add_axes([left, after_bottom - STRIP_H, COL_W, STRIP_H])
    lbl_a.set_facecolor("#EEF6F0")
    lbl_a.set_axis_off()
    lbl_a.text(0.5, 0.55,
               f"{hname}  —  −{red_pct:.1f}% concrete removed  ·  constraints satisfied",
               ha="center", va="center", transform=lbl_a.transAxes,
               fontsize=11, color="#1A6B3C", fontweight="bold")

    # Vertical divider between columns
    if col_i > 0:
        div = fig.add_axes([left, LEG_H, 0.003, 1.0 - HEADER_H - LEG_H])
        div.set_facecolor(CARD)
        div.set_axis_off()
        div.axvline(0.5, color="#C0CCE0", lw=1.2)

# ── Legend strip ──────────────────────────────────────────────────────────────
leg_ax = fig.add_axes([0.0, 0.0, 1.0, LEG_H])
leg_ax.set_facecolor(NAVY)
leg_ax.set_axis_off()

legend_items = [
    mpatches.Patch(facecolor=COL_EXT,   edgecolor="white", lw=0.8,
                   label="Exterior wall  (load-bearing, kept at 156 mm)"),
    mpatches.Patch(facecolor=COL_INT,   edgecolor="white", lw=0.8,
                   label="Interior wall  (non-structural, thinned to 78 mm)"),
    mpatches.Patch(facecolor=COL_ROOF,  edgecolor="white", lw=0.8,
                   label="Roof  (retained)"),
    mpatches.Patch(facecolor=COL_FLOOR, edgecolor="white", lw=0.8,
                   label="Floor slab  (retained)"),
]
leg_ax.legend(
    handles=legend_items,
    loc="center",
    ncol=4,
    fontsize=11.5,
    frameon=False,
    labelcolor=WHITE,
    handlelength=1.6,
    handleheight=1.1,
    columnspacing=2.0,
    bbox_to_anchor=(0.5, 0.5),
)

out_path = OUT_DIR / "fig_intro_house_comparison.png"
fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0,
            facecolor=CARD, edgecolor="none")
plt.close(fig)
print(f"Saved → {out_path}")
