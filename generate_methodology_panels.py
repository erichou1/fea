"""Generate each methodology panel as an individual same-size PNG figure.

Outputs (poster_images_extracted/panels/):
  panel_01_dataset.png
  panel_02_meshing.png
  panel_03_fea.png
  panel_04_voxelization.png
  panel_05_surrogate.png
  panel_06_optimization.png
  panel_07_results.png
  panel_00_banner.png      ← pipeline ribbon only
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
from pathlib import Path

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("poster_images_extracted/panels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared palette ────────────────────────────────────────────────────────────
NAVY     = "#062B7A"
BLUE     = "#1A4FAA"
LBLUE    = "#D9E5FB"
TEAL     = "#0BA6B7"
GOLD     = "#CFA535"
RED      = "#D7263D"
DARK     = "#0B1736"
WHITE    = "#FFFFFF"
PANEL_BG = "#F4F7FF"
WALL     = "#4477CC"
INTERIOR = "#E88843"
ROOF     = "#54A24B"
SLAB     = "#D6B48A"

# ── Uniform figure size ───────────────────────────────────────────────────────
PW, PH = 10.0, 6.5   # inches — every panel uses this
DPI = 220

HEADER_H   = 0.10    # fraction of figure height for the navy title bar
MARGIN     = 0.03    # figure-coord margin around card
BODY_PAD   = 0.04    # extra inside padding below header
CAPTION_H  = 0.13    # fraction reserved at bottom for caption + bullets
BAR_FRAC   = HEADER_H


# ═════════════════════════════════════════════════════════════════════════════
#  LOW-LEVEL PRIMITIVES  (operate in figure-transform coordinates 0–1)
# ═════════════════════════════════════════════════════════════════════════════

def _card(fig, x, y, w, h, fc=PANEL_BG, ec=NAVY, lw=2.2, r=0.018, zo=1):
    fig.add_artist(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        transform=fig.transFigure, clip_on=False, zorder=zo,
    ))


def _header(fig, x, y, w, h, title, fc=NAVY, zo=2):
    """Navy header bar, top of card: y is BOTTOM of bar, y+h is top."""
    fig.add_artist(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size=0.018",
        facecolor=fc, edgecolor=fc, linewidth=0,
        transform=fig.transFigure, clip_on=False, zorder=zo,
    ))
    # square off the bottom half so it merges with card body
    fig.add_artist(mpatches.Rectangle(
        (x, y), w, h * 0.45,
        facecolor=fc, edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=zo,
    ))
    fig.text(x + w / 2, y + h * 0.50, title,
             ha="center", va="center", fontsize=16, fontweight="bold",
             color=WHITE, transform=fig.transFigure, zorder=zo + 1)


def _farrow(fig, x0, y0, x1, y1, color=DARK, lw=1.8, scale=12, zo=12):
    fig.add_artist(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="Simple,head_length=0.7,head_width=0.7,tail_width=0.28",
        mutation_scale=scale, facecolor=color, edgecolor=color,
        linewidth=lw, transform=fig.transFigure, clip_on=False, zorder=zo,
    ))


def _caption(fig, cx, y_bot, text, bullets):
    """Italic caption + bullet points at the bottom of a panel."""
    fig.text(cx, y_bot + 0.070, text,
             ha="center", va="top", color=DARK,
             fontsize=9.0, fontstyle="italic", linespacing=1.4,
             transform=fig.transFigure)
    for i, b in enumerate(bullets):
        fig.text(cx - 0.30, y_bot + 0.040 - i * 0.022, f"• {b}",
                 ha="left", va="center", color=BLUE,
                 fontsize=8.5, transform=fig.transFigure)


def _make_figure():
    """Return a blank figure + body rect (x, y, w, h) in figure fractions."""
    fig = plt.figure(figsize=(PW, PH), facecolor=WHITE)
    cx, cy, cw, ch = MARGIN, MARGIN, 1 - 2 * MARGIN, 1 - 2 * MARGIN
    _card(fig, cx, cy, cw, ch)
    hx, hy, hw, hh = cx, cy + ch - HEADER_H, cw, HEADER_H
    # body area: below header, above caption zone
    bx = cx + BODY_PAD
    by = cy + CAPTION_H
    bw = cw - 2 * BODY_PAD
    bh = ch - HEADER_H - CAPTION_H - BODY_PAD
    return fig, (cx, cy, cw, ch), (hx, hy, hw, hh), (bx, by, bw, bh)


def _finalize(fig, title, caption, bullets, outname):
    cx, cy, cw, ch = MARGIN, MARGIN, 1 - 2 * MARGIN, 1 - 2 * MARGIN
    hx, hy, hw, hh = cx, cy + ch - HEADER_H, cw, HEADER_H
    _header(fig, hx, hy, hw, hh, title)
    _caption(fig, 0.5, cy + 0.005, caption, bullets)
    path = OUT_DIR / outname
    plt.savefig(str(path), dpi=DPI, bbox_inches="tight",
                facecolor=WHITE, edgecolor="none")
    plt.close(fig)
    print(f"  Saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 0 — Pipeline Banner
# ═════════════════════════════════════════════════════════════════════════════

def panel_banner():
    fig = plt.figure(figsize=(PW, 2.8), facecolor=WHITE)
    cx, cy, cw, ch = MARGIN, MARGIN, 1 - 2 * MARGIN, 1 - 2 * MARGIN
    _card(fig, cx, cy, cw, ch, fc=LBLUE, ec=NAVY, lw=2.5)
    fig.text(0.5, cy + ch * 0.82, "ENGINEERING METHODOLOGY",
             ha="center", va="center", fontsize=22, fontweight="bold",
             color=NAVY, transform=fig.transFigure)

    steps = ["Dataset\nGeneration", "Geometric\nModeling", "FEA\nSimulation",
             "Voxelization &\nPreprocessing", "Surrogate\nTraining",
             "Topology\nOptimization", "Structural\nValidation"]
    colors = [NAVY, BLUE, "#1565C0", TEAL, "#6A0DAD", "#A3111A", GOLD]
    N = len(steps)
    bw = 0.100; bh = 0.28; gap = (cw - N * bw) / (N - 1)
    by_c = cy + ch * 0.28
    for i, (s, sc) in enumerate(zip(steps, colors)):
        bx = cx + i * (bw + gap)
        fig.add_artist(FancyBboxPatch(
            (bx, by_c - bh / 2), bw, bh,
            boxstyle="round,pad=0,rounding_size=0.010",
            facecolor=sc, edgecolor="none",
            transform=fig.transFigure, clip_on=False, zorder=3,
        ))
        fig.text(bx + bw / 2, by_c, s,
                 ha="center", va="center", fontsize=9, fontweight="bold",
                 color=WHITE, linespacing=1.2, transform=fig.transFigure, zorder=4)
        if i < N - 1:
            ax0 = bx + bw + 0.004
            ax1 = bx + bw + gap - 0.004
            _farrow(fig, ax0, by_c, ax1, by_c, color=DARK, lw=1.5, scale=9)

    plt.savefig(str(OUT_DIR / "panel_00_banner.png"), dpi=DPI,
                bbox_inches="tight", facecolor=WHITE, edgecolor="none")
    plt.close(fig)
    print(f"  Saved → {OUT_DIR / 'panel_00_banner.png'}")


# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 1 — Dataset Generation
# ═════════════════════════════════════════════════════════════════════════════

def panel_dataset():
    fig, card_rect, hdr, body = _make_figure()
    bx, by, bw, bh = body

    # Three sub-axes + 2 arrows
    n = 3
    pad = 0.025
    sw = (bw - (n + 1) * pad) / n
    sh = bh * 0.72
    sy = by + bh * 0.14

    def sub_ax(col):
        return fig.add_axes([bx + pad + col * (sw + pad), sy, sw, sh])

    # ① wireframe
    ax = sub_ax(0)
    ax.set_xlim(-1.3, 1.8); ax.set_ylim(-0.4, 1.7); ax.axis("off")
    fp = np.array([[-1, 0], [1, 0], [1.5, 0.5], [-0.5, 0.5], [-1, 0]])
    ax.plot(fp[:, 0], fp[:, 1], color=BLUE, lw=2.0)
    wh = 0.9
    cb = [(-1, 0), (1, 0), (1.5, 0.5), (-0.5, 0.5)]
    ct = [(x, y + wh) for x, y in cb]
    for (x0, y0), (x1, y1) in zip(cb, ct):
        ax.plot([x0, x1], [y0, y1], color=BLUE, lw=2.0)
    tr = ct + [ct[0]]
    ax.plot([p[0] for p in tr], [p[1] for p in tr], color=BLUE, lw=2.0)
    rcx = np.mean([p[0] for p in ct])
    rcy = max(p[1] for p in ct) + 0.70
    for x, y in ct:
        ax.plot([x, rcx], [y, rcy], color=TEAL, lw=2.2)
    ax.set_title("① Wireframe", fontsize=11, color=DARK, pad=3, fontweight="bold")

    # ② extruded
    ax = sub_ax(1)
    ax.set_xlim(-0.2, 1.5); ax.set_ylim(-0.1, 1.5); ax.axis("off")
    from matplotlib.patches import Polygon as MPoly
    ax.add_patch(mpatches.FancyBboxPatch((0, 0.05), 1.0, 0.75,
                 boxstyle="square,pad=0", facecolor=WALL + "88", edgecolor=WALL, lw=1.8))
    ax.add_patch(MPoly([[1.0, 0.05], [1.3, 0.22], [1.3, 0.97], [1.0, 0.80]],
                 closed=True, facecolor=WALL + "55", edgecolor=WALL, lw=1.4))
    ax.add_patch(MPoly([[0.0, 0.80], [0.30, 0.97], [1.3, 0.97], [1.0, 0.80]],
                 closed=True, facecolor=WALL + "33", edgecolor=WALL, lw=1.4))
    ax.add_patch(MPoly([[0.0, 0.80], [0.5, 1.35], [1.0, 0.80]],
                 closed=True, facecolor=ROOF + "99", edgecolor=ROOF, lw=1.8))
    ax.set_title("② Volumetric Solid", fontsize=11, color=DARK, pad=3, fontweight="bold")

    # ③ labeled parts
    ax = sub_ax(2)
    ax.set_xlim(-0.1, 1.3); ax.set_ylim(-0.1, 1.35); ax.axis("off")
    ax.add_patch(mpatches.FancyBboxPatch((0, 0), 1.0, 0.15,
                 boxstyle="square,pad=0", facecolor=SLAB, edgecolor=DARK, lw=0.9))
    ax.add_patch(mpatches.FancyBboxPatch((0, 0.15), 1.0, 0.60,
                 boxstyle="square,pad=0", facecolor=WALL, edgecolor=DARK, lw=0.9))
    ax.add_patch(mpatches.FancyBboxPatch((0.35, 0.15), 0.30, 0.60,
                 boxstyle="square,pad=0", facecolor=INTERIOR, edgecolor=DARK, lw=0.9))
    ax.add_patch(MPoly([[0, 0.75], [0.5, 1.25], [1.0, 0.75]],
                 closed=True, facecolor=ROOF, edgecolor=DARK, lw=0.9))
    for txt, xy, c in [("Floor", (0.5, 0.075), WHITE), ("Walls", (0.18, 0.45), WHITE),
                        ("Interior", (0.50, 0.45), WHITE), ("Roof", (0.50, 0.95), WHITE)]:
        ax.text(xy[0], xy[1], txt, ha="center", va="center",
                fontsize=9, color=c, fontweight="bold")
    ax.set_title("③ Labeled Parts", fontsize=11, color=DARK, pad=3, fontweight="bold")

    # arrows between sub-axes
    ay_mid = sy + sh * 0.50
    for col in range(2):
        x0 = bx + pad + col * (sw + pad) + sw + 0.004
        x1 = x0 + pad - 0.008
        _farrow(fig, x0, ay_mid, x1, ay_mid, lw=2.0, scale=13)

    _finalize(fig, "\u2460  Dataset Generation",
              "Residential building wireframes are extruded into volumetric\n"
              "structural solids with labeled components.",
              ["Input: 3DWire building wireframes",
               "Output: volumetric building structures",
               "Labels: exterior / interior / roof / floor"],
              "panel_01_dataset.png")


# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 2 — Structural Meshing
# ═════════════════════════════════════════════════════════════════════════════

def panel_meshing():
    fig, card_rect, hdr, body = _make_figure()
    bx, by, bw, bh = body

    pad = 0.030
    sw = (bw - 3 * pad) / 2
    sh = bh * 0.74
    sy = by + bh * 0.12
    arr_cx = bx + pad + sw + pad / 2

    # Left: house silhouette
    from matplotlib.patches import Polygon as MPoly
    axL = fig.add_axes([bx + pad, sy, sw, sh])
    axL.set_xlim(0, 1.3); axL.set_ylim(0, 1.2); axL.axis("off")
    axL.add_patch(mpatches.Rectangle((0.05, 0.05), 0.90, 0.65,
                  facecolor=WALL + "88", edgecolor=WALL, lw=2.0))
    axL.add_patch(MPoly([[0.05, 0.70], [0.50, 1.10], [0.95, 0.70]],
                  closed=True, facecolor=ROOF + "99", edgecolor=ROOF, lw=2.0))
    axL.set_title("House Geometry", fontsize=12, color=DARK, pad=3, fontweight="bold")

    _farrow(fig, arr_cx - 0.004, sy + sh * 0.5,
            arr_cx + 0.004, sy + sh * 0.5, lw=2.2, scale=14)

    # Right: Delaunay mesh
    axR = fig.add_axes([bx + pad + sw + pad, sy, sw, sh])
    axR.set_xlim(0, 1); axR.set_ylim(0, 1); axR.axis("off")
    rng = np.random.default_rng(42)
    px = np.concatenate([rng.uniform(0.05, 0.95, 60), [0.5]])
    py = np.concatenate([rng.uniform(0.05, 0.80, 60), [0.98]])
    mask = (py < 0.80) | (np.abs(px - 0.5) < (0.98 - py) * 0.60)
    pts = np.column_stack([px[mask], py[mask]])
    try:
        tri = Delaunay(pts)
        colors = plt.get_cmap("Blues")(np.linspace(0.25, 0.88, len(tri.simplices)))
        pc = PolyCollection(pts[tri.simplices], facecolors=colors,
                            edgecolors=NAVY, linewidths=0.6)
        axR.add_collection(pc)
    except Exception:
        pass
    axR.set_title("Tetrahedral Mesh", fontsize=12, color=DARK, pad=3, fontweight="bold")

    _finalize(fig, "\u2461  Structural Meshing",
              "Volumetric building geometries converted into tetrahedral meshes\n"
              "for finite element simulation.",
              ["Mesh generation: Gmsh",
               "50,000–200,000 elements per structure",
               "Refinement near wall junctions"],
              "panel_02_meshing.png")


# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 3 — FEA Simulation
# ═════════════════════════════════════════════════════════════════════════════

def panel_fea():
    fig, card_rect, hdr, body = _make_figure()
    bx, by, bw, bh = body

    pad = 0.022
    sw = bw * 0.32
    sh = bh * 0.68
    sy = by + bh * 0.16
    eq_cx = bx + sw + pad + bw * 0.10
    hm_x = bx + sw + pad + bw * 0.24

    # Left: mesh
    axM = fig.add_axes([bx + pad, sy, sw, sh])
    axM.set_xlim(0, 1); axM.set_ylim(0, 1); axM.axis("off")
    rng = np.random.default_rng(7)
    pts = rng.uniform(0.05, 0.95, (60, 2))
    try:
        tri = Delaunay(pts)
        pc = PolyCollection(pts[tri.simplices],
                            facecolors="#C8D8F5", edgecolors=BLUE, linewidths=0.7)
        axM.add_collection(pc)
    except Exception:
        pass
    axM.set_title("FE Mesh", fontsize=12, color=DARK, pad=3, fontweight="bold")

    # Equation Ku=f
    _farrow(fig, bx + pad + sw + 0.006, sy + sh * 0.5,
            eq_cx - 0.010, sy + sh * 0.5, lw=2.0, scale=13)
    fig.text(eq_cx, sy + sh * 0.5,
             r"$\mathbf{K}\mathbf{u}=\mathbf{f}$",
             ha="center", va="center", fontsize=22, color=NAVY,
             fontweight="bold", transform=fig.transFigure)
    _farrow(fig, eq_cx + 0.055, sy + sh * 0.5,
            hm_x - 0.008, sy + sh * 0.5, lw=2.0, scale=13)

    # Right: stress heatmap
    axS = fig.add_axes([hm_x, sy, sw, sh])
    Z = gaussian_filter(np.random.default_rng(9).random((24, 24)), sigma=3.0)
    axS.imshow(Z, cmap="jet", origin="lower", aspect="auto",
               extent=[0, 1, 0, 1], interpolation="bilinear")
    axS.axis("off")
    axS.set_title("Stress Field", fontsize=12, color=DARK, pad=3, fontweight="bold")

    # Colorbar
    cbx = bx + bw * 0.55
    cax = fig.add_axes([cbx, by + 0.010, bw * 0.36, 0.022])
    cb = plt.colorbar(
        plt.cm.ScalarMappable(mcolors.Normalize(0, 1), cmap="jet"),
        cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["Low", "Med", "High"])
    cb.ax.tick_params(labelsize=8, size=0, colors=DARK)
    cb.outline.set_edgecolor(DARK); cb.outline.set_linewidth(0.8)
    cax.set_title("Von Mises Stress", fontsize=8.5, color=DARK, pad=2)

    _finalize(fig, "\u2462  Finite Element Simulation",
              "Linear elastic FEA computes stress, displacement, and structural\n"
              "compliance for each building design.",
              ["Peak von Mises stress",
               "Maximum nodal displacement",
               "Global structural compliance"],
              "panel_03_fea.png")


# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 4 — Voxelization & Preprocessing
# ═════════════════════════════════════════════════════════════════════════════

def panel_voxelization():
    fig, card_rect, hdr, body = _make_figure()
    bx, by, bw, bh = body

    n = 3
    pad = 0.025
    sw = (bw - (n + 1) * pad) / n
    sh = bh * 0.70
    sy = by + bh * 0.15

    # ① tet mesh
    ax = fig.add_axes([bx + pad, sy, sw, sh])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    rng = np.random.default_rng(3)
    pts = rng.uniform(0.08, 0.92, (45, 2))
    try:
        tri = Delaunay(pts)
        pc = PolyCollection(pts[tri.simplices],
                            facecolors="#D0DCF5", edgecolors=BLUE, linewidths=0.6)
        ax.add_collection(pc)
    except Exception:
        pass
    ax.set_title("① Tet Mesh", fontsize=11, color=DARK, pad=3, fontweight="bold")

    # ② uniform voxel grid
    ax = fig.add_axes([bx + pad * 2 + sw, sy, sw, sh])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    nv = 9
    for ix in range(nv):
        for iz in range(nv):
            fc = "#BBCFEE" if (ix + iz) % 2 == 0 else "#8BAEE0"
            ax.add_patch(mpatches.Rectangle(
                (ix / nv + 0.005, iz / nv + 0.005),
                1 / nv - 0.012, 1 / nv - 0.012,
                facecolor=fc, edgecolor=DARK, lw=0.4))
    ax.set_title("② Voxel Grid", fontsize=11, color=DARK, pad=3, fontweight="bold")

    # ③ labeled channels
    ax = fig.add_axes([bx + pad * 3 + sw * 2, sy, sw, sh])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    cmap_data = {0: WALL, 1: INTERIOR, 2: ROOF, 3: SLAB}
    rng2 = np.random.default_rng(8)
    labels = rng2.integers(0, 4, (nv, nv))
    for ix in range(nv):
        for iz in range(nv):
            fc = cmap_data[labels[ix, iz]] + "CC"
            ax.add_patch(mpatches.Rectangle(
                (ix / nv + 0.005, iz / nv + 0.005),
                1 / nv - 0.012, 1 / nv - 0.012,
                facecolor=fc, edgecolor=DARK, lw=0.3))
    ax.set_title("③ Labeled Channels", fontsize=11, color=DARK, pad=3, fontweight="bold")

    ay_mid = sy + sh * 0.50
    for col in range(2):
        x0 = bx + pad * (col + 1) + sw * (col + 1) + 0.004
        x1 = x0 + pad - 0.008
        _farrow(fig, x0, ay_mid, x1, ay_mid, lw=2.0, scale=13)

    _finalize(fig, "\u2463  Voxelization & Preprocessing",
              "Structural meshes voxelized onto 3-D grids; part labels\n"
              "encoded as multi-channel input tensors.",
              ["128\u00b3 voxel resolution",
               "7 input channels per voxel",
               "Part labels one-hot encoded"],
              "panel_04_voxelization.png")


# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 5 — Surrogate Model Training
# ═════════════════════════════════════════════════════════════════════════════

def panel_surrogate():
    fig, card_rect, hdr, body = _make_figure()
    bx, by, bw, bh = body

    ax = fig.add_axes([bx, by + bh * 0.08, bw, bh * 0.82])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    layers = [
        ("128\u00b3\nVoxel Input", "#2176AE", 0.10),
        ("3D Conv\n\u00d73",       "#1A5276", 0.09),
        ("Residual\nBlocks",       "#6A0DAD", 0.09),
        ("Global\nAvg Pool",       "#0E6655", 0.08),
        ("MLP\nHead",              DARK,      0.08),
        ("\u03c3 / \u03b4 / C\nOutput", RED, 0.08),
    ]
    n = len(layers)
    xs = np.linspace(0.06, 0.94, n)
    yc = 0.58

    for i, (lbl, fc, wr) in enumerate(layers):
        xb = xs[i] - wr / 2
        ax.add_patch(FancyBboxPatch((xb, yc - 0.24), wr, 0.48,
                     boxstyle="round,pad=0.01,rounding_size=0.03",
                     facecolor=fc, edgecolor=WHITE, lw=1.6))
        ax.text(xs[i], yc, lbl, ha="center", va="center",
                fontsize=9.5, color=WHITE, fontweight="bold",
                linespacing=1.25, multialignment="center")
        if i < n - 1:
            ax.annotate("",
                        xy=(xs[i + 1] - layers[i + 1][2] / 2 - 0.008, yc),
                        xytext=(xs[i] + wr / 2 + 0.008, yc),
                        arrowprops=dict(arrowstyle="-|>", color=DARK,
                                        lw=1.6, mutation_scale=12))

    ax.text(0.50, 0.10,
            "\u00d75 independent models  \u2192  deep ensemble  \u2192  uncertainty estimates",
            ha="center", va="center", fontsize=10, color=DARK, fontstyle="italic")

    _finalize(fig, "\u2464  Surrogate Model Training",
              "Deep ensemble of 3D CNNs trained to predict structural responses\n"
              "without running expensive finite element simulations.",
              ["5 independent CNN models",
               "Trained on 11,178 FEA simulations",
               "Predicts stress, displacement & compliance"],
              "panel_05_surrogate.png")


# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 6 — SASTO Optimization
# ═════════════════════════════════════════════════════════════════════════════

def panel_optimization():
    fig, card_rect, hdr, body = _make_figure()
    bx, by, bw, bh = body

    ax = fig.add_axes([bx, by + bh * 0.05, bw, bh * 0.85])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    steps = [
        ("Initial\nStructure",    "#2176AE"),
        ("Surrogate\nEval",       "#6A0DAD"),
        ("Sensitivity\nAnalysis", "#1565C0"),
        ("Voxel\nRemoval",        TEAL),
        ("Constraint\nCheck",     GOLD),
        ("Done?",                 RED),
    ]
    n = len(steps)
    bw2 = 0.120; bh2 = 0.34; gap = (1.0 - n * bw2) / (n - 1)
    yc = 0.66

    for i, (lbl, fc) in enumerate(steps):
        xb = i * (bw2 + gap)
        ax.add_patch(FancyBboxPatch((xb, yc - bh2 / 2), bw2, bh2,
                     boxstyle="round,pad=0.01,rounding_size=0.04",
                     facecolor=fc, edgecolor=WHITE, lw=1.4))
        ax.text(xb + bw2 / 2, yc, lbl, ha="center", va="center",
                fontsize=9.5, color=WHITE, fontweight="bold",
                linespacing=1.2, multialignment="center")
        if i < n - 1:
            ax.annotate("",
                        xy=(xb + bw2 + gap - 0.008, yc),
                        xytext=(xb + bw2 + 0.008, yc),
                        arrowprops=dict(arrowstyle="-|>", color=DARK,
                                        lw=1.4, mutation_scale=11))

    # loop-back arrow
    ax.annotate("",
                xy=(2 * (bw2 + gap) + bw2 / 2, yc - bh2 / 2 - 0.01),
                xytext=((n - 1) * (bw2 + gap) + bw2 / 2, yc - bh2 / 2 - 0.01),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6,
                                mutation_scale=10,
                                connectionstyle="arc3,rad=-0.30"))
    ax.text(0.50, 0.20, "iterate until convergence",
            ha="center", va="center", fontsize=9.5, color=RED, fontstyle="italic")

    # objective equation
    ax.text(0.50, 0.92,
            r"$J(\rho)=w_V\!\frac{V}{V_0}+w_S\!\frac{S}{V_0}+P_{\mathrm{constraint}}$",
            ha="center", va="center", fontsize=16, color=NAVY, fontweight="bold")

    _finalize(fig, "\u2465  SASTO Optimization",
              "Surrogate-guided topology optimization iteratively removes material\n"
              "while enforcing structural constraints and printability.",
              ["Part-aware sensitivity analysis",
               "Printability & connectivity constraints",
               "23\u201392\u00d7 faster than classical SIMP"],
              "panel_06_optimization.png")


# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 7 — Optimized Structures
# ═════════════════════════════════════════════════════════════════════════════

def _render_ply(ply_path, ax3d, color_mode="part", elev=22, azim=-55):
    mesh = trimesh.load(str(ply_path), force="mesh", process=False)
    verts = np.array(mesh.vertices, dtype=float)
    faces = np.array(mesh.faces)
    lo, hi = verts.min(0), verts.max(0)
    span_r = hi - lo; span = span_r.max()
    verts = (verts - (lo + hi) / 2.0) / span
    pv = verts[faces]

    n0, n1, n2 = pv[:, 0], pv[:, 1], pv[:, 2]
    nrm = np.cross(n1 - n0, n2 - n0)
    mag = np.linalg.norm(nrm, axis=1, keepdims=True); mag[mag == 0] = 1.0
    nrm /= mag
    light = np.array([np.cos(np.radians(35)) * np.cos(np.radians(-35)),
                      np.cos(np.radians(35)) * np.sin(np.radians(-35)),
                      np.sin(np.radians(35))])
    lam = np.clip(nrm @ light, 0, 1)
    ctr = pv.mean(1)
    zn = (ctr[:, 2] - ctr[:, 2].min()) / (np.ptp(ctr[:, 2]) + 1e-9)

    if color_mode == "stress":
        rad = np.sqrt(ctr[:, 0] ** 2 + ctr[:, 1] ** 2)
        rad /= rad.max() + 1e-9
        s = np.clip(0.58 * (1 - zn) + 0.42 * rad, 0, 1)
        base = plt.get_cmap("jet")(s)[:, :3]
    else:
        base = np.column_stack([0.25 + 0.20 * zn, 0.45 + 0.25 * zn, 0.82 + 0.10 * zn])

    amb = 0.82
    shd = np.clip(base * (amb + (1 - amb) * lam[:, None]), 0, 1)
    pc = Poly3DCollection(pv, zsort="average")
    pc.set_facecolor(shd); pc.set_edgecolor("none")
    ax3d.add_collection3d(pc)
    pad = 0.06
    ax3d.set_xlim(-0.5 - pad, 0.5 + pad)
    ax3d.set_ylim(-0.5 - pad, 0.5 + pad)
    ax3d.set_zlim(-0.5 - pad, 0.5 + pad)
    ax3d.set_box_aspect([span_r[0], span_r[1], span_r[2]])
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_facecolor(WHITE)
    for a in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0)); a.pane.set_edgecolor((1, 1, 1, 0))
    ax3d.set_axis_off()


def panel_results():
    fig, card_rect, hdr, body = _make_figure()
    bx, by, bw, bh = body

    models = [
        ("figures/screenshot_stls/REF_original_colored.ply",  "part",   "Baseline House"),
        ("figures/screenshot_stls/REF_SASTO_PA_colored.ply",  "stress", "SASTO-PA Optimized"),
        ("figures/screenshot_stls/REF_SASTO_U_colored.ply",   "part",   "SASTO-U Optimized"),
    ]
    n = len(models)
    pad = 0.022
    sw = (bw - (n + 1) * pad) / n
    sh = bh * 0.68
    sy = by + bh * 0.22
    ay_mid = sy + sh * 0.50

    for i, (ply, cmode, lbl) in enumerate(models):
        ax3 = fig.add_axes([bx + pad + i * (sw + pad), sy, sw, sh],
                           projection="3d")
        try:
            _render_ply(ply, ax3, color_mode=cmode)
        except Exception:
            ax3.text2D(0.5, 0.5, "not found", ha="center", va="center",
                       transform=ax3.transAxes, color=RED, fontsize=10)
        fig.text(bx + pad + i * (sw + pad) + sw / 2, sy - 0.024,
                 lbl, ha="center", va="top", fontsize=10.5,
                 color=DARK, fontweight="bold", transform=fig.transFigure)
        if i < 2:
            x0 = bx + pad + i * (sw + pad) + sw + 0.004
            x1 = x0 + pad - 0.008
            _farrow(fig, x0, ay_mid, x1, ay_mid, lw=2.2, scale=14)

    # stat badges
    stats = ["Up to 45% material reduction", "23–92× faster than SIMP",
             "0 constraint violations", "Printable connectivity"]
    sw_b = (bw - 0.01) / len(stats)
    badge_y = by + 0.005
    for i, st in enumerate(stats):
        bxb = bx + i * sw_b
        fig.add_artist(FancyBboxPatch(
            (bxb + 0.002, badge_y), sw_b - 0.006, 0.030,
            boxstyle="round,pad=0,rounding_size=0.006",
            facecolor=NAVY, edgecolor="none",
            transform=fig.transFigure, clip_on=False, zorder=5,
        ))
        fig.text(bxb + sw_b / 2, badge_y + 0.015, f"\u2713  {st}",
                 ha="center", va="center", fontsize=8.5, color=WHITE,
                 fontweight="bold", transform=fig.transFigure, zorder=6)

    _finalize(fig, "\u2466  Optimized Structures",
              "SASTO removes unnecessary material while preserving structural safety\n"
              "and printable connectivity.",
              [],   # bullets replaced by badges above
              "panel_07_results.png")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating methodology panels...")
    panel_banner()
    panel_dataset()
    panel_meshing()
    panel_fea()
    panel_voxelization()
    panel_surrogate()
    panel_optimization()
    panel_results()
    print("All panels saved to", OUT_DIR)
