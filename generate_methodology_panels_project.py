"""Generate polished methodology panel PNGs using real project figures.

Outputs → poster_images_extracted/panels/panel_00_banner.png … panel_07_results.png
Design: each panel has a coloured header ribbon at the top, the main figure
fills most of the body, and a thin caption line sits at the very bottom.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from pathlib import Path
from PIL import Image
from scipy.spatial import Delaunay

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
OUT_DIR = BASE / "poster_images_extracted" / "panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR        = BASE / "figures"
POSTER_HQ      = BASE / "poster_final" / "renders_hq"
POSTER_FINAL   = BASE / "poster_final"
WIRE_PATH      = BASE / "optimization" / "data" / "3dwire_raw" / "00472.npz"
OPT_DIR        = BASE / "fea_ml" / "runs" / "v3" / "optimization_128"
PLY_PART       = BASE / "figures" / "screenshot_stls" / "REF_original_colored.ply"

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY     = "#062B7A"
BLUE     = "#1A4FAA"
LBLUE    = "#D9E5FB"
TEAL     = "#0BA6B7"
GOLD     = "#CFA535"
RED      = "#D7263D"
DARK     = "#0B1736"
WHITE    = "#FFFFFF"
PANEL_BG = "#F7F9FD"
WALL     = "#4477CC"
INTERIOR = "#E88843"
ROOF     = "#54A24B"
SLAB     = "#D6B48A"

# Step colours — one per methodology step
STEP_COLORS = [NAVY, "#1565C0", TEAL, "#6A0DAD", "#2E7D32", "#A3111A", GOLD]

# ── Panel dimensions ──────────────────────────────────────────────────────────
PW, PH = 11.0, 7.0           # figure inches
DPI    = 220
MARGIN = 0.025                # outer margin (fraction of figure)
HDR_H  = 0.095                # header ribbon height
CAP_H  = 0.065                # caption strip height at bottom
BODY_PAD = 0.025              # inner padding inside body


# ═══════════════════════════════════════════════════════════════════════════════
# Low-level drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _rounded_rect(fig, x, y, w, h, *, fc=PANEL_BG, ec=NAVY, lw=2.2,
                  r=0.015, zo=1):
    """Add a rounded rectangle artist to *fig* in figure-fraction coords."""
    fig.add_artist(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        transform=fig.transFigure, clip_on=False, zorder=zo,
    ))


def _header_ribbon(fig, x, y, w, h, title, *, fc=NAVY, fs=17, zo=4):
    """Draw the header band with rounded-top corners and centred title."""
    # top rounded rectangle
    fig.add_artist(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.015",
        facecolor=fc, edgecolor=fc, linewidth=0,
        transform=fig.transFigure, clip_on=False, zorder=zo,
    ))
    # flat bottom half to merge with card body
    fig.add_artist(mpatches.Rectangle(
        (x, y), w, h * 0.42,
        facecolor=fc, edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=zo,
    ))
    fig.text(x + w / 2, y + h * 0.50, title,
             ha="center", va="center", fontsize=fs, fontweight="bold",
             color=WHITE, family="sans-serif",
             transform=fig.transFigure, zorder=zo + 1)


def _caption_strip(fig, x, y, w, text, *, fs=9.5, color=DARK, zo=5):
    """One-line (or two-line) italic caption centred at the bottom."""
    fig.text(x + w / 2, y, text,
             ha="center", va="center", fontsize=fs, fontstyle="italic",
             color=color, family="sans-serif", linespacing=1.35,
             transform=fig.transFigure, zorder=zo)


def _ax(fig, rect, projection=None, zo=6):
    """Create axes with transparent background at a high z-order."""
    kw = {"projection": projection} if projection else {}
    ax = fig.add_axes(rect, **kw)
    ax.set_zorder(zo)
    ax.patch.set_alpha(0)
    return ax


def _arrow(fig, x0, y0, x1, y1, *, color=DARK, lw=2.2, ms=14, zo=15):
    """Fat direction arrow between two points (figure-fraction coords)."""
    fig.add_artist(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="Simple,head_length=0.8,head_width=0.8,tail_width=0.3",
        mutation_scale=ms,
        facecolor=color, edgecolor=color, linewidth=lw,
        transform=fig.transFigure, clip_on=False, zorder=zo,
    ))


def _label_below(fig, cx, y, text, *, fs=10.5, color=DARK, bold=True, zo=8):
    fig.text(cx, y, text,
             ha="center", va="top", fontsize=fs, color=color,
             fontweight="bold" if bold else "normal",
             family="sans-serif",
             transform=fig.transFigure, zorder=zo)


# ── Pipeline-row helpers ──────────────────────────────────────────────────────

def _step_box(fig, x, y, w, h, text, *, fc=BLUE, tc=WHITE, fs=9.0,
              bold=True, r=0.008, zo=10):
    """Small labelled rounded box for mini-pipeline steps."""
    fig.add_artist(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=zo,
    ))
    fig.text(x + w / 2, y + h / 2, text,
             ha="center", va="center", fontsize=fs,
             fontweight="bold" if bold else "normal",
             color=tc, family="sans-serif", linespacing=1.15,
             transform=fig.transFigure, zorder=zo + 1)


def _outline_box(fig, x, y, w, h, text, *, ec=BLUE, tc=BLUE, fs=8.5,
                 bold=True, r=0.007, lw=1.8, zo=10):
    """Outline-only (hollow) rounded box for pipeline sub-items."""
    fig.add_artist(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=WHITE, edgecolor=ec, linewidth=lw,
        transform=fig.transFigure, clip_on=False, zorder=zo,
    ))
    fig.text(x + w / 2, y + h / 2, text,
             ha="center", va="center", fontsize=fs,
             fontweight="bold" if bold else "normal",
             color=tc, family="sans-serif", linespacing=1.15,
             transform=fig.transFigure, zorder=zo + 1)


def _hbrace(fig, x0, x1, y, *, depth=0.022, color=NAVY, lw=2.0, zo=12):
    """Draw a horizontal curly brace from x0 to x1 at height y, tip pointing DOWN."""
    from matplotlib.path import Path as MPath
    mid = (x0 + x1) / 2
    q = depth * 0.55
    verts = [
        (x0, y),
        (x0 + q, y), (mid - q, y - depth + q), (mid, y - depth),
        (mid, y - depth),
        (mid + q, y - depth + q), (x1 - q, y), (x1, y),
    ]
    codes = [
        MPath.MOVETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
    ]
    path = MPath(verts, codes)
    patch = mpatches.PathPatch(
        path, facecolor="none", edgecolor=color,
        linewidth=lw, capstyle="round",
        transform=fig.transFigure, clip_on=False, zorder=zo,
    )
    fig.add_artist(patch)
    return mid, y - depth          # tip coords


def _hbrace_up(fig, x0, x1, y, *, depth=0.022, color=NAVY, lw=2.0, zo=12):
    """Horizontal curly brace, tip pointing UP."""
    from matplotlib.path import Path as MPath
    mid = (x0 + x1) / 2
    q = depth * 0.55
    verts = [
        (x0, y),
        (x0 + q, y), (mid - q, y + depth - q), (mid, y + depth),
        (mid, y + depth),
        (mid + q, y + depth - q), (x1 - q, y), (x1, y),
    ]
    codes = [
        MPath.MOVETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
    ]
    path = MPath(verts, codes)
    patch = mpatches.PathPatch(
        path, facecolor="none", edgecolor=color,
        linewidth=lw, capstyle="round",
        transform=fig.transFigure, clip_on=False, zorder=zo,
    )
    fig.add_artist(patch)
    return mid, y + depth


def _vbrace(fig, x, y0, y1, *, depth=0.020, color=NAVY, lw=2.0, zo=12):
    """Vertical curly brace (right-pointing tip) from y0 to y1 at x."""
    from matplotlib.path import Path as MPath
    mid = (y0 + y1) / 2
    q = depth * 0.55
    verts = [
        (x, y0),
        (x, y0 + q), (x + depth - q, mid - q), (x + depth, mid),
        (x + depth, mid),
        (x + depth - q, mid + q), (x, y1 - q), (x, y1),
    ]
    codes = [
        MPath.MOVETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
    ]
    path = MPath(verts, codes)
    patch = mpatches.PathPatch(
        path, facecolor="none", edgecolor=color,
        linewidth=lw, capstyle="round",
        transform=fig.transFigure, clip_on=False, zorder=zo,
    )
    fig.add_artist(patch)
    return x + depth, mid


def _bracket_h(fig, x0, x1, y, *, depth=0.012, color=NAVY, lw=1.6, zo=12):
    """Simple square bracket spanning x0..x1, sitting at y, pointing down."""
    from matplotlib.lines import Line2D
    for seg in [
        ([x0, x0], [y, y - depth]),
        ([x0, x1], [y - depth, y - depth]),
        ([x1, x1], [y, y - depth]),
    ]:
        fig.add_artist(Line2D(
            seg[0], seg[1],
            color=color, linewidth=lw,
            transform=fig.transFigure, clip_on=False, zorder=zo,
        ))
    return (x0 + x1) / 2, y - depth


def _darrow(fig, x, y0, y1, *, color=DARK, lw=1.8, ms=11, zo=15):
    """Downward (or upward) vertical arrow."""
    _arrow(fig, x, y0, x, y1, color=color, lw=lw, ms=ms, zo=zo)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure / panel scaffolding
# ═══════════════════════════════════════════════════════════════════════════════

def _new_panel(title, caption, *, pw=PW, ph=PH, hdr_fc=NAVY, fs_title=17):
    """Create a new panel figure and return (fig, body_rect).

    *body_rect* = (bx, by, bw, bh) is the rectangle available for content,
    already accounting for header, caption strip, and inner padding.
    """
    fig = plt.figure(figsize=(pw, ph), facecolor=WHITE)

    # outer card
    cx = MARGIN
    cy = MARGIN
    cw = 1 - 2 * MARGIN
    ch = 1 - 2 * MARGIN
    _rounded_rect(fig, cx, cy, cw, ch, fc=PANEL_BG, ec=NAVY, lw=2.5)

    # header ribbon
    hx, hy = cx, cy + ch - HDR_H
    _header_ribbon(fig, hx, hy, cw, HDR_H, title, fc=hdr_fc, fs=fs_title)

    # caption strip
    cap_cy = cy + CAP_H * 0.5
    _caption_strip(fig, cx, cap_cy, cw, caption)

    # usable body
    bx = cx + BODY_PAD
    by = cy + CAP_H + BODY_PAD * 0.5
    bw = cw - 2 * BODY_PAD
    bh = ch - HDR_H - CAP_H - BODY_PAD * 1.5
    return fig, (bx, by, bw, bh)


def _save(fig, name):
    out = OUT_DIR / name
    plt.savefig(str(out), dpi=DPI, bbox_inches="tight",
                facecolor=WHITE, edgecolor="none")
    plt.close(fig)
    print(f"  saved {out.name}  ({out.stat().st_size // 1024} KB)")


# ═══════════════════════════════════════════════════════════════════════════════
# Image loaders
# ═══════════════════════════════════════════════════════════════════════════════

def _trim(img: Image.Image, tol=10) -> Image.Image:
    """Auto-crop whitespace borders."""
    arr = np.asarray(img.convert("RGBA"))
    rgb = arr[..., :3].astype(np.int16)
    corners = np.array([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
    bg = np.median(corners, axis=0)
    diff = np.abs(rgb - bg[None, None, :]).max(axis=2)
    alpha = arr[..., 3]
    mask = (diff > tol) & (alpha > 0)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    return img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def _load(path, *, crop=None, trim=True):
    img = Image.open(path).convert("RGBA")
    if crop is not None:
        w, h = img.size
        x0, y0, x1, y1 = crop
        img = img.crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))
    if trim:
        img = _trim(img)
    return np.asarray(img)


def _imshow(fig, rect, path, *, crop=None, trim=True, zo=6):
    """Embed an external figure image into the panel, filling *rect*."""
    ax = _ax(fig, rect, zo=zo)
    img = Image.open(path).convert("RGBA")
    if crop is not None:
        w, h = img.size
        x0, y0, x1, y1 = crop
        img = img.crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))
    if trim:
        img = _trim(img)
    # Downsample large images to avoid slow rendering
    max_px = 2400
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    arr = np.asarray(img)
    ax.imshow(arr, interpolation="bilinear")
    ax.axis("off")
    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# 3-D rendering helpers (for dataset / meshing panels)
# ═══════════════════════════════════════════════════════════════════════════════

def _mesh_face_colors(mesh):
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        vc = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=float) / 255.0
        return vc[mesh.faces].mean(axis=1)
    return np.tile(np.array([[0.35, 0.55, 0.85]]), (len(mesh.faces), 1))


def _cleanup_3d(ax3d, elev=22, azim=-58, aspect=None):
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_facecolor(WHITE)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
    ax3d.set_axis_off()
    if aspect is not None:
        ax3d.set_box_aspect(aspect)


def _draw_wireframe(ax3d, vertices, lines, elev=25, azim=-60):
    verts = np.asarray(vertices, dtype=float)
    lines_arr = np.asarray(lines)
    z_min, z_range = verts[:, 2].min(), np.ptp(verts[:, 2]) + 1e-9
    for e in lines_arr:
        p0, p1 = verts[e[0]], verts[e[1]]
        zf = ((p0[2] + p1[2]) * 0.5 - z_min) / z_range
        c = SLAB if zf < 0.15 else (ROOF if zf > 0.65 else WALL)
        ax3d.plot3D(*zip(p0, p1), color=c, linewidth=2.4)
    ax3d.scatter3D(verts[:, 0], verts[:, 1], verts[:, 2],
                   color=RED, s=12, depthshade=False, zorder=10)
    lo, hi = verts.min(0), verts.max(0)
    span = hi - lo; pad = span * 0.06
    ax3d.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax3d.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax3d.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    _cleanup_3d(ax3d, elev=elev, azim=azim, aspect=span / span.max())


def _draw_part_mesh(ax3d, ply_path, elev=22, azim=-58):
    mesh = trimesh.load(str(ply_path), force="mesh", process=False)
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    lo, hi = verts.min(0), verts.max(0)
    span = hi - lo; mx = span.max()
    verts = (verts - (lo + hi) / 2) / mx
    poly_v = verts[faces]
    # simple shading
    n0, n1, n2 = poly_v[:, 0], poly_v[:, 1], poly_v[:, 2]
    normals = np.cross(n1 - n0, n2 - n0)
    mag = np.linalg.norm(normals, axis=1, keepdims=True); mag[mag == 0] = 1
    normals /= mag
    light = np.array([np.cos(np.radians(35)) * np.cos(np.radians(-35)),
                       np.cos(np.radians(35)) * np.sin(np.radians(-35)),
                       np.sin(np.radians(35))])
    lam = np.clip(normals @ light, 0, 1)
    base = _mesh_face_colors(mesh)
    shaded = np.clip(base * (0.82 + 0.18 * lam[:, None]), 0, 1)
    pc = Poly3DCollection(poly_v, zsort="average")
    pc.set_facecolor(shaded); pc.set_edgecolor("none")
    ax3d.add_collection3d(pc)
    pad = 0.07
    ax3d.set_xlim(-0.5 - pad, 0.5 + pad)
    ax3d.set_ylim(-0.5 - pad, 0.5 + pad)
    ax3d.set_zlim(-0.5 - pad, 0.5 + pad)
    _cleanup_3d(ax3d, elev=elev, azim=azim, aspect=span / mx)


def _draw_voxels(ax3d, occ, part, elev=22, azim=-55, stride=3):
    occ_s = occ[::stride, ::stride, ::stride].astype(bool)
    part_s = part[::stride, ::stride, ::stride]
    fc = np.zeros(occ_s.shape + (4,))
    cmap = {1: WALL, 2: INTERIOR, 3: ROOF, 4: SLAB}
    for k, c in cmap.items():
        m = occ_s & (part_s == k)
        fc[m, :3] = mcolors.to_rgb(c)
        fc[m, 3] = 0.98
    ec = np.zeros_like(fc); ec[..., :3] = 1; ec[..., 3] = 0.12
    ax3d.voxels(occ_s, facecolors=fc, edgecolors=ec, linewidth=0.14)
    _cleanup_3d(ax3d, elev=elev, azim=azim, aspect=list(occ_s.shape))


def _draw_tet_mesh(ax3d, occ, max_pts=260, elev=18, azim=-60):
    pts = np.argwhere(occ).astype(float)
    rng = np.random.default_rng(12)
    if len(pts) > max_pts:
        idx = rng.choice(len(pts), size=max_pts, replace=False)
        pts = pts[idx]
    pts = (pts - pts.mean(0)) / max(np.ptp(pts, axis=0))
    try:
        tri = Delaunay(pts)
        edges = set()
        for s in tri.simplices:
            for a, b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
                edges.add((min(s[a], s[b]), max(s[a], s[b])))
        segs = [(pts[i], pts[j]) for i, j in edges]
        ax3d.add_collection3d(
            Line3DCollection(segs, colors=BLUE, linewidths=0.4, alpha=0.65))
    except Exception:
        ax3d.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2], s=5, color=BLUE)
    ax3d.set_xlim(-0.55, 0.55); ax3d.set_ylim(-0.55, 0.55)
    ax3d.set_zlim(-0.55, 0.55)
    _cleanup_3d(ax3d, elev=elev, azim=azim, aspect=[1, 1, 0.8])


# ═══════════════════════════════════════════════════════════════════════════════
# Panel generators
# ═══════════════════════════════════════════════════════════════════════════════

def panel_banner():
    """Panel 0 — horizontal pipeline overview banner."""
    fig = plt.figure(figsize=(PW, 3.0), facecolor=WHITE)
    cx, cy, cw, ch = MARGIN, MARGIN, 1 - 2 * MARGIN, 1 - 2 * MARGIN
    _rounded_rect(fig, cx, cy, cw, ch, fc=LBLUE, ec=NAVY, lw=2.5)

    # Title
    fig.text(0.5, cy + ch * 0.88, "ENGINEERING METHODOLOGY",
             ha="center", va="center", fontsize=24, fontweight="bold",
             color=NAVY, family="sans-serif",
             transform=fig.transFigure, zorder=3)

    steps = [
        "\u2460 Dataset\n   Generation",
        "\u2461 Structural\n   Meshing",
        "\u2462 FEA\n   Simulation",
        "\u2463 Voxelization\n   & Encoding",
        "\u2464 Surrogate\n   Training",
        "\u2465 SASTO\n   Optimization",
        "\u2466 Structural\n   Results",
    ]
    N = len(steps)
    bw_box = 0.105
    bh_box = 0.34
    arrow_gap = 0.028
    total_w = N * bw_box + (N - 1) * arrow_gap
    x_start = cx + (cw - total_w) / 2
    y_center = cy + ch * 0.35

    for i, (label, sc) in enumerate(zip(steps, STEP_COLORS)):
        bx = x_start + i * (bw_box + arrow_gap)
        # box
        fig.add_artist(FancyBboxPatch(
            (bx, y_center - bh_box / 2), bw_box, bh_box,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor=sc, edgecolor="none",
            transform=fig.transFigure, clip_on=False, zorder=3,
        ))
        fig.text(bx + bw_box / 2, y_center, label,
                 ha="center", va="center", fontsize=9.5, fontweight="bold",
                 color=WHITE, family="sans-serif", linespacing=1.15,
                 transform=fig.transFigure, zorder=4)
        # arrow to next box
        if i < N - 1:
            ax0 = bx + bw_box + 0.003
            ax1 = bx + bw_box + arrow_gap - 0.003
            _arrow(fig, ax0, y_center, ax1, y_center,
                   color=DARK, lw=1.8, ms=11)

    _save(fig, "panel_00_banner.png")


def panel_dataset():
    """Panel 1 — Dataset: wireframe -> volumetric model -> labeled voxels."""
    fig, body = _new_panel(
        "\u2460  Dataset Generation",
        "3DWire residential wireframes \u2192 volumetric structural models \u2192 "
        "labeled 128\u00b3 voxel grids (exterior / interior / roof / slab)",
        hdr_fc=STEP_COLORS[0],
    )
    bx, by, bw, bh = body
    d = np.load(WIRE_PATH, allow_pickle=True)
    vertices, lines_data = d["vertices"], d["lines"]
    occ = np.load(OPT_DIR / "fixed_occ.npz")["data"]
    part = np.load(OPT_DIR / "fixed_part.npz")["data"]

    # three columns
    n_cols = 3
    gap = 0.045       # gap between columns (room for arrows)
    sw = (bw - (n_cols - 1) * gap) / n_cols
    sh = bh * 0.82
    sy = by + bh * 0.09

    # col 1 — wireframe
    ax1 = _ax(fig, [bx, sy, sw, sh], projection="3d")
    _draw_wireframe(ax1, vertices, lines_data)
    _label_below(fig, bx + sw / 2, sy - 0.005, "3DWire Wireframe")

    # arrow
    _arrow(fig, bx + sw + 0.006, sy + sh * 0.5,
           bx + sw + gap - 0.006, sy + sh * 0.5,
           color=NAVY, lw=2.5, ms=15)

    # col 2 — surface mesh
    ax2 = _ax(fig, [bx + sw + gap, sy, sw, sh], projection="3d")
    _draw_part_mesh(ax2, PLY_PART)
    _label_below(fig, bx + sw + gap + sw / 2, sy - 0.005, "Part-Colored Mesh")

    # arrow
    _arrow(fig, bx + 2 * sw + gap + 0.006, sy + sh * 0.5,
           bx + 2 * sw + 2 * gap - 0.006, sy + sh * 0.5,
           color=NAVY, lw=2.5, ms=15)

    # col 3 — voxels
    ax3 = _ax(fig, [bx + 2 * (sw + gap), sy, sw, sh], projection="3d")
    _draw_voxels(ax3, occ, part, stride=4)
    _label_below(fig, bx + 2 * (sw + gap) + sw / 2, sy - 0.005, "Labeled Voxel Grid")

    _save(fig, "panel_01_dataset.png")


def panel_meshing():
    """Panel 2 — Structural meshing pipeline."""
    fig, body = _new_panel(
        "\u2461  Structural Meshing",
        "Volumetric structural models are discretized into hexahedral finite "
        "elements compatible with the SfePy solver",
        hdr_fc=STEP_COLORS[1],
    )
    bx, by, bw, bh = body
    occ = np.load(OPT_DIR / "fixed_occ.npz")["data"]

    # ── mini-pipeline row at top ──
    pipe_y = by + bh - 0.05
    pipe_h = 0.038
    box_w  = 0.15
    gap_a  = 0.035  # arrow gap
    labels = ["Surface\nGeometry", "Hex8\nDiscretization", "FE Mesh"]
    colors = [NAVY, STEP_COLORS[1], TEAL]
    total  = len(labels) * box_w + (len(labels) - 1) * gap_a
    sx     = bx + (bw - total) / 2
    for i, (lbl, clr) in enumerate(zip(labels, colors)):
        xi = sx + i * (box_w + gap_a)
        _step_box(fig, xi, pipe_y, box_w, pipe_h, lbl, fc=clr, fs=8.5)
        if i < len(labels) - 1:
            _arrow(fig, xi + box_w + 0.004, pipe_y + pipe_h / 2,
                   xi + box_w + gap_a - 0.004, pipe_y + pipe_h / 2,
                   color=DARK, lw=2.0, ms=12)

    # ── brace under pipeline pointing to figures ──
    brace_y = pipe_y - 0.008
    _hbrace(fig, sx + 0.01, sx + total - 0.01, brace_y,
            depth=0.018, color=NAVY, lw=1.6)

    # ── two 3D renders below ──
    rend_gap = 0.07
    rw = (bw - rend_gap) / 2
    rh = bh * 0.60
    ry = by + bh * 0.05

    ax1 = _ax(fig, [bx, ry, rw, rh], projection="3d")
    _draw_part_mesh(ax1, PLY_PART)
    _label_below(fig, bx + rw / 2, ry - 0.005, "Surface Model")

    _arrow(fig, bx + rw + 0.010, ry + rh * 0.5,
           bx + rw + rend_gap - 0.010, ry + rh * 0.5,
           color=NAVY, lw=2.5, ms=15)

    ax2 = _ax(fig, [bx + rw + rend_gap, ry, rw, rh], projection="3d")
    _draw_tet_mesh(ax2, occ[::3, ::3, ::3] > 0)
    _label_below(fig, bx + rw + rend_gap + rw / 2, ry - 0.005,
                 "Tetrahedral Mesh")

    _save(fig, "panel_02_meshing.png")


def panel_fea():
    """Panel 3 — FEA pipeline: inputs → solver → outputs."""
    fig, body = _new_panel(
        "\u2462  Finite Element Simulation",
        "Linear-elastic FEA (SfePy) computes stress, displacement, "
        "and compliance for every building geometry",
        hdr_fc=STEP_COLORS[2],
    )
    bx, by, bw, bh = body

    # ── top row: three input boxes → curly brace → solver box → fan out ──
    row_y = by + bh - 0.048
    bxh = 0.032  # box height
    ibw = 0.125  # input box width
    igap = 0.018
    inputs = ["Hex8 Mesh", "Boundary\nConditions", "Material\nProperties"]
    inp_total = len(inputs) * ibw + (len(inputs) - 1) * igap
    ix0 = bx + 0.02
    for i, lbl in enumerate(inputs):
        xi = ix0 + i * (ibw + igap)
        _outline_box(fig, xi, row_y, ibw, bxh, lbl, ec=NAVY, tc=NAVY, fs=7.5)

    # curly brace under the 3 inputs
    brace_top = row_y - 0.005
    tip_x, tip_y = _hbrace(fig, ix0 + 0.01, ix0 + inp_total - 0.01,
                            brace_top, depth=0.016, color=NAVY, lw=1.5)

    # arrow from brace tip down to solver box
    solv_w, solv_h = 0.16, 0.032
    solv_x = tip_x - solv_w / 2
    solv_y = tip_y - 0.026
    _darrow(fig, tip_x, tip_y - 0.003, solv_y + solv_h + 0.003,
            color=NAVY, lw=1.6, ms=9)
    _step_box(fig, solv_x, solv_y, solv_w, solv_h,
              "SfePy FEA Solver", fc=STEP_COLORS[2], fs=8.5)

    # arrow from solver to outputs bracket
    out_labels = ["\u03c3 Stress", "u Displacement", "C Compliance"]
    obw = 0.095
    ogap = 0.012
    out_total = len(out_labels) * obw + (len(out_labels) - 1) * ogap
    ox0 = bx + bw - out_total - 0.02
    oy  = row_y
    for i, lbl in enumerate(out_labels):
        xi = ox0 + i * (obw + ogap)
        _step_box(fig, xi, oy, obw, bxh, lbl, fc=TEAL, fs=7.5)

    # bracket above outputs
    bracket_y = oy + bxh + 0.005
    _bracket_h(fig, ox0 + 0.005, ox0 + out_total - 0.005,
               bracket_y, depth=-0.010, color=TEAL, lw=1.4)
    fig.text(ox0 + out_total / 2, bracket_y + 0.016, "Outputs",
             ha="center", va="center", fontsize=8, color=TEAL,
             fontweight="bold", family="sans-serif",
             transform=fig.transFigure, zorder=12)

    # horizontal arrow: solver → outputs
    _arrow(fig, solv_x + solv_w + 0.006, solv_y + solv_h / 2,
           ox0 - 0.008, oy + bxh / 2,
           color=NAVY, lw=2.0, ms=12)

    # ── main FEA figure ──
    fig_h = bh * 0.58
    fig_y = by + bh * 0.02
    _imshow(fig, [bx, fig_y, bw, fig_h],
            FIG_DIR / "fig16_fea_stress_placeholder.png")

    _save(fig, "panel_03_fea.png")


def panel_voxelization():
    """Panel 4 — Voxelization pipeline: mesh → 128³ grid → 7-ch tensor."""
    fig, body = _new_panel(
        "\u2463  Voxelization & Preprocessing",
        "Structural meshes are rasterized onto a 128\u00b3 voxel grid and "
        "encoded as 7-channel tensors for the surrogate network",
        hdr_fc=STEP_COLORS[3],
    )
    bx, by, bw, bh = body

    # ── pipeline across top ──
    pipe_y = by + bh - 0.05
    pipe_h = 0.038
    entries = [
        ("Triangle\nMesh",    NAVY),
        ("128\u00b3 Voxel\nRasterization", STEP_COLORS[3]),
        ("7-Channel\nTensor",  TEAL),
    ]
    box_w = 0.145
    gap_a = 0.038
    total = len(entries) * box_w + (len(entries) - 1) * gap_a
    sx = bx + (bw - total) / 2 - 0.10  # shift left to make room for channel list
    for i, (lbl, clr) in enumerate(entries):
        xi = sx + i * (box_w + gap_a)
        _step_box(fig, xi, pipe_y, box_w, pipe_h, lbl, fc=clr, fs=8.5)
        if i < len(entries) - 1:
            _arrow(fig, xi + box_w + 0.004, pipe_y + pipe_h / 2,
                   xi + box_w + gap_a - 0.004, pipe_y + pipe_h / 2,
                   color=DARK, lw=2.0, ms=12)

    # ── vertical curly brace + channel list to the right ──
    last_x = sx + (len(entries) - 1) * (box_w + gap_a) + box_w
    _arrow(fig, last_x + 0.006, pipe_y + pipe_h / 2,
           last_x + 0.026, pipe_y + pipe_h / 2,
           color=DARK, lw=1.6, ms=9)
    ch_x = last_x + 0.032
    channels = ["Occupancy", "Part ID", "Wall dist",
                "Roof dist", "Slab dist", "Height", "Normals"]
    ch_fs = 7.0
    ch_lh = 0.017  # line height
    ch_top = pipe_y + pipe_h / 2 + len(channels) * ch_lh / 2
    for j, ch in enumerate(channels):
        cy = ch_top - j * ch_lh
        fig.text(ch_x + 0.025, cy, ch, ha="left", va="center",
                 fontsize=ch_fs, color=DARK, family="sans-serif",
                 transform=fig.transFigure, zorder=12)
    brace_tip_x, _ = _vbrace(fig, ch_x + 0.005,
                              ch_top - (len(channels) - 1) * ch_lh - 0.004,
                              ch_top + 0.004,
                              depth=0.015, color=STEP_COLORS[3], lw=1.5)

    # ── main voxel figure ──
    fig_h = bh * 0.58
    fig_y = by + bh * 0.02
    _imshow(fig, [bx, fig_y, bw, fig_h],
            FIG_DIR / "fig_voxel_house.png")

    _save(fig, "panel_04_voxelization.png")


def panel_surrogate():
    """Panel 5 — Surrogate pipeline: voxel input → ensemble → predictions."""
    arch_path = POSTER_FINAL / "fig04_architecture.png"
    if not arch_path.exists():
        arch_path = FIG_DIR / "fig2_architecture.png"

    fig, body = _new_panel(
        "\u2464  Surrogate Model Training",
        "A 5-member deep ensemble of 3D CNNs predicts structural "
        "responses directly from voxelized buildings (11,178 FEA training runs)",
        hdr_fc=STEP_COLORS[4],
    )
    bx, by, bw, bh = body

    # ── pipeline row at top ──
    pipe_y = by + bh - 0.050
    pipe_h = 0.038
    # left: input box
    in_w = 0.14
    _step_box(fig, bx + 0.01, pipe_y, in_w, pipe_h,
              "Voxel Input\n(7 channels)", fc=NAVY, fs=8.5)

    # arrow
    _arrow(fig, bx + 0.01 + in_w + 0.005, pipe_y + pipe_h / 2,
           bx + 0.01 + in_w + 0.030, pipe_y + pipe_h / 2,
           color=DARK, lw=2.0, ms=12)

    # centre: 5 small ensemble member boxes with brace below
    ens_x0 = bx + 0.01 + in_w + 0.038
    ens_bw = 0.065
    ens_gap = 0.008
    ens_labels = ["CNN 1", "CNN 2", "CNN 3", "CNN 4", "CNN 5"]
    for i, el in enumerate(ens_labels):
        xi = ens_x0 + i * (ens_bw + ens_gap)
        _outline_box(fig, xi, pipe_y, ens_bw, pipe_h, el,
                     ec=STEP_COLORS[4], tc=STEP_COLORS[4], fs=7.5)
    ens_end = ens_x0 + len(ens_labels) * (ens_bw + ens_gap) - ens_gap

    # curly brace under ensemble
    brace_top = pipe_y - 0.005
    tip_x, tip_y = _hbrace(fig, ens_x0 + 0.008, ens_end - 0.008,
                            brace_top, depth=0.016, color=STEP_COLORS[4], lw=1.5)
    fig.text(tip_x, tip_y - 0.010, "Deep Ensemble",
             ha="center", va="center", fontsize=8, color=STEP_COLORS[4],
             fontweight="bold", family="sans-serif",
             transform=fig.transFigure, zorder=12)

    # arrow from ensemble to output
    _arrow(fig, ens_end + 0.005, pipe_y + pipe_h / 2,
           ens_end + 0.030, pipe_y + pipe_h / 2,
           color=DARK, lw=2.0, ms=12)

    # right: output boxes
    out_x = ens_end + 0.038
    outs = [("\u03c3, u, C", TEAL), ("Uncertainty", "#A3111A")]
    for j, (ol, oc) in enumerate(outs):
        ow = 0.09
        oy = pipe_y + j * (pipe_h + 0.006) - (len(outs) - 1) * (pipe_h + 0.006) / 2
        _step_box(fig, out_x, oy, ow, pipe_h, ol, fc=oc, fs=8.0)

    # ── main architecture figure ──
    fig_h = bh * 0.55
    fig_y = by + bh * 0.02
    _imshow(fig, [bx, fig_y, bw, fig_h], arch_path)

    _save(fig, "panel_05_surrogate.png")


def panel_optimization():
    """Panel 6 — SASTO iterative optimisation loop."""
    flow_path = POSTER_FINAL / "fig05_sasto_flowchart.png"
    if not flow_path.exists():
        flow_path = FIG_DIR / "fig_sasto_pipeline.png"

    fig, body = _new_panel(
        "\u2465  SASTO Optimization",
        "Surrogate-accelerated sensitivity-driven topology optimization "
        "removes material while preserving connectivity & constraints",
        hdr_fc=STEP_COLORS[5],
    )
    bx, by, bw, bh = body

    # ── iterative pipeline across top ──
    pipe_y = by + bh - 0.050
    pipe_h = 0.034
    steps = [
        ("Initial\nDesign",    NAVY),
        ("Surrogate\nEval",    STEP_COLORS[4]),
        ("Sensitivity\nCalc",  STEP_COLORS[2]),
        ("Voxel\nRemoval",    STEP_COLORS[5]),
        ("Constraint\nCheck",  TEAL),
    ]
    sbw = 0.105
    sgap = 0.028
    total = len(steps) * sbw + (len(steps) - 1) * sgap
    sx = bx + (bw - total) / 2
    box_centers = []
    for i, (lbl, clr) in enumerate(steps):
        xi = sx + i * (sbw + sgap)
        _step_box(fig, xi, pipe_y, sbw, pipe_h, lbl, fc=clr, fs=7.5)
        box_centers.append((xi, xi + sbw))
        if i < len(steps) - 1:
            _arrow(fig, xi + sbw + 0.004, pipe_y + pipe_h / 2,
                   xi + sbw + sgap - 0.004, pipe_y + pipe_h / 2,
                   color=DARK, lw=1.8, ms=10)

    # ── loop-back bracket from last box back to second box ──
    from matplotlib.lines import Line2D
    loop_y = pipe_y + pipe_h + 0.008
    loop_top = loop_y + 0.018
    x_from = box_centers[-1][0] + sbw / 2  # centre of last box
    x_to   = box_centers[1][0] + sbw / 2   # centre of second box
    for seg in [
        ([x_from, x_from], [pipe_y + pipe_h, loop_top]),   # up from last
        ([x_from, x_to],   [loop_top, loop_top]),           # across
    ]:
        fig.add_artist(Line2D(
            seg[0], seg[1], color=STEP_COLORS[5], linewidth=1.8,
            transform=fig.transFigure, clip_on=False, zorder=12,
        ))
    # arrow head going down into second box
    _darrow(fig, x_to, loop_top, pipe_y + pipe_h + 0.002,
            color=STEP_COLORS[5], lw=1.8, ms=10)
    fig.text((x_from + x_to) / 2, loop_top + 0.008, "iterate until convergence",
             ha="center", va="center", fontsize=7.5, color=STEP_COLORS[5],
             fontstyle="italic", family="sans-serif",
             transform=fig.transFigure, zorder=12)

    # ── main flowchart figure ──
    fig_h = bh * 0.53
    fig_y = by + bh * 0.02
    _imshow(fig, [bx, fig_y, bw, fig_h], flow_path)

    _save(fig, "panel_06_optimization.png")


def panel_results():
    """Panel 7 — Side-by-side HQ renders: baseline / SASTO-U / SASTO-PA."""
    fig, body = _new_panel(
        "\u2466  Optimized Structures",
        "Final designs preserve connectivity with up to 45% material "
        "reduction  \u2022  23\u201392\u00d7 faster than classical SIMP  \u2022  "
        "0 structural-constraint violations",
        hdr_fc=STEP_COLORS[6],
    )
    bx, by, bw, bh = body

    imgs = [
        POSTER_HQ / "original_solid.png",
        POSTER_HQ / "sasto_u_solid.png",
        POSTER_HQ / "sasto_pa_solid.png",
    ]
    labels = ["Baseline", "SASTO-Uniform", "SASTO-Part-Aware"]
    n = len(imgs)
    gap = 0.045
    sw = (bw - (n - 1) * gap) / n
    sh = bh * 0.78
    sy = by + bh * 0.14

    for i, (path, lbl) in enumerate(zip(imgs, labels)):
        ix = bx + i * (sw + gap)
        # light border behind image
        _rounded_rect(fig, ix - 0.004, sy - 0.004,
                      sw + 0.008, sh + 0.008,
                      fc="#EEF2FA", ec="#B0BDD4", lw=1.0, r=0.008, zo=5)
        _imshow(fig, [ix, sy, sw, sh], path, zo=7)
        _label_below(fig, ix + sw / 2, sy - 0.012, lbl, fs=11)
        if i < n - 1:
            _arrow(fig, ix + sw + 0.006, sy + sh * 0.5,
                   ix + sw + gap - 0.006, sy + sh * 0.5,
                   color=NAVY, lw=2.5, ms=15)

    _save(fig, "panel_07_results.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating polished methodology panels ...")
    panel_banner()
    panel_dataset()
    panel_meshing()
    panel_fea()
    panel_voxelization()
    panel_surrogate()
    panel_optimization()
    panel_results()
    print(f"\nAll panels saved to {OUT_DIR}")
