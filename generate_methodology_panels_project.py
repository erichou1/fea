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
    """Panel 2 — Structural meshing: surface model -> tet mesh."""
    fig, body = _new_panel(
        "\u2461  Structural Meshing",
        "Structural surface models are discretized into finite elements "
        "for simulation with SfePy",
        hdr_fc=STEP_COLORS[1],
    )
    bx, by, bw, bh = body
    occ = np.load(OPT_DIR / "fixed_occ.npz")["data"]

    gap = 0.06
    sw = (bw - gap) / 2
    sh = bh * 0.82
    sy = by + bh * 0.09

    ax1 = _ax(fig, [bx, sy, sw, sh], projection="3d")
    _draw_part_mesh(ax1, PLY_PART)
    _label_below(fig, bx + sw / 2, sy - 0.005, "Surface Model")

    _arrow(fig, bx + sw + 0.008, sy + sh * 0.5,
           bx + sw + gap - 0.008, sy + sh * 0.5,
           color=NAVY, lw=2.5, ms=15)

    ax2 = _ax(fig, [bx + sw + gap, sy, sw, sh], projection="3d")
    _draw_tet_mesh(ax2, occ[::3, ::3, ::3] > 0)
    _label_below(fig, bx + sw + gap + sw / 2, sy - 0.005,
                 "Tetrahedral Mesh")

    _save(fig, "panel_02_meshing.png")


def panel_fea():
    """Panel 3 — FEA stress visualisation (project-generated figure)."""
    fig, body = _new_panel(
        "\u2462  Finite Element Simulation",
        "Linear-elastic FEA (SfePy) computes stress, displacement, and "
        "compliance for each building under structural loading",
        hdr_fc=STEP_COLORS[2],
    )
    bx, by, bw, bh = body
    _imshow(fig, [bx, by, bw, bh],
            FIG_DIR / "fig16_fea_stress_placeholder.png")
    _save(fig, "panel_03_fea.png")


def panel_voxelization():
    """Panel 4 — Voxelization & multi-channel encoding."""
    fig, body = _new_panel(
        "\u2463  Voxelization & Preprocessing",
        "Structural meshes are voxelized to 128\u00b3 and encoded as "
        "7-channel tensors with preserved part labels",
        hdr_fc=STEP_COLORS[3],
    )
    bx, by, bw, bh = body
    _imshow(fig, [bx, by, bw, bh],
            FIG_DIR / "fig_voxel_house.png")
    _save(fig, "panel_04_voxelization.png")


def panel_surrogate():
    """Panel 5 — Deep-ensemble surrogate architecture."""
    arch_path = POSTER_FINAL / "fig04_architecture.png"
    if not arch_path.exists():
        arch_path = FIG_DIR / "fig2_architecture.png"

    fig, body = _new_panel(
        "\u2464  Surrogate Model Training",
        "A 5-member deep ensemble of 3D CNNs predicts structural "
        "responses directly from voxelized buildings (trained on 11,178 FEA runs)",
        hdr_fc=STEP_COLORS[4],
    )
    bx, by, bw, bh = body
    _imshow(fig, [bx, by, bw, bh], arch_path)
    _save(fig, "panel_05_surrogate.png")


def panel_optimization():
    """Panel 6 — SASTO flowchart."""
    flow_path = POSTER_FINAL / "fig05_sasto_flowchart.png"
    if not flow_path.exists():
        flow_path = FIG_DIR / "fig_sasto_pipeline.png"

    fig, body = _new_panel(
        "\u2465  SASTO Optimization",
        "Surrogate-accelerated sensitivity-driven topology optimization "
        "removes material while preserving connectivity and constraints",
        hdr_fc=STEP_COLORS[5],
    )
    bx, by, bw, bh = body
    _imshow(fig, [bx, by, bw, bh], flow_path)
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
