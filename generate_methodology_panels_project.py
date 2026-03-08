"""Generate methodology panel PNGs using real project figures and project data.

Outputs overwrite poster_images_extracted/panels/panel_00_banner.png ... panel_07_results.png.
All panels have the same size.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from pathlib import Path
from PIL import Image
from scipy.spatial import Delaunay

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
OUT_DIR = BASE / "poster_images_extracted" / "panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR = BASE / "figures"
POSTER_HQ = BASE / "poster_final" / "renders_hq"
POSTER_FINAL = BASE / "poster_final"
WIRE_PATH = BASE / "optimization" / "data" / "3dwire_raw" / "00472.npz"
OPT_DIR = BASE / "fea_ml" / "runs" / "v3" / "optimization_128"
PLY_PART = BASE / "figures" / "screenshot_stls" / "REF_original_colored.ply"

# ── Style ─────────────────────────────────────────────────────────────────────
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

PW, PH = 10.0, 6.5
DPI = 220
MARGIN = 0.03
HEADER_H = 0.10
CAPTION_H = 0.14
BODY_PAD = 0.035


def _card(fig, x, y, w, h, fc=PANEL_BG, ec=NAVY, lw=2.2, r=0.018, zo=1):
    fig.add_artist(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        transform=fig.transFigure, clip_on=False, zorder=zo,
    ))


def _header(fig, x, y, w, h, title, fc=NAVY, zo=2):
    fig.add_artist(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.018",
        facecolor=fc, edgecolor=fc, linewidth=0,
        transform=fig.transFigure, clip_on=False, zorder=zo,
    ))
    fig.add_artist(mpatches.Rectangle(
        (x, y), w, h * 0.45,
        facecolor=fc, edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=zo,
    ))
    fig.text(x + w / 2, y + h * 0.50, title,
             ha="center", va="center", fontsize=16, fontweight="bold",
             color=WHITE, transform=fig.transFigure, zorder=zo + 1)


def _axes(fig, rect, projection=None, z=6):
    if projection is None:
        ax = fig.add_axes(rect)
    else:
        ax = fig.add_axes(rect, projection=projection)
    ax.set_zorder(z)
    ax.patch.set_alpha(0)
    return ax


def _farrow(fig, x0, y0, x1, y1, color=DARK, lw=1.8, scale=12, zo=12):
    fig.add_artist(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="Simple,head_length=0.7,head_width=0.7,tail_width=0.28",
        mutation_scale=scale, facecolor=color, edgecolor=color,
        linewidth=lw, transform=fig.transFigure, clip_on=False, zorder=zo,
    ))


def _make_figure():
    fig = plt.figure(figsize=(PW, PH), facecolor=WHITE)
    cx, cy, cw, ch = MARGIN, MARGIN, 1 - 2 * MARGIN, 1 - 2 * MARGIN
    _card(fig, cx, cy, cw, ch)
    _header(fig, cx, cy + ch - HEADER_H, cw, HEADER_H, "")
    bx = cx + BODY_PAD
    by = cy + CAPTION_H
    bw = cw - 2 * BODY_PAD
    bh = ch - HEADER_H - CAPTION_H - BODY_PAD
    return fig, (cx, cy, cw, ch), (bx, by, bw, bh)


def _finish(fig, title, caption, bullets, outname):
    cx, cy, cw, ch = MARGIN, MARGIN, 1 - 2 * MARGIN, 1 - 2 * MARGIN
    _header(fig, cx, cy + ch - HEADER_H, cw, HEADER_H, title)
    fig.text(0.5, cy + 0.073, caption,
             ha="center", va="top", color=DARK,
             fontsize=9.0, fontstyle="italic", linespacing=1.35,
             transform=fig.transFigure)
    for i, b in enumerate(bullets):
        fig.text(0.20, cy + 0.038 - i * 0.022, f"• {b}",
                 ha="left", va="center", color=BLUE,
                 fontsize=8.5, transform=fig.transFigure)
    out = OUT_DIR / outname
    plt.savefig(str(out), dpi=DPI, bbox_inches="tight",
                facecolor=WHITE, edgecolor="none")
    plt.close(fig)
    print(f"  Saved → {out}")


def _trim_image(img: Image.Image, tol=8) -> Image.Image:
    arr = np.asarray(img.convert("RGBA"))
    rgb = arr[..., :3].astype(np.int16)
    corners = np.array([
        rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]
    ])
    bg = np.median(corners, axis=0)
    diff = np.abs(rgb - bg[None, None, :]).max(axis=2)
    alpha = arr[..., 3]
    mask = (diff > tol) & (alpha > 0)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return img.crop((x0, y0, x1, y1))


def _load_image(path, crop=None, trim=True):
    img = Image.open(path).convert("RGBA")
    if crop is not None:
        w, h = img.size
        x0, y0, x1, y1 = crop
        img = img.crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))
    if trim:
        img = _trim_image(img)
    return np.asarray(img)


def _show_image(fig, rect, path, crop=None, trim=True):
    ax = _axes(fig, rect)
    arr = _load_image(path, crop=crop, trim=trim)
    ax.imshow(arr)
    ax.axis("off")
    return ax


def _mesh_face_colors(mesh):
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        vc = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=float) / 255.0
        return vc[mesh.faces].mean(axis=1)
    return np.tile(np.array([[0.35, 0.55, 0.85]]), (len(mesh.faces), 1))


def _render_part_mesh(ax3d, ply_path, elev=22, azim=-58):
    mesh = trimesh.load(str(ply_path), force="mesh", process=False)
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    lo, hi = verts.min(0), verts.max(0)
    span_real = hi - lo
    span = span_real.max()
    verts = (verts - (lo + hi) / 2.0) / span
    poly_v = verts[faces]

    n0, n1, n2 = poly_v[:, 0], poly_v[:, 1], poly_v[:, 2]
    normals = np.cross(n1 - n0, n2 - n0)
    mag = np.linalg.norm(normals, axis=1, keepdims=True)
    mag[mag == 0] = 1.0
    normals = normals / mag
    light = np.array([
        np.cos(np.radians(35)) * np.cos(np.radians(-35)),
        np.cos(np.radians(35)) * np.sin(np.radians(-35)),
        np.sin(np.radians(35)),
    ])
    lambert = np.clip(normals @ light, 0, 1)
    base = _mesh_face_colors(mesh)
    shaded = np.clip(base * (0.84 + 0.16 * lambert[:, None]), 0, 1)

    poly = Poly3DCollection(poly_v, zsort="average")
    poly.set_facecolor(shaded)
    poly.set_edgecolor("none")
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


def _render_wireframe(ax3d, vertices, lines, elev=25, azim=-60):
    verts = np.asarray(vertices, dtype=float)
    lines = np.asarray(lines)
    for e in lines:
        p0, p1 = verts[e[0]], verts[e[1]]
        zf = ((p0[2] + p1[2]) * 0.5 - verts[:, 2].min()) / (np.ptp(verts[:, 2]) + 1e-9)
        if zf < 0.15:
            c = np.array(mcolors.to_rgb(SLAB))
        elif zf > 0.65:
            c = np.array(mcolors.to_rgb(ROOF))
        else:
            c = np.array(mcolors.to_rgb(WALL))
        ax3d.plot3D(*zip(p0, p1), color=c, linewidth=2.0)
    ax3d.scatter3D(verts[:, 0], verts[:, 1], verts[:, 2], color=RED, s=8, depthshade=False)
    lo, hi = verts.min(0), verts.max(0)
    span = hi - lo
    pad = span * 0.04
    ax3d.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax3d.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax3d.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    ax3d.set_box_aspect(span / span.max())
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_facecolor(WHITE)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
    ax3d.set_axis_off()


def _render_voxels(ax3d, occ, part, elev=22, azim=-55, stride=3):
    occ_s = occ[::stride, ::stride, ::stride].astype(bool)
    part_s = part[::stride, ::stride, ::stride]
    fc = np.zeros(occ_s.shape + (4,), dtype=float)
    cmap = {
        1: WALL,
        2: INTERIOR,
        3: ROOF,
        4: SLAB,
    }
    for k, c in cmap.items():
        mask = occ_s & (part_s == k)
        rgb = mcolors.to_rgb(c)
        fc[mask, :3] = rgb
        fc[mask, 3] = 0.98
    ec = np.zeros_like(fc)
    ec[..., :3] = 1.0
    ec[..., 3] = 0.12
    ax3d.voxels(occ_s, facecolors=fc, edgecolors=ec, linewidth=0.14)
    ax3d.set_box_aspect(list(occ_s.shape))
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_facecolor(WHITE)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
    ax3d.set_axis_off()


def _render_tet_mesh(ax3d, occ, max_points=240, elev=18, azim=-60):
    pts = np.argwhere(occ)
    rng = np.random.default_rng(12)
    if len(pts) > max_points:
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
    pts = pts.astype(float)
    pts = (pts - pts.mean(0)) / np.max(np.ptp(pts, axis=0))
    try:
        tet = Delaunay(pts)
        simplices = tet.simplices
        edge_idx = set()
        for tet4 in simplices:
            pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
            for a, b in pairs:
                i, j = sorted((int(tet4[a]), int(tet4[b])))
                edge_idx.add((i, j))
        segs = [(pts[i], pts[j]) for i, j in edge_idx]
        lc = Line3DCollection(segs, colors=BLUE, linewidths=0.35, alpha=0.65)
        ax3d.add_collection3d(lc)
    except Exception:
        ax3d.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2], s=4, color=BLUE)
    ax3d.set_xlim(-0.55, 0.55)
    ax3d.set_ylim(-0.55, 0.55)
    ax3d.set_zlim(-0.55, 0.55)
    ax3d.set_box_aspect([1, 1, 0.8])
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_facecolor(WHITE)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
    ax3d.set_axis_off()


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
    byc = cy + ch * 0.28
    for i, (s, sc) in enumerate(zip(steps, colors)):
        bx = cx + i * (bw + gap)
        fig.add_artist(FancyBboxPatch(
            (bx, byc - bh / 2), bw, bh,
            boxstyle="round,pad=0,rounding_size=0.010",
            facecolor=sc, edgecolor="none",
            transform=fig.transFigure, clip_on=False, zorder=3,
        ))
        fig.text(bx + bw / 2, byc, s, ha="center", va="center",
                 fontsize=9, fontweight="bold", color=WHITE,
                 linespacing=1.2, transform=fig.transFigure, zorder=4)
        if i < N - 1:
            _farrow(fig, bx + bw + 0.004, byc, bx + bw + gap - 0.004, byc,
                    color=DARK, lw=1.5, scale=9)
    out = OUT_DIR / "panel_00_banner.png"
    plt.savefig(str(out), dpi=DPI, bbox_inches="tight", facecolor=WHITE, edgecolor="none")
    plt.close(fig)
    print(f"  Saved → {out}")


def panel_dataset():
    fig, _, body = _make_figure()
    bx, by, bw, bh = body
    d = np.load(WIRE_PATH, allow_pickle=True)
    vertices, lines = d["vertices"], d["lines"]
    occ = np.load(OPT_DIR / "fixed_occ.npz")["data"]
    part = np.load(OPT_DIR / "fixed_part.npz")["data"]

    n = 3; pad = 0.018
    sw = (bw - (n + 1) * pad) / n
    sh = bh * 0.72
    sy = by + bh * 0.16

    ax1 = _axes(fig, [bx + pad, sy, sw, sh], projection="3d")
    _render_wireframe(ax1, vertices, lines)
    ax1.set_title("3DWire Wireframe", fontsize=11.5, color=DARK, pad=4, fontweight="bold")

    ax2 = _axes(fig, [bx + pad * 2 + sw, sy, sw, sh], projection="3d")
    _render_part_mesh(ax2, PLY_PART)
    ax2.set_title("Volumetric House", fontsize=11.5, color=DARK, pad=4, fontweight="bold")

    ax3 = _axes(fig, [bx + pad * 3 + sw * 2, sy, sw, sh], projection="3d")
    _render_voxels(ax3, occ, part, stride=4)
    ax3.set_title("Labeled Voxel Grid", fontsize=11.5, color=DARK, pad=4, fontweight="bold")

    ay = sy + sh * 0.50
    _farrow(fig, bx + pad + sw + 0.004, ay, bx + pad * 2 + sw - 0.004, ay, lw=2.0, scale=13)
    _farrow(fig, bx + pad * 2 + sw * 2 + 0.004, ay, bx + pad * 3 + sw * 2 - 0.004, ay, lw=2.0, scale=13)

    _finish(fig, "\u2460  Dataset Generation",
            "Residential building wireframes are converted into volumetric\nstructural models and voxelized with part labels.",
            ["Input: 3DWire building wireframes",
             "Output: volumetric structural models",
             "Parts: exterior / interior / roof / floor"],
            "panel_01_dataset.png")


def panel_meshing():
    fig, _, body = _make_figure()
    bx, by, bw, bh = body
    occ = np.load(OPT_DIR / "fixed_occ.npz")["data"]

    pad = 0.025
    sw = (bw - 3 * pad) / 2
    sh = bh * 0.72
    sy = by + bh * 0.17

    ax1 = _axes(fig, [bx + pad, sy, sw, sh], projection="3d")
    _render_part_mesh(ax1, PLY_PART)
    ax1.set_title("Structural Surface Model", fontsize=12, color=DARK, pad=4, fontweight="bold")

    ax2 = _axes(fig, [bx + pad * 2 + sw, sy, sw, sh], projection="3d")
    _render_tet_mesh(ax2, occ[::3, ::3, ::3] > 0)
    ax2.set_title("Representative Tetrahedral Mesh", fontsize=12, color=DARK, pad=4, fontweight="bold")

    ay = sy + sh * 0.50
    _farrow(fig, bx + pad + sw + 0.006, ay, bx + pad * 2 + sw - 0.006, ay, lw=2.0, scale=13)

    _finish(fig, "\u2461  Structural Meshing",
            "Volumetric building structures are discretized into finite elements\nfor structural simulation and response evaluation.",
            ["Surface geometry from project structural model",
             "Representative tetrahedral discretization",
             "Used for SfePy-based FEA simulation"],
            "panel_02_meshing.png")


def panel_fea():
    fig, _, body = _make_figure()
    bx, by, bw, bh = body
    _show_image(fig, [bx + 0.01, by + 0.02, bw - 0.02, bh - 0.02],
                FIG_DIR / "fig16_fea_stress_placeholder.png")
    _finish(fig, "\u2462  Finite Element Simulation",
            "Each building is simulated under structural loading conditions to\ncompute stress, displacement, and compliance.",
            ["Linear elastic FEA with SfePy",
             "Outputs: stress, displacement, compliance",
             "Project-generated stress visualization"],
            "panel_03_fea.png")


def panel_voxelization():
    fig, _, body = _make_figure()
    bx, by, bw, bh = body
    _show_image(fig, [bx + 0.01, by + 0.02, bw - 0.02, bh - 0.02], FIG_DIR / "fig_voxel_house.png")
    _finish(fig, "\u2463  Voxelization & Preprocessing",
            "Structural meshes are voxelized onto a regular 128\u00b3 grid and\nencoded as multi-channel tensors for learning.",
            ["128\u00b3 voxel resolution",
             "7-channel learning input",
             "Part labels preserved in the grid"],
            "panel_04_voxelization.png")


def panel_surrogate():
    fig, _, body = _make_figure()
    bx, by, bw, bh = body
    arch_path = POSTER_FINAL / "fig04_architecture.png"
    if not arch_path.exists():
        arch_path = FIG_DIR / "fig2_architecture.png"
    _show_image(fig, [bx + 0.01, by + 0.01, bw - 0.02, bh - 0.01], arch_path)
    _finish(fig, "\u2464  Surrogate Model Training",
            "A deep ensemble of 3D convolutional networks learns to predict\nstructural responses directly from voxelized buildings.",
            ["5-member deep ensemble",
             "Trained on 11,178 project simulations",
             "Predicts structural response without FEA"],
            "panel_05_surrogate.png")


def panel_optimization():
    fig, _, body = _make_figure()
    bx, by, bw, bh = body
    flow_path = POSTER_FINAL / "fig05_sasto_flowchart.png"
    if not flow_path.exists():
        flow_path = FIG_DIR / "fig_sasto_pipeline.png"
    _show_image(fig, [bx + 0.02, by + 0.01, bw - 0.04, bh - 0.01], flow_path)
    _finish(fig, "\u2465  SASTO Optimization",
            "Surrogate-accelerated topology optimization removes unnecessary\nmaterial while preserving structural constraints and connectivity.",
            ["Sensitivity-driven voxel removal",
             "Constraint-aware iterative refinement",
             "Topology remains printable and connected"],
            "panel_06_optimization.png")


def panel_results():
    fig, _, body = _make_figure()
    bx, by, bw, bh = body

    imgs = [
        POSTER_HQ / "original_solid.png",
        POSTER_HQ / "sasto_u_solid.png",
        POSTER_HQ / "sasto_pa_solid.png",
    ]
    labels = ["Baseline House", "SASTO-U", "SASTO-PA"]
    n = 3; pad = 0.018
    sw = (bw - (n + 1) * pad) / n
    sh = bh * 0.68
    sy = by + bh * 0.20

    for i, (path, lbl) in enumerate(zip(imgs, labels)):
        _show_image(fig, [bx + pad + i * (sw + pad), sy, sw, sh], path)
        fig.text(bx + pad + i * (sw + pad) + sw / 2, sy - 0.020, lbl,
                 ha="center", va="top", fontsize=10.5, color=DARK,
                 fontweight="bold", transform=fig.transFigure)
        if i < 2:
            ay = sy + sh * 0.50
            x0 = bx + pad + i * (sw + pad) + sw + 0.004
            x1 = x0 + pad - 0.008
            _farrow(fig, x0, ay, x1, ay, lw=2.0, scale=13)

    stats = ["Up to 45% material reduction", "23–92× faster than SIMP",
             "0 structural constraint violations"]
    for i, st in enumerate(stats):
        fig.text(0.50, by + 0.034 - i * 0.020, f"• {st}",
                 ha="center", va="center", fontsize=8.8, color=BLUE,
                 transform=fig.transFigure)

    _finish(fig, "\u2466  Optimized Structures",
            "The final designs preserve manufacturable connectivity while achieving\nsubstantial material reduction relative to the baseline structure.",
            [],
            "panel_07_results.png")


if __name__ == "__main__":
    print("Generating project-based methodology panels...")
    panel_banner()
    panel_dataset()
    panel_meshing()
    panel_fea()
    panel_voxelization()
    panel_surrogate()
    panel_optimization()
    panel_results()
    print(f"All saved to {OUT_DIR}")
