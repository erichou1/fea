"""
Dataset Generation Pipeline — five separate stage figures.

Stages:
  1. fig_stage1_npz.png          — Raw NPZ data (grid of numbers)
  2. fig_stage2_wireframe.png    — 2-D floor-plan wireframe
  3. fig_stage3_extrusion.png    — Volumetric extrusion (voxel stack)
  4. fig_stage4_watertight.png   — Watertight solid STL mesh
  5. fig_stage5_partlabels.png   — Part-labeled voxel model
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import trimesh
from pathlib import Path
from scipy.ndimage import binary_erosion, label as scipy_label
from skimage.measure import marching_cubes

# ── paths ──────────────────────────────────────────────────────────────────
SAMPLE  = Path("fea_ml/data/runs_real_128/00000")
STL     = Path("fea_ml/runs/v3/optimization_128/original_sharp.stl")
OUT     = Path("figures")
OUT.mkdir(exist_ok=True)

BG   = "#0d0d0d"   # used only for ax facecolor, NOT fig (transparent)
GRAY = "#e0e0e0"
ACC  = "#4fc3f7"

# ── helpers ────────────────────────────────────────────────────────────────
def _ax_clean(ax, title, desc, title_color=ACC):
    ax.set_title(title, color=title_color, fontsize=14, fontweight="bold", pad=10)
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.text(0.5, -0.06, desc, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color="#aaaaaa", wrap=True,
            multialignment="center")


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=180, bbox_inches="tight",
                facecolor="none", transparent=True)
    plt.close(fig)
    print(f"Saved → {path}")


# ── load data ───────────────────────────────────────────────────────────────
occ  = np.load(SAMPLE / "occ.npz")["data"]   # (128,128,128) uint8
part = np.load(SAMPLE / "part.npz")["data"]  # (128,128,128) int  0-4

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — NPZ raw data
# ═══════════════════════════════════════════════════════════════════════════
def stage1_npz():
    """Clean grid of numbers — boundary strip so both 0s and 1s visible."""
    # use z=48 (bottom wall slice) — guaranteed wall content
    sl = occ[:, :, 48]
    rows = np.where(sl.any(axis=1))[0]
    cols = np.where(sl.any(axis=0))[0]
    r0, c0 = rows[0], cols[0]

    CROP = 14
    rs = max(r0 - 1, 0)
    cs = max(c0 - 1, 0)
    crop = sl[rs:rs+CROP, cs:cs+CROP]

    fig, ax = plt.subplots(figsize=(6.8, 7.0), facecolor="none")
    ax.set_facecolor(BG)
    ax.set_xlim(-0.5, CROP - 0.5)
    ax.set_ylim(CROP - 0.5, -0.5)   # top-to-bottom
    ax.set_aspect("equal")

    for r in range(CROP):
        for c in range(CROP):
            val = int(crop[r, c])
            if val == 1:
                fc, ec, tc, fw = "#0d3348", "#4fc3f7", "#cceeff", "bold"
            else:
                fc, ec, tc, fw = "#141414", "#282828", "#505050", "normal"
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       facecolor=fc, edgecolor=ec, linewidth=0.9))
            ax.text(c, r, str(val), ha="center", va="center",
                    fontsize=11, color=tc,
                    fontfamily="monospace", fontweight=fw)

    # ── two clean side-panel annotations (no overlapping arrows) ──
    # Solid legend block — top-right outside grid
    ax.add_patch(plt.Rectangle((CROP + 0.2, 0.5), 1.5, 1.5,
                               facecolor="#0d3348", edgecolor="#4fc3f7", lw=1.1,
                               clip_on=False))
    ax.text(CROP + 0.95, 1.25, "1", ha="center", va="center",
            fontsize=11, color="#cceeff", fontfamily="monospace",
            fontweight="bold", clip_on=False)
    ax.text(CROP + 2.0, 1.25, "solid voxel", ha="left", va="center",
            fontsize=8.5, color="#4fc3f7", clip_on=False)

    # Empty legend block
    ax.add_patch(plt.Rectangle((CROP + 0.2, 3.0), 1.5, 1.5,
                               facecolor="#141414", edgecolor="#383838", lw=1.1,
                               clip_on=False))
    ax.text(CROP + 0.95, 3.75, "0", ha="center", va="center",
            fontsize=11, color="#505050", fontfamily="monospace",
            fontweight="normal", clip_on=False)
    ax.text(CROP + 2.0, 3.75, "empty voxel", ha="left", va="center",
            fontsize=8.5, color="#606060", clip_on=False)

    # row / col index labels
    for i in range(0, CROP, 2):
        ax.text(-0.7, i, str(rs + i), ha="right", va="center",
                fontsize=7, color="#3a6070", fontfamily="monospace")
        ax.text(i, -0.7, str(cs + i), ha="center", va="bottom",
                fontsize=7, color="#3a6070", fontfamily="monospace")
    ax.text(-1.5, CROP / 2 - 0.5, "row", ha="center", va="center",
            fontsize=7.5, color="#3a6070", rotation=90)
    ax.text(CROP / 2 - 0.5, -1.5, "col", ha="center", va="center",
            fontsize=7.5, color="#3a6070")

    ax.set_xlim(-2, CROP + 7)
    ax.set_ylim(CROP + 0.8, -2.2)

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax.set_title(".npz  —  Raw Voxel Array  (z = 48 slice)", color=ACC,
                 fontsize=13, fontweight="bold", pad=10)
    fig.text(0.5, 0.01,
             "128³ binary occupancy grid stored as a compressed NumPy archive.\n"
             "Each value is 1 (solid) or 0 (void).  "
             f"Shown: {CROP}×{CROP} boundary crop   |   {int(occ.sum()):,} solid voxels total.",
             ha="center", fontsize=7.5, color="#aaaaaa")
    save(fig, "fig_stage1_npz.png")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — Wireframe (2-D floor-plan)
# ═══════════════════════════════════════════════════════════════════════════
def stage2_wireframe():
    """Detailed top-down 2-D floor plan with part-coloured filled regions."""
    from skimage import measure
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    # ── crop to house bounding box ──────────────────────────────────────────
    ys, xs, zs = np.where(occ > 0)
    pad = 3
    r0c, r1c = max(ys.min()-pad, 0), min(ys.max()+pad+1, 128)
    c0c, c1c = max(xs.min()-pad, 0), min(xs.max()+pad+1, 128)

    def crop2d(arr2d):
        return arr2d[r0c:r1c, c0c:c1c]

    # Part colour scheme (for 2-D view)
    PART_CFG = [
        (4, "#1a4a22", "#2e7a38", 0.45, "Floor slab"),      # floor — green fill
        (1, "#0a2d3d", "#4fc3f7", 0.60, "Exterior wall"),   # ext wall — blue fill
        (2, "#0d1e30", "#2288aa", 0.55, "Interior wall"),   # int wall — darker blue
        (3, "#3a1505", "#e05a2b", 0.50, "Roof"),             # roof — orange fill
    ]

    # Z slices: use correct z for each part type
    Z_SLICES = {4: 48, 1: 55, 2: 55, 3: 72}

    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="none")
    ax.set_facecolor(BG)
    ax.set_aspect("equal")

    H = r1c - r0c
    W = c1c - c0c

    # fine grid
    step = 8
    for i in range(0, W + 1, step):
        ax.axvline(i, color="#141e22", lw=0.35, zorder=0)
    for i in range(0, H + 1, step):
        ax.axhline(i, color="#141e22", lw=0.35, zorder=0)

    # ── draw each part layer ────────────────────────────────────────────────
    legend_handles = []
    for part_id, fc_col, ec_col, alpha, label in PART_CFG:
        z_idx = Z_SLICES[part_id]
        sl = crop2d((part[:, :, z_idx] == part_id).astype(float))
        if sl.sum() < 2:
            continue
        contours = measure.find_contours(sl, 0.5)
        for cnt in contours:
            cx, cy = cnt[:, 1], cnt[:, 0]
            # filled polygon
            ax.fill(cx, cy, color=fc_col, alpha=alpha, zorder=2)
            # outline
            lw = 2.4 if part_id == 1 else 1.5
            ax.plot(cx, cy, color=ec_col, lw=lw, alpha=0.95,
                    solid_capstyle="round", zorder=3)
        legend_handles.append(
            mpatches.Patch(facecolor=fc_col, edgecolor=ec_col, label=label,
                           linewidth=1.2, alpha=0.9))

    # ── dimension lines ─────────────────────────────────────────────────────
    dim_kw = dict(color="#3a5a6a", lw=0.9, linestyle="--")
    txt_kw = dict(color="#5a8a9a", fontsize=7, ha="center", va="center",
                  fontfamily="monospace",
                  bbox=dict(fc=BG, ec="none", pad=1.5))
    # horizontal span
    ax.annotate("", xy=(W-1, H+2.5), xytext=(1, H+2.5),
                arrowprops=dict(arrowstyle="<->", color="#3a5a6a", lw=0.9))
    ax.text(W/2, H+2.5, f"{W} vox", **txt_kw)
    # vertical span
    ax.annotate("", xy=(W+2.5, 1), xytext=(W+2.5, H-1),
                arrowprops=dict(arrowstyle="<->", color="#3a5a6a", lw=0.9))
    ax.text(W+4, H/2, f"{H} vox", rotation=90, **txt_kw)

    # ── compass ─────────────────────────────────────────────────────────────
    cx_c, cy_c = W - 5, 5
    ax.annotate("", xy=(cx_c, cy_c - 3.5), xytext=(cx_c, cy_c),
                arrowprops=dict(arrowstyle="-|>", color="#4fc3f7", lw=1.3,
                                mutation_scale=10))
    ax.text(cx_c, cy_c - 4.2, "N", color="#4fc3f7", fontsize=8,
            ha="center", va="top", fontweight="bold")

    ax.set_xlim(-1, W + 7)
    ax.set_ylim(H + 6, -1)

    ax.legend(handles=legend_handles, loc="lower left",
              facecolor="#0a1418", edgecolor="#2a4050",
              labelcolor=GRAY, fontsize=8, framealpha=0.92,
              handlelength=1.3, borderpad=0.7)

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_title("Wireframe  —  2-D Floor Plan", color=ACC, fontsize=13,
                 fontweight="bold", pad=10)
    fig.text(0.5, 0.005,
             "Plan-view contours extracted from horizontal part-label slices.\n"
             "Regions are colour-coded by structural role: floor, exterior wall,\n"
             "interior partitions, and roof projection.",
             ha="center", fontsize=7.5, color="#aaaaaa")
    save(fig, "fig_stage2_wireframe.png")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — Volumetric extrusion
# ═══════════════════════════════════════════════════════════════════════════
def stage3_extrusion():
    """3-D isometric view of the voxel grid shown as stacked slabs."""
    fig = plt.figure(figsize=(6, 5.5), facecolor="none")
    ax  = fig.add_subplot(111, projection="3d", facecolor=BG)

    DS = 8          # downsample factor: 128 → 16
    N  = 128 // DS
    vol = occ[:N*DS, :N*DS, :N*DS].reshape(N, DS, N, DS, N, DS).max(axis=(1, 3, 5))

    filled = np.argwhere(vol > 0)
    zvals  = filled[:, 2] / N          # normalise 0-1 for colour

    cmap = plt.cm.cool
    norm = plt.Normalize(0, 1)

    # Draw each voxel as a rectangular bar from z=0 to z=voxel top
    # (gives "extrusion" feel)
    bar_w = 0.75
    for (xi, yi, zi) in filled:
        color = cmap(norm(zi / N))
        ax.bar3d(xi, yi, 0, bar_w, bar_w, zi + 1,
                 color=color, alpha=0.65, shade=True, edgecolor="none")

    ax.set_box_aspect([1, 1, 0.55])
    ax.view_init(elev=28, azim=-50)
    ax.set_axis_off()
    ax.set_title("Volumetric Extrusion", color=ACC, fontsize=13,
                 fontweight="bold", pad=6)
    fig.text(0.5, 0.02,
             "2-D floor plan polygons extruded upward into a full 3-D voxel volume.\n"
             "Each column of voxels corresponds to a vertical structural element\n"
             "(wall, column, or roof slab). Resolution: 128 × 128 × 128.",
             ha="center", fontsize=7.5, color="#aaaaaa")
    save(fig, "fig_stage3_extrusion.png")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4 — Watertight solid STL
# ═══════════════════════════════════════════════════════════════════════════
def stage4_watertight():
    """Smooth solid mesh built from occ.npz (sample 00000) via marching cubes."""
    from skimage.measure import marching_cubes
    from scipy.ndimage import gaussian_filter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    occ_s = gaussian_filter(occ.astype(float), sigma=0.8)
    verts, faces, _, _ = marching_cubes(occ_s, level=0.5, step_size=2)

    # centre & normalise
    verts = verts - verts.mean(axis=0)
    scale = (verts.max(axis=0) - verts.min(axis=0)).max()
    verts /= scale

    fig = plt.figure(figsize=(6, 5.5), facecolor="none")
    ax  = fig.add_subplot(111, projection="3d", facecolor=BG)

    tris = verts[faces]

    # Normal-based shading
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(lens > 0, lens, 1)
    light = np.array([0.4, 0.5, 0.8])
    light /= np.linalg.norm(light)
    brightness = np.clip(normals @ light, 0.0, 1.0)

    base_r, base_g, base_b = 0.60, 0.82, 0.97
    fc = np.stack([base_r * (0.65 + 0.35 * brightness),
                   base_g * (0.65 + 0.35 * brightness),
                   base_b * (0.65 + 0.35 * brightness),
                   np.full(len(faces), 0.93)], axis=1)

    poly = Poly3DCollection(tris, facecolors=np.clip(fc, 0, 1),
                            edgecolors="none", linewidth=0)
    ax.add_collection3d(poly)

    pad = 0.55
    ax.set_xlim(-pad, pad); ax.set_ylim(-pad, pad); ax.set_zlim(-pad, pad)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=18, azim=-55)
    ax.set_axis_off()

    ax.set_title("Watertight Solid", color=ACC, fontsize=13,
                 fontweight="bold", pad=6)
    fig.text(0.5, 0.02,
             "Marching-cubes isosurface extracted from the occ.npz voxel grid,\n"
             "producing a smooth, manifold, watertight surface mesh.\n"
             f"Mesh: {len(faces):,} triangles from sample 00000",
             ha="center", fontsize=7.5, color="#aaaaaa")
    save(fig, "fig_stage4_watertight.png")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5 — Part-labeled model
# ═══════════════════════════════════════════════════════════════════════════
def stage5_partlabels():
    """Watertight mesh (marching cubes on occ) with triangles coloured by part.npz."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    PART_COLORS = {
        0: np.array([0.08, 0.08, 0.08]),   # empty
        1: np.array([0.20, 0.60, 0.95]),   # exterior wall — blue
        2: np.array([0.95, 0.52, 0.12]),   # interior wall — orange
        3: np.array([0.22, 0.72, 0.35]),   # roof — green
        4: np.array([0.76, 0.65, 0.42]),   # floor — tan
    }
    PART_LABELS = {1: "Exterior Wall", 2: "Interior Wall",
                   3: "Roof", 4: "Floor / Slab"}

    # ── build mesh from occ (same sample as part) via marching cubes ──
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    occ_smooth = gaussian_filter(occ.astype(float), sigma=0.8)
    verts_mc, faces_mc, _, _ = marching_cubes(occ_smooth, level=0.5, step_size=2)

    # ── dilate part labels so stray centroids always resolve to a real part ──
    # For each non-zero label, expand it by 4 voxels into unlabeled space
    from scipy.ndimage import grey_dilation
    part_filled = part.copy()
    for pid in [4, 1, 2, 3]:   # priority: floor first, then walls, then roof
        mask = (part_filled == 0) & (occ > 0)
        dilated = grey_dilation(part_filled == pid, size=9).astype(bool)
        part_filled[mask & dilated] = pid
    # Any remaining occ=1 part=0 → nearest via full dilation
    still_empty = (part_filled == 0) & (occ > 0)
    if still_empty.any():
        from scipy.ndimage import label as nd_label
        # fallback: assign to nearest known label via distance transform
        known = part_filled > 0
        _, nearest = distance_transform_edt(~known, return_indices=True)
        part_filled[still_empty] = part_filled[
            nearest[0][still_empty], nearest[1][still_empty], nearest[2][still_empty]]

    tris      = verts_mc[faces_mc]
    centroids = tris.mean(axis=1)
    idx       = np.clip(np.round(centroids).astype(int), 0, 127)
    face_parts = part_filled[idx[:, 0], idx[:, 1], idx[:, 2]]

    # ── Phong shading ──
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    normals = np.cross(v1 - v0, v2 - v0).astype(float)
    lens    = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(lens > 0, lens, 1.0)
    light   = np.array([0.4, 0.4, 0.75])
    light  /= np.linalg.norm(light)
    shade   = 0.65 + 0.35 * np.clip(normals @ light, 0.0, 1.0)  # (F,)

    base_rgb = np.array([PART_COLORS[p] for p in face_parts])    # (F, 3)
    fc_rgba  = np.concatenate([
        np.clip(base_rgb * shade[:, None], 0, 1),
        np.full((len(faces_mc), 1), 0.93)
    ], axis=1)

    # ── centre & scale for display ──
    vc = verts_mc - verts_mc.mean(axis=0)
    sc = (vc.max(axis=0) - vc.min(axis=0)).max()
    vc /= sc
    tris_c = vc[faces_mc]

    fig = plt.figure(figsize=(6.5, 5.8), facecolor="none")
    ax  = fig.add_subplot(111, projection="3d", facecolor=BG)

    poly = Poly3DCollection(tris_c, facecolors=fc_rgba,
                            edgecolors="none", linewidth=0)
    ax.add_collection3d(poly)

    pad = 0.55
    ax.set_xlim(-pad, pad); ax.set_ylim(-pad, pad); ax.set_zlim(-pad, pad)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=18, azim=-52)
    ax.set_axis_off()

    handles = [mpatches.Patch(facecolor=PART_COLORS[pid], label=lbl,
                              edgecolor="#555", linewidth=0.8)
               for pid, lbl in PART_LABELS.items()]
    ax.legend(handles=handles, loc="upper left",
              facecolor="#111111", edgecolor="#333333",
              labelcolor=GRAY, fontsize=8.5, framealpha=0.88,
              handlelength=1.4, borderpad=0.7)

    ax.set_title("Part-Labeled Model", color=ACC, fontsize=13,
                 fontweight="bold", pad=6)
    fig.text(0.5, 0.01,
             "Isosurface mesh coloured by semantic part label from part.npz.\n"
             "Each triangle's centroid is mapped to the voxel grid to assign:\n"
             "exterior wall, interior wall, roof, or floor slab.",
             ha="center", fontsize=7.5, color="#aaaaaa")
    save(fig, "fig_stage5_partlabels.png")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2b — 3-D wireframe from NPZ
# ═══════════════════════════════════════════════════════════════════════════
def stage_wireframe_3d():
    """3-D wireframe skeleton of the house extracted from occ.npz."""
    from skimage.measure import marching_cubes
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from scipy.ndimage import gaussian_filter

    # coarse marching cubes so edge count stays manageable
    occ_s = gaussian_filter(occ.astype(float), sigma=1.0)
    verts, faces, _, _ = marching_cubes(occ_s, level=0.5, step_size=3)

    # build unique edge set from triangles
    edge_set = set()
    for f in faces:
        for a, b in [(f[0],f[1]), (f[1],f[2]), (f[2],f[0])]:
            edge_set.add((min(a,b), max(a,b)))
    edges = np.array(list(edge_set), dtype=int)   # (E, 2)

    # segments: (E, 2, 3)
    segs = verts[edges]          # each row: [[x0,y0,z0],[x1,y1,z1]]

    # centre & normalise
    vc = verts - verts.mean(axis=0)
    sc = (vc.max(axis=0) - vc.min(axis=0)).max()
    vc /= sc
    segs_c = vc[edges]           # centred segments

    # colour edges by mid-point z height
    midz    = segs_c[:, :, 2].mean(axis=1)   # (E,)
    norm_z  = (midz - midz.min()) / ((midz.max() - midz.min()) + 1e-9)
    cmap    = plt.cm.cool
    colors  = cmap(norm_z)
    colors[:, 3] = 0.55 + 0.40 * norm_z    # higher = more opaque

    fig = plt.figure(figsize=(6, 5.8), facecolor="none")
    ax  = fig.add_subplot(111, projection="3d", facecolor=BG)

    lc = Line3DCollection(segs_c, colors=colors, linewidths=0.45)
    ax.add_collection3d(lc)

    pad = 0.55
    ax.set_xlim(-pad, pad); ax.set_ylim(-pad, pad); ax.set_zlim(-pad, pad)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=-52)
    ax.set_axis_off()

    ax.set_title("3-D Wireframe  (from NPZ)", color=ACC, fontsize=13,
                 fontweight="bold", pad=6)
    fig.text(0.5, 0.02,
             "Marching-cubes isosurface rendered as edges only, derived directly\n"
             "from the binary occ.npz voxel grid.  Colour encodes height.\n"
             f"Surface: {len(verts):,} vertices, {len(faces):,} triangles   →   {len(edges):,} unique edges",
             ha="center", fontsize=7.5, color="#aaaaaa")
    save(fig, "fig_stage_wireframe_3d.png")


# ── run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Stage 1 — NPZ raw data …")
    stage1_npz()

    print("Stage 2 — Wireframe …")
    stage2_wireframe()

    print("Stage 2b — 3-D wireframe …")
    stage_wireframe_3d()

    print("Stage 3 — Volumetric extrusion …")
    stage3_extrusion()

    print("Stage 4 — Watertight solid …")
    stage4_watertight()

    print("Stage 5 — Part labels …")
    stage5_partlabels()

    print("\nAll stage figures saved to figures/")
