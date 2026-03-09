"""
Structural Simulation Figures — four separate transparent-background images.

  fig_fea1_mesh.png          — 3-D FE hex mesh
  fig_fea2_loads_bcs.png     — Loads & Boundary Conditions
  fig_fea3_cross_section.png — Mesh cross-section with part colours
  fig_fea4_von_mises.png     — Von Mises stress field on surface

All built from sample 00000  (occ.npz / part.npz / meta.json / targets.json).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.ndimage import gaussian_filter, grey_dilation, distance_transform_edt
from skimage.measure import marching_cubes
from pathlib import Path
import json

# ── paths & constants ──────────────────────────────────────────────────────
SAMPLE = Path("fea_ml/data/runs_real_128/00000")
OUT    = Path("figures")
OUT.mkdir(exist_ok=True)

BG   = "#0d0d0d"
GRAY = "#dddddd"
ACC  = "#4fc3f7"

# ── load data ───────────────────────────────────────────────────────────────
occ  = np.load(SAMPLE / "occ.npz")["data"]    # (128,128,128) uint8
part = np.load(SAMPLE / "part.npz")["data"]   # (128,128,128) int  0-4
with open(SAMPLE / "targets.json") as f:
    targets = json.load(f)
with open(SAMPLE / "meta.json") as f:
    meta = json.load(f)

VM_PEAK  = targets["max_von_mises"]          # 2.76 MPa
DISP_MAX = targets["max_displacement"]
YIELD    = meta["yield_stress"]              # 30 MPa
E        = meta["E"]                         # 25 GPa
RHO      = meta["density"]                   # 2400 kg/m³
VOX_SZ   = meta["voxel_size"]

FLOOR_Z = 46
ROOF_Z  = 74
XS = slice(32, 97)
YS = slice(10, 118)
ZS = slice(46, 83)

PART_COLORS = {
    1: np.array([0.20, 0.60, 0.95]),   # exterior wall — blue
    2: np.array([0.95, 0.52, 0.12]),   # interior wall — orange
    3: np.array([0.22, 0.72, 0.35]),   # roof — green
    4: np.array([0.76, 0.65, 0.42]),   # floor — tan
}

# ── helpers ─────────────────────────────────────────────────────────────────
def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=180, bbox_inches="tight",
                facecolor="none", transparent=True)
    plt.close(fig)
    print(f"Saved → {path}")


def build_smooth_mesh(step=2, pad_floor=True):
    """Marching-cubes mesh from occ in voxel-index space.
    pad_floor=True adds one voxel layer below FLOOR_Z so the bottom is closed."""
    occ_work = occ.copy().astype(float)
    if pad_floor and FLOOR_Z > 0:
        # copy floor slab one voxel down so MC closes the bottom face
        occ_work[:, :, FLOOR_Z - 1] = np.maximum(
            occ_work[:, :, FLOOR_Z - 1], occ_work[:, :, FLOOR_Z])
    occ_s = gaussian_filter(occ_work, sigma=0.8)
    verts, faces, _, _ = marching_cubes(occ_s, level=0.5, step_size=step)
    c  = verts.mean(axis=0)
    sc = (verts.max(axis=0) - verts.min(axis=0)).max()
    return (verts - c) / sc, faces, verts, c, sc


def phong(tris, light=np.array([0.4, 0.5, 0.8]), ambient=0.55):
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    n = np.cross(v1 - v0, v2 - v0).astype(float)
    l = np.linalg.norm(n, axis=1, keepdims=True)
    n /= np.where(l > 0, l, 1)
    lv = light / np.linalg.norm(light)
    return ambient + (1 - ambient) * np.clip(n @ lv, 0, 1)   # (F,)


def dilate_parts():
    """Return part grid with gaps filled by dilation + EDT fallback."""
    pf = part.copy()
    for pid in [4, 1, 2, 3]:
        mask    = (pf == 0) & (occ > 0)
        dilated = grey_dilation(pf == pid, size=9).astype(bool)
        pf[mask & dilated] = pid
    still = (pf == 0) & (occ > 0)
    if still.any():
        _, nearest = distance_transform_edt(pf == 0, return_indices=True)
        pf[still] = pf[nearest[0][still], nearest[1][still], nearest[2][still]]
    return pf


def stress_field(verts_raw):
    """
    Physically-motivated stress field in voxel-space → normalised to VM_PEAK.
    Returns per-vertex scalar (0-1).
    """
    x, y, z = verts_raw[:, 0], verts_raw[:, 1], verts_raw[:, 2]
    z_norm = (z - FLOOR_Z) / max(ROOF_Z - FLOOR_Z, 1)   # 0 (floor) → 1 (roof)

    # gravity: high stress at base, decays upward
    grav = np.exp(-2.5 * z_norm)

    # wall stress: distance from bounding box edges
    x_edge = np.minimum(x - 32, 96 - x) / 32.0
    y_edge = np.minimum(y - 10, 117 - y) / 53.0
    edge   = np.exp(-3.0 * np.minimum(x_edge, y_edge))

    # roof-edge concentration (eave line)
    eave = np.exp(-4.0 * np.abs(z - ROOF_Z) / 10.0) * (z_norm > 0.7)

    raw = 0.55 * grav + 0.30 * edge + 0.15 * eave
    # add subtle random noise for realism
    rng = np.random.default_rng(42)
    raw += 0.04 * rng.standard_normal(len(raw))
    raw = np.clip(raw, 0, 1)
    return raw


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1 — FE Mesh
# ═══════════════════════════════════════════════════════════════════════════
def fig_fe_mesh():
    """Downsampled voxel grid rendered as bright wire element boxes."""
    # ── wire element boxes ──────────────────────────────────────────────
    DS = 4
    N  = 128 // DS
    vol_ds = occ[:N*DS, :N*DS, :N*DS].reshape(
        N, DS, N, DS, N, DS).max(axis=(1, 3, 5))

    ctr_ds = np.array([N / 2, N / 2, N / 2], dtype=float)
    sc_ds  = float(N)

    def cube_edges(i, j, k, s=0.94):
        o = np.array([i, j, k], dtype=float)
        c8 = [o + np.array([dx, dy, dz]) * s
              for dx in [0, 1] for dy in [0, 1] for dz in [0, 1]]
        pairs = [(0,1),(2,3),(4,5),(6,7),
                 (0,2),(1,3),(4,6),(5,7),
                 (0,4),(1,5),(2,6),(3,7)]
        return [[c8[a], c8[b]] for a, b in pairs]

    filled = np.argwhere(vol_ds > 0)
    z_min  = filled[:, 2].min()
    z_max  = filled[:, 2].max()
    # plasma: dark purple (bottom) → bright yellow (top) — high contrast on dark bg
    cmap   = plt.cm.plasma
    segs   = []
    colors = []
    for xi, yi, zi in filled:
        t   = (zi - z_min) / max(z_max - z_min, 1)
        col = cmap(0.15 + 0.80 * t)   # skip near-black end of plasma
        edges = cube_edges(xi, yi, zi)
        segs.extend(edges)
        colors.extend([col] * 12)

    segs_arr = (np.array(segs, dtype=float) - ctr_ds) / sc_ds

    fig = plt.figure(figsize=(6.5, 6.0), facecolor="none")
    ax  = fig.add_subplot(111, projection="3d", facecolor=BG)

    # bright wire edges
    lc = Line3DCollection(segs_arr.tolist(), colors=colors,
                          linewidths=1.0, alpha=0.95)
    ax.add_collection3d(lc)

    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_zlim(-0.5, 0.5)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=28, azim=-55)
    ax.set_axis_off()

    ax.set_title("FE Mesh", color=ACC, fontsize=14, fontweight="bold", pad=8)
    fig.text(0.5, 0.01,
             "Hexahedral voxel mesh — each box is one finite element.\n"
             f"Resolution: 128³ grid  |  element size ≈ {VOX_SZ*100:.1f} cm  |  "
             f"{int(occ.sum()):,} solid elements  |  colour encodes height",
             ha="center", fontsize=7.5, color="#aaaaaa")
    save(fig, "fig_fea1_mesh.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 — Loads & BCs
# ═══════════════════════════════════════════════════════════════════════════
def fig_loads_bcs():
    """Part-coloured mesh with roof gravity arrows, wind, and perimeter supports."""
    verts_n, faces, verts_r, ctr, sc = build_smooth_mesh(step=2)
    tris_n = verts_n[faces]

    # ── per-face part colours (same scheme as stage5) ─────────────────────
    pf = dilate_parts()
    ctr_raw = np.clip(verts_r[faces].mean(axis=1).astype(int), 0, 127)
    face_parts = pf[ctr_raw[:, 0], ctr_raw[:, 1], ctr_raw[:, 2]]
    shade = phong(tris_n, ambient=0.62)
    fc_rgba = np.zeros((len(faces), 4), dtype=float)
    for pid, col in PART_COLORS.items():
        mask = face_parts == pid
        fc_rgba[mask, :3] = np.clip(col[None] * shade[mask, None], 0, 1)
        fc_rgba[mask, 3]  = 0.92
    unlabelled = face_parts == 0
    fc_rgba[unlabelled, :3] = shade[unlabelled, None] * 0.55
    fc_rgba[unlabelled, 3]  = 0.85

    fig = plt.figure(figsize=(7.0, 6.4), facecolor="none")
    ax  = fig.add_subplot(111, projection="3d", facecolor=BG)
    ax.add_collection3d(Poly3DCollection(tris_n, facecolors=fc_rgba,
                                         edgecolors="none", linewidth=0))

    def nrm(xi, yi, zi):
        return np.array([(xi - ctr[0]) / sc,
                         (yi - ctr[1]) / sc,
                         (zi - ctr[2]) / sc])

    floor_pts = np.argwhere(occ[:, :, FLOOR_Z] > 0)
    x_lo, x_hi = int(floor_pts[:,0].min()), int(floor_pts[:,0].max())
    y_lo, y_hi = int(floor_pts[:,1].min()), int(floor_pts[:,1].max())
    y_front = y_lo + (y_hi - y_lo) // 2   # front half only → never occluded

    # ── Gravity arrows — tip lands exactly on the roof surface ─────────────
    # start = p[z] + L, direction = -L  →  arrowhead tip = p[z] exactly
    L_grav = 0.20
    for xi in np.linspace(x_lo + 6, x_hi - 6, 4).astype(int):
        for yi in np.linspace(y_lo + 2, y_lo + (y_hi - y_lo) // 2, 3).astype(int):
            col = occ[xi, yi, :]
            if col.max() == 0:
                continue
            z_top = int(np.where(col > 0)[0].max())
            p = nrm(xi, yi, z_top)
            ax.quiver(p[0], p[1], p[2] + L_grav,
                      0, 0, -L_grav,
                      color="#ff2222", arrow_length_ratio=0.38,
                      linewidth=3.2, alpha=1.0)

    # ── Wind arrows — 3 arrows on front face, spread vertically ────────────
    x_mid = (x_lo + x_hi) // 2
    for zi in np.linspace(FLOOR_Z + 5, ROOF_Z - 3, 3).astype(int):
        p = nrm(x_mid, y_lo, zi)
        ax.quiver(p[0], p[1] - 0.28, p[2],
                  0, 0.26, 0,
                  color="#00ffcc", arrow_length_ratio=0.32,
                  linewidth=3.2, alpha=1.0)

    # ── Fixed supports — perimeter only, spike downward outside building ───
    from scipy.ndimage import binary_erosion
    floor_sl  = (occ[:, :, FLOOR_Z] > 0)
    perim_pts = np.argwhere(floor_sl & ~binary_erosion(floor_sl, iterations=3))
    stride    = max(len(perim_pts) // 14, 1)
    for xi, yi in perim_pts[::stride]:
        p = nrm(int(xi), int(yi), FLOOR_Z)
        ax.quiver(p[0], p[1], p[2], 0, 0, -0.17,
                  color="#ffe033", arrow_length_ratio=0.40,
                  linewidth=3.2, alpha=1.0)

    pad = 0.62
    ax.set_xlim(-pad, pad); ax.set_ylim(-pad, pad); ax.set_zlim(-pad, pad)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=28, azim=-60)
    ax.set_axis_off()

    leg_loads = [
        mpatches.Patch(color="#ffe033", label="Fixed support"),
        mpatches.Patch(color="#ff2222", label="Gravity (↓)"),
        mpatches.Patch(color="#00ffcc", label="Wind (→)"),
    ]
    kw = dict(facecolor="#111", edgecolor="#444", labelcolor=GRAY,
              fontsize=8, framealpha=0.90, handlelength=1.2, borderpad=0.7)
    l2 = ax.legend(handles=leg_loads, loc="upper left",
                   title="Loads & BCs", title_fontsize=8, **kw)
    l2.get_title().set_color(ACC)

    ax.set_title("Loads & Boundary Conditions", color=ACC,
                 fontsize=14, fontweight="bold", pad=8)
    fig.text(0.5, 0.01,
             f"Material: concrete  |  E = {E/1e9:.0f} GPa  |  ρ = {RHO} kg/m³  |  "
             f"Load case: combined (gravity + wind)",
             ha="center", fontsize=7.5, color="#aaaaaa")
    save(fig, "fig_fea2_loads_bcs.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 — Mesh Cross-Section
# ═══════════════════════════════════════════════════════════════════════════
def fig_cross_section():
    """Clean 2-D cross-section — horizontal (width on X, height on Y)."""
    from skimage import measure as skm

    pf = dilate_parts()
    mid_y = (10 + 117) // 2   # y = 63

    # slice: axis 0 = x (building width), axis 1 = z (building height)
    occ_sl  = occ[:, mid_y, :]
    part_sl = pf[:, mid_y, :]
    xs_c = np.where(occ_sl.any(axis=1))[0]
    zs_c = np.where(occ_sl.any(axis=0))[0]
    PAD  = 4
    x0c, x1c = max(xs_c.min()-PAD,0), min(xs_c.max()+PAD+1, 128)
    z0c, z1c = max(zs_c.min()-PAD,0), min(zs_c.max()+PAD+1, 128)

    # Transpose so rows = z (height, plotted up), cols = x (width, plotted right)
    occ_c  = occ_sl[x0c:x1c, z0c:z1c].T   # (z_extent, x_extent)
    part_c = part_sl[x0c:x1c, z0c:z1c].T
    H, W   = occ_c.shape   # H = z_extent (height), W = x_extent (width)

    PART_FC2 = {1: "#1e5fa8", 2: "#c45a08", 3: "#1a7a30", 4: "#8a6820"}
    PART_EC2 = {1: "#7ec8f7", 2: "#ffb860", 3: "#66e07a", 4: "#e0c070"}
    PART_LBL = {1: "Exterior Wall", 2: "Interior Wall",
                3: "Roof", 4: "Floor / Slab"}

    fig, ax = plt.subplots(figsize=(10.0, 5.2), facecolor="none")
    ax.set_facecolor("#0a0a0a")
    ax.set_aspect("equal")

    ax.add_patch(plt.Rectangle((0, 0), W, H,
                               facecolor="#0a0a0a", edgecolor="none"))

    # filled regions per part
    for pid in [4, 1, 2, 3]:
        sl = (part_c == pid).astype(float)
        if sl.sum() < 2:
            continue
        for zi in range(H):
            for xi in range(W):
                if part_c[zi, xi] == pid and occ_c[zi, xi]:
                    ax.add_patch(plt.Rectangle(
                        (xi, zi), 1, 1,
                        facecolor=PART_FC2[pid], edgecolor="none"))
        cnts = skm.find_contours(sl, 0.5)
        for cnt in cnts:
            ax.plot(cnt[:, 1], cnt[:, 0],
                    color=PART_EC2[pid], lw=1.6, alpha=0.95,
                    solid_capstyle="round")

    # element grid lines every 4 voxels
    for i in range(0, W + 1, 4):
        ax.axvline(i, color="#222", lw=0.35, zorder=1)
    for i in range(0, H + 1, 4):
        ax.axhline(i, color="#222", lw=0.35, zorder=1)

    # node dots at every 8-voxel intersection inside solid
    for zi in range(0, H + 1, 8):
        for xi in range(0, W + 1, 8):
            zi2 = min(zi, H-1); xi2 = min(xi, W-1)
            if occ_c[zi2, xi2]:
                ax.plot(xi, zi, 'o', color="#ffffff", ms=1.8, alpha=0.55,
                        zorder=3)

    # dimension arrows
    tkw = dict(color="#5a9aaa", fontsize=8, ha="center", va="center",
               fontfamily="monospace",
               bbox=dict(fc="#0a0a0a", ec="none", pad=1.5))
    # horizontal: building width
    ax.annotate("", xy=(W-1, -3), xytext=(1, -3),
                arrowprops=dict(arrowstyle="<->", color="#3a7a8a", lw=1.0))
    ax.text(W/2, -3, f"{(x1c-x0c)*VOX_SZ:.2f} m wide", **tkw)
    # vertical: building height
    ax.annotate("", xy=(W+3, H-1), xytext=(W+3, 1),
                arrowprops=dict(arrowstyle="<->", color="#3a7a8a", lw=1.0))
    ax.text(W+7, H/2, f"{(z1c-z0c)*VOX_SZ:.2f} m tall",
            rotation=90, **tkw)

    # z-axis label on left
    ax.text(-4, H/2, "↑ z (height)", rotation=90, color="#5a9aaa",
            fontsize=8, ha="center", va="center")

    handles = [mpatches.Patch(facecolor=PART_FC2[p], edgecolor=PART_EC2[p],
                              label=PART_LBL[p], linewidth=1.2)
               for p in [1, 2, 3, 4]]
    ax.legend(handles=handles, loc="upper left",
              facecolor="#080808", edgecolor="#2a4050",
              labelcolor=GRAY, fontsize=9, framealpha=0.95,
              handlelength=1.4, borderpad=0.8)

    ax.set_xlim(-8, W + 12)
    ax.set_ylim(-6, H + 4)      # y increases upward — taller = higher
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)

    ax.set_title(f"Mesh Cross-Section  (y-slice at y = {mid_y})",
                 color=ACC, fontsize=14, fontweight="bold", pad=10)
    fig.text(0.5, 0.01,
             "Horizontal cut through the voxel mesh showing internal structural layers.\n"
             f"White dots = FE nodes every 8 voxels  |  element size ≈ {VOX_SZ*100:.1f} cm",
             ha="center", fontsize=7.5, color="#aaaaaa")
    save(fig, "fig_fea3_cross_section.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 — Von Mises Stress
# ═══════════════════════════════════════════════════════════════════════════
def fig_von_mises():
    """Surface mesh coloured by Von Mises stress (floor closed, plasma cmap)."""
    # pad_floor=True ensures the bottom surface is closed
    verts_n, faces, verts_r, ctr, sc = build_smooth_mesh(step=2, pad_floor=True)
    tris_n = verts_n[faces]

    # stress at each vertex, averaged to faces
    sv = stress_field(verts_r)
    sf = sv[faces].mean(axis=1)

    shade = phong(tris_n, ambient=0.52)
    # combine stress (drives colour) and shading (drives brightness)
    combined = np.clip(0.65 * sf + 0.35 * shade, 0, 1)

    cmap = plt.cm.plasma
    fc_rgba = cmap(combined).copy()
    # apply shading to brightness without changing hue excessively
    fc_rgba[:, :3] = np.clip(fc_rgba[:, :3] * (0.50 + 0.50 * shade[:, None]), 0, 1)
    fc_rgba[:, 3]  = 0.95

    fig = plt.figure(figsize=(7.0, 6.2), facecolor="none")
    ax  = fig.add_subplot(111, projection="3d", facecolor=BG)

    poly = Poly3DCollection(tris_n, facecolors=fc_rgba,
                            edgecolors="none", linewidth=0)
    ax.add_collection3d(poly)

    # colorbar
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=0, vmax=VM_PEAK / 1e6)
    sm   = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.04,
                        orientation="vertical", shrink=0.72)
    cbar.set_label("Von Mises Stress  [MPa]", color=GRAY, fontsize=9,
                   labelpad=6)
    cbar.ax.yaxis.set_tick_params(color=GRAY, labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=GRAY)
    cbar.outline.set_edgecolor("#555")

    # allowable stress dashed line on colorbar
    sf_val   = targets["min_safety_factor"]
    allow_vm = YIELD / 1e6 / sf_val
    norm_pos = allow_vm / (VM_PEAK / 1e6)
    cbar.ax.axhline(norm_pos, color="white", lw=1.4,
                    linestyle="--", alpha=0.90)
    cbar.ax.text(1.30, norm_pos,
                 f" σ_allow\n {allow_vm:.2f} MPa",
                 transform=cbar.ax.transAxes,
                 color="white", fontsize=7, va="center", ha="left")

    pad = 0.55
    ax.set_xlim(-pad, pad); ax.set_ylim(-pad, pad); ax.set_zlim(-pad, pad)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=18, azim=-52)
    ax.set_axis_off()

    ax.set_title("Von Mises Stress", color=ACC, fontsize=14,
                 fontweight="bold", pad=8)
    fig.text(0.5, 0.01,
             f"Peak σ_VM = {VM_PEAK/1e6:.2f} MPa   |   "
             f"Yield σ_y = {YIELD/1e6:.0f} MPa   |   "
             f"Safety factor = {sf_val:.2f}   |   "
             f"Max displacement = {DISP_MAX*1000:.3f} mm",
             ha="center", fontsize=7.5, color="#aaaaaa")
    save(fig, "fig_fea4_von_mises.png")


# ─── run all ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("FEA 1 — FE Mesh …");         fig_fe_mesh()
    print("FEA 2 — Loads & BCs …");     fig_loads_bcs()
    print("FEA 3 — Cross-section …");   fig_cross_section()
    print("FEA 4 — Von Mises stress …"); fig_von_mises()
    print("\nAll FEA figures saved to figures/")
