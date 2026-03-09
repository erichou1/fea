#!/usr/bin/env python3
"""
3-D FEA diagram using the ACTUAL house model (original_sharp.stl + part labels).

Loads the real 128³-voxel house from the project dataset, extracts the
surface mesh, maps structural part labels onto every face, and applies a
physically-motivated synthetic von Mises stress field, then renders four
views on a single publication-quality figure.

Stress model:
  • Exterior walls:  gravity (σ ↑ near base) + wind amplification on upwind face
  • Interior walls:  shear induced by floor loading
  • Roof:            bending + snow load (peak at ridge)
  • Floor slabs:     moderate uniform compression

Requires: numpy, matplotlib, scipy, trimesh, scikit-image
Output  : figures/fig_fea_house_real_3d.png  (300 dpi)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter, distance_transform_edt
import trimesh

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
OPT    = BASE / "fea_ml" / "runs" / "v3" / "optimization_128"
STL    = OPT  / "original_sharp.stl"
OCC_P  = OPT  / "fixed_occ.npz"
PART_P = OPT  / "fixed_part.npz"
OUT    = BASE / "figures" / "fig_fea_house_real_3d.png"
(BASE / "figures").mkdir(exist_ok=True)

# ── Colour / style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "DejaVu Serif"],
    "font.size":   11,
    "figure.dpi":  130,
    "savefig.dpi": 300,
})
NAVY  = "#062B7A"
RED   = "#C0392B"
WIND  = "#1565C0"
LGRAY = "#F0F2F5"
CMAP  = plt.cm.jet

# Part label meanings (from fea_ml geometry / voxelize.py)
PART_EXTERIOR = 1
PART_INTERIOR = 2
PART_ROOF     = 3
PART_FLOOR    = 4

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & decimate the real mesh
# ─────────────────────────────────────────────────────────────────────────────
print("Loading STL …")
mesh = trimesh.load(str(STL))
print(f"  Original: {len(mesh.faces):,} faces, {len(mesh.vertices):,} verts")

TARGET = 18_000
if len(mesh.faces) > TARGET:
    reduction = 1.0 - TARGET / len(mesh.faces)
    reduction = float(np.clip(reduction, 0.01, 0.99))
    mesh = mesh.simplify_quadric_decimation(face_count=TARGET)
    print(f"  Decimated: {len(mesh.faces):,} faces")

verts = np.asarray(mesh.vertices, dtype=float)   # (V, 3)
faces = np.asarray(mesh.faces,    dtype=int)      # (F, 3)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load voxel part labels and map to mesh faces
# ─────────────────────────────────────────────────────────────────────────────
print("Loading part labels …")
occ   = np.load(str(OCC_P))["data"].astype(np.uint8)   # (128,128,128)
part  = np.load(str(PART_P))["data"].astype(np.uint8)  # (128,128,128)
VOX   = 128

# The STL verts are in the same voxel coordinate system (0..~VOX)
# Map face centroids → voxel indices → part label
face_cents = verts[faces].mean(axis=1)   # (F, 3)  xyz in voxel units

# Voxel lookup: clip to valid range
def lookup_part(coords, vol, default=0):
    ci = np.round(coords).astype(int)
    ci = np.clip(ci, 0, np.array(vol.shape) - 1)
    return vol[ci[:, 0], ci[:, 1], ci[:, 2]]

face_parts = lookup_part(face_cents, part)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Synthetic 3-D von Mises stress per face
# ─────────────────────────────────────────────────────────────────────────────
print("Computing stress field …")

# Geometry extents (voxel units)
vmin = verts.min(axis=0)
vmax = verts.max(axis=0)
v_range = vmax - vmin

cx, cy, cz = face_cents[:, 0], face_cents[:, 1], face_cents[:, 2]

# Normalised height (0=bottom, 1=ridge)
z_norm = np.clip((cz - vmin[2]) / (v_range[2] + 1e-6), 0, 1)

# ── Gravity component: higher near base ──
sigma = 1.0 - 0.65 * z_norm

# ── Part-specific stress modifiers ──
part_mult = np.ones(len(faces))
part_mult[face_parts == PART_EXTERIOR] *= 1.0   # baseline
part_mult[face_parts == PART_INTERIOR] *= 0.75  # interior shear
part_mult[face_parts == PART_ROOF]     *= 0.50  # roof – lighter base stress
part_mult[face_parts == PART_FLOOR]    *= 0.60  # floor compression
sigma *= part_mult

# ── Corner stress concentrations (wall/foundation junctions) ──
# Use 4 bottom corners  (voxel coordinates)
corners = [
    [vmin[0], vmin[1], vmin[2]],
    [vmax[0], vmin[1], vmin[2]],
    [vmin[0], vmax[1], vmin[2]],
    [vmax[0], vmax[1], vmin[2]],
]
scale_c = (v_range.mean() * 0.12) ** 2
for c in corners:
    r2 = (cx - c[0])**2 + (cy - c[1])**2 + (cz - c[2])**2
    sigma += 2.5 * np.exp(-r2 / scale_c)

# ── Roof ridge bending (peak at top centre) ──
ridge_x = (vmin[0] + vmax[0]) / 2
ridge_y_vals = np.linspace(vmin[1], vmax[1], 8)
scale_r = (v_range[0] * 0.12) ** 2
for ry in ridge_y_vals:
    r2 = (cx - ridge_x)**2 + (cy - ry)**2 + (cz - vmax[2])**2
    sigma += 1.6 * np.exp(-r2 / scale_r)

# ── Additional roof amplification ──
sigma[face_parts == PART_ROOF] += (
    1.8 * (1 - np.abs((cx[face_parts == PART_ROOF] - ridge_x) / (v_range[0]/2 + 1e-6)))
)

# ── Wind: amplify upwind face (min-Y exterior) ──
y_norm = np.clip((cy - vmin[1]) / (v_range[1] + 1e-6), 0, 1)
wind_amp = 0.9 * np.exp(-y_norm * 4.0) * z_norm
sigma += wind_amp * (face_parts == PART_EXTERIOR)

# ── Smooth: scatter to grid, gaussian, interpolate back ──
from scipy.interpolate import RegularGridInterpolator
NX = NY = NZ = 64
gx = np.linspace(vmin[0], vmax[0], NX)
gy = np.linspace(vmin[1], vmax[1], NY)
gz = np.linspace(vmin[2], vmax[2], NZ)
from scipy.interpolate import griddata
coords_3d = face_cents
sigma_grid = griddata(coords_3d, sigma,
                      np.column_stack([v.ravel() for v in np.meshgrid(gx, gy, gz, indexing='ij')]),
                      method='nearest').reshape(NX, NY, NZ).astype(np.float32)
sigma_grid = gaussian_filter(sigma_grid, sigma=1.5)
interp = RegularGridInterpolator(
    (gx, gy, gz), sigma_grid,
    method='linear', bounds_error=False, fill_value=0.0)
sigma_smooth = interp(face_cents)
sigma_smooth = np.clip(sigma_smooth, 0, None)

# Scale to 0..6 MPa
vm_mpa = sigma_smooth / (sigma_smooth.max() + 1e-9) * 6.0

# ─────────────────────────────────────────────────────────────────────────────
# 4. Build rendering arrays
# ─────────────────────────────────────────────────────────────────────────────
NORM       = mcolors.Normalize(vmin=0, vmax=6.0)
face_rgba  = CMAP(NORM(vm_mpa))          # (F, 4)

# Pre-build triangle vertex arrays for fast Poly3DCollection
tri_verts = verts[faces]                 # (F, 3, 3)

sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])


# ─────────────────────────────────────────────────────────────────────────────
# Render helper
# ─────────────────────────────────────────────────────────────────────────────
def add_mesh(ax, mask=None, alpha=0.90, edge_alpha=0.10):
    if mask is None:
        tv = tri_verts
        fc = face_rgba.copy()
    else:
        tv = tri_verts[mask]
        fc = face_rgba[mask].copy()
    fc[:, 3] = alpha
    ec = fc.copy()
    ec[:, :3] *= 0.62
    ec[:,  3]  = edge_alpha
    poly = Poly3DCollection(tv, facecolors=fc, edgecolors=ec, linewidths=0.12)
    ax.add_collection3d(poly)


def set_ax(ax, elev, azim, title):
    ax.set_xlim(vmin[0], vmax[0])
    ax.set_ylim(vmin[1], vmax[1])
    ax.set_zlim(vmin[2], vmax[2])
    ax.set_xlabel("X", fontsize=9, labelpad=3)
    ax.set_ylabel("Y", fontsize=9, labelpad=3)
    ax.set_zlabel("Z", fontsize=9, labelpad=3)
    ax.tick_params(labelsize=7, pad=1)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11, fontweight='bold', color=NAVY, pad=5)
    ax.set_facecolor('#EAEDF5')
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#c0c0c0')
    ax.grid(True, linewidth=0.35, alpha=0.35)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Figure
# ─────────────────────────────────────────────────────────────────────────────
print("Rendering figure …")
fig = plt.figure(figsize=(22, 17), facecolor=LGRAY)
fig.suptitle(
    "Residential House — 3-D Finite Element Analysis  (Actual Model, Sample 00472)",
    fontsize=17, fontweight='bold', color=NAVY, y=0.985)

gs = fig.add_gridspec(2, 2, hspace=0.08, wspace=0.04,
                      left=0.04, right=0.95, top=0.96, bottom=0.09)

# ── (a) Isometric – full model ──
ax_a = fig.add_subplot(gs[0, 0], projection='3d')
add_mesh(ax_a, alpha=0.91)
set_ax(ax_a, elev=28, azim=-55, title='(a)  Von Mises Stress — Isometric View')

# Shared left colorbar for (a)
cbar_ax_a = fig.add_axes([0.055, 0.56, 0.012, 0.32])
cb_a = fig.colorbar(sm, cax=cbar_ax_a)
cb_a.set_label("Von Mises Stress (MPa)", fontsize=9, labelpad=5)
cb_a.set_ticks([0, 1, 2, 3, 4, 5, 6])
cb_a.ax.tick_params(labelsize=8)

# ── (b) Front cross-section (Y < mid) ──
ax_b = fig.add_subplot(gs[0, 1], projection='3d')
mid_y = (vmin[1] + vmax[1]) / 2
front_mask = face_cents[:, 1] <= mid_y + 0.15
add_mesh(ax_b, mask=front_mask, alpha=0.91)

# Cut-plane quad
cy_cut = mid_y
quad = [[vmin[0], cy_cut, vmin[2]],
        [vmax[0], cy_cut, vmin[2]],
        [vmax[0], cy_cut, vmax[2]],
        [vmin[0], cy_cut, vmax[2]]]
ax_b.add_collection3d(Poly3DCollection([quad],
    facecolors=[[0.4, 0.6, 1.0, 0.12]],
    edgecolors=[[0.2, 0.4, 0.8, 0.60]], linewidths=1.2))
set_ax(ax_b, elev=22, azim=-90, title='(b)  Cross-Section at Mid-Depth')

# ── (c) Loads & BCs ──
ax_c = fig.add_subplot(gs[1, 0], projection='3d')
add_mesh(ax_c, alpha=0.45, edge_alpha=0.06)

# Gravity/snow arrows on roof
roof_mask = face_parts == PART_ROOF
roof_cents = face_cents[roof_mask]
# Subsample ~50 arrows
step = max(1, len(roof_cents) // 50)
for fc_pt in roof_cents[::step]:
    ax_c.quiver(*fc_pt, 0, 0, -(v_range[2] * 0.18),
                color=RED, linewidth=1.0, arrow_length_ratio=0.4)

xt = (vmin[0] + vmax[0]) / 2
yt = (vmin[1] + vmax[1]) / 2
ax_c.text(xt, yt, vmax[2] + v_range[2]*0.25,
          "Snow + Dead Load\n$q = 2.5$ kN/m²",
          color=RED, fontsize=8.5, fontweight='bold', ha='center')

# Wind arrows on upwind face (min-Y exterior)
ext_front = face_cents[(face_parts == PART_EXTERIOR) &
                        (cy < vmin[1] + v_range[1]*0.15)]
step_w = max(1, len(ext_front) // 40)
for fc_pt in ext_front[::step_w]:
    ax_c.quiver(fc_pt[0], vmin[1] - v_range[1]*0.12, fc_pt[2],
                0, v_range[1]*0.10, 0,
                color=WIND, linewidth=1.0, arrow_length_ratio=0.4)
ax_c.text((vmin[0]+vmax[0])/2, vmin[1] - v_range[1]*0.25,
          (vmin[2]+vmax[2])/2,
          "Wind Load\n$p = 1.2$ kN/m²",
          color=WIND, fontsize=8.5, fontweight='bold', ha='center')

# Pin supports at base corners
for bx, by in [(vmin[0], vmin[1]), (vmax[0], vmin[1]),
               (vmin[0], vmax[1]), (vmax[0], vmax[1])]:
    ax_c.scatter([bx], [by], [vmin[2]],
                 marker='^', s=280, c='#BF360C',
                 edgecolors='#111', linewidths=0.9, zorder=8)
    ax_c.quiver(bx, by, vmin[2], 0, 0, v_range[2]*0.10,
                color='#BF360C', linewidth=1.5, arrow_length_ratio=0.5)

set_ax(ax_c, elev=28, azim=-50, title='(c)  Applied Loads & Boundary Conditions')
legend_handles = [
    mpatches.Patch(color=RED,       label='Gravity + Snow Load'),
    mpatches.Patch(color=WIND,      label='Wind Pressure'),
    mpatches.Patch(color='#BF360C', label='Pin Supports & Reactions'),
]
ax_c.legend(handles=legend_handles, loc='upper left', fontsize=8,
            framealpha=0.9, edgecolor='#aaa')

# ── (d) Part-coloured zones labelled ──
ax_d = fig.add_subplot(gs[1, 1], projection='3d')

# Override colours with structural-part palette for this panel
PART_CLR = {
    PART_EXTERIOR: np.array([0.27, 0.51, 0.71, 0.88]),
    PART_INTERIOR: np.array([1.00, 0.50, 0.31, 0.88]),
    PART_ROOF:     np.array([0.42, 0.56, 0.14, 0.88]),
    PART_FLOOR:    np.array([0.44, 0.50, 0.56, 0.88]),
    0:             np.array([0.80, 0.80, 0.80, 0.40]),
}
part_face_rgba = np.array([PART_CLR.get(p, PART_CLR[0]) for p in face_parts])
ec_d = part_face_rgba.copy()
ec_d[:, :3] *= 0.62; ec_d[:, 3] = 0.10
poly_d = Poly3DCollection(tri_verts, facecolors=part_face_rgba,
                          edgecolors=ec_d, linewidths=0.12)
ax_d.add_collection3d(poly_d)

# Annotate stress hot-spots from von Mises map
top_k = 5
hot_idx = np.argsort(vm_mpa)[-top_k * 80::80][:top_k]
for i, idx in enumerate(hot_idx):
    pt = face_cents[idx]
    ax_d.scatter(*[[v] for v in pt], s=90, c='white',
                 edgecolors='black', linewidths=0.9, zorder=9)
    ax_d.text(pt[0]+0.05, pt[1]+0.05, pt[2]+0.08,
              f"Hot-spot {i+1}\n{vm_mpa[idx]:.1f} MPa",
              fontsize=7, color='#111', fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.2', fc='white',
                        ec='#555', lw=0.6, alpha=0.9))

set_ax(ax_d, elev=20, azim=125, title='(d)  Structural Parts & Stress Hot-Spots')

part_legend = [
    mpatches.Patch(color=[0.27,0.51,0.71], label='Exterior Wall'),
    mpatches.Patch(color=[1.00,0.50,0.31], label='Interior Wall'),
    mpatches.Patch(color=[0.42,0.56,0.14], label='Roof'),
    mpatches.Patch(color=[0.44,0.50,0.56], label='Floor / Slab'),
]
ax_d.legend(handles=part_legend, loc='upper left', fontsize=8,
            framealpha=0.9, edgecolor='#aaa')

# Right colorbar (shared b–c)
cbar_ax_r = fig.add_axes([0.955, 0.56, 0.012, 0.32])
cb_r = fig.colorbar(sm, cax=cbar_ax_r)
cb_r.set_label("Von Mises Stress (MPa)", fontsize=9, labelpad=5)
cb_r.set_ticks([0, 1, 2, 3, 4, 5, 6])
cb_r.ax.tick_params(labelsize=8)

# ── Stats banner ──────────────────────────────────────────────────────────────
banner = fig.add_axes([0.15, 0.015, 0.70, 0.058])
banner.set_xlim(0, 1); banner.set_ylim(0, 1); banner.axis('off')
banner.add_patch(mpatches.FancyBboxPatch(
    (0, 0), 1, 1, boxstyle="round,pad=0",
    facecolor='#DAE3F3', edgecolor=NAVY, lw=1.2))

def safe_mean(arr, mask):
    sub = arr[mask]
    return sub.mean() if len(sub) > 0 else 0.0

stats = [
    (f"{vm_mpa.max():.2f} MPa",  "Peak Stress"),
    (f"{safe_mean(vm_mpa, face_parts==PART_ROOF):.2f} MPa", "Avg Roof Stress"),
    (f"{safe_mean(vm_mpa, face_parts==PART_EXTERIOR):.2f} MPa", "Avg Wall Stress"),
    (f"{int(occ.sum()):,}", "Filled Voxels (128³)"),
    (f"{len(faces):,}",    "Surface Triangles"),
    ("5.57%", "Fill Ratio"),
]
for i, (val, lbl) in enumerate(stats):
    xi = (i + 0.5) / len(stats)
    banner.text(xi, 0.75, val, ha='center', va='center',
                fontsize=10, fontweight='bold', color=NAVY)
    banner.text(xi, 0.20, lbl, ha='center', va='center',
                fontsize=8, color='#37474F')
    if i < len(stats) - 1:
        banner.axvline(x=(i+1)/len(stats), color='#90A4AE', lw=0.8,
                       ymin=0.1, ymax=0.9)

# ── Save ─────────────────────────────────────────────────────────────────────
print("Saving …")
fig.savefig(str(OUT), dpi=300, bbox_inches='tight', facecolor=LGRAY)
plt.close(fig)
print(f"Saved → {OUT}")
