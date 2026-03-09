#!/usr/bin/env python3
"""
3-D FEA diagram: house structure under finite-element analysis.

Four subplots on one figure:
  (a) Isometric – von Mises stress colourmap on all exterior surfaces
  (b) Front elevation (cross-section cut at mid-depth)
  (c) Boundary conditions & loads – isometric annotated view
  (d) Rear-right perspective – second viewing angle

Geometry (metres)
  Width  X : 0 → W = 10
  Depth  Y : 0 → D = 8
  Wall   Z : 0 → H = 5
  Slab   Z : -GH → 0   (foundation, GH = 0.5)
  Ridge  Z : H → H+RH = 8   (ridge at X = W/2, full depth)

Requires: numpy, matplotlib, scipy
Output  : figures/fig_fea_house_3d.png  (300 dpi)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "fig_fea_house_3d.png"

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.dpi": 130,
    "savefig.dpi": 300,
})

NAVY  = "#062B7A"
RED   = "#C0392B"
WIND  = "#1565C0"
LGRAY = "#F0F2F5"
DGRAY = "#37474F"

# ── House dimensions ──────────────────────────────────────────────────────────
W  = 10.0   # width  (X)
D  =  8.0   # depth  (Y)
H  =  5.0   # wall height (Z)
GH =  0.5   # foundation slab below Z=0
RH =  3.0   # roof ridge height above Z=H

# ─────────────────────────────────────────────────────────────────────────────
# Panel tessellation helper
# ─────────────────────────────────────────────────────────────────────────────
def make_panel(c00, c10, c11, c01, nu=10, nv=10):
    """
    Bilinear-quad panel → list of triangle vertices.
    c** are (3,) corner arrays in order: bottom-left, bottom-right,
    top-right, top-left.
    Returns tris: (N, 3, 3), centroids: (N, 3)
    """
    u = np.linspace(0, 1, nu + 1)
    v = np.linspace(0, 1, nv + 1)
    U, V = np.meshgrid(u, v)          # (nv+1, nu+1)

    # Bilinear interpolation
    pts = (np.einsum('ij,k->ijk', (1-U)*(1-V), c00) +
           np.einsum('ij,k->ijk', U*(1-V),     c10) +
           np.einsum('ij,k->ijk', U*V,          c11) +
           np.einsum('ij,k->ijk', (1-U)*V,     c01))   # (nv+1, nu+1, 3)

    tris = []
    for j in range(nv):
        for i in range(nu):
            p00 = pts[j,   i  ]
            p10 = pts[j,   i+1]
            p11 = pts[j+1, i+1]
            p01 = pts[j+1, i  ]
            tris.append([p00, p10, p11])
            tris.append([p00, p11, p01])

    tris = np.array(tris)              # (N, 3, 3)
    cents = tris.mean(axis=1)          # (N, 3)
    return tris, cents


def make_roof_face(apex, e0, e1, nu=10, nv=10):
    """
    Triangular roof face: apex at top, edge pts e0..e1 at eave.
    Returns tris, cents.
    """
    # Parameterise: bottom edge e0→e1, top collapsed to apex
    c00 = e0; c10 = e1; c11 = apex; c01 = apex
    return make_panel(c00, c10, c11, c01, nu=nu, nv=nv)


# ─────────────────────────────────────────────────────────────────────────────
# Build complete house surface
# ─────────────────────────────────────────────────────────────────────────────
panels = {}   # name → (tris, cents)

# Foundation slab faces
panels['found_front'] = make_panel(
    [0,0,-GH],[W,0,-GH],[W,0,0],[0,0,0], nu=12, nv=3)
panels['found_back']  = make_panel(
    [W,D,-GH],[0,D,-GH],[0,D,0],[W,D,0], nu=12, nv=3)
panels['found_left']  = make_panel(
    [0,D,-GH],[0,0,-GH],[0,0,0],[0,D,0], nu=10, nv=3)
panels['found_right'] = make_panel(
    [W,0,-GH],[W,D,-GH],[W,D,0],[W,0,0], nu=10, nv=3)
panels['found_bot']   = make_panel(
    [0,0,-GH],[W,0,-GH],[W,D,-GH],[0,D,-GH], nu=12, nv=10)
panels['found_top']   = make_panel(
    [0,0,0],[W,0,0],[W,D,0],[0,D,0], nu=12, nv=10)

# Walls
panels['wall_front']  = make_panel(
    [0,0,0],[W,0,0],[W,0,H],[0,0,H], nu=14, nv=10)
panels['wall_back']   = make_panel(
    [W,D,0],[0,D,0],[0,D,H],[W,D,H], nu=14, nv=10)
panels['wall_left']   = make_panel(
    [0,D,0],[0,0,0],[0,0,H],[0,D,H], nu=10, nv=10)
panels['wall_right']  = make_panel(
    [W,0,0],[W,D,0],[W,D,H],[W,0,H], nu=10, nv=10)

# Gable end walls (triangular part above H, front & back)
# Front gable: triangle [0,0,H] → [W,0,H] → [W/2,0,H+RH]
panels['gable_front'] = make_panel(
    [0,0,H],[W,0,H],[W/2,0,H+RH],[W/2,0,H+RH], nu=14, nv=8)
panels['gable_back']  = make_panel(
    [W,D,H],[0,D,H],[W/2,D,H+RH],[W/2,D,H+RH], nu=14, nv=8)

# Roof faces
panels['roof_left']   = make_panel(
    [0,0,H],[0,D,H],[W/2,D,H+RH],[W/2,0,H+RH], nu=10, nv=12)
panels['roof_right']  = make_panel(
    [W,0,H],[W/2,0,H+RH],[W/2,D,H+RH],[W,D,H], nu=10, nv=12)

# Interior floor (Z=0, inside)
panels['floor_int']   = make_panel(
    [0,0,0],[W,0,0],[W,D,0],[0,D,0], nu=14, nv=10)

# Collect all triangles and centroids
all_tris  = []
all_cents = []
for name, (tris, cents) in panels.items():
    all_tris.append(tris)
    all_cents.append(cents)

all_tris  = np.vstack(all_tris)   # (N_total, 3, 3)
all_cents = np.vstack(all_cents)  # (N_total, 3)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic 3-D von Mises stress field
# ─────────────────────────────────────────────────────────────────────────────
def stress_field(X, Y, Z):
    """
    Illustrative stress combination:
      - Gravity: higher at base
      - Corner concentrations: wall/foundation junctions
      - Ridge concentration: bending + snow
      - Wind: higher on upwind face (Y=0), increases with height
    """
    Z_norm = np.clip(Z / (H + RH), 0, 1)

    # Gravity (decreases with height)
    s = 1.0 - 0.65 * Z_norm

    # Wall base corners (4 corners at Z=0)
    for cx, cy in [(0,0),(W,0),(0,D),(W,D)]:
        r2 = (X-cx)**2 + (Y-cy)**2 + Z**2
        s += 1.8 * np.exp(-r2 / 3.0)

    # Foundation support reactions (bottom corners)
    for cx, cy in [(0,0),(W,0),(0,D),(W,D)]:
        r2 = (X-cx)**2 + (Y-cy)**2 + (Z+GH)**2
        s += 2.0 * np.exp(-r2 / 2.0)

    # Roof ridge concentration (bending + snow load)
    r2_ridge = (X - W/2)**2 + (Z - (H + RH))**2
    s += 1.8 * np.exp(-r2_ridge / 2.0)

    # Snow: eave stress (where roof meets wall)
    for cx in [0, W]:
        r2 = (X - cx)**2 + (Z - H)**2
        s += 1.0 * np.exp(-r2 / 1.5)

    # Wind (upwind face Y=0, grows with height)
    wind_face = np.exp(-Y**2 / 2.0) * Z_norm * 1.2
    s += wind_face

    return s


# Build on a regular grid then interpolate to triangle centroids
NX, NY, NZ = 60, 50, 55
gx = np.linspace(-0.5, W+0.5, NX)
gy = np.linspace(-0.5, D+0.5, NY)
gz = np.linspace(-GH-0.5, H+RH+0.5, NZ)
GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
S_grid = stress_field(GX, GY, GZ).astype(np.float32)
S_grid = gaussian_filter(S_grid, sigma=1.8)

interp = RegularGridInterpolator(
    (gx, gy, gz), S_grid, method='linear', bounds_error=False,
    fill_value=0.0)

cx, cy, cz = all_cents[:,0], all_cents[:,1], all_cents[:,2]
stress_vals = interp(np.column_stack([cx, cy, cz]))
stress_vals = np.clip(stress_vals, 0, None)
stress_mpa  = stress_vals / stress_vals.max() * 6.0   # scale 0..6 MPa

# ─────────────────────────────────────────────────────────────────────────────
# Colour each triangle by stress
# ─────────────────────────────────────────────────────────────────────────────
CMAP = plt.cm.jet
NORM = mcolors.Normalize(vmin=0, vmax=6.0)
face_colors = CMAP(NORM(stress_mpa))   # (N, 4)

# Slightly darken edge colours for depth cues
edge_colors = face_colors.copy()
edge_colors[:, :3] *= 0.70
edge_colors[:,  3]  = 0.15   # near-transparent edges


# ─────────────────────────────────────────────────────────────────────────────
# Per-panel indices for selective rendering
# ─────────────────────────────────────────────────────────────────────────────
idx_start = {}
offset = 0
for name, (tris, cents) in panels.items():
    idx_start[name] = (offset, offset + len(tris))
    offset += len(tris)

def panel_slice(name):
    s, e = idx_start[name]
    return slice(s, e)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: render a subset of triangles on a 3-D axis
# ─────────────────────────────────────────────────────────────────────────────
def render_tris(ax, tri_idx, alpha=0.92, edge_alpha=0.12,
                highlight_faces=None):
    """
    tri_idx: list/array of triangle indices to plot.
    highlight_faces: optional extra face colour override (same length as tri_idx).
    """
    tris = all_tris[tri_idx]
    fc   = face_colors[tri_idx].copy()
    if highlight_faces is not None:
        fc = highlight_faces
    fc[:, 3] = alpha

    ec   = fc.copy()
    ec[:, :3] *= 0.65
    ec[:,  3]  = edge_alpha

    poly = Poly3DCollection(tris, facecolors=fc, edgecolors=ec, linewidths=0.15)
    ax.add_collection3d(poly)


def set_axes(ax, elev=25, azim=-50, title='', equal=True):
    ax.set_xlim(0, W);  ax.set_ylim(0, D);  ax.set_zlim(-GH, H+RH)
    ax.set_xlabel('X (m)', labelpad=4, fontsize=9)
    ax.set_ylabel('Y (m)', labelpad=4, fontsize=9)
    ax.set_zlabel('Z (m)', labelpad=4, fontsize=9)
    ax.tick_params(labelsize=8, pad=1)
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold',
                     color=NAVY, pad=6)
    ax.set_facecolor('#EAECF2')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#cccccc')
    ax.yaxis.pane.set_edgecolor('#cccccc')
    ax.zaxis.pane.set_edgecolor('#cccccc')
    ax.grid(True, linewidth=0.4, alpha=0.4)


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16), facecolor=LGRAY)
fig.suptitle("Residential House — 3-D Finite Element Analysis",
             fontsize=17, fontweight='bold', color=NAVY, y=0.98)

# Grid layout
gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.05,
                      left=0.04, right=0.96, top=0.94, bottom=0.08)

all_idx = np.arange(len(all_tris))

# ──────────────────────────────────────────────────────────────────────────────
# Panel (a): Full isometric – von Mises stress
# ──────────────────────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0], projection='3d')
render_tris(ax_a, all_idx)
set_axes(ax_a, elev=28, azim=-55,
         title='(a)  Von Mises Stress — Isometric View')

# Colorbar for panel (a)
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])
cbar_ax = fig.add_axes([0.06, 0.56, 0.014, 0.30])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
cbar.set_label('Von Mises Stress (MPa)', fontsize=9, labelpad=6)
cbar.set_ticks([0, 1, 2, 3, 4, 5, 6])
cbar.ax.tick_params(labelsize=8)

# ──────────────────────────────────────────────────────────────────────────────
# Panel (b): Cross-section view (front half only, Y < D/2)
# ──────────────────────────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1], projection='3d')

# Show only panels whose centroids have Y <= D/2 + small buffer
front_half = all_idx[all_cents[:, 1] <= D / 2 + 0.3]
render_tris(ax_b, front_half)

# Cut-plane: semi-transparent rectangle at Y = D/2
cut_y = D / 2
cut_plane_verts = [[-0.1, cut_y, -GH-0.1],
                   [W+0.1, cut_y, -GH-0.1],
                   [W+0.1, cut_y, H+RH+0.1],
                   [-0.1,  cut_y, H+RH+0.1]]
cut_poly = Poly3DCollection([cut_plane_verts],
                             facecolors=[[0.4, 0.6, 1.0, 0.12]],
                             edgecolors=[[0.2, 0.4, 0.8, 0.6]],
                             linewidths=1.2)
ax_b.add_collection3d(cut_poly)

set_axes(ax_b, elev=22, azim=-80,
         title='(b)  Cross-Section Cut at Mid-Depth')

# ──────────────────────────────────────────────────────────────────────────────
# Panel (c): Loads & boundary conditions
# ──────────────────────────────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0], projection='3d')

# Draw surfaces with reduced alpha so annotations show through
render_tris(ax_c, all_idx, alpha=0.55, edge_alpha=0.08)

# ── Gravity + snow load: downward arrows on roof ──
n_x, n_y = 6, 5
xs = np.linspace(0.8, W-0.8, n_x)
ys = np.linspace(0.6, D-0.6, n_y)
arrow_len = 1.1
for xi in xs:
    for yi in ys:
        # Roof surface Z
        zbase_l = H + (RH / (W/2)) * xi
        zbase_r = H + (RH / (W/2)) * (W - xi)
        zroof = min(zbase_l, zbase_r)
        ax_c.quiver(xi, yi, zroof + arrow_len,
                    0, 0, -arrow_len * 0.85,
                    color=RED, linewidth=1.1, arrow_length_ratio=0.35)

ax_c.text(W/2, D/2, H+RH+1.6, "Snow + Dead Load\n$q = 2.5$ kN/m²",
          color=RED, fontsize=8.5, fontweight='bold',
          ha='center', va='bottom', zorder=10)

# ── Wind load: horizontal arrows on front face (Y=0) ──
nz = 5
zs_wind = np.linspace(0.6, H-0.6, nz)
xs_wind = np.linspace(1.5, W-1.5, 4)
for zi in zs_wind:
    for xi in xs_wind:
        ax_c.quiver(xi, -0.9, zi,
                    0, 0.85, 0,
                    color=WIND, linewidth=1.1, arrow_length_ratio=0.35)

ax_c.text(W/2, -1.5, H/2, "Wind Load\n$p = 1.2$ kN/m²",
          color=WIND, fontsize=8.5, fontweight='bold',
          ha='center', va='center', zorder=10)

# ── Pin supports: triangles at base corners ──
for cx, cy in [(0,0),(W,0),(0,D),(W,D)]:
    ax_c.scatter([cx], [cy], [-GH - 0.05],
                 marker='^', s=220, c='#BF360C', zorder=8,
                 edgecolors='#333', linewidths=0.8)
    # Support reaction arrow (upward)
    ax_c.quiver(cx, cy, -GH-0.05,
                0, 0, 0.6,
                color='#BF360C', linewidth=1.4,
                arrow_length_ratio=0.45)

ax_c.text(W/2, D/2, -GH-0.9, "Pin Supports (free rotation)",
          color='#BF360C', fontsize=8.5, fontweight='bold',
          ha='center', va='top', zorder=10)

set_axes(ax_c, elev=28, azim=-50,
         title='(c)  Applied Loads & Boundary Conditions')

# ── Legend ──
legend_handles = [
    mpatches.Patch(color=RED,       label='Gravity + Snow Load'),
    mpatches.Patch(color=WIND,      label='Wind Pressure (Y face)'),
    mpatches.Patch(color='#BF360C', label='Pin Supports'),
]
ax_c.legend(handles=legend_handles, loc='upper left',
            fontsize=8, framealpha=0.88, edgecolor='#aaa')

# ──────────────────────────────────────────────────────────────────────────────
# Panel (d): Rear-right perspective + high-stress zones labelled
# ──────────────────────────────────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1], projection='3d')
render_tris(ax_d, all_idx, alpha=0.88, edge_alpha=0.10)

# Annotate 3 high-stress regions
annotation_pts = [
    ([0, 0, 0],      "Base corner\nstress conc."),
    ([W/2, D/2, H+RH], "Ridge bending\npeak"),
    ([W, D, 0],       "Foundation\nreaction"),
]
for pt, lbl in annotation_pts:
    ax_d.scatter(*[[v] for v in pt], s=80, c='white',
                 edgecolors='black', linewidths=1.0, zorder=9)
    ax_d.text(pt[0]+0.4, pt[1]+0.4, pt[2]+0.5, lbl,
              fontsize=7.5, color='#111', fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.25', fc='white',
                        ec='#555', lw=0.7, alpha=0.85))

set_axes(ax_d, elev=18, azim=130,
         title='(d)  Rear-Right View — Stress Concentrations')

# ── Colourbar (shared for c & d) ──
cbar_ax2 = fig.add_axes([0.96, 0.08, 0.014, 0.35])
cbar2 = fig.colorbar(sm, cax=cbar_ax2, orientation='vertical')
cbar2.set_label('Von Mises Stress (MPa)', fontsize=9, labelpad=6)
cbar2.set_ticks([0, 1, 2, 3, 4, 5, 6])
cbar2.ax.tick_params(labelsize=8)

# ── Stats banner ──────────────────────────────────────────────────────────────
banner_ax = fig.add_axes([0.18, 0.015, 0.64, 0.055])
banner_ax.set_xlim(0, 1); banner_ax.set_ylim(0, 1); banner_ax.axis('off')
banner_ax.add_patch(mpatches.FancyBboxPatch(
    (0, 0), 1, 1, boxstyle="round,pad=0",
    facecolor='#DAE3F3', edgecolor=NAVY, lw=1.2))

stats = [
    (f"{stress_mpa.max():.2f} MPa",  "Peak Stress"),
    (f"{stress_mpa.mean():.2f} MPa", "Mean Stress"),
    (f"{len(all_tris):,}",            "Surface Elements"),
    (f"{W:.0f}×{D:.0f}×{H+RH:.0f} m",  "House Dimensions"),
    ("E=30 GPa",                      "Concrete (walls)"),
    ("E=11 GPa",                      "Timber (roof)"),
]
for i, (val, lbl) in enumerate(stats):
    xi = (i + 0.5) / len(stats)
    banner_ax.text(xi, 0.75, val, ha='center', va='center',
                   fontsize=10, fontweight='bold', color=NAVY)
    banner_ax.text(xi, 0.22, lbl, ha='center', va='center',
                   fontsize=8, color=DGRAY)
    if i < len(stats) - 1:
        banner_ax.axvline(x=(i+1)/len(stats), color='#90A4AE',
                          lw=0.8, ymin=0.1, ymax=0.9)

# ── Save ─────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PATH, dpi=300, bbox_inches='tight', facecolor=LGRAY)
plt.close(fig)
print(f"Saved → {OUT_PATH}")
