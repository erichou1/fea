#!/usr/bin/env python3
"""
Generate a figure of a house under FEA (Finite Element Analysis) simulation.

Shows:
  (a) 2-D cross-section of the house with a Delaunay FE mesh coloured by
      synthetic von Mises stress (illustrative field derived from a simple
      cantilever-like load path).
  (b) Annotated schematic: boundary conditions (pinned supports at foundation
      corners, uniform gravity + snow load on roof, lateral wind pressure on
      one side wall).

Requires only: numpy, matplotlib, scipy.
Output: figures/fig_fea_house.png  (300 dpi)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.collections as mc
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
from pathlib import Path

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "fig_fea_house.png"

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

NAVY   = "#062B7A"
BLUE   = "#1A4FAA"
LBLUE  = "#D9E5FB"
GOLD   = "#F5A623"
TEAL   = "#00897B"
RED    = "#C0392B"
LGRAY  = "#F4F4F6"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  House geometry (2-D cross-section polygon)
# ─────────────────────────────────────────────────────────────────────────────
#   Dimensions (metres)
W  = 10.0   # full width
H  = 5.0    # wall height
RH = 3.0    # roof ridge height above wall top
GH = 0.5    # foundation slab thickness

# Key vertices  (x, y), origin at bottom-left of foundation
foundation = np.array([
    [0,    -GH],
    [W,    -GH],
    [W,     0.0],
    [0,     0.0],
])

walls_outer = np.array([
    [0,    0.0],
    [W,    0.0],
    [W,    H],
    [0,    H],
])

roof = np.array([
    [0,     H],
    [W/2,   H + RH],
    [W,     H],
])

# Interior nodes – dense inside the house geometry for a good mesh
rng = np.random.default_rng(42)

def in_house(pts):
    """Return boolean mask: is each point inside the house cross-section?"""
    x, y = pts[:, 0], pts[:, 1]
    in_found  = (x >= 0) & (x <= W) & (y >= -GH) & (y <= H)
    # roof triangle mask: y <= ridge line
    roof_left  = (y - H) <= (RH / (W/2)) * x       # left face
    roof_right = (y - H) <= (RH / (W/2)) * (W - x) # right face
    in_roof   = in_found | (
        (x >= 0) & (x <= W) & (y >= H) & roof_left & roof_right
    )
    return in_roof

n_interior = 1200
candidates   = rng.uniform([-0.3, -GH - 0.1], [W + 0.3, H + RH + 0.1],
                            size=(n_interior * 6, 2))
mask = in_house(candidates)
interior_pts = candidates[mask][:n_interior]

# Boundary nodes (traced along each face)
def linspace_pts(a, b, n):
    t = np.linspace(0, 1, n, endpoint=False)
    return a[None] + t[:, None] * (b[None] - a[None])

bdry_pts = np.vstack([
    linspace_pts(np.array([0, -GH]),   np.array([W,  -GH]),  25),   # bottom
    linspace_pts(np.array([W, -GH]),   np.array([W,   H]),   20),   # right wall
    linspace_pts(np.array([W,  H]),    np.array([W/2, H+RH]),18),   # right roof
    linspace_pts(np.array([W/2,H+RH]),np.array([0,   H]),   18),   # left roof
    linspace_pts(np.array([0,  H]),    np.array([0,  -GH]),  20),   # left wall
])

# Interior wall / floor details  (thinner lines of nodes)
# Floor line at y=0
floor_pts = linspace_pts(np.array([0.3, 0.0]), np.array([W-0.3, 0.0]), 20)
# Interior partition wall at x=W/2
part_pts  = linspace_pts(np.array([W/2, 0.0]), np.array([W/2, H]),     15)

all_pts = np.vstack([bdry_pts, floor_pts, part_pts, interior_pts])

# Delaunay triangulation
tri  = Delaunay(all_pts)
pts  = all_pts
simp = tri.simplices   # (N_tri, 3)

# Keep only simplices whose centroid is inside the house
centroids = pts[simp].mean(axis=1)
keep = in_house(centroids)
simp = simp[keep]

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Synthetic von Mises stress field
#     Stress model (illustrative):
#       • Foundation/slab: high compression at base corners, moderate centre
#       • Walls: gradient – high at base, lower at top
#       • Roof: peak at ridge (bending + snow), lower at eaves
#       • Partition: moderate shear
# ─────────────────────────────────────────────────────────────────────────────
x, y = pts[:, 0], pts[:, 1]

# Normalised height 0..1 inside the house body
y_norm = np.clip(y / (H + RH), 0, 1)

# Base stress: gravity → higher at bottom
sigma_gravity = 1.0 - 0.7 * y_norm

# Corner concentrations (four corners of the wall box)
corners = np.array([[0, 0], [W, 0], [0, H], [W, H]])
corner_stress = np.zeros(len(pts))
for cx, cy in corners:
    r2 = (x - cx)**2 + (y - cy)**2
    corner_stress += 1.5 * np.exp(-r2 / 2.5)

# Roof ridge concentration
r2_ridge = (x - W/2)**2 + (y - (H + RH))**2
ridge_stress = 2.0 * np.exp(-r2_ridge / 1.5)

# Wind pressure on left wall (increases with height)
wind_stress = np.where((x < 0.6) & (y > 0), 0.8 * y_norm, 0.0)

# Foundation corners (support reactions)
for fc in [[0, -GH], [W, -GH]]:
    r2 = (x - fc[0])**2 + (y - fc[1])**2
    corner_stress += 2.0 * np.exp(-r2 / 1.5)

# Combine
vm = (sigma_gravity * 0.4
      + corner_stress * 0.35
      + ridge_stress * 0.15
      + wind_stress * 0.10)

# Smooth slightly
# Interpolate to a grid, smooth, then back to point values via bilinear interp
from scipy.interpolate import griddata
gx = np.linspace(x.min(), x.max(), 200)
gy = np.linspace(y.min(), y.max(), 200)
GX, GY = np.meshgrid(gx, gy)
vm_grid = griddata((x, y), vm, (GX, GY), method='linear', fill_value=0)
vm_grid = gaussian_filter(vm_grid, sigma=2.5)
vm_smooth = griddata((GX.ravel(), GY.ravel()), vm_grid.ravel(), (x, y),
                     method='linear', fill_value=0)
vm_smooth = np.clip(vm_smooth, 0, None)

# Scale to MPa range 0..6
vm_mpa = vm_smooth / vm_smooth.max() * 6.0

# Face (centroid) stress as average of vertex stresses
vm_face = vm_mpa[simp].mean(axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Figure layout: two panels side by side
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8.5),
                         gridspec_kw={'width_ratios': [1, 1]},
                         facecolor=LGRAY)
fig.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.12, wspace=0.14)

# ── Panel (a): FEA mesh coloured by von Mises stress ──────────────────────
ax = axes[0]
ax.set_facecolor(LGRAY)

# Draw filled triangles coloured by stress
cmap = plt.cm.jet
norm = mcolors.Normalize(vmin=0, vmax=6.0)

tria  = Triangulation(pts[:, 0], pts[:, 1], triangles=simp)
tc = ax.tripcolor(tria, vm_mpa, shading='gouraud', cmap=cmap, norm=norm, zorder=2)

# Overlay mesh edges – very thin, semi-transparent
ax.triplot(tria, color='black', linewidth=0.18, alpha=0.25, zorder=3)

# Foundation hatching
f_patch = mpatches.Polygon(
    np.array([[0, -GH], [W, -GH], [W, 0], [0, 0]]),
    closed=True, facecolor='none', edgecolor='#333333',
    hatch='////', linewidth=0.8, zorder=4, alpha=0.55)
ax.add_patch(f_patch)

# House outline (for visual clarity)
house_outline = np.array([
    [0, -GH], [W, -GH], [W, 0], [W, H], [W/2, H + RH],
    [0, H], [0, 0], [0, -GH]
])
ax.plot(house_outline[:, 0], house_outline[:, 1],
        color='#111111', lw=1.4, zorder=5)

# Partition wall line
ax.plot([W/2, W/2], [0, H], color='#111111', lw=1.0, ls='--', zorder=5, alpha=0.7)
# Floor slab line
ax.plot([0, W], [0, 0], color='#111111', lw=1.2, zorder=5)

# Colorbar
cbar = fig.colorbar(tc, ax=ax, orientation='vertical', pad=0.02, fraction=0.035)
cbar.set_label("Von Mises Stress (MPa)", fontsize=11, labelpad=8)
cbar.ax.tick_params(labelsize=10)
cbar.set_ticks([0, 1, 2, 3, 4, 5, 6])

ax.set_xlim(-1.0, W + 1.0)
ax.set_ylim(-GH - 0.8, H + RH + 0.8)
ax.set_aspect('equal')
ax.set_xlabel("Width (m)", fontsize=11)
ax.set_ylabel("Height (m)", fontsize=11)
ax.set_title("(a)  Von Mises Stress Distribution — FEA", fontsize=13,
             fontweight='bold', color=NAVY, pad=8)

# Tick labels
ax.set_xticks([0, W/4, W/2, 3*W/4, W])
ax.set_xticklabels(["0", "2.5", "5.0", "7.5", "10.0"])
ax.set_yticks([-GH, 0, H/2, H, H + RH/2, H + RH])
ax.set_yticklabels([f"{-GH:.1f}", "0", "2.5", "5.0",
                    f"{H+RH/2:.1f}", f"{H+RH:.1f}"])

# ── Panel (b): Boundary condition schematic ────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor(LGRAY)
ax2.set_xlim(-1.5, W + 1.5)
ax2.set_ylim(-GH - 1.2, H + RH + 1.2)
ax2.set_aspect('equal')
ax2.set_xlabel("Width (m)", fontsize=11)
ax2.set_ylabel("Height (m)", fontsize=11)
ax2.set_title("(b)  Structural Loads & Boundary Conditions", fontsize=13,
              fontweight='bold', color=NAVY, pad=8)

# -- House geometry fill (light colours by zone) --
from matplotlib.patches import Polygon as MPoly

# Foundation slab
ax2.add_patch(MPoly([[0, -GH], [W, -GH], [W, 0], [0, 0]],
              closed=True, facecolor='#B0BEC5', edgecolor='#333', lw=1.2,
              hatch='////', alpha=0.7, zorder=2))

# Walls
ax2.add_patch(MPoly([[0, 0], [W, 0], [W, H], [0, H]],
              closed=True, facecolor='#ECEFF1', edgecolor='#333', lw=1.2,
              zorder=2))

# Roof
ax2.add_patch(MPoly([[0, H], [W/2, H+RH], [W, H]],
              closed=True, facecolor='#CFD8DC', edgecolor='#333', lw=1.4,
              zorder=2))

# Partition wall
ax2.plot([W/2, W/2], [0, H], color='#607D8B', lw=1.5, ls='--', zorder=3)

# Interior floor
ax2.plot([0, W], [0, 0], color='#455A64', lw=2, zorder=3)

# ── Gravity + snow load on roof (downward arrows) ──
n_arrows = 9
xs_roof = np.linspace(0.6, W - 0.6, n_arrows)
arrow_len = 0.9
for xi in xs_roof:
    # Map xi to roof surface y
    if xi <= W/2:
        y_roof = H + (RH / (W/2)) * xi
    else:
        y_roof = H + (RH / (W/2)) * (W - xi)
    ax2.annotate("", xy=(xi, y_roof + 0.08),
                 xytext=(xi, y_roof + arrow_len),
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5,
                                 mutation_scale=10))

ax2.text(W/2, H + RH + 1.0, "Self-weight + Snow Load\n$q = 2.5$ kN/m²",
         ha='center', va='bottom', fontsize=10, color=RED, fontweight='bold')

# ── Wind load on left wall (horizontal arrows) ──
ys_wind = np.linspace(0.5, H - 0.5, 6)
for yi in ys_wind:
    ax2.annotate("", xy=(0.1, yi), xytext=(-1.1, yi),
                 arrowprops=dict(arrowstyle="-|>", color='#1565C0', lw=1.5,
                                 mutation_scale=10))
ax2.text(-1.25, H/2, "Wind\n$p = 1.2$\nkN/m²", ha='center', va='center',
         fontsize=9.5, color='#1565C0', fontweight='bold')

# ── Pinned supports at foundation corners ──
def draw_pin(ax, xc, yc, size=0.22, color='#BF360C'):
    """Draw a pinned support triangle."""
    tri = mpatches.RegularPolygon((xc, yc - size * 0.6), numVertices=3,
                                  radius=size * 0.9, orientation=np.pi,
                                  facecolor=color, edgecolor='#333', lw=1)
    ax.add_patch(tri)
    ax.plot([xc - size, xc + size], [yc - size * 1.5, yc - size * 1.5],
            color='#333', lw=2)
    # diagonal hatch lines
    for dx in np.linspace(-size, size, 6):
        ax.plot([xc + dx, xc + dx - 0.15], [yc - size*1.5, yc - size*1.8],
                color='#333', lw=0.8)

draw_pin(ax2, 0,  -GH)
draw_pin(ax2, W,  -GH)
ax2.text(W/2, -GH - 0.85, "Pin Supports (no translation)",
         ha='center', va='top', fontsize=10, color='#BF360C', fontweight='bold')

# ── Dimension annotations ──
dim_style = dict(arrowstyle='<->', color='#444', lw=1.1, mutation_scale=8)

# Width dimension
yy = -GH - 0.5
ax2.annotate("", xy=(W, yy), xytext=(0, yy),
             arrowprops=dict(arrowstyle='<->', color='#444', lw=1.1, mutation_scale=8))
ax2.text(W/2, yy - 0.18, "L = 10.0 m", ha='center', va='top', fontsize=9.5, color='#444')

# Wall height dimension
xx = W + 0.8
ax2.annotate("", xy=(xx, H), xytext=(xx, 0),
             arrowprops=dict(arrowstyle='<->', color='#444', lw=1.1, mutation_scale=8))
ax2.text(xx + 0.12, H/2, "h = 5.0 m", ha='left', va='center', fontsize=9.5,
         color='#444', rotation=90)

# Roof height
ax2.annotate("", xy=(W + 0.8, H + RH), xytext=(W + 0.8, H),
             arrowprops=dict(arrowstyle='<->', color='#444', lw=1.1, mutation_scale=8))
ax2.text(W + 0.95, H + RH/2, "r = 3.0 m", ha='left', va='center', fontsize=9.5,
         color='#444', rotation=90)

# ── Material labels ──
ax2.text(1.0, 2.2, "Concrete\nE = 30 GPa\nν = 0.20", ha='center', va='center',
         fontsize=8.5, color='#37474F', style='italic',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#90A4AE', lw=0.8))
ax2.text(W - 1.0, 2.2, "Brick\nE = 15 GPa\nν = 0.25", ha='center', va='center',
         fontsize=8.5, color='#37474F', style='italic',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#90A4AE', lw=0.8))
ax2.text(W/2, H + RH/2 + 0.2, "Timber\nE = 11 GPa\nν = 0.30",
         ha='center', va='center', fontsize=8.5, color='#37474F', style='italic',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#90A4AE', lw=0.8))

ax2.set_xticks([0, W/4, W/2, 3*W/4, W])
ax2.set_xticklabels(["0", "2.5", "5.0", "7.5", "10.0"])
ax2.set_yticks([-GH, 0, H/2, H, H + RH/2, H + RH])
ax2.set_yticklabels([f"{-GH:.1f}", "0", "2.5", "5.0",
                     f"{H+RH/2:.1f}", f"{H+RH:.1f}"])

# ── Legend for panel (b) ──
legend_handles = [
    mpatches.Patch(facecolor='#ECEFF1', edgecolor='#333', label="Masonry Walls"),
    mpatches.Patch(facecolor='#CFD8DC', edgecolor='#333', label="Timber Roof"),
    mpatches.Patch(facecolor='#B0BEC5', edgecolor='#333', hatch='////', label="Concrete Foundation"),
    mpatches.Patch(facecolor=RED,       edgecolor='none', alpha=0.8, label="Gravity + Snow Load"),
    mpatches.Patch(facecolor='#1565C0', edgecolor='none', alpha=0.8, label="Wind Load"),
    mpatches.Patch(facecolor='#BF360C', edgecolor='#333', label="Pin Supports"),
]
ax2.legend(handles=legend_handles, loc='lower right', fontsize=8.5,
           framealpha=0.92, edgecolor='#90A4AE', ncol=2)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Residential House Structure — Finite Element Analysis",
    fontsize=16, fontweight='bold', color=NAVY, y=0.96,
)

# ── Stats inset (bottom centre) ───────────────────────────────────────────────
stats_ax = fig.add_axes([0.33, 0.01, 0.34, 0.09])
stats_ax.set_xlim(0, 1); stats_ax.set_ylim(0, 1); stats_ax.axis('off')
stats_ax.set_facecolor('#DDE3EE')
stats_ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0",
                                  facecolor='#DDE3EE', edgecolor=NAVY, lw=1.2,
                                  zorder=0))
stats = [
    (f"{vm_mpa.max():.2f} MPa", "Peak Stress"),
    (f"{int(len(simp)):,}", "Elements"),
    (f"{int(len(pts)):,}", "Nodes"),
    (f"{vm_mpa.mean():.2f} MPa", "Mean Stress"),
]
for i, (val, lbl) in enumerate(stats):
    xi = (i + 0.5) / 4
    stats_ax.text(xi, 0.74, val, ha='center', va='center',
                  fontsize=12, fontweight='bold', color=NAVY)
    stats_ax.text(xi, 0.24, lbl, ha='center', va='center',
                  fontsize=8.5, color='#444')
    if i < 3:
        stats_ax.axvline(x=(i+1)/4, color='#90A4AE', lw=0.8, ymin=0.1, ymax=0.9)

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PATH, dpi=300, bbox_inches='tight', facecolor=LGRAY)
plt.close(fig)
print(f"Saved → {OUT_PATH}")
