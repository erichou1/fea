"""
Generate 5 clean diagram icons for the Problem Framing conceptual diagram.
Output: poster_images_extracted/diagram_icons/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── palette ──────────────────────────────────────────────────────────────────
NAVY   = "#062B7A"
BLUE   = "#0A3D9A"
LBLUE  = "#C5D4F5"
TEAL   = "#008C9E"
GOLD   = "#CFA535"
RED    = "#D7263D"
DARK   = "#0B1736"
WHITE  = "#FFFFFF"
CARD   = "#F7F9FC"

OUT = "poster_images_extracted/diagram_icons"
os.makedirs(OUT, exist_ok=True)

W, H = 3.2, 2.6   # inches per icon
DPI  = 220

# ─────────────────────────────────────────────────────────────────────────────
# 1. VOXEL STRUCTURE  — 3D stacked voxel grid
# ─────────────────────────────────────────────────────────────────────────────
def draw_voxel(ax, x, y, z, color, alpha=1.0, lw=0.5):
    """Draw a unit cube at grid position (x,y,z)."""
    r = [[x,x+1],[y,y+1],[z,z+1]]
    verts = [
        [(r[0][0],r[1][0],r[2][0]),(r[0][1],r[1][0],r[2][0]),
         (r[0][1],r[1][1],r[2][0]),(r[0][0],r[1][1],r[2][0])],  # bottom
        [(r[0][0],r[1][0],r[2][1]),(r[0][1],r[1][0],r[2][1]),
         (r[0][1],r[1][1],r[2][1]),(r[0][0],r[1][1],r[2][1])],  # top
        [(r[0][0],r[1][0],r[2][0]),(r[0][1],r[1][0],r[2][0]),
         (r[0][1],r[1][0],r[2][1]),(r[0][0],r[1][0],r[2][1])],  # front
        [(r[0][0],r[1][1],r[2][0]),(r[0][1],r[1][1],r[2][0]),
         (r[0][1],r[1][1],r[2][1]),(r[0][0],r[1][1],r[2][1])],  # back
        [(r[0][0],r[1][0],r[2][0]),(r[0][0],r[1][1],r[2][0]),
         (r[0][0],r[1][1],r[2][1]),(r[0][0],r[1][0],r[2][1])],  # left
        [(r[0][1],r[1][0],r[2][0]),(r[0][1],r[1][1],r[2][0]),
         (r[0][1],r[1][1],r[2][1]),(r[0][1],r[1][0],r[2][1])],  # right
    ]
    poly = Poly3DCollection(verts, alpha=alpha, linewidths=lw)
    poly.set_facecolor(color)
    poly.set_edgecolor(DARK)
    ax.add_collection3d(poly)

# Grid positions for a hollow building shell
shell = []
for xi in range(4):
    for yi in range(4):
        for zi in range(3):
            is_wall = (xi==0 or xi==3 or yi==0 or yi==3)
            is_floor = (zi==0)
            is_roof  = (zi==2)
            if is_wall or is_floor or is_roof:
                shell.append((xi, yi, zi))

fig = plt.figure(figsize=(W, H), facecolor=CARD)
ax  = fig.add_subplot(111, projection='3d', facecolor=CARD)
for (xi,yi,zi) in shell:
    col = BLUE if zi < 2 else TEAL
    draw_voxel(ax, xi, yi, zi, col, alpha=0.85)

ax.set_xlim(0,4); ax.set_ylim(0,4); ax.set_zlim(0,3)
ax.set_box_aspect([4,4,3])
ax.view_init(elev=22, azim=-50)
ax.axis('off')
ax.set_title("Voxel Structure", fontsize=11, fontweight='bold',
             color=DARK, pad=4)
plt.tight_layout(pad=0.2)
plt.savefig(f"{OUT}/icon_voxel.png", dpi=DPI, bbox_inches='tight',
            facecolor=CARD)
plt.close()
print("1/5  icon_voxel.png")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DEEP ENSEMBLE SURROGATE — 5 mini neural-net columns
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(W, H), facecolor=CARD)
ax.set_facecolor(CARD)
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

# 5 ensemble members stacked slightly offset
neuron_sizes = [4, 6, 6, 3]   # nodes per layer
layer_x = np.linspace(0.12, 0.88, len(neuron_sizes))

for member in range(5):
    dy_off = (member - 2) * 0.03
    alpha  = 0.55 + member * 0.06
    col    = BLUE

    prev_positions = None
    for li, (lx, n) in enumerate(zip(layer_x, neuron_sizes)):
        positions = np.linspace(0.18 + dy_off, 0.82 + dy_off, n)
        # draw connections
        if prev_positions is not None:
            for py in prev_positions:
                for cy in positions:
                    ax.plot([layer_x[li-1], lx], [py, cy],
                            color=LBLUE, lw=0.4, alpha=alpha*0.7, zorder=1)
        # draw neurons
        for cy in positions:
            circle = plt.Circle((lx, cy), 0.028, color=col,
                                 alpha=alpha, zorder=3)
            ax.add_patch(circle)
        prev_positions = positions

# Brace label
ax.text(0.5, 0.04, "5× independent members  →  mean μ, std σ",
        ha='center', va='bottom', fontsize=7.5, color=DARK, style='italic')
ax.text(0.5, 0.94, "Deep Ensemble Surrogate",
        ha='center', va='top', fontsize=11, fontweight='bold', color=DARK)

# input/output labels
ax.text(layer_x[0],  0.11, "input\n7-ch", ha='center', fontsize=6.5,
        color=DARK, style='italic')
ax.text(layer_x[-1], 0.11, "σ, u, C", ha='center', fontsize=6.5,
        color=DARK, style='italic')

plt.tight_layout(pad=0.2)
plt.savefig(f"{OUT}/icon_ensemble.png", dpi=DPI, bbox_inches='tight',
            facecolor=CARD)
plt.close()
print("2/5  icon_ensemble.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SENSITIVITY GRADIENT — 4×4 heatmap with remove/keep overlay
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
sens = np.array([
    [0.8,  0.7,  0.6,  0.75],
    [0.3,  -0.4, -0.5, 0.3 ],
    [0.4,  -0.6, -0.7, 0.35],
    [0.85, 0.6,  0.55, 0.8 ],
])

fig, ax = plt.subplots(figsize=(W, H), facecolor=CARD)
ax.set_facecolor(CARD)

import matplotlib.colors as mcolors
cmap = mcolors.LinearSegmentedColormap.from_list(
    'sens', [(0.0, TEAL), (0.5, '#F7F9FC'), (1.0, RED)])
im = ax.imshow(sens, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

for i in range(4):
    for j in range(4):
        v = sens[i,j]
        sym = "✕" if v > 0 else "▲"
        col = RED if v > 0 else TEAL
        ax.text(j, i, sym, ha='center', va='center',
                fontsize=14, color=col, fontweight='bold')

ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_edgecolor(BLUE)

# legend
ax.text(-0.45, 1.5, "✕ remove\n(s > 0)",
        ha='left', va='center', fontsize=7, color=RED,
        transform=ax.transData)
ax.text(3.45, 1.5, "▲ keep\n(s < 0)",
        ha='right', va='center', fontsize=7, color=TEAL,
        transform=ax.transData)

ax.set_title("Sensitivity Gradient", fontsize=11, fontweight='bold',
             color=DARK, pad=6)
plt.tight_layout(pad=0.3)
plt.savefig(f"{OUT}/icon_sensitivity.png", dpi=DPI, bbox_inches='tight',
            facecolor=CARD)
plt.close()
print("3/5  icon_sensitivity.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TOPOLOGY-SAFE REMOVAL — 26-conn fragment (red X) vs 6-conn ok (teal ✓)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(W, H), facecolor=CARD)
fig.patch.set_facecolor(CARD)

def grid_squares(ax, grid, color, title, verdict, v_color):
    ax.set_facecolor(CARD)
    n = grid.shape[0]
    for i in range(n):
        for j in range(n):
            if grid[i,j]:
                rect = FancyBboxPatch((j+0.05, n-i-1+0.05), 0.88, 0.88,
                                      boxstyle="round,pad=0.05",
                                      facecolor=color, edgecolor=DARK, lw=0.8)
                ax.add_patch(rect)
    ax.set_xlim(0,n); ax.set_ylim(0,n)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=8.5, fontweight='bold', color=DARK, pad=3)
    ax.text(n/2, -0.55, verdict, ha='center', va='top',
            fontsize=9, fontweight='bold', color=v_color)

bad = np.array([
    [1,1,1,1],
    [1,0,0,0],
    [0,0,0,1],
    [1,1,1,1],
])
good = np.array([
    [1,1,1,1],
    [1,0,0,1],
    [1,0,0,1],
    [1,1,1,1],
])

grid_squares(axes[0], bad,  RED,  "26-adj",    "✕ fragments", RED)
grid_squares(axes[1], good, TEAL, "6-simple",  "✓ connected", TEAL)

# diagonal arrows on bad grid to show the problem
axes[0].annotate("", xy=(3.5, 0.5), xytext=(0.5, 3.5),
                 arrowprops=dict(arrowstyle='->', color=RED, lw=1.2,
                                 linestyle='dashed'))

fig.suptitle("Topology-Safe Removal", fontsize=11, fontweight='bold',
             color=DARK, y=1.0)
plt.tight_layout(pad=0.3)
plt.savefig(f"{OUT}/icon_topology.png", dpi=DPI, bbox_inches='tight',
            facecolor=CARD)
plt.close()
print("4/5  icon_topology.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. OPTIMIZED STRUCTURE — isometric hollow building outline
# ─────────────────────────────────────────────────────────────────────────────
thin_shell = []
for xi in range(4):
    for yi in range(4):
        for zi in range(3):
            is_ext_wall = ((xi==0 or xi==3) or (yi==0 or yi==3))
            # Interior walls: only outer ring, thinned
            #   keep exterior; skip interior if not floor/roof
            is_floor = (zi==0)
            is_roof  = (zi==2)
            if is_ext_wall or is_floor or is_roof:
                # thin interior partitions to single voxel
                is_inner_wall = is_ext_wall and not is_floor and not is_roof
                # skip some interior voxels to show thinning
                keep = True
                if is_inner_wall and xi in (0,3) and yi in (1,2):
                    keep = (zi < 2)  # keep only lower
                thin_shell.append((xi, yi, zi, keep))

fig = plt.figure(figsize=(W, H), facecolor=CARD)
ax  = fig.add_subplot(111, projection='3d', facecolor=CARD)

for (xi,yi,zi,keep) in thin_shell:
    col   = TEAL if zi == 2 else BLUE
    alpha = 0.85 if keep else 0.25
    draw_voxel(ax, xi, yi, zi, col, alpha=alpha)

ax.set_xlim(0,4); ax.set_ylim(0,4); ax.set_zlim(0,3)
ax.set_box_aspect([4,4,3])
ax.view_init(elev=22, azim=-50)
ax.axis('off')
ax.set_title("Optimized Structure", fontsize=11, fontweight='bold',
             color=DARK, pad=4)

# volume reduction badge
ax.text2D(0.72, 0.08, "−45%\nmaterial", transform=ax.transAxes,
          ha='center', va='bottom', fontsize=9, fontweight='bold',
          color=GOLD)

plt.tight_layout(pad=0.2)
plt.savefig(f"{OUT}/icon_optimized.png", dpi=DPI, bbox_inches='tight',
            facecolor=CARD)
plt.close()
print("5/5  icon_optimized.png")
print(f"\nAll icons saved to  {OUT}/")
