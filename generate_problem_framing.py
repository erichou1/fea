"""
Generate a 3-panel Problem Framing diagram for the SASTO poster.

Panel 1 (left)   – Voxelized Structure: 3D house + ρ(x) ∈ {0,1}
Panel 2 (middle) – PDE + Deep Ensemble Surrogate (3D CNN sketch)
Panel 3 (right)  – Predicted Structural Response (stress & compliance heat-map)

Output: poster_images_extracted/problem_framing.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh
from PIL import Image
from pathlib import Path

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY   = "#062B7A"
BLUE   = "#0A3D9A"
LBLUE  = "#C5D4F5"
TEAL   = "#008C9E"
GOLD   = "#CFA535"
RED    = "#D7263D"
DARK   = "#0B1736"
CARD   = "#F7F9FC"
SAND   = "#F5ECD7"
WHITE  = "#FFFFFF"
SECBAR = "#0A3D9A"

OUT = Path("poster_images_extracted/problem_framing.png")

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: render a PLY mesh as matplotlib 3-D with optional stress colormap
# ═══════════════════════════════════════════════════════════════════════════════
def render_ply_to_axes(ax3d, ply_path, cmap_name=None, elev=22, azim=-50):
    """Load PLY and draw it onto an existing Axes3D."""
    mesh = trimesh.load(str(ply_path), force="mesh", process=False)
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # normalise
    lo, hi = verts.min(0), verts.max(0)
    span   = (hi - lo).max()
    verts  = (verts - (lo + hi) / 2) / span

    poly_v = verts[faces]

    # per-face normals for shading
    n0  = poly_v[:, 0]; n1 = poly_v[:, 1]; n2 = poly_v[:, 2]
    nrm = np.cross(n1 - n0, n2 - n0)
    mag = np.linalg.norm(nrm, axis=1, keepdims=True)
    mag[mag == 0] = 1.0
    nrm = nrm / mag
    light = np.array([np.cos(np.radians(35))*np.cos(np.radians(-40)),
                      np.cos(np.radians(35))*np.sin(np.radians(-40)),
                      np.sin(np.radians(35))])
    intens = np.clip(nrm @ light, 0, 1)

    if cmap_name:
        # use z-coordinate of face centre as proxy for "stress"
        z_mid   = poly_v[:, :, 2].mean(axis=1)
        z_norm  = (z_mid - z_mid.min()) / (z_mid.max() - z_mid.min() + 1e-9)
        cmap    = plt.get_cmap(cmap_name)
        base_fc = cmap(z_norm)[:, :3]
    else:
        # vertex colours if available
        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
            vc = mesh.visual.vertex_colors[:, :3] / 255.0
            base_fc = vc[faces].mean(axis=1)
        else:
            base_fc = np.tile(matplotlib.colors.to_rgb(TEAL), (len(faces), 1))

    shaded = np.clip(base_fc * (0.30 + 0.70 * intens[:, np.newaxis]), 0, 1)

    col = Poly3DCollection(poly_v, zsort="average")
    col.set_facecolor(shaded)
    col.set_edgecolor("none")
    col.set_alpha(1.0)
    ax3d.add_collection3d(col)

    pad = 0.06
    ax3d.set_xlim(-0.5 - pad, 0.5 + pad)
    ax3d.set_ylim(-0.5 - pad, 0.5 + pad)
    ax3d.set_zlim(-0.5 - pad, 0.5 + pad)
    real_span = hi - lo
    ax3d.set_box_aspect([real_span[0], real_span[1], real_span[2]])
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_axis_off()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN FIGURE
# ═══════════════════════════════════════════════════════════════════════════════
FIG_W, FIG_H = 16.0, 5.6
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=WHITE)

# ── three card-background rectangles ─────────────────────────────────────────
CARD_Y0, CARD_H = 0.04, 0.92   # normalised figure coords
CARD_XS = [0.01, 0.355, 0.695]
CARD_XE = [0.32, 0.66, 0.99]
CARD_COLORS  = [LBLUE, SAND, LBLUE]
EDGE_COLORS  = [BLUE,  GOLD, BLUE]

for x0, x1, fc, ec in zip(CARD_XS, CARD_XE, CARD_COLORS, EDGE_COLORS):
    rect = mpatches.FancyBboxPatch(
        (x0, CARD_Y0), x1 - x0, CARD_H,
        boxstyle="round,pad=0.01",
        facecolor=fc, edgecolor=ec, linewidth=1.8,
        transform=fig.transFigure, clip_on=False, zorder=0,
    )
    fig.add_artist(rect)

# ── Blue connecting arrows between cards ─────────────────────────────────────
for xa, xb in [(CARD_XE[0], CARD_XS[1]), (CARD_XE[1], CARD_XS[2])]:
    xm = (xa + xb) / 2
    ym = CARD_Y0 + CARD_H / 2
    arr = FancyArrowPatch(
        (xa + 0.005, ym), (xb - 0.005, ym),
        arrowstyle="-|>", mutation_scale=22,
        color=BLUE, lw=2.2,
        transform=fig.transFigure, clip_on=False, zorder=10,
    )
    fig.add_artist(arr)

# ─────────────────────────────────────────────────────────────────────────────
#  PANEL 1 — Voxelized Structure
# ─────────────────────────────────────────────────────────────────────────────
ax1 = fig.add_axes([0.02, 0.15, 0.28, 0.70], projection="3d", facecolor=LBLUE)
render_ply_to_axes(ax1,
    "figures/screenshot_stls/REF_original_colored.ply",
    cmap_name=None, elev=18, azim=-48)

# Panel 1 title
fig.text(0.165, 0.925, "Voxelized Structure",
         ha="center", va="center", fontsize=13, fontweight="bold", color=BLUE,
         transform=fig.transFigure)

# Panel 1 equation label
ax1t = fig.add_axes([0.02, 0.05, 0.28, 0.12], facecolor="none")
ax1t.set_axis_off()
ax1t.text(0.5, 0.72, r"$\rho(\mathbf{x}) \in \{0,1\}$",
          ha="center", va="center", fontsize=13, color=DARK, fontweight="bold")
ax1t.text(0.5, 0.28, "Binary material field",
          ha="center", va="center", fontsize=9, color=DARK, fontstyle="italic")

# ─────────────────────────────────────────────────────────────────────────────
#  PANEL 2 — Linear Elasticity PDE + Deep Ensemble CNN
# ─────────────────────────────────────────────────────────────────────────────
ax2 = fig.add_axes([0.365, 0.06, 0.285, 0.88], facecolor="none")
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_axis_off()

# -- PDE sub-box
pde_box = FancyBboxPatch((0.04, 0.52), 0.92, 0.43,
    boxstyle="round,pad=0.02",
    facecolor=CARD, edgecolor=GOLD, linewidth=1.4, zorder=2)
ax2.add_patch(pde_box)
ax2.text(0.50, 0.92, "Linear Elasticity PDE",
         ha="center", va="center", fontsize=10.5, fontweight="bold",
         color=DARK, zorder=5)
ax2.text(0.50, 0.80, r"$\bar{\nabla} \cdot \boldsymbol{\sigma} + \mathbf{b} = \mathbf{0}$",
         ha="center", va="center", fontsize=13, color=DARK, zorder=5)
ax2.annotate("", xy=(0.50, 0.70), xytext=(0.50, 0.74),
             arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.2), zorder=5)
ax2.text(0.50, 0.65, r"$K\mathbf{u} = \mathbf{f}$",
         ha="center", va="center", fontsize=14.5, fontweight="bold",
         color=BLUE, zorder=5)
ax2.text(0.50, 0.56, "FEA discretization",
         ha="center", va="center", fontsize=8.5, color=DARK,
         fontstyle="italic", zorder=5)

# -- CNN sub-box
cnn_box = FancyBboxPatch((0.04, 0.04), 0.92, 0.44,
    boxstyle="round,pad=0.02",
    facecolor=CARD, edgecolor=BLUE, linewidth=1.4, zorder=2)
ax2.add_patch(cnn_box)

# draw a mini 3D CNN architecture
def draw_conv_block(ax, cx, cy, w, h, d, color, alpha=0.80, lw=0.8):
    """Draw a pseudo-3D rectangular block representing a conv feature map."""
    dz = d * 0.012; dx = d * 0.009
    # front face
    front = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
        boxstyle="square,pad=0", facecolor=color,
        edgecolor=DARK, linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(front)
    # top face (parallelogram approximation using polygon)
    top_x = [cx - w/2, cx - w/2 + dx, cx + w/2 + dx, cx + w/2]
    top_y = [cy + h/2, cy + h/2 + dz, cy + h/2 + dz, cy + h/2]
    ax.fill(top_x, top_y, color=matplotlib.colors.to_rgb(color),
            alpha=alpha * 0.6, zorder=3, edgecolor=DARK, linewidth=lw)
    # right face
    right_x = [cx + w/2, cx + w/2 + dx, cx + w/2 + dx, cx + w/2]
    right_y = [cy - h/2, cy - h/2 + dz, cy + h/2 + dz, cy + h/2]
    ax.fill(right_x, right_y, color=matplotlib.colors.to_rgb(color),
            alpha=alpha * 0.45, zorder=3, edgecolor=DARK, linewidth=lw)

blocks = [
    # (cx,  cy,  w,    h,    d,   color)
    (0.14, 0.26, 0.11, 0.30, 5.0, BLUE),        # input
    (0.31, 0.26, 0.085, 0.26, 6.0, TEAL),       # conv1
    (0.46, 0.26, 0.065, 0.21, 7.0, "#6A8FCC"),  # conv2
    (0.60, 0.26, 0.045, 0.15, 6.0, TEAL),       # conv3
    (0.74, 0.26, 0.025, 0.08, 4.0, BLUE),       # fc
    (0.88, 0.26, 0.025, 0.10, 3.0, GOLD),       # output
]
for (cx, cy, w, h, d, col) in blocks:
    draw_conv_block(ax2, cx, cy, w, h, d, col)

# connections between blocks (simple lines at mid-height)
for i in range(len(blocks) - 1):
    b1, b2 = blocks[i], blocks[i+1]
    ax2.annotate("", xy=(b2[0] - b2[2]/2 - 0.003, b2[1]),
                 xytext=(b1[0] + b1[2]/2 + 0.003, b1[1]),
                 arrowprops=dict(arrowstyle="-|>", color=DARK,
                                 lw=0.9, mutation_scale=8), zorder=5)

ax2.text(0.50, 0.48, "Deep Ensemble Surrogate  ·  3D CNN",
         ha="center", va="center", fontsize=9.5, fontweight="bold",
         color=BLUE, zorder=5)
ax2.text(0.50, 0.41, "Learns operator",
         ha="center", va="center", fontsize=8.5, color=DARK,
         fontstyle="italic", zorder=5)
ax2.text(0.50, 0.34,
         r"$\rho(\mathbf{x}) \longrightarrow \{\sigma_{VM},\; C\}$",
         ha="center", va="center", fontsize=11, color=DARK,
         fontweight="bold", zorder=5)
ax2.text(0.50, 0.14,
         r"$5\times$ independent members$\;\Rightarrow\;\mu_C,\,\sigma_C$",
         ha="center", va="center", fontsize=8.5, color=DARK,
         fontstyle="italic", zorder=5)

# Panel 2 title
fig.text(0.508, 0.925, "Surrogate Model",
         ha="center", va="center", fontsize=13, fontweight="bold", color=DARK,
         transform=fig.transFigure)

# ─────────────────────────────────────────────────────────────────────────────
#  PANEL 3 — Predicted Structural Response (stress + compliance heat-maps)
# ─────────────────────────────────────────────────────────────────────────────
# Left 3D: original with stress coloring (jet = high stress at top)
ax3a = fig.add_axes([0.70, 0.18, 0.135, 0.66], projection="3d", facecolor=LBLUE)
render_ply_to_axes(ax3a,
    "figures/screenshot_stls/REF_original_cutaway.ply",
    cmap_name="jet", elev=18, azim=-48)

# Right 3D: optimized with displacement coloring (plasma)
ax3b = fig.add_axes([0.845, 0.18, 0.135, 0.66], projection="3d", facecolor=LBLUE)
render_ply_to_axes(ax3b,
    "figures/screenshot_stls/REF_SASTO_PA_cutaway.ply",
    cmap_name="plasma", elev=18, azim=-48)

# -- stress colourbar (left render)
ax_cb_s = fig.add_axes([0.700, 0.11, 0.130, 0.025])
cb_s = fig.colorbar(
    plt.cm.ScalarMappable(matplotlib.colors.Normalize(0, 1),
                          cmap=plt.get_cmap("jet")),
    cax=ax_cb_s, orientation="horizontal")
cb_s.set_ticks([0, 1]); cb_s.set_ticklabels(["Low", "High"],
    fontsize=7, color=DARK)
ax_cb_s.set_title(r"$\sigma_{VM}$  stress", fontsize=7.5, color=DARK, pad=2)

# -- compliance colourbar (right render)
ax_cb_d = fig.add_axes([0.845, 0.11, 0.130, 0.025])
cb_d = fig.colorbar(
    plt.cm.ScalarMappable(matplotlib.colors.Normalize(0, 1),
                          cmap=plt.get_cmap("plasma")),
    cax=ax_cb_d, orientation="horizontal")
cb_d.set_ticks([0, 1]); cb_d.set_ticklabels(["Low", "High"],
    fontsize=7, color=DARK)
ax_cb_d.set_title(r"Displacement $|\mathbf{u}|$", fontsize=7.5, color=DARK, pad=2)

# C= label
fig.text(0.852, 0.088, r"$C \approx 2.57$",
         ha="left", va="center", fontsize=8.5, color=DARK, fontweight="bold",
         transform=fig.transFigure)

# Panel 3 title
fig.text(0.852, 0.925,
         "Predicted Structural Response",
         ha="center", va="center", fontsize=13, fontweight="bold", color=BLUE,
         transform=fig.transFigure)

# Panel 3 sub-label at bottom
ax3t = fig.add_axes([0.695, 0.035, 0.295, 0.055], facecolor="none")
ax3t.set_axis_off()
ax3t.text(0.5, 0.6, r"$\sigma_{VM}$,  $\mathbf{u}$,  $C$",
          ha="center", va="center", fontsize=10, color=DARK, fontweight="bold")
ax3t.text(0.5, 0.05, "von Mises stress, displacement, compliance",
          ha="center", va="center", fontsize=8, color=DARK, fontstyle="italic")

# ─────────────────────────────────────────────────────────────────────────────
#  Save
# ─────────────────────────────────────────────────────────────────────────────
plt.savefig(str(OUT), dpi=230, bbox_inches="tight",
            facecolor=WHITE, edgecolor="none")
plt.close()
print(f"Saved → {OUT}")
