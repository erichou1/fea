"""
Generate a clean 3-panel Problem Framing diagram for the SASTO poster.
Follows the reference "image â†’ box â†’ image" style (clean, minimal text).

Panel 1 (left)   â€“ Voxelized Structure : uses fig_voxel_house.png (real render)
Panel 2 (middle) â€“ PDE + Deep Ensemble CNN box (minimal, clean)
Panel 3 (right)  â€“ Predicted Structural Response: two bright colormap renders

Output: poster_images_extracted/problem_framing.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh
from PIL import Image
from pathlib import Path

# â”€â”€ Palette â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
NAVY  = "#062B7A"
BLUE  = "#0A3D9A"
LBLUE = "#C5D4F5"
TEAL  = "#008C9E"
GOLD  = "#CFA535"
RED   = "#D7263D"
DARK  = "#0B1736"
CARD  = "#F7F9FF"
SAND  = "#FAF3E0"
WHITE = "#FFFFFF"

OUT = Path("poster_images_extracted/problem_framing.png")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  HELPER: render a PLY mesh with a bright z-height colormap
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def render_ply_colormap(ax3d, ply_path, cmap_name, elev=22, azim=-50,
                        ambient=0.55, bg_color=LBLUE):
    """Load PLY, apply z-height colormap with Lambertian shading, draw on ax3d."""
    mesh = trimesh.load(str(ply_path), force="mesh", process=False)
    verts = np.array(mesh.vertices, dtype=float)
    faces = np.array(mesh.faces)

    # normalise to unit cube
    lo, hi    = verts.min(0), verts.max(0)
    real_span = hi - lo
    span      = real_span.max()
    verts     = (verts - (lo + hi) / 2) / span

    poly_v = verts[faces]   # (F, 3, 3)

    # per-face z-centre â†’ colour
    z_mid  = poly_v[:, :, 2].mean(axis=1)
    z_norm = (z_mid - z_mid.min()) / (z_mid.max() - z_mid.min() + 1e-9)
    cmap   = plt.get_cmap(cmap_name)
    base_fc = cmap(z_norm)[:, :3]   # (F, 3) RGB

    # Lambertian shading from upper-left light
    n0  = poly_v[:, 0]; n1 = poly_v[:, 1]; n2 = poly_v[:, 2]
    nrm = np.cross(n1 - n0, n2 - n0)
    mag = np.linalg.norm(nrm, axis=1, keepdims=True)
    mag[mag == 0] = 1.0
    nrm = nrm / mag
    light   = np.array([np.cos(np.radians(35)) * np.cos(np.radians(-40)),
                         np.cos(np.radians(35)) * np.sin(np.radians(-40)),
                         np.sin(np.radians(35))])
    intens  = np.clip(nrm @ light, 0, 1)
    shaded  = np.clip(base_fc * (ambient + (1.0 - ambient) * intens[:, None]), 0, 1)

    col = Poly3DCollection(poly_v, zsort="average")
    col.set_facecolor(shaded)
    col.set_edgecolor("none")
    col.set_alpha(1.0)
    ax3d.add_collection3d(col)

    pad = 0.08
    ax3d.set_xlim(-0.5 - pad, 0.5 + pad)
    ax3d.set_ylim(-0.5 - pad, 0.5 + pad)
    ax3d.set_zlim(-0.5 - pad, 0.5 + pad)
    ax3d.set_box_aspect([real_span[0], real_span[1], real_span[2]])
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_facecolor(bg_color)
    ax3d.set_axis_off()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  FIGURE LAYOUT
#  Reference style: [image] â”€â”€â–º [box] â”€â”€â–º [image | image]
#  3 main columns with clean card backgrounds
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
FIG_W, FIG_H = 18.0, 5.8
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=WHITE)

# â”€â”€ Column geometry (figure-normalised) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Col A: left image   0.00 â€“ 0.27
# Arrow gap           0.27 â€“ 0.33
# Col B: middle box   0.33 â€“ 0.67
# Arrow gap           0.67 â€“ 0.73
# Col C: right images 0.73 â€“ 1.00

CY0, CH = 0.04, 0.92   # card y-start and height (fig-norm)

def add_card(x0, x1, fc, ec, lw=1.8, radius=0.015):
    rect = mpatches.FancyBboxPatch(
        (x0, CY0), x1 - x0, CH,
        boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        transform=fig.transFigure, clip_on=False, zorder=0)
    fig.add_artist(rect)

add_card(0.005, 0.295, LBLUE, BLUE)
add_card(0.330, 0.670, SAND,  GOLD)
add_card(0.705, 0.995, LBLUE, BLUE)

# â”€â”€ Blue arrows between cards â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for xa, xb in [(0.295, 0.330), (0.670, 0.705)]:
    xm_y = CY0 + CH / 2
    arr  = FancyArrowPatch(
        (xa + 0.004, xm_y), (xb - 0.004, xm_y),
        arrowstyle="-|>", mutation_scale=26,
        color=BLUE, lw=2.5,
        transform=fig.transFigure, clip_on=False, zorder=10)
    fig.add_artist(arr)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  PANEL A â€” Voxelized Structure  (use real rendered PNG)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
vox_img = np.array(Image.open("figures/fig_voxel_house.png").convert("RGB"))
# crop to roughly square centre section
h_img, w_img = vox_img.shape[:2]
# take centre square crop
crop_w = min(w_img, int(h_img * 1.35))
x0c  = (w_img - crop_w) // 2
vox_img = vox_img[:, x0c: x0c + crop_w]

ax_a = fig.add_axes([0.015, 0.155, 0.270, 0.700])
ax_a.imshow(vox_img, aspect="auto", interpolation="lanczos")
ax_a.set_axis_off()

# Panel A labels
fig.text(0.150, 0.923, "Voxelized Structure",
         ha="center", fontsize=14, fontweight="bold", color=BLUE,
         transform=fig.transFigure)
fig.text(0.150, 0.068, r"$\rho(\mathbf{x}) \in \{0,\,1\}$",
         ha="center", fontsize=12, color=DARK, fontweight="bold",
         transform=fig.transFigure)
fig.text(0.150, 0.040, "Binary material representation",
         ha="center", fontsize=8.5, color=DARK, fontstyle="italic",
         transform=fig.transFigure)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  PANEL B â€” PDE + Surrogate box  (clean text layout, two sub-boxes)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ax_b = fig.add_axes([0.340, 0.06, 0.320, 0.88], facecolor="none")
ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1); ax_b.set_axis_off()

# -- PDE sub-box
pde = FancyBboxPatch((0.06, 0.51), 0.88, 0.44,
    boxstyle="round,pad=0.025", facecolor=CARD,
    edgecolor=GOLD, linewidth=1.6, zorder=2)
ax_b.add_patch(pde)
ax_b.text(0.50, 0.92, "Linear Elasticity PDE",
          ha="center", va="center", fontsize=11, fontweight="bold",
          color=DARK, zorder=5)
ax_b.text(0.50, 0.79,
          r"$\bar{\nabla}\!\cdot\!\boldsymbol{\sigma} + \mathbf{b} = \mathbf{0}$",
          ha="center", va="center", fontsize=14, color=DARK, zorder=5)
ax_b.annotate("", xy=(0.50, 0.68), xytext=(0.50, 0.73),
              arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.3), zorder=5)
ax_b.text(0.50, 0.64, r"$K\mathbf{u} = \mathbf{f}$",
          ha="center", va="center", fontsize=16, fontweight="bold",
          color=BLUE, zorder=5)
ax_b.text(0.50, 0.55, "FEA discretization",
          ha="center", va="center", fontsize=9, color=DARK,
          fontstyle="italic", zorder=5)

# -- CNN / surrogate sub-box
cnn = FancyBboxPatch((0.06, 0.04), 0.88, 0.43,
    boxstyle="round,pad=0.025", facecolor=CARD,
    edgecolor=BLUE, linewidth=1.6, zorder=2)
ax_b.add_patch(cnn)
ax_b.text(0.50, 0.445, "Deep Ensemble Surrogate  Â·  3D CNN",
          ha="center", va="center", fontsize=10.5, fontweight="bold",
          color=BLUE, zorder=5)
ax_b.text(0.50, 0.355,
          r"$\rho(\mathbf{x})\;\longrightarrow\;\{\sigma_{VM},\; C\}$",
          ha="center", va="center", fontsize=13.5, color=DARK, zorder=5)
ax_b.text(0.50, 0.245,
          r"$5\!\times$ members$\;\Rightarrow\;\mu_C,\;\sigma_C$",
          ha="center", va="center", fontsize=10, color=DARK,
          fontstyle="italic", zorder=5)
ax_b.text(0.50, 0.13,
          "Eliminates repeated FEA\nduring optimization",
          ha="center", va="center", fontsize=8.5, color=DARK,
          linespacing=1.4, zorder=5)

# Panel B title
fig.text(0.500, 0.923, "Surrogate Model",
         ha="center", fontsize=14, fontweight="bold", color=DARK,
         transform=fig.transFigure)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  PANEL C â€” Predicted Structural Response  (2 Ã— bright colormap 3D renders)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Left render: original with jet stress coloring
ax_c1 = fig.add_axes([0.714, 0.175, 0.132, 0.680], projection="3d")
render_ply_colormap(ax_c1,
    "figures/screenshot_stls/REF_original_cutaway.ply",
    cmap_name="jet", elev=22, azim=-52, ambient=0.55)

# Right render: optimised with inferno displacement coloring
ax_c2 = fig.add_axes([0.854, 0.175, 0.132, 0.680], projection="3d")
render_ply_colormap(ax_c2,
    "figures/screenshot_stls/REF_SASTO_PA_cutaway.ply",
    cmap_name="inferno", elev=22, azim=-52, ambient=0.55)

# Colourbar: stress (jet)
ax_cb1 = fig.add_axes([0.714, 0.110, 0.130, 0.030])
cb1 = fig.colorbar(
    plt.cm.ScalarMappable(mcolors.Normalize(0, 1), cmap=plt.get_cmap("jet")),
    cax=ax_cb1, orientation="horizontal")
cb1.set_ticks([0, 1])
cb1.set_ticklabels(["Low stress", "High stress"], fontsize=7.5, color=DARK)
cb1.ax.tick_params(size=0, colors=DARK)
cb1.outline.set_edgecolor(DARK)
cb1.outline.set_linewidth(0.6)

# Colourbar: displacement (inferno)
ax_cb2 = fig.add_axes([0.854, 0.110, 0.130, 0.030])
cb2 = fig.colorbar(
    plt.cm.ScalarMappable(mcolors.Normalize(0, 1), cmap=plt.get_cmap("inferno")),
    cax=ax_cb2, orientation="horizontal")
cb2.set_ticks([0, 1])
cb2.set_ticklabels(["Low disp.", "High disp."], fontsize=7.5, color=DARK)
cb2.ax.tick_params(size=0, colors=DARK)
cb2.outline.set_edgecolor(DARK)
cb2.outline.set_linewidth(0.6)

# Labels under colorbars
fig.text(0.779, 0.062, r"$\sigma_{VM}$ Â· Original",
         ha="center", fontsize=8.5, color=DARK, fontweight="bold",
         transform=fig.transFigure)
fig.text(0.919, 0.062, r"$|\mathbf{u}|$ Â· SASTO-PA     $C\!\approx\!2.57$",
         ha="center", fontsize=8.5, color=DARK, fontweight="bold",
         transform=fig.transFigure)

# Panel C title
fig.text(0.850, 0.923, "Predicted Structural Response",
         ha="center", fontsize=14, fontweight="bold", color=BLUE,
         transform=fig.transFigure)

# Sub-label
fig.text(0.850, 0.040,
         r"von Mises stress  $\sigma_{VM}$,  displacement  $\mathbf{u}$,  compliance  $C$",
         ha="center", fontsize=8.5, color=DARK, fontstyle="italic",
         transform=fig.transFigure)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Save
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.savefig(str(OUT), dpi=230, bbox_inches="tight",
            facecolor=WHITE, edgecolor="none")
plt.close()
print(f"Saved â†’ {OUT}")
