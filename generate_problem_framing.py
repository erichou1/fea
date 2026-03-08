"""
Generate the poster problem-framing diagram.

New layout:
  [real voxelized house] -> [single surrogate-model panel] -> [full-house stress response]

The middle panel contains:
  - 3D CNN encoder cue
  - latent feature matrix
  - deep ensemble predictor
  - compact physics/training equation strip

Output: poster_images_extracted/problem_framing.png
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path


NAVY = "#062B7A"
BLUE = "#0A3D9A"
LBLUE = "#D9E5FB"
TEAL = "#0BA6B7"
GOLD = "#CFA535"
RED = "#D7263D"
DARK = "#0B1736"
CARD = "#F7F9FF"
WHITE = "#FFFFFF"

OUT = Path("poster_images_extracted/problem_framing.png")


def render_voxelized_house(ax3d, mesh_path, pitch_div=28, elev=24, azim=-55):
    """Voxelize the full-house mesh and render the filled voxel shell."""
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    pitch = float(mesh.extents.max() / pitch_div)
    vox = mesh.voxelized(pitch)
    filled = np.asarray(vox.matrix, dtype=bool)

    sx, sy, sz = filled.shape
    zgrad = np.linspace(0, 1, sz)[None, None, :]
    facecolors = np.zeros(filled.shape + (4,), dtype=float)
    edgecolors = np.zeros_like(facecolors)

    facecolors[..., 0] = 0.08 + 0.10 * zgrad
    facecolors[..., 1] = 0.30 + 0.35 * zgrad
    facecolors[..., 2] = 0.78 + 0.15 * zgrad
    facecolors[..., 3] = 0.98

    edgecolors[..., 0] = 1.0
    edgecolors[..., 1] = 1.0
    edgecolors[..., 2] = 1.0
    edgecolors[..., 3] = 0.12

    ax3d.voxels(
        filled,
        facecolors=facecolors,
        edgecolors=edgecolors,
        linewidth=0.20,
    )
    ax3d.set_box_aspect([sx, sy, sz])
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_facecolor(LBLUE)
    ax3d.set_axis_off()


def render_mesh(ax3d, ply_path, color_mode="stress", elev=22, azim=-55, ambient=0.82):
    """Render a mesh with simple shaded synthetic stress coloring."""
    mesh = trimesh.load(str(ply_path), force="mesh", process=False)
    verts = np.array(mesh.vertices, dtype=float)
    faces = np.array(mesh.faces)
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

    centers = poly_v.mean(axis=1)
    z_norm = (centers[:, 2] - centers[:, 2].min()) / (np.ptp(centers[:, 2]) + 1e-9)
    radial = np.sqrt(centers[:, 0] ** 2 + centers[:, 1] ** 2)
    radial = radial / (radial.max() + 1e-9)

    if color_mode == "stress":
        stress = np.clip(0.58 * (1 - z_norm) + 0.42 * radial, 0, 1)
        base = plt.get_cmap("jet")(stress)[:, :3]
    else:
        base = np.column_stack([
            0.10 + 0.14 * z_norm,
            0.32 + 0.25 * z_norm,
            0.75 + 0.15 * z_norm,
        ])

    shaded = np.clip(base * (ambient + (1 - ambient) * lambert[:, None]), 0, 1)

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
    ax3d.set_facecolor(LBLUE)
    ax3d.set_axis_off()


def add_card(fig, xywh, facecolor, edgecolor, lw=2.2, radius=0.02, zorder=0):
    x, y, w, h = xywh
    fig.add_artist(FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        transform=fig.transFigure,
        clip_on=False,
        zorder=zorder,
    ))


def add_arrow(fig, start, end, color=BLUE, lw=2.6, scale=26, z=15):
    fig.add_artist(FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=scale,
        color=color,
        lw=lw,
        transform=fig.transFigure,
        clip_on=False,
        zorder=z,
    ))


fig = plt.figure(figsize=(20, 5.8), facecolor=WHITE)

left = (0.015, 0.08, 0.215, 0.84)
mid = (0.280, 0.08, 0.440, 0.84)
right = (0.770, 0.08, 0.215, 0.84)

add_card(fig, left, LBLUE, BLUE, zorder=0)
add_card(fig, mid, CARD, GOLD, zorder=0)
add_card(fig, right, LBLUE, BLUE, zorder=0)

add_arrow(fig, (0.235, 0.50), (0.275, 0.50))
add_arrow(fig, (0.725, 0.50), (0.765, 0.50))


# ── Left: actual voxelized structure ─────────────────────────────────────────
ax_left = fig.add_axes([0.032, 0.18, 0.178, 0.64], projection="3d")
render_voxelized_house(ax_left, "figures/screenshot_stls/REF_original_colored.ply")

fig.text(0.1225, 0.905, "Voxelized Structure",
         ha="center", va="center", fontsize=13, fontweight="bold", color=BLUE)
fig.text(0.1225, 0.115,
         r"$\rho \in \{0,1\}^{n_x \times n_y \times n_z}$",
         ha="center", va="center", fontsize=11.5, color=DARK, fontweight="bold")
fig.text(0.1225, 0.082, "binary occupancy tensor",
         ha="center", va="center", fontsize=8.7, color=DARK, fontstyle="italic")


# ── Middle: one coherent surrogate-model panel ───────────────────────────────
ax_mid = fig.add_axes([mid[0], mid[1], mid[2], mid[3]], facecolor="none")
ax_mid.set_xlim(0, 1)
ax_mid.set_ylim(0, 1)
ax_mid.set_axis_off()

ax_mid.text(0.50, 0.945, "Physics-Informed Surrogate Model",
            ha="center", va="center", fontsize=14, fontweight="bold", color=DARK)

# encoder block
enc_box = FancyBboxPatch((0.035, 0.20), 0.255, 0.62,
                         boxstyle="round,pad=0.012,rounding_size=0.025",
                         facecolor="#EEF4FF", edgecolor=BLUE, linewidth=1.8)
ax_mid.add_patch(enc_box)
ax_mid.text(0.162, 0.78, "3D CNN Encoder",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color=BLUE)

for w, h, dx, dy, c in [
    (0.120, 0.28, 0.000, 0.000, "#91B2F0"),
    (0.095, 0.22, 0.030, 0.025, "#5E8BDE"),
    (0.072, 0.17, 0.056, 0.045, TEAL),
]:
    x0 = 0.078 + dx
    y0 = 0.39 + dy
    ax_mid.add_patch(mpatches.Rectangle((x0, y0), w, h,
                                        facecolor=c, edgecolor=WHITE, linewidth=1.0))
    ax_mid.add_patch(mpatches.Rectangle((x0 + 0.016, y0 + 0.016), w, h,
                                        facecolor="none", edgecolor=DARK,
                                        linewidth=0.8, alpha=0.22))
    for gx in np.linspace(x0 + 0.02, x0 + w - 0.02, 4):
        ax_mid.plot([gx, gx], [y0, y0 + h], color=WHITE, lw=0.45, alpha=0.40)
    for gy in np.linspace(y0 + 0.03, y0 + h - 0.03, 4):
        ax_mid.plot([x0, x0 + w], [gy, gy], color=WHITE, lw=0.45, alpha=0.40)

ax_mid.text(0.162, 0.28, r"$\mathbf{z}=E_{\theta}(\rho)$",
            ha="center", va="center", fontsize=11.2, color=DARK, fontweight="bold")
ax_mid.text(0.162, 0.235, "learned structural features",
            ha="center", va="center", fontsize=8.1, color=DARK, fontstyle="italic")

# latent matrix block
lat_box = FancyBboxPatch((0.370, 0.20), 0.235, 0.62,
                         boxstyle="round,pad=0.012,rounding_size=0.025",
                         facecolor="#0D2E73", edgecolor=GOLD, linewidth=1.8)
ax_mid.add_patch(lat_box)
ax_mid.text(0.4875, 0.78, "Latent Feature Matrix",
            ha="center", va="center", fontsize=11.3, fontweight="bold", color=WHITE)

ax_lat = fig.add_axes([mid[0] + mid[2] * 0.405, mid[1] + mid[3] * 0.35,
                       mid[2] * 0.165, mid[3] * 0.31])
latent = np.outer(np.sin(np.linspace(0.2, 2.8, 14)), np.cos(np.linspace(0.1, 2.6, 10))).T
latent += 0.18 * np.outer(np.linspace(0, 1, 10), np.linspace(1, 0, 14))
ax_lat.imshow(latent, cmap="viridis", aspect="auto", interpolation="nearest")
ax_lat.set_xticks([])
ax_lat.set_yticks([])
for spine in ax_lat.spines.values():
    spine.set_color(WHITE)
    spine.set_linewidth(0.9)

ax_mid.text(0.4875, 0.285, r"$\mathbf{z} \in \mathbb{R}^{d \times c}$",
            ha="center", va="center", fontsize=10.8, color=GOLD, fontweight="bold")
ax_mid.text(0.4875, 0.238, r"$K(\rho)\,\mathbf{u}=\mathbf{f}$",
            ha="center", va="center", fontsize=11.2, color=WHITE, fontweight="bold")
ax_mid.text(0.4875, 0.192, "trained against FEA response",
            ha="center", va="center", fontsize=8.0, color=WHITE, fontstyle="italic")

# ensemble block
ens_box = FancyBboxPatch((0.685, 0.20), 0.280, 0.62,
                         boxstyle="round,pad=0.012,rounding_size=0.025",
                         facecolor="#F8F1E0", edgecolor=TEAL, linewidth=1.8)
ax_mid.add_patch(ens_box)
ax_mid.text(0.825, 0.78, "Deep Ensemble Predictor",
            ha="center", va="center", fontsize=11.2, fontweight="bold", color=DARK)

member_x = np.linspace(0.725, 0.875, 5)
for i, mx in enumerate(member_x):
    ax_mid.add_patch(FancyBboxPatch((mx - 0.020, 0.50), 0.040, 0.11,
                                    boxstyle="round,pad=0.006,rounding_size=0.01",
                                    facecolor="#DCEBFA", edgecolor=BLUE, linewidth=1.0))
    ax_mid.text(mx, 0.555, rf"$f_{i+1}$", ha="center", va="center",
                fontsize=8.6, color=BLUE, fontweight="bold")
    ax_mid.plot([mx, mx], [0.45, 0.50], color=TEAL, lw=1.3)
    ax_mid.plot([mx, 0.915], [0.45, 0.34], color=TEAL, lw=0.9, alpha=0.55)

stats = Circle((0.915, 0.34), 0.070, facecolor=TEAL, edgecolor=WHITE, linewidth=1.2)
ax_mid.add_patch(stats)
ax_mid.text(0.915, 0.355, r"$\mu$", ha="center", va="center",
            fontsize=12, color=WHITE, fontweight="bold")
ax_mid.text(0.915, 0.315, r"$\sigma$", ha="center", va="center",
            fontsize=10, color=WHITE, fontweight="bold")

ax_mid.text(0.805, 0.24, r"$\mu = \frac{1}{M}\sum_{m=1}^{M} y_m$",
            ha="center", va="center", fontsize=9.8, color=DARK, fontweight="bold")
ax_mid.text(0.805, 0.195, r"$\sigma^2 = \frac{1}{M}\sum_{m=1}^{M}(y_m-\mu)^2$",
            ha="center", va="center", fontsize=8.8, color=DARK)

# internal arrows and output label
ax_mid.add_patch(FancyArrowPatch((0.300, 0.51), (0.355, 0.51), arrowstyle="-|>",
                                 mutation_scale=18, color=BLUE, lw=2.0))
ax_mid.add_patch(FancyArrowPatch((0.615, 0.51), (0.670, 0.51), arrowstyle="-|>",
                                 mutation_scale=18, color=BLUE, lw=2.0))
ax_mid.text(0.92, 0.63, r"$\hat{\sigma}_{VM},\;\hat{C},\;\hat{\mathbf{u}}$",
            ha="center", va="center", fontsize=10.2, color=RED, fontweight="bold")


# ── Right: full-house structural response ───────────────────────────────────
ax_right = fig.add_axes([0.787, 0.18, 0.178, 0.64], projection="3d")
render_mesh(ax_right, "figures/screenshot_stls/REF_SASTO_PA_colored.ply",
            color_mode="stress", elev=22, azim=-55, ambient=0.84)

fig.text(0.8775, 0.905, "Predicted Structural Response",
         ha="center", va="center", fontsize=13, fontweight="bold", color=BLUE)
fig.text(0.8775, 0.082, "full-house von Mises stress field",
         ha="center", va="center", fontsize=8.7, color=DARK, fontstyle="italic")

ax_cb = fig.add_axes([0.815, 0.115, 0.125, 0.028])
cb = fig.colorbar(
    plt.cm.ScalarMappable(mcolors.Normalize(0, 1), cmap=plt.get_cmap("jet")),
    cax=ax_cb,
    orientation="horizontal",
)
cb.set_ticks([0, 1])
cb.set_ticklabels(["Low", "High"])
cb.ax.tick_params(labelsize=8, size=0, colors=DARK)
cb.outline.set_edgecolor(DARK)
cb.outline.set_linewidth(0.7)
ax_cb.set_title(r"$\sigma_{VM}$", fontsize=8.5, color=DARK, pad=2)


plt.savefig(str(OUT), dpi=240, bbox_inches="tight", facecolor=WHITE, edgecolor="none")
plt.close()
print(f"Saved → {OUT}")
