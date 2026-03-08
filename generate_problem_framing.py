"""Generate a conceptual problem-framing diagram for the poster."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
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
BLACK = "#111111"
WALL = "#4477CC"
INTERIOR = "#E88843"
ROOF = "#54A24B"
SLAB = "#D6B48A"
ARROW_LW = 3.2
ARROW_SCALE = 30

OUT = Path("poster_images_extracted/problem_framing.png")


def render_voxelized_house(ax3d, mesh_path, pitch_div=28, elev=24, azim=-55):
    """Voxelize the full-house mesh and color major building parts heuristically."""
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    pitch = float(mesh.extents.max() / pitch_div)
    vox = mesh.voxelized(pitch)
    try:
        vox = vox.fill()
    except Exception:
        pass
    filled = np.asarray(vox.matrix, dtype=bool)

    sx, sy, sz = filled.shape
    facecolors = np.zeros(filled.shape + (4,), dtype=float)
    edgecolors = np.zeros_like(facecolors)

    x = np.linspace(0, 1, sx)[:, None, None]
    y = np.linspace(0, 1, sy)[None, :, None]
    z = np.linspace(0, 1, sz)[None, None, :]

    roof_mask = filled & (z > 0.78)
    slab_mask = filled & (z < 0.14)
    outer_mask = filled & ((x < 0.15) | (x > 0.85) | (y < 0.15) | (y > 0.85))
    interior_mask = filled & ~(roof_mask | slab_mask | outer_mask)

    exposed = np.zeros_like(filled, dtype=bool)
    exposed[0, :, :] = filled[0, :, :]
    exposed[-1, :, :] = filled[-1, :, :]
    exposed[:, 0, :] = filled[:, 0, :]
    exposed[:, -1, :] = filled[:, -1, :]
    exposed[:, :, 0] = filled[:, :, 0]
    exposed[:, :, -1] = filled[:, :, -1]
    exposed[1:, :, :] |= filled[1:, :, :] & ~filled[:-1, :, :]
    exposed[:-1, :, :] |= filled[:-1, :, :] & ~filled[1:, :, :]
    exposed[:, 1:, :] |= filled[:, 1:, :] & ~filled[:, :-1, :]
    exposed[:, :-1, :] |= filled[:, :-1, :] & ~filled[:, 1:, :]
    exposed[:, :, 1:] |= filled[:, :, 1:] & ~filled[:, :, :-1]
    exposed[:, :, :-1] |= filled[:, :, :-1] & ~filled[:, :, 1:]

    # Avoid seeing interior walls or floor slab through the shell in the conceptual thumbnail.
    visible = exposed & ~interior_mask & ~slab_mask

    def apply_color(mask, hex_color, alpha=0.98):
        rgb = mcolors.to_rgb(hex_color)
        facecolors[mask, 0] = rgb[0]
        facecolors[mask, 1] = rgb[1]
        facecolors[mask, 2] = rgb[2]
        facecolors[mask, 3] = alpha

    apply_color(outer_mask & visible, WALL)
    apply_color(interior_mask & visible, INTERIOR)
    apply_color(roof_mask & visible, ROOF)
    apply_color(slab_mask & visible, SLAB)

    edgecolors[..., 0] = 1.0
    edgecolors[..., 1] = 1.0
    edgecolors[..., 2] = 1.0
    edgecolors[..., 3] = 0.15

    ax3d.voxels(
        visible,
        facecolors=facecolors,
        edgecolors=edgecolors,
        linewidth=0.20,
    )
    ax3d.set_box_aspect([sx, sy, sz])
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_facecolor(WHITE)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
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
    ax3d.set_facecolor(WHITE)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
    ax3d.set_axis_off()


def render_reference_voxel_house(ax3d, elev=28, azim=-54):
    """Render a clean stylized voxel house matching the attached part-colored look."""
    nx, ny, nz = 18, 18, 11
    occ = np.zeros((nx, ny, nz), dtype=bool)
    facecolors = np.zeros((nx, ny, nz, 4), dtype=float)
    edgecolors = np.zeros_like(facecolors)

    def set_block(x0, x1, y0, y1, z0, z1, color, alpha=1.0):
        rgb = mcolors.to_rgb(color)
        occ[x0:x1, y0:y1, z0:z1] = True
        facecolors[x0:x1, y0:y1, z0:z1, :3] = rgb
        facecolors[x0:x1, y0:y1, z0:z1, 3] = alpha
        edgecolors[x0:x1, y0:y1, z0:z1, :3] = np.clip(np.array(rgb) * 0.55, 0, 1)
        edgecolors[x0:x1, y0:y1, z0:z1, 3] = 0.45

    set_block(0, nx, 0, ny, 0, 3, SLAB)
    set_block(0, nx, 0, ny, 3, 7, WALL)
    set_block(8, 9, 0, 1, 3, 7, INTERIOR)

    roof_levels = [
        (0, 18, 0, 18, 7, 8),
        (2, 17, 2, 17, 8, 9),
        (4, 15, 4, 15, 9, 10),
        (6, 13, 6, 13, 10, 11),
    ]
    for x0, x1, y0, y1, z0, z1 in roof_levels:
        set_block(x0, x1, y0, y1, z0, z1, ROOF)
        inset = 2
        occ[x0 + inset:x1 - inset, y0 + inset:y1 - inset, z0:z1] = False
        facecolors[x0 + inset:x1 - inset, y0 + inset:y1 - inset, z0:z1] = 0
        edgecolors[x0 + inset:x1 - inset, y0 + inset:y1 - inset, z0:z1] = 0

    set_block(7, 9, 7, 9, 9, 10, ROOF)

    ax3d.voxels(occ, facecolors=facecolors, edgecolors=edgecolors, linewidth=0.7)
    ax3d.set_box_aspect([1, 1, 0.72])
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_facecolor(WHITE)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
    ax3d.set_axis_off()


def draw_volume_block(ax, x, y, w, h, dx, dy, face, side, top):
    """Draw a simple perspective feature-volume block."""
    front = mpatches.Rectangle((x, y), w, h, facecolor=face, edgecolor=WHITE, linewidth=1.2)
    ax.add_patch(front)
    ax.add_patch(Polygon(
        [[x + w, y], [x + w + dx, y + dy], [x + w + dx, y + h + dy], [x + w, y + h]],
        closed=True, facecolor=side, edgecolor=WHITE, linewidth=1.0,
    ))
    ax.add_patch(Polygon(
        [[x, y + h], [x + dx, y + h + dy], [x + w + dx, y + h + dy], [x + w, y + h]],
        closed=True, facecolor=top, edgecolor=WHITE, linewidth=1.0,
    ))


def draw_house_thumbnail(ax, x, y, s=1.0, removed=None, selected=False):
    """Draw a small house-like voxel thumbnail for conceptual steps."""
    w = 0.070 * s
    h = 0.070 * s
    dx = 0.020 * s
    dy = 0.018 * s

    # walls
    ax.add_patch(mpatches.Rectangle((x, y), w, h, facecolor=WALL, edgecolor=WHITE, linewidth=1.0))
    ax.add_patch(Polygon(
        [[x + w, y], [x + w + dx, y + dy], [x + w + dx, y + h + dy], [x + w, y + h]],
        closed=True, facecolor="#2E5FB1", edgecolor=WHITE, linewidth=0.9,
    ))
    # roof
    ax.add_patch(Polygon(
        [[x, y + h], [x + dx, y + h + dy], [x + w + dx, y + h + dy], [x + w, y + h]],
        closed=True, facecolor=ROOF, edgecolor=WHITE, linewidth=0.9,
    ))

    for gx in np.linspace(x + w * 0.20, x + w * 0.80, 3):
        ax.plot([gx, gx], [y + h * 0.10, y + h * 0.90], color=WHITE, lw=0.5, alpha=0.45)
    for gy in np.linspace(y + h * 0.20, y + h * 0.80, 3):
        ax.plot([x + w * 0.08, x + w * 0.92], [gy, gy], color=WHITE, lw=0.5, alpha=0.45)

    if removed:
        for rx, ry in removed:
            ax.add_patch(mpatches.Rectangle((x + rx * w, y + ry * h), w * 0.14, h * 0.14,
                                            facecolor=RED, edgecolor=WHITE, linewidth=0.5))

    if selected:
        circ = Circle((x + w + dx * 0.65, y + h * 0.28), 0.018 * s,
                      facecolor="#2CA02C", edgecolor=WHITE, linewidth=0.9)
        ax.add_patch(circ)
        ax.plot([x + w + dx * 0.56, x + w + dx * 0.64, x + w + dx * 0.80],
                [y + h * 0.27, y + h * 0.19, y + h * 0.36], color=WHITE, lw=1.2)


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


def add_arrow(fig, start, end, color=BLACK, lw=2.6, scale=26, z=15):
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


def add_outline_arrow(container, start, end, transform, scale=28, lw=2.8, z=15):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="Simple,head_length=1.2,head_width=1.2,tail_width=0.50",
        mutation_scale=scale,
        facecolor=WHITE,
        edgecolor=BLACK,
        linewidth=lw,
        transform=transform,
        clip_on=False,
        joinstyle="miter",
        capstyle="projecting",
        zorder=z,
    )
    container.add_artist(patch)


fig = plt.figure(figsize=(20, 5.8), facecolor=WHITE)

left = (0.015, 0.08, 0.215, 0.84)
mid = (0.280, 0.08, 0.440, 0.84)
right = (0.770, 0.08, 0.215, 0.84)

# keep the background clean like the reference figure

add_outline_arrow(fig, (0.238, 0.50), (0.275, 0.50), transform=fig.transFigure, scale=ARROW_SCALE, lw=ARROW_LW)
add_outline_arrow(fig, (0.725, 0.50), (0.762, 0.50), transform=fig.transFigure, scale=ARROW_SCALE, lw=ARROW_LW)


# ── Left: actual voxelized structure ─────────────────────────────────────────
ax_left = fig.add_axes([0.018, 0.165, 0.228, 0.64], projection="3d")
render_voxelized_house(
    ax_left,
    "figures/screenshot_stls/REF_SASTO_PA_colored.ply",
    pitch_div=30,
    elev=22,
    azim=-55,
)

fig.text(0.132, 0.805, "Voxelized Structure",
         ha="center", va="center", fontsize=13, fontweight="bold", color=DARK)
fig.text(0.132, 0.132, "part-labeled voxel grid of the starting design",
         ha="center", va="center", fontsize=8.8, color=DARK, fontstyle="italic")

legend_y = 0.168
legend_x = [0.080, 0.118, 0.157, 0.193]
legend_colors = [WALL, INTERIOR, ROOF, SLAB]
legend_labels = ["Exterior", "Interior", "Roof", "Floor"]
for x0, c0, txt in zip(legend_x, legend_colors, legend_labels):
    fig.add_artist(mpatches.Rectangle((x0, legend_y), 0.010, 0.016,
                                      transform=fig.transFigure, facecolor=c0,
                                      edgecolor=BLACK, linewidth=0.6, zorder=10))
    fig.text(x0 + 0.013, legend_y + 0.008, txt,
             ha="left", va="center", fontsize=7.0, color=DARK)


# ── Middle: conceptual optimization panel ────────────────────────────────────
ax_mid = fig.add_axes([mid[0], mid[1], mid[2], mid[3]], facecolor="none")
ax_mid.set_xlim(0, 1)
ax_mid.set_ylim(0, 1)
ax_mid.set_axis_off()

ax_mid.text(0.50, 0.79, "Surrogate Model",
            ha="center", va="center", fontsize=14, fontweight="bold", color=DARK)

# simple reference-style flow
enc_box = FancyBboxPatch((0.08, 0.37), 0.18, 0.20,
                         boxstyle="round,pad=0.010,rounding_size=0.018",
                         facecolor="#F2F5FA", edgecolor=BLACK, linewidth=1.8)
ax_mid.add_patch(enc_box)
ax_mid.text(0.17, 0.50, "3D House\nEncoder",
            ha="center", va="center", fontsize=11.5, color=DARK, fontweight="bold", linespacing=1.1, multialignment="center")
ax_mid.text(0.17, 0.31, "voxel geometry to structural features",
            ha="center", va="center", fontsize=7.8, color=DARK, fontstyle="italic")

add_outline_arrow(ax_mid, (0.28, 0.47), (0.39, 0.47), transform=ax_mid.transAxes, scale=ARROW_SCALE, lw=ARROW_LW)

vec_box = FancyBboxPatch((0.42, 0.22), 0.16, 0.50,
                         boxstyle="round,pad=0.008,rounding_size=0.010",
                         facecolor="#FFF6E5", edgecolor=BLACK, linewidth=1.6)
ax_mid.add_patch(vec_box)
ax_mid.text(0.50, 0.76, "Structural Code",
            ha="center", va="center", fontsize=10.5, color=DARK, fontweight="bold")
ax_mid.text(0.50, 0.63, r"$z_{\mathrm{shape}}$", ha="center", va="center", fontsize=11.4, color=DARK, fontweight="bold")
ax_mid.text(0.50, 0.53, r"$z_{\mathrm{load}}$", ha="center", va="center", fontsize=11.4, color=DARK, fontweight="bold")
ax_mid.text(0.50, 0.43, r"$\vdots$", ha="center", va="center", fontsize=16, color=DARK)
ax_mid.text(0.50, 0.33, r"$z_{\mathrm{stiff}}$", ha="center", va="center", fontsize=11.4, color=DARK, fontweight="bold")
ax_mid.text(0.50, 0.16, r"$J = V + \lambda P$",
            ha="center", va="center", fontsize=11.5, color=NAVY, fontweight="bold")

add_outline_arrow(ax_mid, (0.60, 0.47), (0.71, 0.47), transform=ax_mid.transAxes, scale=ARROW_SCALE, lw=ARROW_LW)

dec_box = FancyBboxPatch((0.74, 0.37), 0.18, 0.20,
                         boxstyle="round,pad=0.010,rounding_size=0.018",
                         facecolor="#FDEEEF", edgecolor=BLACK, linewidth=1.8)
ax_mid.add_patch(dec_box)
ax_mid.text(0.83, 0.50, "Structural\nSurrogate",
            ha="center", va="center", fontsize=11.5, color=DARK, fontweight="bold", linespacing=1.1, multialignment="center")
ax_mid.text(0.83, 0.31, "predicts stress, compliance, and displacement",
            ha="center", va="center", fontsize=7.4, color=DARK, fontstyle="italic")


# ── Right: full-house structural response ───────────────────────────────────
ax_right = fig.add_axes([0.787, 0.18, 0.178, 0.64], projection="3d")
render_mesh(ax_right, "figures/screenshot_stls/REF_SASTO_PA_colored.ply",
            color_mode="stress", elev=22, azim=-55, ambient=0.84)

fig.text(0.8775, 0.805, "Predicted Structural Response",
         ha="center", va="center", fontsize=13, fontweight="bold", color=DARK)
fig.text(0.8775, 0.082, "full-house stress prediction",
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
ax_cb.set_title("Stress", fontsize=8.5, color=DARK, pad=2)


plt.savefig(str(OUT), dpi=240, bbox_inches="tight", facecolor=WHITE, edgecolor="none")
plt.close()
print(f"Saved → {OUT}")
