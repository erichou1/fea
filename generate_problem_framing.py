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
ARROW_LW  = 3.2
ARROW_SCALE = 30
ARROW_DX  = 0.045   # arrow length in figure coords
TITLE_Y   = 0.870
SUB_Y     = 0.118   # subtitle row y
LEG_Y     = 0.183   # legend / colorbar row y

# Symmetric layout grid
PANEL_W  = 0.220
PANEL_H  = 0.590
PANEL_Y  = 0.205
GAP      = 0.048    # gap on each side of mid panel
LEFT_X   = 0.010
MID_X    = LEFT_X + PANEL_W + GAP   # 0.285
MID_W    = 1.0 - 2*LEFT_X - 2*PANEL_W - 2*GAP  # 0.430
RIGHT_X  = MID_X + MID_W + GAP     # 0.770
LEFT_CX  = LEFT_X  + PANEL_W / 2   # 0.120
RIGHT_CX = RIGHT_X + PANEL_W / 2   # 0.880

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

    # Make the conceptual voxel thumbnail read as a solid house mass with no open shaft.
    solid = filled.copy()
    for ix in range(solid.shape[0]):
        for iy in range(solid.shape[1]):
            z_idx = np.flatnonzero(filled[ix, iy, :])
            if z_idx.size:
                solid[ix, iy, z_idx.min():z_idx.max() + 1] = True
    filled = solid

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

    visible = exposed

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

# Right panel size (hoisted here so arrow 4 can use _R_X)
_R_W = PANEL_W + 0.010
_R_H = PANEL_H + 0.070
_R_Y = PANEL_Y - 0.035
_R_X = 1.0 - LEFT_X - _R_W

# Single shared arrow_y: vertical midpoint of both 3D panels
arrow_y = PANEL_Y + PANEL_H * 0.50

# ── All 4 identical arrows in figure coordinates ──────────────────────────────
# Box edges in ax_mid fractions (must match FancyBboxPatch positions below)
_ENC_R_AX = 0.26   # enc box right edge in ax_mid  (pushed in to widen gaps)
_VEC_L_AX = 0.41   # vec box left  edge in ax_mid
_VEC_R_AX = 0.59   # vec box right edge in ax_mid
_DEC_L_AX = 0.74   # dec box left  edge in ax_mid  (pushed in to widen gaps)

_enc_r_fig = MID_X + MID_W * _ENC_R_AX
_vec_l_fig = MID_X + MID_W * _VEC_L_AX
_vec_r_fig = MID_X + MID_W * _VEC_R_AX
_dec_l_fig = MID_X + MID_W * _DEC_L_AX

_a1_cx = (LEFT_X + PANEL_W + MID_X)   / 2   # outer-left  gap centre
_a2_cx = (_enc_r_fig + _vec_l_fig)     / 2   # inner gap 1 centre
_a3_cx = (_vec_r_fig + _dec_l_fig)     / 2   # inner gap 2 centre
_a4_cx = (MID_X + MID_W + _R_X)       / 2   # outer-right gap centre (uses actual _R_X)

for _cx in [_a1_cx, _a2_cx, _a3_cx, _a4_cx]:
    _sx = _cx - ARROW_DX / 2
    add_outline_arrow(fig, (_sx, arrow_y), (_sx + ARROW_DX, arrow_y),
                      transform=fig.transFigure, scale=ARROW_SCALE, lw=ARROW_LW)


# ── Left: voxelized structure ─────────────────────────────────────────────────
ax_left = fig.add_axes([LEFT_X, PANEL_Y, PANEL_W, PANEL_H], projection="3d")
render_voxelized_house(
    ax_left,
    "figures/screenshot_stls/REF_SASTO_PA_colored.ply",
    pitch_div=30,
    elev=22,
    azim=-55,
)

fig.text(LEFT_CX, TITLE_Y, "Voxelized Structure",
         ha="center", va="center", fontsize=13, fontweight="bold", color=DARK)
fig.text(LEFT_CX, SUB_Y, "Part-labeled voxel grid of starting design",
         ha="center", va="center", fontsize=8.5, color=DARK, fontstyle="italic")

# Legend centred under left panel
_leg_labels = ["Exterior", "Interior", "Roof", "Floor"]
_leg_colors  = [WALL,       INTERIOR,   ROOF,   SLAB]
_SW = 0.011; _SG = 0.004; _TW = 0.040
_iw = _SW + _SG + _TW
_tot = len(_leg_labels) * _iw + (len(_leg_labels) - 1) * 0.005
_lx0 = LEFT_CX - _tot / 2
for i, (c0, txt) in enumerate(zip(_leg_colors, _leg_labels)):
    xi = _lx0 + i * (_iw + 0.005)
    fig.add_artist(mpatches.Rectangle((xi, LEG_Y), _SW, 0.016,
                                      transform=fig.transFigure, facecolor=c0,
                                      edgecolor=BLACK, linewidth=0.6, zorder=10))
    fig.text(xi + _SW + _SG, LEG_Y + 0.008, txt,
             ha="left", va="center", fontsize=7.2, color=DARK)


# ── Middle: surrogate model panel ─────────────────────────────────────────────
# ax_mid spans full figure height so box coords can reference arrow_y cleanly
MID_Y_AX, MID_H_AX = 0.05, 0.90
ax_mid = fig.add_axes([MID_X, MID_Y_AX, MID_W, MID_H_AX], facecolor="none")
ax_mid.set_xlim(0, 1)
ax_mid.set_ylim(0, 1)
ax_mid.set_axis_off()

fig.text(MID_X + MID_W / 2, TITLE_Y, "Surrogate Model",
         ha="center", va="center", fontsize=14, fontweight="bold", color=DARK)

# Convert shared arrow_y to ax_mid fraction
_ay = (arrow_y - MID_Y_AX) / MID_H_AX
_bh = 0.20   # box height in ax_mid units
_bw = _ENC_R_AX - 0.02   # box width = from left anchor 0.02 to _ENC_R_AX = 0.24

# Encoder box: anchored at left=0.02, right=_ENC_R_AX=0.26
_ENC_CX = (0.02 + _ENC_R_AX) / 2   # = 0.14
enc_box = FancyBboxPatch((0.02, _ay - _bh/2), _bw, _bh,
                         boxstyle="round,pad=0.010,rounding_size=0.018",
                         facecolor="#F2F5FA", edgecolor=BLACK, linewidth=1.8)
ax_mid.add_patch(enc_box)
ax_mid.text(_ENC_CX, _ay, "3D House\nEncoder",
            ha="center", va="center", fontsize=10.5, color=DARK,
            fontweight="bold", linespacing=1.15, multialignment="center")
ax_mid.text(_ENC_CX, _ay - _bh/2 - 0.055,
            "voxel geometry →\nstructural features",
            ha="center", va="center", fontsize=7.5, color=DARK,
            fontstyle="italic", linespacing=1.3)

# Latent vector box: centred at 0.50, edges match _VEC_L_AX/_VEC_R_AX
_VEC_CX = 0.50
_vw = _VEC_R_AX - _VEC_L_AX   # = 0.18
_vh = 0.50
# White background (no visible border)
vec_bg = FancyBboxPatch((_VEC_L_AX + 0.028, _ay - _vh/2), _vw - 0.056, _vh,
                         boxstyle="square,pad=0.0",
                         facecolor=WHITE, edgecolor="none", linewidth=0)
ax_mid.add_patch(vec_bg)
# Square brackets [ ] drawn as L-shaped line segments
_blw   = 3.0          # bracket line width
_btick = 0.030        # horizontal tick length
_bxi_l = _VEC_L_AX + 0.008   # vertical stem x, left bracket
_bxi_r = _VEC_R_AX - 0.008   # vertical stem x, right bracket
_by_top = _ay + _vh / 2 - 0.006
_by_bot = _ay - _vh / 2 + 0.006
# Left bracket  [ : tick-right → down → tick-right
ax_mid.plot([_bxi_l + _btick, _bxi_l, _bxi_l, _bxi_l + _btick],
            [_by_top, _by_top, _by_bot, _by_bot],
            color=DARK, lw=_blw, solid_capstyle='butt', solid_joinstyle='miter')
# Right bracket ] : tick-left ← down → tick-left
ax_mid.plot([_bxi_r - _btick, _bxi_r, _bxi_r, _bxi_r - _btick],
            [_by_top, _by_top, _by_bot, _by_bot],
            color=DARK, lw=_blw, solid_capstyle='butt', solid_joinstyle='miter')
# Vertical stack of z labels, centred on _ay, evenly spaced
_z_labels = [r"$z_{\mathrm{shape}}$", r"$z_{\mathrm{load}}$",
             r"$\vdots$",            r"$z_{\mathrm{stiff}}$"]
_z_fs     = [11.0, 11.0, 18.0, 11.0]
_n_items  = len(_z_labels)
_z_step   = 0.135
_z_top    = _ay + (_n_items - 1) / 2 * _z_step
for k, (lbl, fs) in enumerate(zip(_z_labels, _z_fs), 0):
    ax_mid.text(_VEC_CX, _z_top - k * _z_step, lbl,
                ha="center", va="center", fontsize=fs, color=DARK,
                fontweight="bold" if r"\vdots" not in lbl else "normal")

# Surrogate/Decoder box: anchored at left=_DEC_L_AX=0.74, right=0.98
_DEC_CX = (_DEC_L_AX + 0.98) / 2   # = 0.86
dec_box = FancyBboxPatch((_DEC_L_AX, _ay - _bh/2), _bw, _bh,
                         boxstyle="round,pad=0.010,rounding_size=0.018",
                         facecolor="#FDEEEF", edgecolor=BLACK, linewidth=1.8)
ax_mid.add_patch(dec_box)
ax_mid.text(_DEC_CX, _ay, "Structural\nSurrogate",
            ha="center", va="center", fontsize=10.5, color=DARK,
            fontweight="bold", linespacing=1.15, multialignment="center")
ax_mid.text(_DEC_CX, _ay - _bh/2 - 0.055,
            "predicts stress,\ncompliance & displacement",
            ha="center", va="center", fontsize=7.5, color=DARK,
            fontstyle="italic", linespacing=1.3)

# Objective below the whole mid panel
ax_mid.text(0.50, _ay - _vh/2 - 0.13, r"$J = V + \lambda P$",
            ha="center", va="center", fontsize=11.5, color=NAVY, fontweight="bold")


# ── Right: predicted structural response ──────────────────────────────────────
# Right panel size already defined above (hoisted for arrow centering)
ax_right = fig.add_axes([_R_X, _R_Y, _R_W, _R_H], projection="3d")
render_mesh(ax_right, "figures/screenshot_stls/REF_SASTO_PA_colored.ply",
            color_mode="stress", elev=22, azim=-55, ambient=0.84)

fig.text(RIGHT_CX, TITLE_Y, "Predicted Structural Response",
         ha="center", va="center", fontsize=13, fontweight="bold", color=DARK)
fig.text(RIGHT_CX, SUB_Y, "Full-house stress field prediction",
         ha="center", va="center", fontsize=8.5, color=DARK, fontstyle="italic")

# Stress colorbar centred under right panel, same row as left legend
_cb_w = 0.130
ax_cb = fig.add_axes([RIGHT_CX - _cb_w/2, LEG_Y - 0.002, _cb_w, 0.022])
cb = fig.colorbar(
    plt.cm.ScalarMappable(mcolors.Normalize(0, 1), cmap=plt.get_cmap("jet")),
    cax=ax_cb, orientation="horizontal",
)
cb.set_ticks([0, 1])
cb.set_ticklabels(["Low", "High"])
cb.ax.tick_params(labelsize=8, size=0, colors=DARK)
cb.outline.set_edgecolor(DARK)
cb.outline.set_linewidth(0.7)
ax_cb.set_title("Stress", fontsize=8.0, color=DARK, pad=2)


plt.savefig(str(OUT), dpi=240, bbox_inches="tight", facecolor=WHITE, edgecolor="none")
plt.close()
print(f"Saved → {OUT}")
