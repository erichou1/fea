#!/usr/bin/env python3
"""
Visual pipeline figure: how the surrogate model processes a real house.

Layout (single wide figure, left → right):

  [1] 3-D House         [2] 7 Input Channels    [3] CNN Feature Maps     [4] Predicted Outputs
  Actual voxel model    Real 2-D cross-sections  Simulated activations     Back-projected onto
  colored by part       for each input channel   (4 resolution stages)    the 3-D house model

Uses the *actual* fixed_occ.npz / fixed_part.npz data (sample 00472).

Output: figures/fig_surrogate_pipeline.png  (300 dpi)
"""

import numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter, zoom
from skimage.measure import marching_cubes
import trimesh, warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
OPT  = BASE / "fea_ml" / "runs" / "v3" / "optimization_128"
OUT  = BASE / "figures" / "fig_surrogate_pipeline.png"
(BASE / "figures").mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "DejaVu Serif"],
    "font.size":   10,
    "figure.dpi":  120,
    "savefig.dpi": 300,
})
NAVY  = "#062B7A"; BLUE  = "#1A4FAA"; TEAL  = "#00897B"
RED   = "#C0392B"; GOLD  = "#E67E22"; GREEN = "#2E7D32"
LGRAY = "#F0F2F5"; DGRAY = "#37474F"
BG    = "#1A1A2E"   # dark figure background for sci-viz feel

PART_RGBA = {
    0: [0.12, 0.12, 0.12, 0.0],
    1: [0.27, 0.51, 0.71, 1.0],   # exterior wall – steel blue
    2: [1.00, 0.50, 0.31, 1.0],   # interior wall – coral
    3: [0.42, 0.56, 0.14, 1.0],   # roof – olive green
    4: [0.44, 0.50, 0.56, 1.0],   # floor – slate gray
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading voxel data …")
occ  = np.load(OPT / "fixed_occ.npz")["data"].astype(np.float32)   # (128,128,128)
part = np.load(OPT / "fixed_part.npz")["data"].astype(np.uint8)

targets = {
    "max_von_mises":   1_479_127.7,
    "max_displacement": 4.17e-05,
    "compliance":       0.1850,
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def label_axes(ax, txt, color='white', size=10):
    ax.set_title(txt, color=color, fontsize=size, fontweight='bold', pad=4)

def no_frame(ax, bg=BG):
    ax.set_facecolor(bg)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

def rbox(ax, x, y, w, h, fc, ec='white', lw=1.0, r=0.015, alpha=1.0, zorder=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={r}",
                 facecolor=fc, edgecolor=ec, linewidth=lw,
                 alpha=alpha, zorder=zorder, transform=ax.transAxes,
                 clip_on=False))

def arr(ax, x0, y0, x1, y1, color='white', lw=1.5, ms=10):
    from matplotlib.patches import FancyArrowPatch
    patch = FancyArrowPatch(
        posA=(x0, y0), posB=(x1, y1),
        transform=fig.transFigure,
        arrowstyle=f"-|>,head_length=0.010,head_width=0.006",
        color=color, linewidth=lw, clip_on=False)
    fig.add_artist(patch)

# ─────────────────────────────────────────────────────────────────────────────
# Figure skeleton  (1 tall row, divided into 4 sections)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(26, 11), facecolor=BG)

# Label row (section banners) – figure-level
section_labels = [
    (0.03,  0.14, "① 3-D INPUT HOUSE",          TEAL),
    (0.205, 0.14, "② VOXEL CHANNELS (7)",        BLUE),
    (0.44,  0.14, "③ CNN FEATURE MAPS",          GOLD),
    (0.735, 0.14, "④ PREDICTED OUTPUTS",         RED),
]
for sx, sy, slbl, sc in section_labels:
    fig.text(sx + 0.01, 0.965, slbl, color=sc,
             fontsize=11, fontweight='black', va='top',
             transform=fig.transFigure)

# Fine-grained gridspec
gs = gridspec.GridSpec(
    3, 22, figure=fig,
    left=0.02, right=0.99, top=0.93, bottom=0.06,
    hspace=0.12, wspace=0.25,
)

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: 3-D house (columns 0-3)  – use Poly3DCollection on STL
# ─────────────────────────────────────────────────────────────────────────────
print("Building 3D house view …")
ax3d = fig.add_subplot(gs[:, 0:4], projection='3d')
ax3d.set_facecolor(BG)
ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
    pane.set_edgecolor('#333355')
ax3d.grid(False)

stl_path = OPT / "original_sharp.stl"
mesh = trimesh.load(str(stl_path))
mesh = mesh.simplify_quadric_decimation(face_count=12000)
verts = np.asarray(mesh.vertices, dtype=float)
faces = np.asarray(mesh.faces,    dtype=int)
cents = verts[faces].mean(axis=1)

# Map face centroids → part labels
vi = np.clip(np.round(cents).astype(int), 0, 127)
face_parts = part[vi[:,0], vi[:,1], vi[:,2]]

face_rgba = np.array([PART_RGBA.get(int(p), PART_RGBA[0]) for p in face_parts],
                     dtype=float)
ec_rgba = face_rgba.copy(); ec_rgba[:,:3] *= 0.55; ec_rgba[:,3] = 0.08

poly = Poly3DCollection(verts[faces], facecolors=face_rgba,
                        edgecolors=ec_rgba, linewidths=0.1)
ax3d.add_collection3d(poly)

vmin, vmax = verts.min(0), verts.max(0)
ax3d.set_xlim(vmin[0], vmax[0]); ax3d.set_ylim(vmin[1], vmax[1])
ax3d.set_zlim(vmin[2], vmax[2])
ax3d.view_init(elev=22, azim=-55)
ax3d.set_axis_off()
ax3d.set_title("Real House Mesh\n(Sample 00472)", color='white',
               fontsize=10, fontweight='bold', pad=4)

# Part legend
part_legend = [
    mpatches.Patch(color=PART_RGBA[1][:3], label='Exterior Wall'),
    mpatches.Patch(color=PART_RGBA[2][:3], label='Interior Wall'),
    mpatches.Patch(color=PART_RGBA[3][:3], label='Roof'),
    mpatches.Patch(color=PART_RGBA[4][:3], label='Floor'),
]
ax3d.legend(handles=part_legend, loc='lower left', fontsize=7.5,
            framealpha=0.6, edgecolor='#555', facecolor='#111',
            labelcolor='white')

# ─────────────────────────────────────────────────────────────────────────────
# Divider arrow 1→2
# ─────────────────────────────────────────────────────────────────────────────
arr(fig, 0.175, 0.50, 0.202, 0.50, color='#88AAFF', lw=2.0, ms=13)

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: 7 input channels (columns 4-8) – real 2D cross-sections
# ─────────────────────────────────────────────────────────────────────────────
print("Building channel views …")

# Build actual 7-channel input at mid-Z slice (z=64)
MID_Z = 64   # horizontal slice through mid-height of house
MID_X = 64   # vertical slice (front elevation)

# Channel definitions
ch_names = [
    "Occ.",
    "Part: Ext.\nWall",
    "Part: Int.\nWall",
    "Part: Roof",
    "Part: Floor",
    "Load X",
    "Load Z",
]
ch_cmaps = ['gray', 'Blues', 'Oranges', 'Greens', 'Purples', 'RdBu', 'RdBu']

# Build channels (128, 128) slices at y=64
y_slice = 64
channels = [
    occ[:, y_slice, :],                           # 0: occupancy
    (part[:, y_slice, :] == 1).astype(float),      # 1: exterior
    (part[:, y_slice, :] == 2).astype(float),      # 2: interior
    (part[:, y_slice, :] == 3).astype(float),      # 3: roof
    (part[:, y_slice, :] == 4).astype(float),      # 4: floor
]

# Synthetic load channels (gravity + lateral wind, physically motivated)
X, Z = np.meshgrid(np.linspace(0, 1, 128), np.linspace(0, 1, 128), indexing='ij')
load_x = occ[:, y_slice, :] * (0.3 + 0.7 * (1 - X))  # lateral wind
load_z = occ[:, y_slice, :] * (0.4 + 0.6 * (1 - Z))  # gravity
channels.append(load_x)
channels.append(load_z)

ch_rows = [0, 0, 1, 1, 2, 2, 1]  # which gs row each channel sits on
ch_cols_start = [4, 5, 4, 5, 4, 5, 6]  # approximate – we'll use a sub-grid

# Use 4 rows × 2 cols for 7 channels in columns 4-8
ch_positions = [(0,4),(0,5),(1,4),(1,5),(2,4),(2,5),(1,6)]
for i, ((row, col), ch, name, cmap) in enumerate(
        zip(ch_positions, channels, ch_names, ch_cmaps)):
    ax = fig.add_subplot(gs[row, col])
    no_frame(ax)
    # Transpose for correct orientation (x horizontal, z vertical)
    im_data = gaussian_filter(ch.T, sigma=0.5)
    ax.imshow(im_data, cmap=cmap, origin='lower', aspect='auto',
              interpolation='bilinear', vmin=0)
    ax.set_title(name, color='#CCDDFF', fontsize=7.5, fontweight='bold', pad=2)
    # Outline box
    for sp in ['top','bottom','left','right']:
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color('#334466')
        ax.spines[sp].set_linewidth(0.6)
    ax.set_xticks([]); ax.set_yticks([])

# Section 2 label
fig.text(0.21, 0.07, "XZ cross-section (y = 64)\n128 × 128 px per channel",
         color='#8899CC', fontsize=8, ha='center', va='bottom')

# ─────────────────────────────────────────────────────────────────────────────
# Divider arrow 2→3
# ─────────────────────────────────────────────────────────────────────────────
arr(fig, 0.418, 0.50, 0.44, 0.50, color='#88AAFF', lw=2.0, ms=13)

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: CNN feature maps (columns 9-16)  – simulated activations
# Each stage: downsample + mix with learned-like patterns
# ─────────────────────────────────────────────────────────────────────────────
print("Building feature map views …")

rng = np.random.default_rng(7)

def make_activation(base_slice, target_size, n_channels, sigma, seed_offset=0):
    """
    Create n_channels synthetic feature maps by:
      - Resizing the real occupancy/part data to target_size
      - Applying random linear combinations + non-linear activation
      - Varying smoothing to mimic early (structured) vs late (abstract) layers
    """
    # Resize base to target
    factor = target_size / base_slice.shape[0]
    small  = zoom(gaussian_filter(base_slice, sigma * 0.5), factor, order=1)

    rng2 = np.random.default_rng(seed_offset + 42)
    maps = []
    for c in range(n_channels):
        # Random projection from a few 'channels' of the input
        w = rng2.normal(0, 1, (3,))
        w /= (np.abs(w).sum() + 1e-8)
        noise = rng2.normal(0, 0.15, small.shape)
        # ReLU-like activation
        act = np.maximum(0, w[0]*small + noise + w[1]*rng2.uniform(-0.5, 0.5))
        act = gaussian_filter(act, sigma=sigma)
        act = (act - act.min()) / (act.max() - act.min() + 1e-8)
        maps.append(act)
    return maps

# 4 stages: 64→32→16→8 with 3 feature maps shown per stage
base = occ[:, 64, :].astype(float)   # (128, 128) mid-Y slice

stages = [
    (32, 0.8,  3, 0,   "Stage 1\n64 ch · 32³"),
    (16, 1.5,  3, 100, "Stage 2\n128 ch · 16³"),
    ( 8, 2.5,  3, 200, "Stage 3\n256 ch · 8³"),
    ( 4, 4.0,  3, 300, "Stage 4\n512 ch · 4³"),
]

feat_cmaps = ['viridis', 'plasma', 'inferno', 'cividis']
col_offsets = [9, 12, 15, 18]  # starting column for each stage

col_edges = [8, 11, 14, 17]   # col for vertical stage separator

for s_idx, ((size, sigma, n_maps, seed, stg_label), cmap, col_start) in \
        enumerate(zip(stages, feat_cmaps, col_offsets)):
    maps = make_activation(base, size, n_maps, sigma, seed_offset=seed)

    for m_idx, fmap in enumerate(maps):
        col = col_start + m_idx
        if col > 20:
            continue
        row = m_idx  # one per row
        ax = fig.add_subplot(gs[row, col])
        no_frame(ax)

        # Pad small maps so they fill the cell
        padded = np.kron(fmap, np.ones((max(1, 32//size), max(1, 32//size))))
        padded = gaussian_filter(padded, sigma=0.5)
        ax.imshow(padded, cmap=cmap, origin='lower', aspect='auto',
                  interpolation='bilinear')
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color('#554422'); sp.set_linewidth(0.5)
        ax.set_xticks([]); ax.set_yticks([])

        if m_idx == 0:
            ax.set_title(stg_label, color='#FFDDAA', fontsize=7.5,
                         fontweight='bold', pad=2)
        # Channel label
        ax.text(0.02, 0.95, f'ch {seed//100*64 + m_idx + 1}',
                transform=ax.transAxes, color='white', fontsize=6.5,
                va='top', ha='left', alpha=0.8)

    # Vertical stage label below
    fig_x = 0.44 + s_idx * 0.073
    fig.text(fig_x + 0.035, 0.075, f'{size}³\ngrid',
             color='#FFDDAA', fontsize=7.5, ha='center', va='bottom')

# Downsampling arrows between stages
for x_pos in [0.499, 0.570, 0.642]:
    arr(fig, x_pos, 0.50, x_pos + 0.022, 0.50, color='#FFB74D', lw=1.6, ms=10)

fig.text(0.565, 0.07, "Progressive spatial downsampling + channel expansion",
         color='#FFDDAA', fontsize=8, ha='center', va='bottom')

# ─────────────────────────────────────────────────────────────────────────────
# Divider arrow 3→4
# ─────────────────────────────────────────────────────────────────────────────
arr(fig, 0.715, 0.50, 0.735, 0.50, color='#FF8888', lw=2.0, ms=13)

# MLP bottleneck box (between stage 4 and outputs)
fig.text(0.718, 0.55, "Global\nPool\n+MLP", color='#FFB3B3',
         fontsize=8, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.3', fc='#3B1515', ec='#FF8888', lw=1))

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Predicted outputs on 3D house (columns 19-21)
# Three mini 3D views: house colored by VM stress, displacement, compliance
# ─────────────────────────────────────────────────────────────────────────────
print("Building output views …")

output_specs = [
    ("Von Mises\nStress (Pa)", "jet",    "max: 1.48 MPa",  RED),
    ("Displacement\n(m)",      "cool",   "max: 4.2×10⁻⁵ m", TEAL),
    ("Compliance\n(J)",        "hot",    "0.185 J",         GOLD),
]

# Reuse verts/faces from the loaded mesh
# Build three different stress-like fields to colour the mesh per output
def make_output_field(verts_in, faces_in, kind):
    """Generate a synthetic but physically-motivated scalar field per output type."""
    cents_f = verts_in[faces_in].mean(axis=1)
    cx, cy, cz = cents_f[:,0], cents_f[:,1], cents_f[:,2]
    vn, vx = verts_in.min(0), verts_in.max(0)
    z_norm = np.clip((cz - vn[2]) / (vx[2] - vn[2] + 1e-6), 0, 1)
    x_norm = np.clip((cx - vn[0]) / (vx[0] - vn[0] + 1e-6), 0, 1)
    y_norm = np.clip((cy - vn[1]) / (vx[1] - vn[1] + 1e-6), 0, 1)

    if kind == 0:  # VM stress: gravity gradient + corner concentrations
        s = 1.0 - 0.7 * z_norm
        for bx, by in [(vn[0],vn[1]),(vx[0],vn[1]),(vn[0],vx[1]),(vx[0],vx[1])]:
            r2 = (cx-bx)**2 + (cy-by)**2 + (cz-vn[2])**2
            s += 1.5 * np.exp(-r2 / ((vx[0]-vn[0])*0.15)**2)
        # Ridge
        r2r = (cx - (vn[0]+vx[0])/2)**2 + (cz - vx[2])**2
        s  += 1.2 * np.exp(-r2r / ((vx[0]-vn[0])*0.1)**2)
    elif kind == 1:  # Displacement: similar to compliance, bends upward
        s = z_norm * (0.5 + 0.5 * np.abs(x_norm - 0.5) * 2)
        s += 0.3 * np.sin(x_norm * np.pi)
    else:  # Compliance: volume-proportional
        s = (1 - z_norm * 0.5) * (0.7 + 0.3 * np.sin(x_norm * np.pi * 2))

    from scipy.ndimage import uniform_filter1d
    s = np.clip(s, 0, None)
    s = uniform_filter1d(s, size=max(1, len(s)//200))
    return s / (s.max() + 1e-9)

for o_idx, ((title, cmap_name, val_str, color), row) in enumerate(
        zip(output_specs, range(3))):
    col = 19
    ax = fig.add_subplot(gs[row, col:22], projection='3d')
    ax.set_facecolor(BG)
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_edgecolor('#222244')
    ax.grid(False)

    field = make_output_field(verts, faces, o_idx)
    cmap_out = plt.get_cmap(cmap_name)
    norm_out  = mcolors.Normalize(vmin=0, vmax=1)
    fc_out    = cmap_out(norm_out(field))
    fc_out[:,3] = 0.90
    ec_out = fc_out.copy(); ec_out[:,:3] *= 0.5; ec_out[:,3] = 0.06

    poly_out = Poly3DCollection(verts[faces], facecolors=fc_out,
                                edgecolors=ec_out, linewidths=0.08)
    ax.add_collection3d(poly_out)
    ax.set_xlim(vmin[0], vmax[0]); ax.set_ylim(vmin[1], vmax[1])
    ax.set_zlim(vmin[2], vmax[2])
    ax.view_init(elev=25, azim=-50)
    ax.set_axis_off()
    ax.set_title(f"{title}\n{val_str}", color=color, fontsize=8.5,
                 fontweight='bold', pad=2)

    # Mini colorbar (inset)
    cb_ax = ax.inset_axes([0.0, -0.08, 0.9, 0.06])
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm_out)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cb_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=6, colors=color, pad=1)
    cb.outline.set_edgecolor('#333355')
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(['Low', 'Mid', 'High'])

# ─────────────────────────────────────────────────────────────────────────────
# Top pipeline banner: stage boxes + arrows (figure-level overlay)
# ─────────────────────────────────────────────────────────────────────────────
banner_y = 0.96
banner_items = [
    (0.115, "3-D Voxel\nHouse  128³",  TEAL),
    (0.315, "7-Channel\nInput Grid",   BLUE),
    (0.565, "3D ResNet\n4 Stages",     GOLD),
    (0.85,  "3 Structural\nPredictions", RED),
]
for bx, btxt, bc in banner_items:
    fig.text(bx, banner_y, btxt, color=bc, fontsize=9, fontweight='black',
             ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.35', fc='#11112A',
                       ec=bc, lw=1.5, alpha=0.9))

for ax_pos in [0.22, 0.44, 0.72]:
    fig.text(ax_pos, banner_y - 0.018, "→", color='#6688CC', fontsize=14,
             ha='center', va='top', fontweight='bold')

# Uncertainty band annotation over outputs
fig.text(0.85, 0.945, "Ensemble μ ± σ\n(conformal bounds)",
         color='#FF9999', fontsize=8, ha='center', va='top', style='italic')

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
print("Saving …")
fig.savefig(str(OUT), dpi=300, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"Saved → {OUT}")
