"""
Generate Problem Framing diagram following the reference Encoder-Matrix-Decoder style.

5-element horizontal flow:
  [Voxel house image]  →  [3D CNN Encoder]  →  [Feature matrix]  →  [Ensemble Predictor]  →  [Stress render]

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

# ── Palette ─────────────────────────────────────────────────────────────────
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

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: render PLY with bright z-height colormap (Lambertian shading)
# ═══════════════════════════════════════════════════════════════════════════════
def render_ply_colormap(ax3d, ply_path, cmap_name, elev=22, azim=-50,
                        ambient=0.55, bg_color=LBLUE):
    mesh    = trimesh.load(str(ply_path), force="mesh", process=False)
    verts   = np.array(mesh.vertices, dtype=float)
    faces   = np.array(mesh.faces)
    lo, hi  = verts.min(0), verts.max(0)
    real_sp = hi - lo
    span    = real_sp.max()
    verts   = (verts - (lo + hi) / 2) / span
    poly_v  = verts[faces]

    z_mid   = poly_v[:, :, 2].mean(axis=1)
    z_norm  = (z_mid - z_mid.min()) / (z_mid.max() - z_mid.min() + 1e-9)
    base_fc = plt.get_cmap(cmap_name)(z_norm)[:, :3]

    n0, n1, n2 = poly_v[:, 0], poly_v[:, 1], poly_v[:, 2]
    nrm = np.cross(n1 - n0, n2 - n0)
    mag = np.linalg.norm(nrm, axis=1, keepdims=True)
    mag[mag == 0] = 1.0
    nrm    = nrm / mag
    light  = np.array([np.cos(np.radians(35)) * np.cos(np.radians(-40)),
                       np.cos(np.radians(35)) * np.sin(np.radians(-40)),
                       np.sin(np.radians(35))])
    intens = np.clip(nrm @ light, 0, 1)
    shaded = np.clip(base_fc * (ambient + (1.0 - ambient) * intens[:, None]), 0, 1)

    col = Poly3DCollection(poly_v, zsort="average")
    col.set_facecolor(shaded)
    col.set_edgecolor("none")
    ax3d.add_collection3d(col)

    pad = 0.08
    ax3d.set_xlim(-0.5 - pad, 0.5 + pad)
    ax3d.set_ylim(-0.5 - pad, 0.5 + pad)
    ax3d.set_zlim(-0.5 - pad, 0.5 + pad)
    ax3d.set_box_aspect([real_sp[0], real_sp[1], real_sp[2]])
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_facecolor(bg_color)
    ax3d.set_axis_off()


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE  —  5-element horizontal flow
#  Columns (figure-normalised x):
#    [0.01 – 0.21]  image A (voxel house)
#    [0.24 – 0.42]  box 1 (Encoder)
#    [0.45 – 0.57]  matrix (feature outputs)
#    [0.60 – 0.78]  box 2 (Predictor)
#    [0.81 – 0.99]  image B (stress render)
# ═══════════════════════════════════════════════════════════════════════════════
FIG_W, FIG_H = 20.0, 5.8
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=WHITE)

CY0, CH = 0.04, 0.92  # card vertical extents (fig-norm)

# ── Image card backgrounds ────────────────────────────────────────────────────
def add_card(x0, x1, fc, ec, lw=1.8):
    fig.add_artist(mpatches.FancyBboxPatch(
        (x0, CY0), x1 - x0, CH,
        boxstyle="round,pad=0.012",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        transform=fig.transFigure, clip_on=False, zorder=0))

add_card(0.010, 0.215, LBLUE,  BLUE)   # left image card
add_card(0.795, 0.990, LBLUE,  BLUE)   # right image card

# ── Arrows between all 5 elements ────────────────────────────────────────────
ARROW_Y = CY0 + CH / 2
arrow_gaps = [
    (0.215, 0.240),   # image A  → Encoder
    (0.430, 0.450),   # Encoder  → Matrix
    (0.585, 0.605),   # Matrix   → Predictor
    (0.785, 0.795),   # Predictor → image B
]
for xa, xb in arrow_gaps:
    fig.add_artist(FancyArrowPatch(
        (xa + 0.003, ARROW_Y), (xb - 0.003, ARROW_Y),
        arrowstyle="-|>", mutation_scale=26,
        color=BLUE, lw=2.5,
        transform=fig.transFigure, clip_on=False, zorder=10))

# ──────────────────────────────────────────────────────────────────────────────
#  IMAGE A — Voxelized Structure
# ──────────────────────────────────────────────────────────────────────────────
vox = np.array(Image.open("figures/fig_voxel_house.png").convert("RGB"))
h_v, w_v = vox.shape[:2]
# centre-crop to 4:3 aspect
crop_w = min(w_v, int(h_v * 1.32))
xc     = (w_v - crop_w) // 2
vox    = vox[:, xc: xc + crop_w]

ax_a = fig.add_axes([0.018, 0.16, 0.190, 0.700])
ax_a.imshow(vox, aspect="auto", interpolation="lanczos")
ax_a.set_axis_off()

fig.text(0.113, 0.924, "Voxelized Structure",
         ha="center", fontsize=12.5, fontweight="bold", color=BLUE,
         transform=fig.transFigure)
fig.text(0.113, 0.063, r"$\rho(\mathbf{x}) \in \{0,1\}$",
         ha="center", fontsize=11, color=DARK, fontweight="bold",
         transform=fig.transFigure)
fig.text(0.113, 0.036, "Binary material field",
         ha="center", fontsize=8, color=DARK, fontstyle="italic",
         transform=fig.transFigure)

# ──────────────────────────────────────────────────────────────────────────────
#  BOX 1 — 3D CNN Encoder  (sand card, gold border — like reference Encoder)
# ──────────────────────────────────────────────────────────────────────────────
ax_enc = fig.add_axes([0.242, 0.06, 0.186, 0.88], facecolor="none")
ax_enc.set_xlim(0, 1); ax_enc.set_ylim(0, 1); ax_enc.set_axis_off()

enc_bg = FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
    boxstyle="round,pad=0.02", facecolor=SAND,
    edgecolor=GOLD, linewidth=2.2, zorder=2, clip_on=False)
ax_enc.add_patch(enc_bg)

# Magic wand icon placeholder — horizontal rule with "✦"
ax_enc.text(0.50, 0.88, "3D CNN",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=DARK, zorder=5)
ax_enc.text(0.50, 0.76, "Encoder",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=DARK, zorder=5)

# small stylised funnel / compress icon: wide bar → narrow bar
for i, (bw, by) in enumerate([(0.62, 0.60), (0.46, 0.51), (0.30, 0.43),
                               (0.20, 0.36), (0.14, 0.30)]):
    rect = FancyBboxPatch((0.50 - bw/2, by - 0.033), bw, 0.052,
        boxstyle="round,pad=0.01", facecolor=BLUE,
        edgecolor="none", alpha=0.75 - i*0.04, zorder=4)
    ax_enc.add_patch(rect)

ax_enc.text(0.50, 0.18, r"$\rho(\mathbf{x})$",
            ha="center", va="center", fontsize=11, color=DARK, zorder=5)
ax_enc.text(0.50, 0.09, "compress structure",
            ha="center", va="center", fontsize=8, color=DARK,
            fontstyle="italic", zorder=5)

# bottom magic-wand glyph
ax_enc.text(0.50, 0.02, "✦",
            ha="center", va="bottom", fontsize=11, color=GOLD, zorder=5)

# ──────────────────────────────────────────────────────────────────────────────
#  MATRIX — Feature outputs  (navy box, white text — like reference z matrix)
# ──────────────────────────────────────────────────────────────────────────────
ax_m = fig.add_axes([0.452, 0.16, 0.131, 0.680], facecolor="none")
ax_m.set_xlim(0, 1); ax_m.set_ylim(0, 1); ax_m.set_axis_off()

mat_bg = FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
    boxstyle="round,pad=0.03", facecolor=NAVY,
    edgecolor=GOLD, linewidth=2.5, zorder=2, clip_on=False)
ax_m.add_patch(mat_bg)

entries = [
    (r"$\sigma_{VM}$", "von Mises stress"),
    (r"$\mathbf{u}$",  "displacement"),
    (r"$C$",           "compliance"),
    (r"$\mu_C$",       "mean (5× ens.)"),
    (r"$\sigma_C$",    "uncertainty"),
]
n_ent = len(entries)
for k, (sym, label) in enumerate(entries):
    y = 0.86 - k * (0.76 / (n_ent - 1))
    ax_m.text(0.50, y, sym,
              ha="center", va="center", fontsize=13,
              color=GOLD, fontweight="bold", zorder=5)
    ax_m.text(0.50, y - 0.085, label,
              ha="center", va="center", fontsize=7.0,
              color=WHITE, fontstyle="italic", zorder=5)
    if k < n_ent - 1:
        ax_m.plot([0.12, 0.88], [y - 0.125, y - 0.125],
                  color=GOLD, lw=0.5, alpha=0.4, zorder=4)

fig.text(0.518, 0.924, "Predicted\nOutputs",
         ha="center", fontsize=11, fontweight="bold", color=NAVY,
         linespacing=1.2, transform=fig.transFigure)

# ──────────────────────────────────────────────────────────────────────────────
#  BOX 2 — Ensemble Predictor  (sand card, blue border — like reference Decoder)
# ──────────────────────────────────────────────────────────────────────────────
ax_dec = fig.add_axes([0.607, 0.06, 0.176, 0.88], facecolor="none")
ax_dec.set_xlim(0, 1); ax_dec.set_ylim(0, 1); ax_dec.set_axis_off()

dec_bg = FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
    boxstyle="round,pad=0.02", facecolor=SAND,
    edgecolor=BLUE, linewidth=2.2, zorder=2, clip_on=False)
ax_dec.add_patch(dec_bg)

ax_dec.text(0.50, 0.88, "Ensemble",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=DARK, zorder=5)
ax_dec.text(0.50, 0.76, "Predictor",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=DARK, zorder=5)

# expand icon: narrow → wide bars (reverse of encoder)
for i, (bw, by) in enumerate([(0.14, 0.60), (0.20, 0.53), (0.30, 0.46),
                               (0.46, 0.39), (0.62, 0.32)]):
    rect = FancyBboxPatch((0.50 - bw/2, by - 0.033), bw, 0.052,
        boxstyle="round,pad=0.01", facecolor=TEAL,
        edgecolor="none", alpha=0.60 + i*0.04, zorder=4)
    ax_dec.add_patch(rect)

ax_dec.text(0.50, 0.19, r"$5\!\times$ members  $\Rightarrow \mu_C, \sigma_C$",
            ha="center", va="center", fontsize=8, color=DARK, zorder=5)
ax_dec.text(0.50, 0.09, r"conservative bound $\hat{C}^+$",
            ha="center", va="center", fontsize=8, color=DARK,
            fontstyle="italic", zorder=5)
ax_dec.text(0.50, 0.02, "✦",
            ha="center", va="bottom", fontsize=11, color=TEAL, zorder=5)

# ──────────────────────────────────────────────────────────────────────────────
#  IMAGE B — Predicted Structural Response (3D stress render)
# ──────────────────────────────────────────────────────────────────────────────
ax_b = fig.add_axes([0.800, 0.175, 0.185, 0.680], projection="3d")
render_ply_colormap(ax_b,
    "figures/screenshot_stls/REF_SASTO_PA_cutaway.ply",
    cmap_name="jet", elev=22, azim=-52, ambient=0.55, bg_color=LBLUE)

# Colorbar
ax_cb = fig.add_axes([0.802, 0.110, 0.183, 0.030])
cb = fig.colorbar(
    plt.cm.ScalarMappable(mcolors.Normalize(0, 1), cmap=plt.get_cmap("jet")),
    cax=ax_cb, orientation="horizontal")
cb.set_ticks([0, 1])
cb.set_ticklabels(["Low stress", "High stress"], fontsize=7.5, color=DARK)
cb.ax.tick_params(size=0)
cb.outline.set_edgecolor(DARK); cb.outline.set_linewidth(0.6)

fig.text(0.893, 0.924, "Predicted Structural\nResponse",
         ha="center", fontsize=12.5, fontweight="bold", color=BLUE,
         linespacing=1.2, transform=fig.transFigure)
fig.text(0.893, 0.063, r"$\sigma_{VM}$,  $\mathbf{u}$,  $C$",
         ha="center", fontsize=11, color=DARK, fontweight="bold",
         transform=fig.transFigure)
fig.text(0.893, 0.036, "von Mises stress · displacement · compliance",
         ha="center", fontsize=8, color=DARK, fontstyle="italic",
         transform=fig.transFigure)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig(str(OUT), dpi=230, bbox_inches="tight",
            facecolor=WHITE, edgecolor="none")
plt.close()
print(f"Saved → {OUT}")
