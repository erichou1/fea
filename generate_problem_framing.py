"""
Generate Problem Framing diagram — reference Encoder/Matrix/Decoder style.

5-element horizontal flow:
  [3D voxel house]  →  [3D CNN Encoder box]  →  [2D stress-field heatmap]  →  [Ensemble Decoder box]  →  [stress-colored 3D house]

All images are generated in-script; no external figures used.
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
from pathlib import Path

# ── Palette ──────────────────────────────────────────────────────────────────
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
#  HELPER: render PLY onto a 3-D axes with bright Lambertian shading
#  color_mode: "solid" = uniform blue/grey; "stress" = jet z-height colormap
# ═══════════════════════════════════════════════════════════════════════════════
def render_ply(ax3d, ply_path, color_mode="solid",
               elev=22, azim=-52, ambient=0.80, bg=LBLUE):
    mesh    = trimesh.load(str(ply_path), force="mesh", process=False)
    verts   = np.array(mesh.vertices, dtype=float)
    faces   = np.array(mesh.faces)
    lo, hi  = verts.min(0), verts.max(0)
    real_sp = hi - lo
    span    = real_sp.max()
    verts   = (verts - (lo + hi) / 2) / span
    poly_v  = verts[faces]

    # Lambertian normals
    n0, n1, n2 = poly_v[:, 0], poly_v[:, 1], poly_v[:, 2]
    nrm = np.cross(n1 - n0, n2 - n0)
    mag = np.linalg.norm(nrm, axis=1, keepdims=True); mag[mag == 0] = 1.0
    nrm    = nrm / mag
    light  = np.array([np.cos(np.radians(35))*np.cos(np.radians(-40)),
                       np.cos(np.radians(35))*np.sin(np.radians(-40)),
                       np.sin(np.radians(35))])
    intens = np.clip(nrm @ light, 0, 1)

    if color_mode == "stress":
        cen     = poly_v.mean(axis=1)
        # simulate stress: high at base, moderate at walls, lower at roof
        # use radial distance from building centre + z (walls carry more stress)
        r       = np.sqrt(cen[:,0]**2 + cen[:,1]**2)
        z_norm  = (cen[:,2] - cen[:,2].min()) / (cen[:,2].max() - cen[:,2].min() + 1e-9)
        stress  = np.clip(0.70 * (1 - z_norm) + 0.50 * r / (r.max()+1e-9), 0, 1)
        base_fc = plt.get_cmap("jet")(stress)[:, :3]
    else:  # solid blue
        z_cen   = poly_v[:, :, 2].mean(axis=1)
        z_norm  = (z_cen - z_cen.min()) / (z_cen.max() - z_cen.min() + 1e-9)
        # gradient from steel-blue (walls) to lighter blue (roof)
        r_ch = 0.10 + 0.25 * z_norm
        g_ch = 0.24 + 0.30 * z_norm
        b_ch = 0.60 + 0.25 * z_norm
        base_fc = np.stack([r_ch, g_ch, b_ch], axis=1)

    shaded = np.clip(base_fc * (ambient + (1 - ambient) * intens[:, None]), 0, 1)

    col = Poly3DCollection(poly_v, zsort="average")
    col.set_facecolor(shaded); col.set_edgecolor("none")
    ax3d.add_collection3d(col)

    pad = 0.06
    ax3d.set_xlim(-0.5-pad, 0.5+pad); ax3d.set_ylim(-0.5-pad, 0.5+pad)
    ax3d.set_zlim(-0.5-pad, 0.5+pad)
    ax3d.set_box_aspect([real_sp[0], real_sp[1], real_sp[2]])
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_facecolor(bg); ax3d.set_axis_off()


# ════════════════════════════════════════════════════════════════════════════
#  MAIN FIGURE
# ════════════════════════════════════════════════════════════════════════════
FIG_W, FIG_H = 20.0, 5.8
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=WHITE)

CY0, CH = 0.04, 0.92  # card y/h (fig-norm)

def add_img_card(x0, x1):
    fig.add_artist(mpatches.FancyBboxPatch(
        (x0, CY0), x1-x0, CH, boxstyle="round,pad=0.012",
        facecolor=LBLUE, edgecolor=BLUE, linewidth=1.8,
        transform=fig.transFigure, clip_on=False, zorder=0))

add_img_card(0.010, 0.200)   # left image
add_img_card(0.800, 0.992)   # right image

# ── Arrows ───────────────────────────────────────────────────────────────────
ARROW_Y = CY0 + CH / 2
for xa, xb in [(0.200, 0.225), (0.420, 0.440), (0.595, 0.615), (0.790, 0.800)]:
    fig.add_artist(FancyArrowPatch(
        (xa+0.003, ARROW_Y), (xb-0.003, ARROW_Y),
        arrowstyle="-|>", mutation_scale=28, color=BLUE, lw=2.8,
        transform=fig.transFigure, clip_on=False, zorder=10))

# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE A — Voxelized structure  (bright blue 3D render)
# ══════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_axes([0.017, 0.14, 0.178, 0.730], projection="3d")
render_ply(ax_a, "figures/screenshot_stls/REF_original_colored.ply",
           color_mode="solid", elev=22, azim=-52, ambient=0.82)

fig.text(0.106, 0.926, "Voxelized Structure",
         ha="center", fontsize=12.5, fontweight="bold", color=BLUE,
         transform=fig.transFigure)
fig.text(0.106, 0.064, r"$\rho(\mathbf{x}) \in \{0,\,1\}$",
         ha="center", fontsize=11.5, color=DARK, fontweight="bold",
         transform=fig.transFigure)
fig.text(0.106, 0.036, "Binary material field",
         ha="center", fontsize=8.5, color=DARK, fontstyle="italic",
         transform=fig.transFigure)

# ══════════════════════════════════════════════════════════════════════════════
#  BOX 1 — 3D CNN Encoder  (matches reference Encoder box style)
# ══════════════════════════════════════════════════════════════════════════════
ax_enc = fig.add_axes([0.227, 0.07, 0.191, 0.86], facecolor="none")
ax_enc.set_xlim(0,1); ax_enc.set_ylim(0,1); ax_enc.set_axis_off()

ax_enc.add_patch(FancyBboxPatch((0,0),1,1, boxstyle="round,pad=0.02",
    facecolor=SAND, edgecolor=GOLD, linewidth=2.4, zorder=2, clip_on=False))

ax_enc.text(0.50, 0.88, "3D CNN",   ha="center", va="center",
            fontsize=14, fontweight="bold", color=DARK, zorder=5)
ax_enc.text(0.50, 0.76, "Encoder",  ha="center", va="center",
            fontsize=14, fontweight="bold", color=DARK, zorder=5)

# Conv-block shrinking diagram (simple, clean)
block_data = [(0.60, 0.56), (0.44, 0.47), (0.30, 0.40), (0.20, 0.34), (0.12, 0.29)]
colors_enc  = [BLUE, "#3668C4", "#5080CC", TEAL, "#12B0C2"]
for (bw, by), bc in zip(block_data, colors_enc):
    ax_enc.add_patch(FancyBboxPatch((0.50-bw/2, by-0.038), bw, 0.062,
        boxstyle="square,pad=0.005", facecolor=bc, edgecolor=WHITE,
        linewidth=0.6, alpha=0.92, zorder=4))

ax_enc.text(0.50, 0.20, r"$\rho(\mathbf{x}) \rightarrow \mathbf{z}$",
            ha="center", va="center", fontsize=11, color=DARK, zorder=5)
ax_enc.text(0.50, 0.10, "compress to latent",
            ha="center", va="center", fontsize=8, color=DARK,
            fontstyle="italic", zorder=5)
ax_enc.text(0.50, 0.022, "✦",
            ha="center", va="bottom", fontsize=12, color=GOLD, zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
#  CENTER — 2-D predicted stress cross-section  (like reference z-matrix, but visual)
# ══════════════════════════════════════════════════════════════════════════════
ax_hm = fig.add_axes([0.442, 0.09, 0.151, 0.82], facecolor=NAVY)
ax_hm.set_xlim(0,1); ax_hm.set_ylim(0,1); ax_hm.set_axis_off()

ax_hm.add_patch(FancyBboxPatch((0,0),1,1, boxstyle="round,pad=0.02",
    facecolor=NAVY, edgecolor=GOLD, linewidth=2.4, zorder=2, clip_on=False))

# --- synthetic 2-D stress field (building cross-section, XZ plane) ---
nx, nz = 24, 20
X, Z = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,nz))
# shell: stress concentrated at walls (low x, high x) and base (low z)
wall_L = np.exp(-((X-0.0)**2)/0.04)
wall_R = np.exp(-((X-1.0)**2)/0.04)
base   = np.exp(-((Z-0.0)**2)/0.06)
roof   = np.exp(-((Z-1.0)**2)/0.12) * 0.4
stress = np.clip(wall_L + wall_R + base + roof, 0, 1)
# set interior (non-shell) to zero → hollow building
interior = (X > 0.15) & (X < 0.85) & (Z > 0.18) & (Z < 0.88)
stress[interior] *= 0.08   # near-zero inside

# draw as imshow inside the navy box (axes [0.08,0.52]×[0.52,0.92])
ax_inner = fig.add_axes([0.442+0.151*0.08, 0.09+0.82*0.52,
                          0.151*0.84,       0.82*0.38])
im = ax_inner.imshow(stress, cmap="jet", origin="lower", aspect="auto",
                     vmin=0, vmax=1, interpolation="bilinear")
ax_inner.set_axis_off()

# equation overlay on heatmap
ax_inner.text(0.5, 0.5,
              r"$K\mathbf{u}=\mathbf{f}$",
              ha="center", va="center",
              transform=ax_inner.transAxes,
              fontsize=13, fontweight="bold", color=WHITE, alpha=0.85,
              bbox=dict(boxstyle="round,pad=0.3", facecolor=NAVY, alpha=0.55, edgecolor="none"))

# variable list in lower half of navy box
entries = [
    (r"$\sigma_{VM}$", "von Mises stress"),
    (r"$C$",           "compliance"),
    (r"$\mu_{C}$",     "predicted mean"),
    (r"$\sigma_{C}$",  "uncertainty"),
]
for k, (sym, lbl) in enumerate(entries):
    yy = 0.42 - k * 0.115
    ax_hm.text(0.50, yy, sym, ha="center", va="center",
               fontsize=11.5, color=GOLD, fontweight="bold", zorder=5)
    ax_hm.text(0.50, yy-0.055, lbl, ha="center", va="center",
               fontsize=7.2, color=WHITE, fontstyle="italic", zorder=5)
    if k < len(entries)-1:
        ax_hm.plot([0.10,0.90],[yy-0.082,yy-0.082],
                   color=GOLD, lw=0.5, alpha=0.35, zorder=4)

fig.text(0.518, 0.926, "Predicted\nOutputs",
         ha="center", fontsize=11.5, fontweight="bold", color=NAVY,
         linespacing=1.2, transform=fig.transFigure)

# ══════════════════════════════════════════════════════════════════════════════
#  BOX 2 — Ensemble Predictor / Decoder  (matches reference Decoder box)
# ══════════════════════════════════════════════════════════════════════════════
ax_dec = fig.add_axes([0.597, 0.07, 0.191, 0.86], facecolor="none")
ax_dec.set_xlim(0,1); ax_dec.set_ylim(0,1); ax_dec.set_axis_off()

ax_dec.add_patch(FancyBboxPatch((0,0),1,1, boxstyle="round,pad=0.02",
    facecolor=SAND, edgecolor=BLUE, linewidth=2.4, zorder=2, clip_on=False))

ax_dec.text(0.50, 0.88, "Ensemble",   ha="center", va="center",
            fontsize=14, fontweight="bold", color=DARK, zorder=5)
ax_dec.text(0.50, 0.76, "Predictor",  ha="center", va="center",
            fontsize=14, fontweight="bold", color=DARK, zorder=5)

# expanding decode blocks (reverse of encoder)
block_data2 = [(0.12,0.60),(0.20,0.53),(0.30,0.46),(0.44,0.40),(0.60,0.34)]
colors_dec   = ["#12B0C2", TEAL, "#5080CC", "#3668C4", BLUE]
for (bw, by), bc in zip(block_data2, colors_dec):
    ax_dec.add_patch(FancyBboxPatch((0.50-bw/2, by-0.038), bw, 0.062,
        boxstyle="square,pad=0.005", facecolor=bc, edgecolor=WHITE,
        linewidth=0.6, alpha=0.92, zorder=4))

ax_dec.text(0.50, 0.22,
            r"$\hat{C}^{+} = \mu_{C} + k\,\sigma_{C}$",
            ha="center", va="center", fontsize=10.5, color=DARK, zorder=5)
ax_dec.text(0.50, 0.12, r"$5\!\times$ independent members",
            ha="center", va="center", fontsize=8, color=DARK,
            fontstyle="italic", zorder=5)
ax_dec.text(0.50, 0.022, "✦",
            ha="center", va="bottom", fontsize=12, color=TEAL, zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE B — Stress-coloured 3D render of optimised structure
# ══════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_axes([0.806, 0.14, 0.178, 0.730], projection="3d")
render_ply(ax_b, "figures/screenshot_stls/REF_SASTO_PA_cutaway.ply",
           color_mode="stress", elev=22, azim=-52, ambient=0.78, bg=LBLUE)

# colourbar
ax_cb = fig.add_axes([0.808, 0.090, 0.175, 0.028])
cb = fig.colorbar(
    plt.cm.ScalarMappable(mcolors.Normalize(0,1), cmap=plt.get_cmap("jet")),
    cax=ax_cb, orientation="horizontal")
cb.set_ticks([0,1]); cb.set_ticklabels(["Low", "High"], fontsize=8, color=DARK)
cb.ax.tick_params(size=0); cb.outline.set_edgecolor(DARK); cb.outline.set_linewidth(0.7)
ax_cb.set_title(r"$\sigma_{VM}$ stress", fontsize=8.5, color=DARK, pad=2)

fig.text(0.895, 0.926, "Predicted Structural\nResponse",
         ha="center", fontsize=12.5, fontweight="bold", color=BLUE,
         linespacing=1.2, transform=fig.transFigure)
fig.text(0.895, 0.038, r"von Mises $\sigma_{VM}$  ·  compliance $C$  ·  displacement $\mathbf{u}$",
         ha="center", fontsize=8.5, color=DARK, fontstyle="italic",
         transform=fig.transFigure)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig(str(OUT), dpi=240, bbox_inches="tight",
            facecolor=WHITE, edgecolor="none")
plt.close()
print(f"Saved → {OUT}")
