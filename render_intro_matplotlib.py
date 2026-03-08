"""
Render before/after intro images using matplotlib 3D (no pyrender/osmesa needed).
Loads PLY files from figures/screenshot_stls/, renders isometric cutaway views,
saves to poster_images_extracted/ as:
  REF_original_cutaway_iso.png
  REF_SASTO_PA_cutaway_iso.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import numpy as np
from pathlib import Path

OUT_DIR = Path("poster_images_extracted")
SRC_DIR = Path("figures/screenshot_stls")

# ── Poster palette ────────────────────────────────────────────────────────────
NAVY  = "#062B7A"
GOLD  = "#CFA535"
TEAL  = "#008C9E"
RED   = "#D7263D"
CARD  = "#F7F9FC"
WHITE = "#FFFFFF"

# ── Rendering parameters ──────────────────────────────────────────────────────
ELEV = 22       # degrees elevation for isometric-ish view
AZIM = -55      # azimuth angle
DPI  = 260
IMG_W, IMG_H = 4.2, 3.5   # figure size in inches

def load_ply_mesh(path):
    """Load a PLY file, return trimesh Mesh."""
    mesh = trimesh.load(str(path), force="mesh", process=False)
    return mesh

def get_face_colors(mesh):
    """Extract per-face colors from vertex colors (if available)."""
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        vc = mesh.visual.vertex_colors[:, :3] / 255.0   # (N_verts, 3) float
        # average vertex colors per face
        fc = vc[mesh.faces].mean(axis=1)                # (N_faces, 3)
        return fc
    # fallback: single color based on mesh name
    return None

def render_mesh(ply_path, out_path, label_top, label_sub, label_color,
                header_color, fig_outline_color):
    """Render a single PLY as a matplotlib 3D figure with label panels."""

    mesh = load_ply_mesh(ply_path)

    verts  = np.array(mesh.vertices)
    faces  = np.array(mesh.faces)
    fcolors = get_face_colors(mesh)

    # Normalise to unit cube for consistent framing
    bbox   = verts.max(axis=0) - verts.min(axis=0)
    center = verts.min(axis=0) + bbox / 2
    scale  = bbox.max() if bbox.max() > 0 else 1.0
    verts  = (verts - center) / scale   # centred, ≈ [-0.5, 0.5]

    # Build face vertex list for Poly3DCollection
    poly_verts = verts[faces]   # (N_faces, 3, 3)

    fig = plt.figure(figsize=(IMG_W, IMG_H), facecolor=CARD)

    # ── Header band ────────────────────────────────────────────────────────────
    ax_hdr = fig.add_axes([0.0, 0.88, 1.0, 0.12], facecolor=header_color)
    ax_hdr.set_axis_off()
    ax_hdr.text(0.5, 0.55, label_top,
                ha="center", va="center", transform=ax_hdr.transAxes,
                fontsize=11, color=WHITE, fontweight="bold")
    ax_hdr.text(0.5, 0.10, label_sub,
                ha="center", va="center", transform=ax_hdr.transAxes,
                fontsize=8.5, color=WHITE, fontstyle="italic")

    # ── 3-D axes ───────────────────────────────────────────────────────────────
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.90], projection="3d",
                      facecolor=CARD)

    # matplotlib shade=True requires facecolor at construction time
    base_fc = fcolors if fcolors is not None else np.full((len(faces), 3), 0.0)
    if fcolors is None:
        # Use a teal default colour for every face
        base_fc = np.tile(matplotlib.colors.to_rgb(TEAL), (len(faces), 1))

    # Compute per-face normals for shading
    v0 = poly_verts[:, 0, :]
    v1 = poly_verts[:, 1, :]
    v2 = poly_verts[:, 2, :]
    normals = np.cross(v1 - v0, v2 - v0)
    norms   = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = normals / norms   # unit normals (N_faces, 3)

    # Manual Lambertian shading: dot product with light direction
    light_dir = np.array([np.cos(np.radians(45)) * np.cos(np.radians(225)),
                           np.cos(np.radians(45)) * np.sin(np.radians(225)),
                           np.sin(np.radians(45))])
    intensity  = np.clip(normals @ light_dir, 0, 1)[:, np.newaxis]  # (N_faces, 1)
    shaded_fc  = base_fc * (0.30 + 0.70 * intensity)                # ambient + diffuse
    shaded_fc  = np.clip(shaded_fc, 0, 1)

    col = Poly3DCollection(poly_verts, zsort="average")
    col.set_facecolor(shaded_fc)
    col.set_edgecolor("none")
    col.set_alpha(1.0)

    ax.add_collection3d(col)

    # Frame the axes to the mesh
    pad = 0.08
    ax.set_xlim(-0.5 - pad, 0.5 + pad)
    ax.set_ylim(-0.5 - pad, 0.5 + pad)
    ax.set_zlim(-0.5 - pad, 0.5 + pad)
    ax.set_axis_off()
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_box_aspect([bbox[0], bbox[1], bbox[2]])   # preserve real aspect

    # ── Thin coloured outline ──────────────────────────────────────────────────
    for spine in ["left", "right", "top", "bottom"]:
        fig.add_axes([0.0, 0.0, 1.0, 1.0]).set_axis_off()
    # draw an outline rect
    rect_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    rect_ax.set_axis_off()
    rect_ax.set_xlim(0, 1)
    rect_ax.set_ylim(0, 1)
    for (x0, y0, x1, y1) in [(0, 0, 1, 0), (0, 1, 1, 1),
                               (0, 0, 0, 1), (1, 0, 1, 1)]:
        rect_ax.plot([x0, x1], [y0, y1],
                     color=fig_outline_color, lw=2.5,
                     transform=rect_ax.transAxes)

    plt.savefig(str(out_path), dpi=DPI, bbox_inches="tight", facecolor=CARD)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Render REF_original_cutaway ───────────────────────────────────────────────
print("Rendering REF_original_cutaway_iso.png …")
render_mesh(
    ply_path        = SRC_DIR / "REF_original_cutaway.ply",
    out_path        = OUT_DIR / "REF_original_cutaway_iso.png",
    label_top       = "Before: Reference Design",
    label_sub       = "Uniform thick walls — baseline CubiCasa footprint",
    label_color     = RED,
    header_color    = NAVY,
    fig_outline_color = RED,
)

# ── Render REF_SASTO_PA_cutaway ───────────────────────────────────────────────
print("Rendering REF_SASTO_PA_cutaway_iso.png …")
render_mesh(
    ply_path        = SRC_DIR / "REF_SASTO_PA_cutaway.ply",
    out_path        = OUT_DIR / "REF_SASTO_PA_cutaway_iso.png",
    label_top       = "After: SASTO Part-Aware Optimisation",
    label_sub       = "Thinned walls · compliance ratio verified under independent FEA",
    label_color     = TEAL,
    header_color    = "#0A3D9A",
    fig_outline_color = TEAL,
)

print("Done.")
