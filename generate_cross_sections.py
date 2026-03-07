#!/usr/bin/env python3
"""Generate cross-section renderings of 3D house models.

Loads STL part files (exterior_walls, interior_rooms, roof, floor, attic_floor),
slices them at the Y-midplane with trimesh.slice_plane to produce a clean cut,
and renders isometric views with per-part coloring.

Requires: trimesh, shapely, rtree, mapbox-earcut
"""

import os
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Configuration ──
PARTS_DIR = r"C:\Users\ericx\workspace\topopt_project\optimization\data\3dwire_parts_combined"
OUT_DIR   = r"C:\Users\ericx\workspace\topopt_project\figures"

MODEL_IDS = ["00000", "00010", "00050"]

BG_RGB = (0.902, 0.910, 0.922)  # matches BG_COLOR [230, 232, 235]

PART_STYLES = {
    "exterior_walls":  {"color": (0.72, 0.72, 0.72, 1.0), "label": "Exterior Walls"},
    "interior_rooms":  {"color": (0.45, 0.68, 0.88, 1.0), "label": "Interior Rooms"},
    "roof":            {"color": (0.82, 0.48, 0.33, 1.0), "label": "Roof"},
    "floor":           {"color": (0.60, 0.60, 0.55, 1.0), "label": "Floor"},
    "attic_floor":     {"color": (0.88, 0.78, 0.52, 1.0), "label": "Attic Floor"},
}


def get_global_bounds(model_id):
    """Get the bounding box across ALL parts of a model."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for pname in PART_STYLES:
        p = os.path.join(PARTS_DIR, f"{model_id}_{pname}.stl")
        if os.path.exists(p):
            m = trimesh.load(p, force='mesh')
            if not m.is_empty:
                lo = np.minimum(lo, m.bounds[0])
                hi = np.maximum(hi, m.bounds[1])
    return lo, hi


def add_mesh_to_ax(ax, mesh, color, max_faces=10000):
    """Add a trimesh to a matplotlib 3D axis."""
    verts = mesh.vertices
    faces = mesh.faces
    if len(faces) > max_faces:
        idx = np.random.choice(len(faces), max_faces, replace=False)
        faces = faces[idx]
    polys = verts[faces]
    pc = Poly3DCollection(polys, alpha=color[3], linewidths=0.15,
                          edgecolors=(0.25, 0.25, 0.25, 0.2))
    pc.set_facecolor(color[:3])
    ax.add_collection3d(pc)


def set_equal_axes(ax, lo, hi):
    """Set axis limits proportional to data extents — tighter framing."""
    ranges = hi - lo
    margin = ranges * 0.0  # zero margin for tightest framing
    ax.set_xlim(lo[0] - margin[0], hi[0] + margin[0])
    ax.set_ylim(lo[1] - margin[1], hi[1] + margin[1])
    ax.set_zlim(lo[2] - margin[2], hi[2] + margin[2])
    # Proportional aspect — model fills its natural shape
    ax.set_box_aspect(ranges / ranges.max() if ranges.max() > 0 else [1, 1, 1])
    ax.set_axis_off()


def render_row(model_id, ax_full, ax_cut):
    """Render one model: full view + proper cross-section (sliced at Y midplane)."""
    lo, hi = get_global_bounds(model_id)
    mid_y = (lo[1] + hi[1]) / 2.0

    for pname, style in PART_STYLES.items():
        stl_path = os.path.join(PARTS_DIR, f"{model_id}_{pname}.stl")
        if not os.path.exists(stl_path):
            continue
        mesh = trimesh.load(stl_path, force='mesh')
        if mesh.is_empty:
            continue

        color = style["color"]

        # Full model
        add_mesh_to_ax(ax_full, mesh, color)

        # Sliced: keep the Y < mid_y half, cap=True fills the cut face
        try:
            sliced = mesh.slice_plane(
                plane_origin=[0, mid_y, 0],
                plane_normal=[0, -1, 0],
                cap=True
            )
            if sliced is not None and not sliced.is_empty:
                add_mesh_to_ax(ax_cut, sliced, color)
        except Exception:
            pass

    set_equal_axes(ax_full, lo, hi)
    # For the cut view, use full bounds so model stays same scale
    set_equal_axes(ax_cut, lo, hi)

    ax_full.view_init(elev=20, azim=-60)
    # Cut view: look from +Y direction into the exposed interior face
    ax_cut.view_init(elev=20, azim=85)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    valid_ids = []
    for mid in MODEL_IDS:
        if os.path.exists(os.path.join(PARTS_DIR, f"{mid}_exterior_walls.stl")):
            valid_ids.append(mid)

    if not valid_ids:
        print("ERROR: No valid model IDs found.")
        return

    n = len(valid_ids)
    print(f"Rendering {n} models (full + cross-section)...")

    fig = plt.figure(figsize=(14, 3.5 * n), facecolor=BG_RGB)

    for row, mid in enumerate(valid_ids):
        print(f"  Model {mid} ({row+1}/{n})...")
        ax_full = fig.add_subplot(n, 2, row * 2 + 1, projection='3d')
        ax_cut  = fig.add_subplot(n, 2, row * 2 + 2, projection='3d')
        ax_full.set_facecolor(BG_RGB)
        ax_cut.set_facecolor(BG_RGB)

        render_row(mid, ax_full, ax_cut)

        ax_full.set_title(f"Model {mid} — Full", fontsize=11,
                          fontweight='bold', pad=-14)
        ax_cut.set_title(f"Model {mid} — Cross-Section",
                         fontsize=11, fontweight='bold', pad=-14)

    # Shared legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v["color"][:3], edgecolor='gray',
                             label=v["label"]) for v in PART_STYLES.values()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=10, frameon=True, fancybox=True,
               edgecolor='#cccccc', bbox_to_anchor=(0.5, 0.003))

    plt.suptitle("3D House Models \u2014 Cross-Section Views",
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99], h_pad=0.0, w_pad=0.0)
    plt.subplots_adjust(left=-0.08, right=1.08, hspace=-0.05, wspace=-0.15)

    for ext in ("png",):
        fig.savefig(os.path.join(OUT_DIR, f"fig_cross_sections.{ext}"),
                    dpi=200, bbox_inches='tight',
                    facecolor=BG_RGB, edgecolor='none')
    plt.close(fig)
    print(f"\nSaved to {OUT_DIR}/fig_cross_sections.png")


if __name__ == "__main__":
    main()
