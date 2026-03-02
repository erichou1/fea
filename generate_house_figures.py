#!/usr/bin/env python3
"""Generate optimized house STL figures for the research paper.

Produces:
  1. fig_optimized_gallery.png — Grid of 6 diverse designs (original vs optimized)
  2. fig_type_comparison.png  — Reference case: Original vs SASTO-U vs SASTO-PA

Uses SDF + marching cubes + trimesh for mesh extraction, matplotlib for rendering.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, binary_dilation
from skimage.measure import marching_cubes
import trimesh

# Add fea_ml to path for imports
sys.path.insert(0, str(Path(__file__).parent / "fea_ml"))

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

FEA_ML = BASE_DIR / "fea_ml"
BATCH_DIR = FEA_ML / "runs" / "v3" / "batch_results_all"
DATA_DIR = FEA_ML / "data" / "runs_real_128"
OPT_DIR = FEA_ML / "runs" / "v3" / "optimization_128"

# Part labels (from fea_ml/fea_ml/geometry/voxelize.py)
PART_EMPTY = 0
PART_EXTERIOR = 1   # exterior wall
PART_INTERIOR = 2   # interior wall
PART_ROOF = 3
PART_FLOOR = 4

# RGBA colors for each structural part
PART_COLORS = {
    PART_EXTERIOR: [0.27, 0.51, 0.71, 1.0],   # Steel blue
    PART_INTERIOR: [1.00, 0.50, 0.31, 1.0],    # Coral/orange
    PART_ROOF:     [0.42, 0.56, 0.14, 1.0],    # Olive green
    PART_FLOOR:    [0.44, 0.50, 0.56, 1.0],    # Slate gray
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})


def voxels_to_mesh(occ, blur_sigma=0.6):
    """Convert binary voxel grid to clean trimesh with floor, no floating parts."""
    occ = occ.astype(bool).copy()

    # Find occupied z-range and add floor slab
    z_occupied = np.where(occ.any(axis=(0, 1)))[0]
    if len(z_occupied) == 0:
        return trimesh.Trimesh()
    z_min, z_max = z_occupied[0], z_occupied[-1]

    # Get footprint from lowest occupied slices and add 2-voxel floor slab
    xy_footprint = occ[:, :, z_min:min(z_min + 4, z_max)].any(axis=2)
    xy_footprint = binary_dilation(xy_footprint, iterations=1)
    floor_start = max(0, z_min - 2)
    for dz in range(floor_start, z_min + 1):
        occ[:, :, dz] |= xy_footprint

    # Remove small disconnected voxel clusters
    labeled, n = label(occ)
    if n > 1:
        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
        main_label = np.argmax(sizes) + 1
        occ = (labeled == main_label)

    # SDF: positive inside, negative outside
    dist_in = distance_transform_edt(occ)
    dist_out = distance_transform_edt(~occ)
    sdf = dist_in - dist_out

    if blur_sigma > 0:
        sdf = gaussian_filter(sdf, sigma=blur_sigma)

    # Pad to ensure closed surface
    sdf_padded = np.pad(sdf, 1, mode='constant', constant_values=-1)
    verts, faces, normals, _ = marching_cubes(sdf_padded, level=0.0)
    # Remove padding offset
    verts -= 1.0
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # Keep only largest mesh component (skip for very large meshes — slow)
    if len(mesh.faces) < 500000:
        components = mesh.split()
        if len(components) > 1:
            mesh = max(components, key=lambda c: len(c.faces))

    # Fill holes and fix normals (skip fix_normals for very large meshes - can hang)
    trimesh.repair.fill_holes(mesh)
    if len(mesh.faces) < 200000:
        try:
            trimesh.repair.fix_normals(mesh)
        except Exception:
            pass
    return mesh


def decimate_mesh(mesh, target_faces=10000):
    """Decimate mesh preserving surface coverage (no dots like random subsampling)."""
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        decimated = mesh.simplify_quadric_decimation(face_count=target_faces)
        if len(decimated.faces) > 0:
            return decimated
    except Exception as e:
        print(f"    Decimation failed: {e}")
    return mesh


def get_colors(n_faces, triangles, color_by, alpha):
    """Generate face colors (flat mode only — used when no part labels available)."""
    n = n_faces
    if color_by == 'height':
        centroids_z = triangles.mean(axis=1)[:, 2]
        z_lo, z_hi = centroids_z.min(), centroids_z.max()
        norm_z = (centroids_z - z_lo) / (z_hi - z_lo + 1e-10)
        colors = plt.cm.viridis(norm_z)
    elif color_by == 'original':
        colors = np.full((n, 4), [0.72, 0.76, 0.82, alpha])
    elif color_by == 'optimized':
        colors = np.full((n, 4), [0.20, 0.60, 0.85, alpha])
    elif color_by == 'sasto_pa':
        colors = np.full((n, 4), [0.85, 0.35, 0.20, alpha])
    elif color_by == 'cutout':
        colors = np.full((n, 4), [0.78, 0.75, 0.72, alpha])
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n))
    colors[:, 3] = alpha
    return colors


def get_part_face_colors(centroids_voxel, part_labels, alpha=0.95):
    """Map face centroids (in voxel coordinate space) to per-part RGBA colors.

    Args:
        centroids_voxel: (n_faces, 3) face centroid positions in voxel coords.
        part_labels: 3-D integer array of part labels (same shape as voxel grid).
        alpha: opacity value.

    Returns:
        (n_faces, 4) RGBA color array.
    """
    shape = np.array(part_labels.shape)
    coords = np.clip(np.round(centroids_voxel).astype(int), 0, shape - 1)
    labels = part_labels[coords[:, 0], coords[:, 1], coords[:, 2]]

    # Default gray for any unlabeled voxels
    colors = np.full((len(labels), 4), [0.55, 0.55, 0.55, alpha])
    for label_val, rgba in PART_COLORS.items():
        mask = labels == label_val
        if mask.any():
            colors[mask] = rgba
    colors[:, 3] = alpha
    return colors


def render_mesh(ax, mesh, elev=25, azim=-60, color_by='height', alpha=0.95,
                target_faces=10000, title=None, title_size=13, part_labels=None):
    """Render a trimesh using decimation.  If *part_labels* is provided,
    faces are coloured by structural part; otherwise falls back to *color_by*."""
    render_m = decimate_mesh(mesh, target_faces)
    verts = render_m.vertices.copy()
    faces = render_m.faces

    # Compute part colors in original voxel space BEFORE centering
    if part_labels is not None:
        centroids_orig = verts[faces].mean(axis=1)
        colors = get_part_face_colors(centroids_orig, part_labels, alpha)
    else:
        colors = None

    center = verts.mean(axis=0)
    verts -= center

    triangles = verts[faces]
    if colors is None:
        colors = get_colors(len(faces), triangles, color_by, alpha)

    edge_colors = colors.copy()
    edge_colors[:, :3] *= 0.85
    edge_colors[:, 3] = 0.3

    poly = Poly3DCollection(triangles, facecolors=colors,
                            edgecolors=edge_colors, linewidths=0.1)
    ax.add_collection3d(poly)

    extents = np.abs(verts).max(axis=0) * 1.05
    ax.set_xlim(-extents[0], extents[0])
    ax.set_ylim(-extents[1], extents[1])
    ax.set_zlim(-extents[2], extents[2])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    if title:
        ax.set_title(title, fontsize=title_size, pad=2, fontweight='bold')


def render_cutout(ax, mesh, cut_axis='y', cut_frac=0.5, elev=20, azim=-30,
                  color_by='cutout', alpha=0.95, target_faces=10000,
                  title=None, title_size=13, part_labels=None):
    """Render interior cutout — use trimesh.slice_plane for a clean planar cut.
    If *part_labels* is provided, faces are coloured by structural part."""
    render_m = decimate_mesh(mesh, target_faces)

    # --- clean planar slice (no shards) ---
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[cut_axis]
    v_min = render_m.vertices[:, axis_idx].min()
    v_max = render_m.vertices[:, axis_idx].max()
    cut_val = v_min + cut_frac * (v_max - v_min)

    # plane_normal points in +axis direction; slice_plane keeps the side the
    # normal points AWAY from, i.e. the side > cut_val
    plane_origin = np.zeros(3)
    plane_origin[axis_idx] = cut_val
    plane_normal = np.zeros(3)
    plane_normal[axis_idx] = -1.0          # keep the +axis half

    try:
        sliced = render_m.slice_plane(plane_origin, plane_normal)
    except Exception:
        return
    if sliced is None or len(sliced.faces) == 0:
        return

    verts = sliced.vertices.copy()
    faces = sliced.faces

    # Part colours in original voxel space (before centering)
    if part_labels is not None:
        centroids_orig = verts[faces].mean(axis=1)
        colors = get_part_face_colors(centroids_orig, part_labels, alpha)
    else:
        colors = None

    center = render_m.vertices.mean(axis=0)   # use full-mesh center for consistency
    verts -= center

    triangles = verts[faces]
    if colors is None:
        colors = get_colors(len(faces), triangles, color_by, alpha)

    edge_colors = colors.copy()
    edge_colors[:, :3] *= 0.80
    edge_colors[:, 3] = 0.4

    poly = Poly3DCollection(triangles, facecolors=colors,
                            edgecolors=edge_colors, linewidths=0.15)
    ax.add_collection3d(poly)

    # Use full-mesh extents so camera framing matches sibling panels
    full_verts = render_m.vertices - center
    extents = np.abs(full_verts).max(axis=0) * 1.05
    ax.set_xlim(-extents[0], extents[0])
    ax.set_ylim(-extents[1], extents[1])
    ax.set_zlim(-extents[2], extents[2])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    if title:
        ax.set_title(title, fontsize=title_size, pad=2, fontweight='bold')


def load_sample_meshes(sample_id):
    """Load baseline and optimized occupancy, convert to meshes.

    Returns (base_mesh, opt_mesh, part_labels).
    *part_labels* is the 3-D integer part-label grid (or None).
    The optimized mesh uses blur_sigma=0.0 (no smoothing) so that
    voxel-level optimisation (holes, thinned walls) is clearly visible.
    """
    # Baseline
    base_path = DATA_DIR / sample_id / "occ.npz"
    if not base_path.exists():
        return None, None, None
    base_occ = np.load(base_path)['data']

    # Optimized
    opt_path = BATCH_DIR / sample_id / "optimized_occ.npz"
    if not opt_path.exists():
        return None, None, None
    opt_occ = np.load(opt_path)['data']

    # Part labels (from original design)
    part_labels = None
    part_path = DATA_DIR / sample_id / "part.npz"
    if part_path.exists():
        part_labels = np.load(part_path)['data']

    base_mesh = voxels_to_mesh(base_occ, blur_sigma=0.4)
    opt_mesh = voxels_to_mesh(opt_occ, blur_sigma=0.0)
    return base_mesh, opt_mesh, part_labels


def generate_gallery():
    """Generate a gallery of 4 diverse designs: Original | Optimized | Interior Cutout."""
    print("Generating optimized house gallery...")

    # Select diverse samples across the reduction distribution
    samples_info = []
    batch_files = list(BATCH_DIR.iterdir())
    for d in batch_files:
        summ_path = d / "optimization_summary.json"
        occ_path = d / "optimized_occ.npz"
        if summ_path.exists() and occ_path.exists():
            with open(summ_path) as f:
                s = json.load(f)
            if s.get('constraints_satisfied'):
                base_path = DATA_DIR / s['sample_id'] / "occ.npz"
                if base_path.exists():
                    samples_info.append(s)

    samples_info.sort(key=lambda x: x['volume_reduction_pct'], reverse=True)

    # Pick 4 spread across the distribution
    n = len(samples_info)
    indices = [0, n//4, n//2, 3*n//4]
    selected = [samples_info[i] for i in indices]

    fig = plt.figure(figsize=(16, 18))

    for row, s in enumerate(selected):
        sid = s['sample_id']
        red_pct = s['volume_reduction_pct']
        print(f"  {sid}: {red_pct:.1f}% reduction...")

        base_mesh, opt_mesh, part_labels = load_sample_meshes(sid)
        if base_mesh is None:
            continue

        # Pre-decimate once per design
        base_mesh = decimate_mesh(base_mesh, 5000)
        opt_mesh = decimate_mesh(opt_mesh, 5000)

        n_base = int(np.load(DATA_DIR / sid / "occ.npz")['data'].sum())
        n_opt = int(np.load(BATCH_DIR / sid / "optimized_occ.npz")['data'].sum())

        # Original (left) — coloured by structural part
        ax1 = fig.add_subplot(4, 3, row * 3 + 1, projection='3d')
        t1 = "Original" if row == 0 else ""
        render_mesh(ax1, base_mesh, color_by='original', part_labels=part_labels,
                    title=f"{t1}\n{n_base:,} vox" if row == 0 else f"{n_base:,} vox",
                    title_size=13, target_faces=99999)

        # Optimized (middle) — coloured by structural part (shows thinning/holes)
        ax2 = fig.add_subplot(4, 3, row * 3 + 2, projection='3d')
        t2 = "Optimized" if row == 0 else ""
        render_mesh(ax2, opt_mesh, color_by='height', part_labels=part_labels,
                    title=f"{t2}\n{n_opt:,} vox ($-${red_pct:.1f}%)" if row == 0 else f"{n_opt:,} vox ($-${red_pct:.1f}%)",
                    title_size=13, target_faces=99999)

        # Interior cutout (right) — vertical half-cut, coloured by part
        ax3 = fig.add_subplot(4, 3, row * 3 + 3, projection='3d')
        t3 = "Interior Cutout" if row == 0 else ""
        render_cutout(ax3, opt_mesh, cut_axis='y', cut_frac=0.5,
                      elev=15, azim=-25, part_labels=part_labels,
                      title=f"{t3}\nSample {sid}" if row == 0 else f"Sample {sid}",
                      title_size=13, target_faces=99999)

    fig.suptitle("SASTO-PA Optimization Gallery\nOriginal  |  Optimized  |  Interior Cutout",
                 fontsize=17, fontweight='bold', y=0.99)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02,
                        wspace=0.02, hspace=0.06)

    out_path = OUT_DIR / "fig_optimized_gallery.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def generate_type_comparison():
    """Generate comparison: Original vs SASTO-U (uniform) vs SASTO-PA (part-aware)."""
    print("Generating type comparison (Original vs SASTO-U vs SASTO-PA)...")

    # Reference case — has v11 (PA) and v12 (U) 
    base_path = OPT_DIR / "fixed_occ.npz"
    v11_path = OPT_DIR / "optimized_occ_v11.npz"
    v12_path = OPT_DIR / "optimized_occ_v12.npz"

    if not all(p.exists() for p in [base_path, v11_path, v12_path]):
        print("  ! Reference case files missing for type comparison")
        # Fallback: use STL files directly
        return generate_type_comparison_stl()

    base_occ = np.load(base_path)['data']
    v11_occ = np.load(v11_path)['data']
    v12_occ = np.load(v12_path)['data']

    n_base = int(base_occ.sum())
    n_v11 = int(v11_occ.sum())
    n_v12 = int(v12_occ.sum())

    # Load part labels for the reference case
    part_labels = None
    part_path = OPT_DIR / "fixed_part.npz"
    if part_path.exists():
        part_labels = np.load(part_path)['data']

    # Low blur for optimized meshes → preserves holes / thinned walls
    base_mesh = voxels_to_mesh(base_occ, blur_sigma=0.4)
    v11_mesh = voxels_to_mesh(v11_occ, blur_sigma=0.0)
    v12_mesh = voxels_to_mesh(v12_occ, blur_sigma=0.0)

    # Pre-decimate once
    FACES = 5000
    base_mesh = decimate_mesh(base_mesh, FACES)
    v11_mesh = decimate_mesh(v11_mesh, FACES)
    v12_mesh = decimate_mesh(v12_mesh, FACES)
    print(f"  Decimated to: {len(base_mesh.faces)}, {len(v11_mesh.faces)}, {len(v12_mesh.faces)} faces")

    # 3 columns x 4 view rows: Front, Isometric, Top, Interior Cutout
    views = [
        ("Front", 0, -90),
        ("Isometric", 25, -60),
        ("Top", 90, -90),
    ]

    meshes = [
        (base_mesh, f"Original\n({n_base:,} voxels)", 'original'),
        (v12_mesh, f"SASTO-U (uniform)\n({n_v12:,} voxels, -{100*(n_base-n_v12)/n_base:.1f}%)", 'optimized'),
        (v11_mesh, f"SASTO-PA (part-aware)\n({n_v11:,} voxels, -{100*(n_base-n_v11)/n_base:.1f}%)", 'sasto_pa'),
    ]

    fig = plt.figure(figsize=(18, 20))

    for col, (mesh, label_text, cmode) in enumerate(meshes):
        for row, (view_name, elev, azim) in enumerate(views):
            ax = fig.add_subplot(4, 3, row * 3 + col + 1, projection='3d')
            title = label_text if row == 0 else view_name
            render_mesh(ax, mesh, elev=elev, azim=azim, color_by=cmode,
                        title=title, title_size=14, target_faces=99999,
                        part_labels=part_labels)

        # Row 4: Interior cutout — clean vertical half-cut, part-coloured
        ax = fig.add_subplot(4, 3, 9 + col + 1, projection='3d')
        render_cutout(ax, mesh, cut_axis='y', cut_frac=0.5,
                      elev=15, azim=-25, color_by=cmode,
                      title="Interior Cutout", title_size=14, target_faces=99999,
                      part_labels=part_labels)

    fig.suptitle("Reference Case (Sample 00472): Optimization Type Comparison\n"
                 "SASTO-U (uniform, min = 2 voxels) vs SASTO-PA (interior min = 1 voxel)",
                 fontsize=17, fontweight='bold', y=0.99)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02,
                        wspace=0.02, hspace=0.06)

    out_path = OUT_DIR / "fig_type_comparison.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def generate_type_comparison_stl():
    """Fallback: use pre-generated STL files for type comparison."""
    import trimesh

    stl_files = {
        "Original": OPT_DIR / "original_sharp.stl",
        "SASTO-U": OPT_DIR / "optimized_v12_sharp.stl",
        "SASTO-PA": OPT_DIR / "optimized_v11_sharp.stl",
    }

    meshes = []
    for label, path in stl_files.items():
        if path.exists():
            mesh = trimesh.load(path)
            mesh = decimate_mesh(mesh, 5000)
            meshes.append((mesh, label))
        else:
            print(f"  ! Missing: {path}")

    if len(meshes) < 3:
        print("  ! Not enough STL files")
        return

    views = [("Front", 0, -90), ("Isometric", 25, -60), ("Top", 90, -90)]

    fig = plt.figure(figsize=(18, 20))
    for col, (mesh, label_text) in enumerate(meshes):
        cmode = 'original' if col == 0 else ('optimized' if col == 1 else 'sasto_pa')
        for row, (view_name, elev, azim) in enumerate(views):
            ax = fig.add_subplot(4, 3, row * 3 + col + 1, projection='3d')
            title = label_text if row == 0 else view_name
            render_mesh(ax, mesh, elev=elev, azim=azim, color_by=cmode,
                        title=title, title_size=14, target_faces=99999)

        # Row 4: Interior cutout — clean vertical half-cut
        ax = fig.add_subplot(4, 3, 9 + col + 1, projection='3d')
        render_cutout(ax, mesh, cut_axis='y', cut_frac=0.5,
                      elev=15, azim=-25, color_by=cmode,
                      title="Interior Cutout", title_size=14, target_faces=99999)

    fig.suptitle("Reference Case: Original \u2192 SASTO-U \u2192 SASTO-PA",
                 fontsize=17, fontweight='bold', y=0.99)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02,
                        wspace=0.02, hspace=0.06)

    out_path = OUT_DIR / "fig_type_comparison.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    print("=" * 60)
    print("Generating Optimized House Figures")
    print("=" * 60)

    generate_type_comparison()
    generate_gallery()

    print("\nDone!")


if __name__ == "__main__":
    main()
