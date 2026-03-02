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

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
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

    # Keep only largest mesh component
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


def render_mesh(ax, mesh, elev=25, azim=-60, color_by='height', alpha=0.95,
                max_faces=10000, title=None):
    """Render a trimesh on a matplotlib 3D axis."""
    verts = mesh.vertices.copy()
    faces = mesh.faces

    # Center
    center = verts.mean(axis=0)
    verts -= center

    # Subsample faces for rendering speed
    if len(faces) > max_faces:
        idx = np.random.RandomState(42).choice(len(faces), max_faces, replace=False)
        faces = faces[idx]

    triangles = verts[faces]

    if color_by == 'height':
        centroids_z = triangles.mean(axis=1)[:, 2]
        z_min, z_max = centroids_z.min(), centroids_z.max()
        norm_z = (centroids_z - z_min) / (z_max - z_min + 1e-10)
        colors = plt.cm.viridis(norm_z)
    elif color_by == 'original':
        colors = np.full((len(faces), 4), [0.7, 0.75, 0.8, alpha])
    elif color_by == 'optimized':
        colors = np.full((len(faces), 4), [0.2, 0.6, 0.85, alpha])
    elif color_by == 'sasto_pa':
        colors = np.full((len(faces), 4), [0.85, 0.35, 0.2, alpha])
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(faces)))

    colors[:, 3] = alpha

    poly = Poly3DCollection(triangles, facecolors=colors,
                            edgecolors='none', linewidths=0.0)
    ax.add_collection3d(poly)

    extents = np.abs(verts).max(axis=0) * 1.1
    ax.set_xlim(-extents[0], extents[0])
    ax.set_ylim(-extents[1], extents[1])
    ax.set_zlim(-extents[2], extents[2])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    if title:
        ax.set_title(title, fontsize=10, pad=-5)


def load_sample_meshes(sample_id):
    """Load baseline and optimized occupancy, convert to meshes."""
    # Baseline
    base_path = DATA_DIR / sample_id / "occ.npz"
    if not base_path.exists():
        return None, None
    base_occ = np.load(base_path)['data']

    # Optimized
    opt_path = BATCH_DIR / sample_id / "optimized_occ.npz"
    if not opt_path.exists():
        return None, None
    opt_occ = np.load(opt_path)['data']

    base_mesh = voxels_to_mesh(base_occ)
    opt_mesh = voxels_to_mesh(opt_occ)
    return base_mesh, opt_mesh


def generate_gallery():
    """Generate a gallery of 6 diverse designs (original vs optimized)."""
    print("Generating optimized house gallery...")

    # Select 6 diverse samples: 2 high-reduction, 2 mid, 2 low
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

    # Pick 6 spread across the distribution
    n = len(samples_info)
    indices = [0, 2, n//4, n//2, 3*n//4, n-5]
    selected = [samples_info[i] for i in indices]

    fig = plt.figure(figsize=(18, 20))

    for row, s in enumerate(selected):
        sid = s['sample_id']
        red_pct = s['volume_reduction_pct']
        print(f"  {sid}: {red_pct:.1f}% reduction...")

        base_mesh, opt_mesh = load_sample_meshes(sid)
        if base_mesh is None:
            continue

        n_base = int(np.load(DATA_DIR / sid / "occ.npz")['data'].sum())
        n_opt = int(np.load(BATCH_DIR / sid / "optimized_occ.npz")['data'].sum())

        # Original (left)
        ax1 = fig.add_subplot(6, 3, row * 3 + 1, projection='3d')
        render_mesh(ax1, base_mesh, color_by='original',
                    title=f"Original ({n_base:,} voxels)")

        # Optimized (middle) 
        ax2 = fig.add_subplot(6, 3, row * 3 + 2, projection='3d')
        render_mesh(ax2, opt_mesh, color_by='optimized',
                    title=f"Optimized ({n_opt:,} voxels, -{red_pct:.1f}%)")

        # Optimized isometric (right)
        ax3 = fig.add_subplot(6, 3, row * 3 + 3, projection='3d')
        render_mesh(ax3, opt_mesh, elev=15, azim=-135, color_by='height',
                    title=f"Sample {sid} (isometric)")

    fig.suptitle("Gallery of SASTO-PA Optimized Houses\n(Original → Optimized → Isometric View)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = OUT_DIR / "fig_optimized_gallery.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
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

    base_mesh = voxels_to_mesh(base_occ)
    v11_mesh = voxels_to_mesh(v11_occ)
    v12_mesh = voxels_to_mesh(v12_occ)

    # 3 columns × 3 view rows
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

    fig = plt.figure(figsize=(16, 16))

    for col, (mesh, label, cmode) in enumerate(meshes):
        for row, (view_name, elev, azim) in enumerate(views):
            ax = fig.add_subplot(3, 3, row * 3 + col + 1, projection='3d')
            title = label if row == 0 else view_name
            render_mesh(ax, mesh, elev=elev, azim=azim, color_by=cmode,
                        title=title)

    fig.suptitle("Reference Case (Sample 00472): Optimization Type Comparison\n"
                 "SASTO-U (uniform min thickness = 2 voxels) vs SASTO-PA (interior min = 1 voxel)",
                 fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = OUT_DIR / "fig_type_comparison.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
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
            meshes.append((mesh, label))
        else:
            print(f"  ! Missing: {path}")

    if len(meshes) < 3:
        print("  ! Not enough STL files")
        return

    views = [("Front", 0, -90), ("Isometric", 25, -60), ("Top", 90, -90)]

    fig = plt.figure(figsize=(16, 16))
    for col, (mesh, label) in enumerate(meshes):
        cmode = 'original' if col == 0 else ('optimized' if col == 1 else 'height')
        for row, (view_name, elev, azim) in enumerate(views):
            ax = fig.add_subplot(3, 3, row * 3 + col + 1, projection='3d')
            title = label if row == 0 else view_name
            render_mesh(ax, mesh, elev=elev, azim=azim, color_by=cmode,
                        title=title, max_faces=30000)

    fig.suptitle("Reference Case: Original → SASTO-U → SASTO-PA",
                 fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = OUT_DIR / "fig_type_comparison.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
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
