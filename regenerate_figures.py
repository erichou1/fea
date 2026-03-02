#!/usr/bin/env python3
"""Regenerate all STL exports and figures with improved mesh quality.

Fixes:
  - Adds solid floor slab to all meshes (original + optimized)
  - Removes floating disconnected components (keeps largest only)
  - Fills holes and repairs normals
  - Generates cross-section style figure for v11/v12 reference case
  - Fixes diverse gallery layout (proper sizing, no overflow)
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

sys.path.insert(0, str(Path(__file__).parent / "fea_ml"))

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
STL_DIR = OUT_DIR / "stl_exports"
STL_DIR.mkdir(exist_ok=True)

FEA_ML = BASE_DIR / "fea_ml"
BATCH_DIR = FEA_ML / "runs" / "v3" / "batch_results_all"
DATA_DIR = FEA_ML / "data" / "runs_real_128"
OPT_DIR = FEA_ML / "runs" / "v3" / "optimization_128"
SIMP_JSON = FEA_ML / "runs" / "v3" / "simp_benchmark.json"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 200,
    "savefig.dpi": 200,
})


# ——————————————— Improved Mesh Utilities ———————————————

def voxels_to_mesh_clean(occ, blur_sigma=0.6, add_floor=True, keep_largest=True):
    """Convert voxel grid to clean trimesh with floor, no floating parts, no holes."""
    occ = occ.astype(bool).copy()

    # Find occupied z-range
    z_occupied = np.where(occ.any(axis=(0, 1)))[0]
    if len(z_occupied) == 0:
        return None
    z_min, z_max = z_occupied[0], z_occupied[-1]

    # Add a solid floor slab at the bottom of the occupied volume
    if add_floor:
        # Get the footprint from the lowest few occupied slices
        xy_footprint = occ[:, :, z_min:min(z_min + 4, z_max)].any(axis=2)
        # Dilate slightly for a continuous, slightly overhanging floor
        xy_footprint = binary_dilation(xy_footprint, iterations=1)
        # Fill 2 voxel-layers below the house as floor slab
        floor_start = max(0, z_min - 2)
        for dz in range(floor_start, z_min + 1):
            occ[:, :, dz] |= xy_footprint

    # Remove small disconnected voxel clusters before meshing
    if keep_largest:
        labeled, n = label(occ)
        if n > 1:
            sizes = [(labeled == i).sum() for i in range(1, n + 1)]
            main_label = np.argmax(sizes) + 1
            occ = (labeled == main_label)

    # SDF + marching cubes
    dist_in = distance_transform_edt(occ)
    dist_out = distance_transform_edt(~occ)
    sdf = dist_in - dist_out

    if blur_sigma > 0:
        sdf = gaussian_filter(sdf, sigma=blur_sigma)

    sdf_padded = np.pad(sdf, 1, mode='constant', constant_values=-1)
    verts, faces, normals, _ = marching_cubes(sdf_padded, level=0.0)
    verts -= 1.0
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # Keep only largest mesh component after marching cubes
    if keep_largest:
        components = mesh.split()
        if len(components) > 1:
            mesh = max(components, key=lambda c: len(c.faces))

    # Fill holes and fix normals
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)

    return mesh


def render_mesh(ax, mesh, elev=25, azim=-60, color_by='height', alpha=0.95,
                max_faces=8000, title=None, title_size=10):
    """Render a trimesh on a matplotlib 3D axis."""
    verts = mesh.vertices.copy()
    faces = mesh.faces

    center = verts.mean(axis=0)
    verts -= center

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
        colors = np.full((len(faces), 4), [0.15, 0.55, 0.35, alpha])
    elif color_by == 'simp':
        colors = np.full((len(faces), 4), [0.85, 0.35, 0.2, alpha])
    elif color_by == 'edge_case':
        colors = np.full((len(faces), 4), [0.9, 0.6, 0.2, alpha])
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
        ax.set_title(title, fontsize=title_size, pad=-2)


def render_cross_section(ax, mesh, cut_axis='y', cut_frac=0.5, elev=0, azim=0,
                         color_by='height', alpha=0.95, max_faces=8000, title=None,
                         title_size=10):
    """Render a cross-section view of a mesh (cut away half to show interior)."""
    verts = mesh.vertices.copy()
    faces = mesh.faces
    center = verts.mean(axis=0)
    verts -= center

    # Determine cut plane
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[cut_axis]
    cut_val = cut_frac * (verts[:, axis_idx].max() - verts[:, axis_idx].min()) + verts[:, axis_idx].min()

    # Keep only triangles whose centroid is on the visible side
    centroids = verts[faces].mean(axis=1)
    mask = centroids[:, axis_idx] > cut_val
    faces = faces[mask]

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
        colors = np.full((len(faces), 4), [0.15, 0.55, 0.35, alpha])
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
        ax.set_title(title, fontsize=title_size, pad=-2)


def load_occ(sample_id, which='baseline'):
    """Load occupancy grid."""
    if which == 'baseline':
        p = DATA_DIR / sample_id / "occ.npz"
    else:
        p = BATCH_DIR / sample_id / "optimized_occ.npz"
    if p.exists():
        return np.load(p)['data']
    return None


# ——————————————— 1. Re-export all STL files ———————————————

def export_stl_files():
    """Re-export all STL files with improved mesh quality."""
    print("\n=== Re-exporting STL files with improved quality ===")

    # Re-export reference case STLs
    print("  Reference case...")
    ref_base = np.load(OPT_DIR / "fixed_occ.npz")['data']
    ref_v11 = np.load(OPT_DIR / "optimized_occ_v11.npz")['data']
    ref_v12 = np.load(OPT_DIR / "optimized_occ_v12.npz")['data']

    for name, occ in [("ref_original", ref_base), ("ref_v11_pa", ref_v11), ("ref_v12_u", ref_v12)]:
        mesh = voxels_to_mesh_clean(occ.copy())
        path = STL_DIR / f"{name}.stl"
        mesh.export(str(path))
        comps = len(mesh.split())
        print(f"    {name}: {len(mesh.faces)} faces, {comps} component(s), watertight={mesh.is_watertight}")

    # Export gallery designs
    samples_info = []
    for d in sorted(BATCH_DIR.iterdir()):
        sp = d / "optimization_summary.json"
        op = d / "optimized_occ.npz"
        if sp.exists() and op.exists():
            with open(sp) as f:
                s = json.load(f)
            if s.get('constraints_satisfied'):
                bp = DATA_DIR / s['sample_id'] / "occ.npz"
                if bp.exists():
                    samples_info.append(s)

    samples_info.sort(key=lambda x: x['volume_reduction_pct'], reverse=True)
    n = len(samples_info)
    indices = [0, 2, n // 4, n // 2, 3 * n // 4, n - 5]
    selected = [samples_info[i] for i in indices]

    # Also add SIMP top 3
    simp_data = json.load(open(SIMP_JSON))
    simp_ids = {e['sample_id'] for e in simp_data[:3]}

    all_ids = set()
    for s in selected:
        all_ids.add(s['sample_id'])
    for sid in simp_ids:
        all_ids.add(sid)

    exported = []
    for s in samples_info:
        sid = s['sample_id']
        if sid not in all_ids:
            continue
        all_ids.discard(sid)

        red = s['volume_reduction_pct']
        print(f"  Exporting {sid} ({red:.1f}% reduction)...")

        base_occ = load_occ(sid, 'baseline')
        opt_occ = load_occ(sid, 'optimized')
        if base_occ is None or opt_occ is None:
            continue

        base_mesh = voxels_to_mesh_clean(base_occ.copy())
        opt_mesh = voxels_to_mesh_clean(opt_occ.copy())

        base_path = STL_DIR / f"{sid}_original.stl"
        opt_path = STL_DIR / f"{sid}_optimized.stl"
        base_mesh.export(str(base_path))
        opt_mesh.export(str(opt_path))

        bc = len(base_mesh.split())
        oc = len(opt_mesh.split())
        print(f"    Original: {len(base_mesh.faces)} faces, {bc} comp, watertight={base_mesh.is_watertight}")
        print(f"    Optimized: {len(opt_mesh.faces)} faces, {oc} comp, watertight={opt_mesh.is_watertight}")
        exported.append((sid, red))

    print(f"  Total STL files: {len(list(STL_DIR.glob('*.stl')))}")
    return exported


# ——————————————— 2. Cross-section figure for v11/v12 ———————————————

def generate_cross_section_figure():
    """Generate cross-section comparison: Original vs SASTO-U (v12) vs SASTO-PA (v11).

    Similar to fig_cross_sections but showing the optimization results.
    """
    print("\n=== Generating cross-section comparison figure ===")

    base_occ = np.load(OPT_DIR / "fixed_occ.npz")['data']
    v11_occ = np.load(OPT_DIR / "optimized_occ_v11.npz")['data']
    v12_occ = np.load(OPT_DIR / "optimized_occ_v12.npz")['data']

    n_b = int(base_occ.sum())
    n_u = int(v12_occ.sum())
    n_pa = int(v11_occ.sum())

    base_mesh = voxels_to_mesh_clean(base_occ.copy())
    v12_mesh = voxels_to_mesh_clean(v12_occ.copy())
    v11_mesh = voxels_to_mesh_clean(v11_occ.copy())

    # Layout: 3 columns (Original, SASTO-U, SASTO-PA) x 3 rows (isometric, front, cross-section)
    fig = plt.figure(figsize=(15, 14))

    labels = [
        f"Original\n({n_b:,} voxels)",
        f"SASTO-U (v12)\n({n_u:,} vox, $-${100*(n_b-n_u)/n_b:.1f}%)",
        f"SASTO-PA (v11)\n({n_pa:,} vox, $-${100*(n_b-n_pa)/n_b:.1f}%)",
    ]
    meshes = [base_mesh, v12_mesh, v11_mesh]
    cmodes = ['original', 'optimized', 'sasto_pa']

    row_labels = ["Isometric View", "Front Elevation", "Y-Midplane\nCross-Section"]
    view_params = [
        (25, -60),   # isometric
        (0, -90),    # front
        (15, 0),     # cross-section view angle (from the side to see the cut)
    ]

    for col in range(3):
        mesh = meshes[col]
        cmode = cmodes[col]

        # Row 1: Isometric
        ax = fig.add_subplot(3, 3, col + 1, projection='3d')
        render_mesh(ax, mesh, elev=25, azim=-60, color_by=cmode,
                    title=labels[col], title_size=11, max_faces=10000)

        # Row 2: Front elevation
        ax = fig.add_subplot(3, 3, 3 + col + 1, projection='3d')
        render_mesh(ax, mesh, elev=0, azim=-90, color_by=cmode,
                    title="Front" if col > 0 else "Front Elevation",
                    title_size=10, max_faces=10000)

        # Row 3: Cross-section (cut at Y midplane)
        ax = fig.add_subplot(3, 3, 6 + col + 1, projection='3d')
        render_cross_section(ax, mesh, cut_axis='y', cut_frac=0.3,
                             elev=15, azim=-30, color_by=cmode,
                             title="Cross-Section" if col > 0 else "Y-Midplane Section",
                             title_size=10, max_faces=10000)

    # Row labels on the left
    for i, lbl in enumerate(row_labels):
        fig.text(0.01, 0.83 - i * 0.33, lbl, fontsize=11, fontweight='bold',
                 va='center', rotation=90)

    fig.suptitle("Reference Case (Sample 00472): Optimization Type Comparison\n"
                 "Three views: isometric, front elevation, and Y-midplane cross-section",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0.04, 0, 1, 0.96])

    out_path = OUT_DIR / "fig_cross_section_comparison.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ——————————————— 3. Fixed diverse STL gallery ———————————————

def generate_diverse_stl_gallery():
    """Multi-design gallery — properly sized to fit on a page."""
    print("\n=== Generating diverse STL gallery (fixed layout) ===")

    # Pick 4 diverse feasible designs
    samples_info = []
    for d in sorted(BATCH_DIR.iterdir()):
        sp = d / "optimization_summary.json"
        op = d / "optimized_occ.npz"
        if sp.exists() and op.exists():
            with open(sp) as f:
                s = json.load(f)
            if s.get('constraints_satisfied'):
                bp = DATA_DIR / s['sample_id'] / "occ.npz"
                if bp.exists():
                    samples_info.append(s)

    samples_info.sort(key=lambda x: x['volume_reduction_pct'], reverse=True)
    n = len(samples_info)
    # 4 designs spread across the spectrum
    indices = [1, n // 4, n // 2, 3 * n // 4]
    selected = [samples_info[i] for i in indices]

    # 4 rows x 2 columns (Original | Optimized), compact
    fig = plt.figure(figsize=(10, 14))

    for row, s in enumerate(selected):
        sid = s['sample_id']
        red = s['volume_reduction_pct']
        print(f"  Row {row}: {sid} ({red:.1f}% reduction)")

        base_occ = load_occ(sid, 'baseline')
        opt_occ = load_occ(sid, 'optimized')
        if base_occ is None or opt_occ is None:
            continue

        n_base = int(base_occ.sum())
        n_opt = int(opt_occ.sum())

        base_mesh = voxels_to_mesh_clean(base_occ.copy())
        opt_mesh = voxels_to_mesh_clean(opt_occ.copy())

        # Original (left column)
        ax1 = fig.add_subplot(4, 2, row * 2 + 1, projection='3d')
        render_mesh(ax1, base_mesh, color_by='original',
                    title=f"Original ({n_base:,} vox)", title_size=9,
                    max_faces=8000)

        # Optimized (right column)
        ax2 = fig.add_subplot(4, 2, row * 2 + 2, projection='3d')
        render_mesh(ax2, opt_mesh, color_by='height',
                    title=f"Optimized ({n_opt:,} vox, $-${red:.1f}%)", title_size=9,
                    max_faces=8000)

        # Row label
        fig.text(0.01, 0.88 - row * 0.245, f"{sid}\n({red:.0f}%)",
                 fontsize=9, fontweight='bold', va='center', rotation=90)

    fig.suptitle("SASTO-PA Optimization Gallery: Original vs Optimized\n"
                 "Four designs spanning 18--45% material reduction",
                 fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0.04, 0.01, 1, 0.96])

    out_path = OUT_DIR / "fig_diverse_stl_gallery.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ——————————————— 4. Failure gallery (also fixed) ———————————————

def generate_failure_gallery():
    """Edge-case gallery: low-reduction feasible + high-reduction infeasible."""
    print("\n=== Generating failure/edge-case gallery ===")

    samples = []
    for d in sorted(BATCH_DIR.iterdir()):
        sp = d / "optimization_summary.json"
        op = d / "optimized_occ.npz"
        if sp.exists() and op.exists():
            with open(sp) as f:
                s = json.load(f)
            bp = DATA_DIR / s['sample_id'] / "occ.npz"
            if bp.exists():
                samples.append(s)

    feasible = [s for s in samples if s.get('constraints_satisfied')]
    infeasible = [s for s in samples if not s.get('constraints_satisfied')]

    feasible.sort(key=lambda x: x['volume_reduction_pct'])
    infeasible.sort(key=lambda x: x['volume_reduction_pct'], reverse=True)

    low_feasible = feasible[:3]
    high_infeasible = infeasible[:3]

    all_cases = [(s, 'Low Feasible', 'edge_case') for s in low_feasible] + \
                [(s, 'High Infeasible', 'simp') for s in high_infeasible]

    # 6 rows x 2 cols (Original | Optimized)
    fig = plt.figure(figsize=(10, 18))

    for row, (s, category, cmode) in enumerate(all_cases):
        sid = s['sample_id']
        red = s['volume_reduction_pct']
        feas = s.get('constraints_satisfied', False)
        print(f"  {category}: {sid} ({red:.1f}%, feasible={feas})")

        base_occ = load_occ(sid, 'baseline')
        opt_occ = load_occ(sid, 'optimized')
        if base_occ is None or opt_occ is None:
            continue

        base_mesh = voxels_to_mesh_clean(base_occ.copy())
        opt_mesh = voxels_to_mesh_clean(opt_occ.copy())

        n_base = int(base_occ.sum())
        n_opt = int(opt_occ.sum())

        ax1 = fig.add_subplot(6, 2, row * 2 + 1, projection='3d')
        render_mesh(ax1, base_mesh, color_by='original',
                    title=f"Original ({n_base:,} vox)", title_size=9,
                    max_faces=8000)

        status_sym = "Pass" if feas else "Fail"
        ax2 = fig.add_subplot(6, 2, row * 2 + 2, projection='3d')
        render_mesh(ax2, opt_mesh, color_by=cmode,
                    title=f"Optimized ({red:+.1f}%, {status_sym})", title_size=9,
                    max_faces=8000)

        fig.text(0.01, 0.92 - row * 0.163, f"{sid}\n{category}",
                 fontsize=8, fontweight='bold', va='center', rotation=90)

    fig.suptitle("Edge Cases: Low-Reduction Feasible (top 3) vs High-Reduction Infeasible (bottom 3)",
                 fontsize=12, fontweight='bold', y=0.99)

    # Add divider text
    fig.text(0.5, 0.505, "--- Infeasible designs below ---",
             fontsize=10, ha='center', color='red', fontstyle='italic')

    plt.tight_layout(rect=[0.05, 0, 1, 0.97])

    out_path = OUT_DIR / "fig_failure_gallery.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ——————————————— Main ———————————————

def main():
    print("=" * 60)
    print("Regenerating Figures with Improved Mesh Quality")
    print("=" * 60)

    export_stl_files()
    generate_cross_section_figure()
    generate_diverse_stl_gallery()
    generate_failure_gallery()

    print("\n" + "=" * 60)
    print("All figures regenerated!")
    print(f"STL files: {STL_DIR}")
    print(f"Figures: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
