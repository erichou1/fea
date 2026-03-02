#!/usr/bin/env python3
"""Regenerate all STL exports and figures with improved mesh quality.

Fixes v2:
  - Uses mesh decimation (not random subsampling) -> solid renders, no dots
  - Adds interior cutout views showing optimized wall structure
  - Larger figures filling whitespace, bigger fonts
  - Proper page-fitting for galleries
  - Floor slabs, no floating parts, watertight STLs
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

# Part labels (from fea_ml/fea_ml/geometry/voxelize.py)
PART_EMPTY = 0
PART_EXTERIOR = 1   # exterior wall
PART_INTERIOR = 2   # interior wall
PART_ROOF = 3
PART_FLOOR = 4

# RGBA colours per structural part
PART_COLORS = {
    PART_EXTERIOR: [0.27, 0.51, 0.71, 1.0],   # Steel blue
    PART_INTERIOR: [1.00, 0.50, 0.31, 1.0],    # Coral/orange
    PART_ROOF:     [0.42, 0.56, 0.14, 1.0],    # Olive green
    PART_FLOOR:    [0.44, 0.50, 0.56, 1.0],    # Slate gray
}

# Larger base font sizes
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})


# --------------- Improved Mesh Utilities ---------------

def voxels_to_mesh_clean(occ, blur_sigma=0.6, add_floor=True, keep_largest=True):
    """Convert voxel grid to clean trimesh with floor, no floating parts, no holes."""
    occ = occ.astype(bool).copy()

    z_occupied = np.where(occ.any(axis=(0, 1)))[0]
    if len(z_occupied) == 0:
        return None
    z_min, z_max = z_occupied[0], z_occupied[-1]

    if add_floor:
        xy_footprint = occ[:, :, z_min:min(z_min + 4, z_max)].any(axis=2)
        xy_footprint = binary_dilation(xy_footprint, iterations=1)
        floor_start = max(0, z_min - 2)
        for dz in range(floor_start, z_min + 1):
            occ[:, :, dz] |= xy_footprint

    if keep_largest:
        labeled, n = label(occ)
        if n > 1:
            sizes = [(labeled == i).sum() for i in range(1, n + 1)]
            main_label = np.argmax(sizes) + 1
            occ = (labeled == main_label)

    dist_in = distance_transform_edt(occ)
    dist_out = distance_transform_edt(~occ)
    sdf = dist_in - dist_out
    if blur_sigma > 0:
        sdf = gaussian_filter(sdf, sigma=blur_sigma)

    sdf_padded = np.pad(sdf, 1, mode='constant', constant_values=-1)
    verts, faces, normals, _ = marching_cubes(sdf_padded, level=0.0)
    verts -= 1.0
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    if keep_largest and len(mesh.faces) < 500000:
        components = mesh.split()
        if len(components) > 1:
            mesh = max(components, key=lambda c: len(c.faces))

    trimesh.repair.fill_holes(mesh)
    if len(mesh.faces) < 200000:
        try:
            trimesh.repair.fix_normals(mesh)
        except Exception:
            pass

    return mesh


def decimate_mesh(mesh, target_faces=10000):
    """Decimate mesh to target face count using quadric decimation.
    This preserves surface coverage (no dots) unlike random subsampling."""
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        decimated = mesh.simplify_quadric_decimation(face_count=target_faces)
        if len(decimated.faces) > 0:
            return decimated
    except Exception as e:
        print(f"    Decimation failed: {e}")
    return mesh


def get_colors(faces, triangles, color_by, alpha):
    """Generate face colors (flat mode — used when no part labels available)."""
    n = len(faces)
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
    elif color_by == 'simp':
        colors = np.full((n, 4), [0.80, 0.30, 0.25, alpha])
    elif color_by == 'edge_case':
        colors = np.full((n, 4), [0.90, 0.60, 0.20, alpha])
    elif color_by == 'cutout':
        colors = np.full((n, 4), [0.78, 0.75, 0.72, alpha])
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n))
    colors[:, 3] = alpha
    return colors


def get_part_face_colors(centroids_voxel, part_labels, alpha=0.95):
    """Map face centroids (in voxel coordinate space) to per-part RGBA colours."""
    shape = np.array(part_labels.shape)
    coords = np.clip(np.round(centroids_voxel).astype(int), 0, shape - 1)
    labels = part_labels[coords[:, 0], coords[:, 1], coords[:, 2]]

    colors = np.full((len(labels), 4), [0.55, 0.55, 0.55, alpha])
    for label_val, rgba in PART_COLORS.items():
        mask = labels == label_val
        if mask.any():
            colors[mask] = rgba
    colors[:, 3] = alpha
    return colors


def render_mesh(ax, mesh, elev=25, azim=-60, color_by='height', alpha=0.95,
                target_faces=10000, title=None, title_size=13, part_labels=None):
    """Render a trimesh on a matplotlib 3D axis.  If *part_labels* is
    provided, faces are coloured by structural part; otherwise uses *color_by*."""
    render_m = decimate_mesh(mesh, target_faces)
    verts = render_m.vertices.copy()
    faces = render_m.faces

    # Part colours in original voxel space (before centering)
    if part_labels is not None:
        centroids_orig = verts[faces].mean(axis=1)
        colors = get_part_face_colors(centroids_orig, part_labels, alpha)
    else:
        colors = None

    center = verts.mean(axis=0)
    verts -= center

    triangles = verts[faces]
    if colors is None:
        colors = get_colors(faces, triangles, color_by, alpha)

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
    """Render a cutout view — use trimesh.slice_plane for a clean planar cut.
    If *part_labels* is provided, faces are coloured by structural part."""
    render_m = decimate_mesh(mesh, target_faces)

    # --- clean planar slice (no shards) ---
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[cut_axis]
    v_min = render_m.vertices[:, axis_idx].min()
    v_max = render_m.vertices[:, axis_idx].max()
    cut_val = v_min + cut_frac * (v_max - v_min)

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

    center = render_m.vertices.mean(axis=0)
    verts -= center

    triangles = verts[faces]
    if colors is None:
        colors = get_colors(faces, triangles, color_by, alpha)

    edge_colors = colors.copy()
    edge_colors[:, :3] *= 0.80
    edge_colors[:, 3] = 0.4

    poly = Poly3DCollection(triangles, facecolors=colors,
                            edgecolors=edge_colors, linewidths=0.15)
    ax.add_collection3d(poly)

    extents = np.abs(verts).max(axis=0) * 1.05
    ax.set_xlim(-extents[0], extents[0])
    ax.set_ylim(-extents[1], extents[1])
    ax.set_zlim(-extents[2], extents[2])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    if title:
        ax.set_title(title, fontsize=title_size, pad=2, fontweight='bold')


def load_occ(sample_id, which='baseline'):
    """Load occupancy grid."""
    if which == 'baseline':
        p = DATA_DIR / sample_id / "occ.npz"
    else:
        p = BATCH_DIR / sample_id / "optimized_occ.npz"
    if p.exists():
        return np.load(p)['data']
    return None


# --------------- 1. Re-export all STL files ---------------

def export_stl_files():
    """Re-export all STL files with improved mesh quality."""
    print("\n=== Re-exporting STL files with improved quality ===")

    print("  Reference case...")
    ref_base = np.load(OPT_DIR / "fixed_occ.npz")['data']
    ref_v11 = np.load(OPT_DIR / "optimized_occ_v11.npz")['data']
    ref_v12 = np.load(OPT_DIR / "optimized_occ_v12.npz")['data']

    for name, occ in [("ref_original", ref_base), ("ref_v11_pa", ref_v11), ("ref_v12_u", ref_v12)]:
        mesh = voxels_to_mesh_clean(occ.copy())
        path = STL_DIR / f"{name}.stl"
        mesh.export(str(path))
        comps = len(mesh.split())
        print(f"    {name}: {len(mesh.faces)} faces, {comps} comp, watertight={mesh.is_watertight}")

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

    simp_data = json.load(open(SIMP_JSON))
    simp_ids = {e['sample_id'] for e in simp_data[:3]}

    all_ids = set()
    for s in selected:
        all_ids.add(s['sample_id'])
    for sid in simp_ids:
        all_ids.add(sid)

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

        base_mesh = voxels_to_mesh_clean(base_occ.copy(), blur_sigma=0.4)
        opt_mesh = voxels_to_mesh_clean(opt_occ.copy(), blur_sigma=0.0)

        base_mesh.export(str(STL_DIR / f"{sid}_original.stl"))
        opt_mesh.export(str(STL_DIR / f"{sid}_optimized.stl"))

        print(f"    Orig: {len(base_mesh.faces)} faces, Opt: {len(opt_mesh.faces)} faces")

    print(f"  Total STL files: {len(list(STL_DIR.glob('*.stl')))}")


# --------------- 2. Cross-section figure (BIGGER, fills page) ---------------

def generate_cross_section_figure():
    """Generate cross-section comparison: Original vs SASTO-U vs SASTO-PA.
    3 columns x 3 rows (isometric, front, interior cutout). Large figure."""
    print("\n=== Generating cross-section comparison figure ===")

    base_occ = np.load(OPT_DIR / "fixed_occ.npz")['data']
    v11_occ = np.load(OPT_DIR / "optimized_occ_v11.npz")['data']
    v12_occ = np.load(OPT_DIR / "optimized_occ_v12.npz")['data']

    n_b = int(base_occ.sum())
    n_u = int(v12_occ.sum())
    n_pa = int(v11_occ.sum())

    # Load part labels for the reference case
    part_labels = None
    part_path = OPT_DIR / "fixed_part.npz"
    if part_path.exists():
        part_labels = np.load(part_path)['data']

    # Pre-decimate meshes ONCE
    # Low blur for optimised meshes preserves holes / thinned walls
    FACES = 5000
    print("  Pre-decimating meshes...")
    base_mesh = decimate_mesh(voxels_to_mesh_clean(base_occ.copy(), blur_sigma=0.4), FACES)
    v12_mesh = decimate_mesh(voxels_to_mesh_clean(v12_occ.copy(), blur_sigma=0.0), FACES)
    v11_mesh = decimate_mesh(voxels_to_mesh_clean(v11_occ.copy(), blur_sigma=0.0), FACES)
    print(f"  Decimated to: {len(base_mesh.faces)}, {len(v12_mesh.faces)}, {len(v11_mesh.faces)} faces")

    # LARGE figure to fill page
    fig = plt.figure(figsize=(18, 16))

    col_labels = [
        f"Original\n({n_b:,} voxels)",
        f"SASTO-U\n({n_u:,} vox, $-${100*(n_b-n_u)/n_b:.1f}%)",
        f"SASTO-PA\n({n_pa:,} vox, $-${100*(n_b-n_pa)/n_b:.1f}%)",
    ]
    meshes = [base_mesh, v12_mesh, v11_mesh]
    cmodes = ['original', 'optimized', 'sasto_pa']

    for col in range(3):
        mesh = meshes[col]
        cmode = cmodes[col]

        # Row 1: Isometric view
        ax = fig.add_subplot(3, 3, col + 1, projection='3d')
        render_mesh(ax, mesh, elev=25, azim=-60, color_by=cmode,
                    title=col_labels[col], title_size=15, target_faces=99999,
                    part_labels=part_labels)

        # Row 2: Front elevation
        ax = fig.add_subplot(3, 3, 3 + col + 1, projection='3d')
        render_mesh(ax, mesh, elev=5, azim=-90, color_by=cmode,
                    title="Front Elevation" if col == 0 else "Front",
                    title_size=14, target_faces=99999,
                    part_labels=part_labels)

        # Row 3: Interior cutout — vertical half-cut, part-coloured
        ax = fig.add_subplot(3, 3, 6 + col + 1, projection='3d')
        render_cutout(ax, mesh, cut_axis='y', cut_frac=0.5,
                      elev=15, azim=-25, color_by=cmode,
                      title="Interior Cutout" if col == 0 else "Cutout",
                      title_size=14, target_faces=99999,
                      part_labels=part_labels)

    # Row labels
    row_names = ["Isometric\nView", "Front\nElevation", "Interior\nCutout"]
    for i, lbl in enumerate(row_names):
        fig.text(0.01, 0.82 - i * 0.32, lbl, fontsize=14, fontweight='bold',
                 va='center', rotation=90)

    fig.suptitle("Reference Case (Sample 00472): Optimization Type Comparison\n"
                 "Isometric, front elevation, and interior cutout views",
                 fontsize=18, fontweight='bold', y=0.99)
    plt.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.02,
                        wspace=0.02, hspace=0.08)

    out_path = OUT_DIR / "fig_cross_section_comparison.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# --------------- 3. Diverse STL gallery (fits page, 3 columns) ---------------

def generate_diverse_stl_gallery():
    """Gallery: Original | Optimized | Interior Cutout for 4 designs.
    3 columns, 4 rows, properly sized to fit one page."""
    print("\n=== Generating diverse STL gallery ===")

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
    indices = [1, n // 4, n // 2, 3 * n // 4]
    selected = [samples_info[i] for i in indices]

    # 4 rows x 3 columns
    fig = plt.figure(figsize=(16, 18))

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

        # Low blur for optimised mesh → visible holes / thinning
        base_mesh = voxels_to_mesh_clean(base_occ.copy(), blur_sigma=0.4)
        opt_mesh = voxels_to_mesh_clean(opt_occ.copy(), blur_sigma=0.0)

        # Load part labels
        part_labels = None
        part_path = DATA_DIR / sid / "part.npz"
        if part_path.exists():
            part_labels = np.load(part_path)['data']

        # Pre-decimate once per design (avoid repeated expensive operation)
        FACES = 5000
        base_mesh = decimate_mesh(base_mesh, FACES)
        opt_mesh = decimate_mesh(opt_mesh, FACES)

        # Col 1: Original — coloured by structural part
        ax1 = fig.add_subplot(4, 3, row * 3 + 1, projection='3d')
        t1 = "Original" if row == 0 else ""
        render_mesh(ax1, base_mesh, color_by='original', part_labels=part_labels,
                    title=f"{t1}\n{n_base:,} vox" if row == 0 else f"{n_base:,} vox",
                    title_size=13, target_faces=99999)

        # Col 2: Optimized — coloured by part (holes/thinning visible)
        ax2 = fig.add_subplot(4, 3, row * 3 + 2, projection='3d')
        t2 = "Optimized" if row == 0 else ""
        render_mesh(ax2, opt_mesh, color_by='height', part_labels=part_labels,
                    title=f"{t2}\n{n_opt:,} vox ($-${red:.1f}%)" if row == 0 else f"{n_opt:,} vox ($-${red:.1f}%)",
                    title_size=13, target_faces=99999)

        # Col 3: Interior cutout — clean vertical half-cut, part-coloured
        ax3 = fig.add_subplot(4, 3, row * 3 + 3, projection='3d')
        t3 = "Interior Cutout" if row == 0 else ""
        render_cutout(ax3, opt_mesh, cut_axis='y', cut_frac=0.5,
                      elev=15, azim=-25, part_labels=part_labels,
                      title=f"{t3}\nSample {sid}" if row == 0 else f"Sample {sid}",
                      title_size=13, target_faces=99999)

        # Row label
        fig.text(0.01, 0.88 - row * 0.235, f"{sid}\n({red:.0f}%)",
                 fontsize=12, fontweight='bold', va='center', rotation=90)

    fig.suptitle("SASTO-PA Optimization Gallery\nOriginal  |  Optimized  |  Interior Cutout",
                 fontsize=17, fontweight='bold', y=0.99)
    plt.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.02,
                        wspace=0.02, hspace=0.06)

    out_path = OUT_DIR / "fig_diverse_stl_gallery.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# --------------- 4. Failure/edge-case gallery ---------------

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

    feasible = sorted([s for s in samples if s.get('constraints_satisfied')],
                      key=lambda x: x['volume_reduction_pct'])
    infeasible = sorted([s for s in samples if not s.get('constraints_satisfied')],
                        key=lambda x: x['volume_reduction_pct'], reverse=True)

    low_feasible = feasible[:3]
    high_infeasible = infeasible[:3]

    all_cases = [(s, 'Low Feasible', 'edge_case') for s in low_feasible] + \
                [(s, 'High Infeasible', 'simp') for s in high_infeasible]

    # 6 rows x 3 cols (Original | Optimized | Cutout)
    fig = plt.figure(figsize=(16, 22))

    for row, (s, category, cmode) in enumerate(all_cases):
        sid = s['sample_id']
        red = s['volume_reduction_pct']
        feas = s.get('constraints_satisfied', False)
        print(f"  {category}: {sid} ({red:.1f}%, feasible={feas})")

        base_occ = load_occ(sid, 'baseline')
        opt_occ = load_occ(sid, 'optimized')
        if base_occ is None or opt_occ is None:
            continue

        # No blur for optimised mesh → visible holes / thinning
        base_mesh = voxels_to_mesh_clean(base_occ.copy(), blur_sigma=0.4)
        opt_mesh = voxels_to_mesh_clean(opt_occ.copy(), blur_sigma=0.0)

        # Load part labels
        part_labels_local = None
        part_path = DATA_DIR / sid / "part.npz"
        if part_path.exists():
            part_labels_local = np.load(part_path)['data']

        # Pre-decimate once per design
        FACES = 4000
        base_mesh = decimate_mesh(base_mesh, FACES)
        opt_mesh = decimate_mesh(opt_mesh, FACES)

        n_base = int(base_occ.sum())
        status_sym = "PASS" if feas else "FAIL"

        ax1 = fig.add_subplot(6, 3, row * 3 + 1, projection='3d')
        render_mesh(ax1, base_mesh, color_by='original', part_labels=part_labels_local,
                    title=f"Original ({n_base:,})" if row == 0 else f"{n_base:,} vox",
                    title_size=12, target_faces=99999)

        ax2 = fig.add_subplot(6, 3, row * 3 + 2, projection='3d')
        render_mesh(ax2, opt_mesh, color_by=cmode, part_labels=part_labels_local,
                    title=f"Optimized ({red:+.1f}%, {status_sym})",
                    title_size=12, target_faces=99999)

        ax3 = fig.add_subplot(6, 3, row * 3 + 3, projection='3d')
        render_cutout(ax3, opt_mesh, cut_axis='y', cut_frac=0.5,
                      elev=15, azim=-25, part_labels=part_labels_local,
                      title=f"Interior ({sid})",
                      title_size=12, target_faces=99999)

        fig.text(0.01, 0.91 - row * 0.155, f"{sid}\n{category}",
                 fontsize=10, fontweight='bold', va='center', rotation=90)

    fig.suptitle("Edge Cases: Low-Reduction Feasible (top 3) vs High-Reduction Infeasible (bottom 3)",
                 fontsize=15, fontweight='bold', y=0.99)

    fig.text(0.5, 0.51, "--- Infeasible designs below ---",
             fontsize=13, ha='center', color='red', fontstyle='italic')

    plt.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.02,
                        wspace=0.02, hspace=0.06)

    out_path = OUT_DIR / "fig_failure_gallery.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# --------------- Main ---------------

def main():
    print("=" * 60)
    print("Regenerating Figures with Improved Mesh Quality (v2)")
    print("  - Solid renders via mesh decimation (no dots)")
    print("  - Interior cutout views")
    print("  - Bigger figures and fonts")
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
