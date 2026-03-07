"""
Export part-coloured STL files + cutaway (sliced) STLs for viewer screenshots.

Colours match the paper figures:
  Exterior wall = Steel blue   (69, 130, 181)
  Interior wall = Coral/orange (255, 128, 79)
  Roof          = Olive green  (107, 143, 36)
  Floor         = Slate gray   (112, 128, 143)

Output goes to figures/stl_exports_colored/
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, binary_dilation
from skimage.measure import marching_cubes
import trimesh

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "figures" / "stl_exports_colored"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEA_ML = BASE_DIR / "fea_ml"
BATCH_DIR = FEA_ML / "runs" / "v3" / "batch_results_all"
DATA_DIR = FEA_ML / "data" / "runs_real_128"
OPT_DIR = FEA_ML / "runs" / "v3" / "optimization_128"

# Part labels
PART_EMPTY = 0
PART_EXTERIOR = 1
PART_INTERIOR = 2
PART_ROOF = 3
PART_FLOOR = 4

# RGB colours per part (0-255)
PART_COLORS_RGB = {
    PART_EXTERIOR: [69, 130, 181],    # Steel blue
    PART_INTERIOR: [255, 128, 79],    # Coral/orange
    PART_ROOF:     [107, 143, 36],    # Olive green
    PART_FLOOR:    [112, 128, 143],   # Slate gray
}
DEFAULT_COLOR = [140, 140, 140]  # grey for unlabelled


def voxels_to_mesh(occ, blur_sigma=0.0, add_floor=True):
    """Convert binary voxel grid to trimesh."""
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

    labeled, n = label(occ)
    if n > 1:
        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
        main_label = np.argmax(sizes) + 1
        occ = (labeled == main_label)

    if blur_sigma > 0:
        dist_in = distance_transform_edt(occ)
        dist_out = distance_transform_edt(~occ)
        sdf = dist_in - dist_out
        sdf = gaussian_filter(sdf, sigma=blur_sigma)
        sdf_padded = np.pad(sdf, 1, mode='constant', constant_values=-1)
        verts, faces, normals, _ = marching_cubes(sdf_padded, level=0.0)
    else:
        # Fast path: skip distance transform, just use binary volume directly
        occ_padded = np.pad(occ.astype(np.float32), 1, mode='constant', constant_values=0)
        verts, faces, normals, _ = marching_cubes(occ_padded, level=0.5)
    verts -= 1.0
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    return mesh


def color_mesh_by_parts(mesh, part_labels):
    """Assign per-face vertex colours based on part_labels voxel grid."""
    face_centroids = mesh.triangles_center
    shape = np.array(part_labels.shape)
    coords = np.clip(np.round(face_centroids).astype(int), 0, shape - 1)
    labels = part_labels[coords[:, 0], coords[:, 1], coords[:, 2]]

    # Build per-face colors (RGBA, 0-255)
    face_colors = np.zeros((len(labels), 4), dtype=np.uint8)
    face_colors[:, 3] = 255  # opaque
    for label_val, rgb in PART_COLORS_RGB.items():
        mask = labels == label_val
        if mask.any():
            face_colors[mask, :3] = rgb
    # unlabelled faces
    unlabelled = ~np.isin(labels, list(PART_COLORS_RGB.keys()))
    face_colors[unlabelled, :3] = DEFAULT_COLOR

    mesh.visual.face_colors = face_colors
    return mesh


def decimate_mesh(mesh, target_faces=20000):
    """Decimate mesh while preserving colours."""
    if len(mesh.faces) <= target_faces:
        return mesh
    decimated = mesh.simplify_quadric_decimation(face_count=target_faces)
    return decimated


def load_occ(sample_id, kind='baseline'):
    """Load occupancy grid."""
    if kind == 'baseline':
        p = DATA_DIR / sample_id / "occ.npz"
    else:
        candidates = list(BATCH_DIR.glob(f"*/{sample_id}"))
        if not candidates:
            # Try finding via optimization_summary
            for d in BATCH_DIR.iterdir():
                sp = d / "optimization_summary.json"
                if sp.exists():
                    with open(sp) as f:
                        s = json.load(f)
                    if s.get('sample_id') == sample_id:
                        p = d / "optimized_occ.npz"
                        break
            else:
                return None
        else:
            p = candidates[0] / "optimized_occ.npz" if (candidates[0] / "optimized_occ.npz").exists() else None
            if p is None:
                return None
    if not p.exists():
        return None
    return np.load(str(p))['data']


def find_opt_dir(sample_id):
    """Find batch result directory for a sample."""
    for d in BATCH_DIR.iterdir():
        sp = d / "optimization_summary.json"
        if sp.exists():
            with open(sp) as f:
                s = json.load(f)
            if s.get('sample_id') == sample_id:
                return d
    return None


def load_part_labels(sample_id, kind='baseline'):
    """Load part labels for a sample."""
    if kind == 'baseline':
        p = DATA_DIR / sample_id / "part.npz"
    else:
        opt_d = find_opt_dir(sample_id)
        if opt_d is None:
            return None
        p = opt_d / "fixed_part.npz"
        if not p.exists():
            # Fall back to baseline part labels
            p = DATA_DIR / sample_id / "part.npz"
    if not p.exists():
        return None
    return np.load(str(p))['data']


def export_colored_and_cutaway(name, mesh, part_labels, target_faces=20000):
    """Export coloured full + cutaway in GLB format (colors work in all viewers)."""
    print(f"    Colouring and decimating...")

    # Colour the mesh
    mesh = color_mesh_by_parts(mesh, part_labels)

    # Decimate
    mesh = decimate_mesh(mesh, target_faces)

    # Export coloured full as GLB (best color support)
    path_glb = OUT_DIR / f"{name}_colored.glb"
    mesh.export(str(path_glb), file_type='glb')
    print(f"    Saved: {path_glb.name} ({len(mesh.faces)} faces)")

    # Create cutaway — slice at 50% along Y axis
    bounds = mesh.bounds
    y_mid = (bounds[0, 1] + bounds[1, 1]) / 2.0
    plane_origin = [0, y_mid, 0]
    plane_normal = [0, -1, 0]  # keep the +Y half (back half)

    sliced = mesh.slice_plane(plane_origin, plane_normal)
    if sliced is not None and len(sliced.faces) > 0:
        # Re-colour cutaway faces using part labels
        sliced = color_mesh_by_parts(sliced, part_labels)
        path_cut = OUT_DIR / f"{name}_cutaway.glb"
        sliced.export(str(path_cut), file_type='glb')
        print(f"    Saved: {path_cut.name} ({len(sliced.faces)} faces)")
    else:
        print(f"    WARNING: cutaway slice produced empty mesh for {name}")


def main():
    print("=" * 60)
    print("Exporting Part-Coloured + Cutaway STL Files")
    print("=" * 60)

    # ---- Reference case (type comparison) ----
    print("\n--- Reference case (type comparison) ---")
    ref_part = None
    ref_part_path = OPT_DIR / "fixed_part.npz"
    if ref_part_path.exists():
        ref_part = np.load(ref_part_path)['data']
        print(f"  Loaded ref part labels: {ref_part.shape}")

    ref_base = np.load(OPT_DIR / "fixed_occ.npz")['data']
    ref_v11 = np.load(OPT_DIR / "optimized_occ_v11.npz")['data']
    ref_v12 = np.load(OPT_DIR / "optimized_occ_v12.npz")['data']

    for tag, occ, sigma in [("ref_original", ref_base, 0.4),
                             ("ref_v12_u", ref_v12, 0.0),
                             ("ref_v11_pa", ref_v11, 0.0)]:
        print(f"  {tag}...")
        mesh = voxels_to_mesh(occ.copy(), blur_sigma=sigma)
        if mesh is None:
            print(f"    SKIP: empty mesh")
            continue
        if ref_part is not None:
            export_colored_and_cutaway(tag, mesh, ref_part)
        else:
            print(f"    No part labels, exporting grey")
            mesh = decimate_mesh(mesh, 20000)
            mesh.export(str(OUT_DIR / f"{tag}_colored.stl"))

    # ---- Gallery samples ----
    print("\n--- Gallery samples ---")
    gallery_ids = ["04203", "08018", "05728", "01440"]

    for sid in gallery_ids:
        print(f"\n  Sample {sid}:")

        # Original
        base_occ = load_occ(sid, 'baseline')
        opt_occ = load_occ(sid, 'optimized')
        if base_occ is None or opt_occ is None:
            print(f"    SKIP: missing data")
            continue

        base_part = load_part_labels(sid, 'baseline')
        opt_part = load_part_labels(sid, 'optimized')

        # Use whichever part labels are available
        part_for_base = base_part if base_part is not None else opt_part
        part_for_opt = opt_part if opt_part is not None else base_part

        base_mesh = voxels_to_mesh(base_occ.copy(), blur_sigma=0.4)
        opt_mesh = voxels_to_mesh(opt_occ.copy(), blur_sigma=0.0)

        if base_mesh is not None and part_for_base is not None:
            print(f"  Original:")
            export_colored_and_cutaway(f"{sid}_original", base_mesh, part_for_base)
        elif base_mesh is not None:
            print(f"  Original (no part labels, grey):")
            base_mesh = decimate_mesh(base_mesh, 20000)
            base_mesh.export(str(OUT_DIR / f"{sid}_original_colored.stl"))

        if opt_mesh is not None and part_for_opt is not None:
            print(f"  Optimized:")
            export_colored_and_cutaway(f"{sid}_optimized", opt_mesh, part_for_opt)
        elif opt_mesh is not None:
            print(f"  Optimized (no part labels, grey):")
            opt_mesh = decimate_mesh(opt_mesh, 20000)
            opt_mesh.export(str(OUT_DIR / f"{sid}_optimized_colored.stl"))

    print(f"\n{'='*60}")
    total = len(list(OUT_DIR.glob('*.glb')))
    print(f"Done! {total} GLB files in: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
