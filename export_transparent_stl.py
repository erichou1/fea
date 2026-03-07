"""
Export GLB files with transparent exterior shell for optimized models.

For SASTO-U and SASTO-PA, the exterior walls and roof are made semi-transparent
(alpha=60) so the thinned interior walls are clearly visible through the shell.

Output goes to figures/screenshot_stls/
"""

import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, binary_dilation
from skimage.measure import marching_cubes
import trimesh
import json

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "figures" / "screenshot_stls"
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

# RGB colours per part (0-255) — same as paper
PART_COLORS_RGB = {
    PART_EXTERIOR: [69, 130, 181],    # Steel blue
    PART_INTERIOR: [255, 128, 79],    # Coral/orange
    PART_ROOF:     [107, 143, 36],    # Olive green
    PART_FLOOR:    [112, 128, 143],   # Slate gray
}
DEFAULT_COLOR = [140, 140, 140]


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
        occ_padded = np.pad(occ.astype(np.float32), 1, mode='constant', constant_values=0)
        verts, faces, normals, _ = marching_cubes(occ_padded, level=0.5)
    verts -= 1.0
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    return mesh


def color_mesh_opaque(mesh, part_labels):
    """Colour mesh fully opaque by part labels."""
    face_centroids = mesh.triangles_center
    shape = np.array(part_labels.shape)
    coords = np.clip(np.round(face_centroids).astype(int), 0, shape - 1)
    labels = part_labels[coords[:, 0], coords[:, 1], coords[:, 2]]

    face_colors = np.zeros((len(labels), 4), dtype=np.uint8)
    face_colors[:, 3] = 255
    for label_val, rgb in PART_COLORS_RGB.items():
        mask = labels == label_val
        if mask.any():
            face_colors[mask, :3] = rgb
    unlabelled = ~np.isin(labels, list(PART_COLORS_RGB.keys()))
    face_colors[unlabelled, :3] = DEFAULT_COLOR
    mesh.visual.face_colors = face_colors
    return mesh


def color_mesh_transparent_shell(mesh, part_labels, shell_alpha=60, interior_alpha=255):
    """
    Colour mesh with transparent exterior/roof so interior walls are visible.
    
    - Exterior walls: alpha=shell_alpha (semi-transparent)
    - Roof: alpha=shell_alpha (semi-transparent)  
    - Interior walls: alpha=interior_alpha (opaque, bright orange)
    - Floor: alpha=255 (opaque)
    """
    face_centroids = mesh.triangles_center
    shape = np.array(part_labels.shape)
    coords = np.clip(np.round(face_centroids).astype(int), 0, shape - 1)
    labels = part_labels[coords[:, 0], coords[:, 1], coords[:, 2]]

    face_colors = np.zeros((len(labels), 4), dtype=np.uint8)
    
    # Exterior walls — semi-transparent
    mask_ext = labels == PART_EXTERIOR
    face_colors[mask_ext, :3] = PART_COLORS_RGB[PART_EXTERIOR]
    face_colors[mask_ext, 3] = shell_alpha
    
    # Interior walls — OPAQUE, bright
    mask_int = labels == PART_INTERIOR
    face_colors[mask_int, :3] = PART_COLORS_RGB[PART_INTERIOR]
    face_colors[mask_int, 3] = interior_alpha
    
    # Roof — semi-transparent
    mask_roof = labels == PART_ROOF
    face_colors[mask_roof, :3] = PART_COLORS_RGB[PART_ROOF]
    face_colors[mask_roof, 3] = shell_alpha
    
    # Floor — opaque
    mask_floor = labels == PART_FLOOR
    face_colors[mask_floor, :3] = PART_COLORS_RGB[PART_FLOOR]
    face_colors[mask_floor, 3] = 255
    
    # Unlabelled
    unlabelled = ~np.isin(labels, list(PART_COLORS_RGB.keys()))
    face_colors[unlabelled, :3] = DEFAULT_COLOR
    face_colors[unlabelled, 3] = shell_alpha
    
    mesh.visual.face_colors = face_colors
    return mesh


def decimate_mesh(mesh, target_faces=20000):
    if len(mesh.faces) <= target_faces:
        return mesh
    return mesh.simplify_quadric_decimation(face_count=target_faces)


def export_mesh(mesh, name, suffix=""):
    """Export as GLB (best color/transparency support) + PLY."""
    tag = f"{name}{suffix}"
    
    # GLB
    path_glb = OUT_DIR / f"{tag}.glb"
    mesh.export(str(path_glb), file_type='glb')
    print(f"    {path_glb.name} ({len(mesh.faces)} faces)")
    
    # PLY (also supports vertex colors + alpha)
    path_ply = OUT_DIR / f"{tag}.ply"
    mesh.export(str(path_ply), file_type='ply')


def load_occ(sample_id, kind='baseline'):
    if kind == 'baseline':
        p = DATA_DIR / sample_id / "occ.npz"
    else:
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
    if not p.exists():
        return None
    return np.load(str(p))['data']


def load_part_labels(sample_id, kind='baseline'):
    if kind == 'baseline':
        p = DATA_DIR / sample_id / "part.npz"
    else:
        for d in BATCH_DIR.iterdir():
            sp = d / "optimization_summary.json"
            if sp.exists():
                with open(sp) as f:
                    s = json.load(f)
                if s.get('sample_id') == sample_id:
                    pp = d / "fixed_part.npz"
                    if pp.exists():
                        return np.load(str(pp))['data']
                    break
        p = DATA_DIR / sample_id / "part.npz"
    if not p.exists():
        return None
    return np.load(str(p))['data']


def make_cutaway(mesh, part_labels):
    """Slice at Y midplane to reveal interior."""
    bounds = mesh.bounds
    y_mid = (bounds[0, 1] + bounds[1, 1]) / 2.0
    sliced = mesh.slice_plane([0, y_mid, 0], [0, -1, 0])
    if sliced is not None and len(sliced.faces) > 0:
        sliced = color_mesh_opaque(sliced, part_labels)
        return sliced
    return None


def main():
    print("=" * 60)
    print("Exporting Coloured + Transparent-Shell GLB/PLY Files")
    print("=" * 60)

    ref_part = np.load(OPT_DIR / "fixed_part.npz")['data']
    ref_base = np.load(OPT_DIR / "fixed_occ.npz")['data']
    ref_v11 = np.load(OPT_DIR / "optimized_occ_v11.npz")['data']
    ref_v12 = np.load(OPT_DIR / "optimized_occ_v12.npz")['data']

    # ── Reference Original: opaque coloured ──
    print("\n  REF Original (opaque, coloured)...")
    mesh_orig = voxels_to_mesh(ref_base.copy(), blur_sigma=0.4)
    mesh_orig = color_mesh_opaque(mesh_orig, ref_part)
    mesh_orig = decimate_mesh(mesh_orig, 20000)
    export_mesh(mesh_orig, "REF_original", "_colored")

    # Cutaway
    cut = make_cutaway(voxels_to_mesh(ref_base.copy(), blur_sigma=0.4), ref_part)
    if cut is not None:
        cut = decimate_mesh(cut, 20000)
        export_mesh(cut, "REF_original", "_cutaway")

    # ── Reference SASTO-U: transparent shell ──
    print("\n  REF SASTO-U (transparent shell)...")
    mesh_u = voxels_to_mesh(ref_v12.copy(), blur_sigma=0.0)
    mesh_u_trans = color_mesh_transparent_shell(mesh_u.copy(), ref_part, shell_alpha=60)
    mesh_u_trans = decimate_mesh(mesh_u_trans, 20000)
    export_mesh(mesh_u_trans, "REF_SASTO_U", "_transparent")

    # Also opaque
    mesh_u_opaque = color_mesh_opaque(voxels_to_mesh(ref_v12.copy(), blur_sigma=0.0), ref_part)
    mesh_u_opaque = decimate_mesh(mesh_u_opaque, 20000)
    export_mesh(mesh_u_opaque, "REF_SASTO_U", "_colored")

    # Cutaway
    cut_u = make_cutaway(voxels_to_mesh(ref_v12.copy(), blur_sigma=0.0), ref_part)
    if cut_u is not None:
        cut_u = decimate_mesh(cut_u, 20000)
        export_mesh(cut_u, "REF_SASTO_U", "_cutaway")

    # ── Reference SASTO-PA: transparent shell ──
    print("\n  REF SASTO-PA (transparent shell)...")
    mesh_pa = voxels_to_mesh(ref_v11.copy(), blur_sigma=0.0)
    mesh_pa_trans = color_mesh_transparent_shell(mesh_pa.copy(), ref_part, shell_alpha=60)
    mesh_pa_trans = decimate_mesh(mesh_pa_trans, 20000)
    export_mesh(mesh_pa_trans, "REF_SASTO_PA", "_transparent")

    # Also opaque
    mesh_pa_opaque = color_mesh_opaque(voxels_to_mesh(ref_v11.copy(), blur_sigma=0.0), ref_part)
    mesh_pa_opaque = decimate_mesh(mesh_pa_opaque, 20000)
    export_mesh(mesh_pa_opaque, "REF_SASTO_PA", "_colored")

    # Cutaway
    cut_pa = make_cutaway(voxels_to_mesh(ref_v11.copy(), blur_sigma=0.0), ref_part)
    if cut_pa is not None:
        cut_pa = decimate_mesh(cut_pa, 20000)
        export_mesh(cut_pa, "REF_SASTO_PA", "_cutaway")

    # ── Gallery samples ──
    gallery_ids = ["04203", "08018", "05728", "01440"]

    for sid in gallery_ids:
        print(f"\n  Sample {sid}:")
        base_occ = load_occ(sid, 'baseline')
        opt_occ = load_occ(sid, 'optimized')
        if base_occ is None or opt_occ is None:
            print(f"    SKIP: missing data")
            continue

        part = load_part_labels(sid, 'baseline')
        opt_part = load_part_labels(sid, 'optimized')
        part_for_base = part if part is not None else opt_part
        part_for_opt = opt_part if opt_part is not None else part

        if part_for_base is None:
            print(f"    SKIP: no part labels")
            continue

        # Original — opaque coloured
        print(f"    Original (opaque):")
        base_mesh = voxels_to_mesh(base_occ.copy(), blur_sigma=0.4)
        if base_mesh is not None:
            base_mesh = color_mesh_opaque(base_mesh, part_for_base)
            base_mesh = decimate_mesh(base_mesh, 20000)
            export_mesh(base_mesh, f"{sid}_original", "_colored")

        # Optimized — transparent shell
        print(f"    Optimized (transparent shell):")
        opt_mesh = voxels_to_mesh(opt_occ.copy(), blur_sigma=0.0)
        if opt_mesh is not None and part_for_opt is not None:
            opt_trans = color_mesh_transparent_shell(opt_mesh.copy(), part_for_opt, shell_alpha=60)
            opt_trans = decimate_mesh(opt_trans, 20000)
            export_mesh(opt_trans, f"{sid}_optimized", "_transparent")

            # Also opaque
            opt_opaque = color_mesh_opaque(voxels_to_mesh(opt_occ.copy(), blur_sigma=0.0), part_for_opt)
            opt_opaque = decimate_mesh(opt_opaque, 20000)
            export_mesh(opt_opaque, f"{sid}_optimized", "_colored")

            # Cutaway
            cut_opt = make_cutaway(voxels_to_mesh(opt_occ.copy(), blur_sigma=0.0), part_for_opt)
            if cut_opt is not None:
                cut_opt = decimate_mesh(cut_opt, 20000)
                export_mesh(cut_opt, f"{sid}_optimized", "_cutaway")

    print(f"\n{'='*60}")
    total_glb = len(list(OUT_DIR.glob('*.glb')))
    total_ply = len(list(OUT_DIR.glob('*.ply')))
    print(f"Done! {total_glb} GLB + {total_ply} PLY files in: {OUT_DIR}")
    print(f"{'='*60}")
    print(f"\nFor fig_model_comparison panels (c) and (d), open the *_transparent.glb files.")
    print(f"The exterior/roof are semi-transparent so you can see the thinned interior walls.")
    print(f"\nFor fig12_stl_comparison, use the *_colored.glb (original) and *_cutaway.glb (optimized).")


if __name__ == "__main__":
    main()
