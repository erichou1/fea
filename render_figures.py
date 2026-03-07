"""
Render publication-quality part-coloured house images using pyrender offscreen.

Styled to match 3dviewer.net:
  - Smooth shading (vertex colours + smooth=True)
  - Soft 3-point lighting (no harsh wash-out)
  - Light gray background
  - Higher poly count for smoother surfaces
  - Blur sigma on ALL meshes for clean marching-cubes output
  - Cutaway views for optimized models (reveals interior walls clearly)

Figures produced:
  - fig_model_comparison.png
  - fig12_stl_comparison.png
  - fig_optimized_gallery.png
  - fig_diverse_stl_gallery.png
  - fig_wireframe_pipeline.png
  - fig_type_comparison.png
  - fig_cross_section_comparison.png
"""

import json
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, binary_dilation
from skimage.measure import marching_cubes
import trimesh
import pyrender
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ──
BASE_DIR = Path(__file__).parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

FEA_ML = BASE_DIR / "fea_ml"
BATCH_DIR = FEA_ML / "runs" / "v3" / "batch_results_all"
DATA_DIR = FEA_ML / "data" / "runs_real_128"
OPT_DIR = FEA_ML / "runs" / "v3" / "optimization_128"
WIRE_DIR = BASE_DIR / "optimization" / "data" / "3dwire_raw"

# ── Part colours (0-255 RGBA) — saturated for clarity ──
PART_COLORS = {
    0: np.array([160, 160, 160, 255], dtype=np.uint8),  # unlabelled
    1: np.array([55, 125, 190, 255], dtype=np.uint8),    # exterior wall — stronger blue (more opaque feel)
    2: np.array([240, 120, 60, 255], dtype=np.uint8),    # interior wall — orange
    3: np.array([100, 160, 50, 255], dtype=np.uint8),    # roof — green
    4: np.array([190, 165, 120, 255], dtype=np.uint8),   # floor — tan
}

# 3dviewer.net-style background (light gray, not pure white)
BG_COLOR = [230, 232, 235, 255]

RENDER_W, RENDER_H = 1600, 1200
TARGET_FACES = 35000   # smooth surfaces
BLUR_SIGMA = 0.5       # blur ALL meshes for smooth marching-cubes


# ═══════════════════════════════════════════════════════════════
# Mesh generation
# ═══════════════════════════════════════════════════════════════

def voxels_to_mesh(occ, blur_sigma=BLUR_SIGMA, add_floor=True):
    """Binary voxel grid -> smooth trimesh via SDF + marching cubes."""
    occ = occ.astype(bool).copy()
    z_occ = np.where(occ.any(axis=(0, 1)))[0]
    if len(z_occ) == 0:
        return None
    z_min, z_max = z_occ[0], z_occ[-1]

    if add_floor:
        fp = occ[:, :, z_min:min(z_min + 4, z_max)].any(axis=2)
        fp = binary_dilation(fp, iterations=1)
        for dz in range(max(0, z_min - 2), z_min + 1):
            occ[:, :, dz] |= fp

    lb, n = label(occ)
    if n > 1:
        sizes = [(lb == i).sum() for i in range(1, n + 1)]
        occ = lb == (np.argmax(sizes) + 1)

    # Always use SDF for smooth surface
    sdf = distance_transform_edt(occ) - distance_transform_edt(~occ)
    sdf = gaussian_filter(sdf, sigma=max(blur_sigma, 0.3))
    sdf_pad = np.pad(sdf, 1, mode='constant', constant_values=-1)
    verts, faces, normals, _ = marching_cubes(sdf_pad, level=0.0)
    verts -= 1.0

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    return mesh


def face_colors_to_vertex_colors(mesh, face_colors):
    """
    Convert per-face colors to per-vertex colors by averaging
    surrounding face colors for each vertex. This enables smooth=True
    rendering in pyrender, which is key to the 3dviewer.net look.
    """
    n_verts = len(mesh.vertices)
    vc_sum = np.zeros((n_verts, 4), dtype=np.float64)
    vc_count = np.zeros(n_verts, dtype=np.float64)

    for fi, face in enumerate(mesh.faces):
        for vi in face:
            vc_sum[vi] += face_colors[fi].astype(np.float64)
            vc_count[vi] += 1.0

    # Avoid division by zero
    vc_count = np.maximum(vc_count, 1.0)
    vc = (vc_sum / vc_count[:, None]).astype(np.uint8)
    return vc


def dilate_part_labels(part_labels):
    """Propagate part labels into empty space via nearest-neighbor.
    This ensures mesh faces near the surface get the correct part color
    instead of label 0 (gray) when SDF smoothing expands the surface."""
    occupied = part_labels > 0
    if not occupied.any():
        return part_labels
    # distance_transform_edt with return_indices gives the coordinates
    # of the nearest occupied voxel for every empty voxel
    _, nearest_idx = distance_transform_edt(~occupied, return_indices=True)
    dilated = part_labels[nearest_idx[0], nearest_idx[1], nearest_idx[2]]
    return dilated


def color_mesh(mesh, part_labels):
    """Assign per-vertex colors from part labels (via face centroid lookup).
    Uses dilated part labels so the expanded SDF surface gets correct colors.
    Returns mesh with vertex_colors set for smooth Phong rendering."""
    dilated = dilate_part_labels(part_labels)

    centroids = mesh.triangles_center
    shape = np.array(dilated.shape)
    coords = np.clip(np.round(centroids).astype(int), 0, shape - 1)
    labels = dilated[coords[:, 0], coords[:, 1], coords[:, 2]]

    # Build face colors
    fc = np.zeros((len(labels), 4), dtype=np.uint8)
    for lbl, rgba in PART_COLORS.items():
        mask = labels == lbl
        if mask.any():
            fc[mask] = rgba

    # Convert to vertex colors for smooth Phong shading
    vc = face_colors_to_vertex_colors(mesh, fc)
    mesh.visual.vertex_colors = vc
    return mesh


def decimate(mesh, target=TARGET_FACES):
    if len(mesh.faces) <= target:
        return mesh
    return mesh.simplify_quadric_decimation(face_count=target)


def cutaway_mesh(mesh, part_labels, axis=1, fraction=0.45):
    """Slice mesh at 45% along axis to reveal interior while preserving side walls."""
    bounds = mesh.bounds
    mid = bounds[0, axis] + (bounds[1, axis] - bounds[0, axis]) * fraction
    origin = [0, 0, 0]
    normal = [0, 0, 0]
    origin[axis] = mid
    normal[axis] = 1  # keep the back half
    sliced = mesh.slice_plane(origin, normal)
    if sliced is not None and len(sliced.faces) > 0:
        sliced = color_mesh(sliced, part_labels)
        return sliced
    return None


# ═══════════════════════════════════════════════════════════════
# Rendering (3dviewer.net style)
# ═══════════════════════════════════════════════════════════════

def look_at(eye, target, up=np.array([0, 0, 1])):
    """Compute camera pose (4x4) looking from eye toward target."""
    fwd = np.array(target, dtype=float) - np.array(eye, dtype=float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0, 1, 0], dtype=float)
        right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    true_up /= np.linalg.norm(true_up)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -fwd
    pose[:3, 3] = eye
    return pose


def compute_camera_poses(mesh, distance_factor=2.0):
    """Compute standard camera poses. Slightly tighter framing than before."""
    center = mesh.centroid
    extent = mesh.extents.max()
    d = extent * distance_factor

    poses = {}
    # Isometric — classic 3/4 view like the user's screenshot
    eye = center + np.array([d * 0.65, -d * 0.75, d * 0.45])
    poses['isometric'] = look_at(eye, center)

    # Front
    eye = center + np.array([0, -d, d * 0.1])
    poses['front'] = look_at(eye, center)

    # Side
    eye = center + np.array([d, 0, d * 0.1])
    poses['side'] = look_at(eye, center)

    # Top
    eye = center + np.array([0, 0, d])
    poses['top'] = look_at(eye, center, up=np.array([0, -1, 0]))

    # Cutaway front — horizontal angle matching front view
    poses['cutaway_front'] = poses['front']

    return poses, d


# ---- 3dviewer.net-matching edge + lighting constants ----
EDGE_COLOR = (100, 100, 105)  # medium-gray edges — visible but subtle
CREASE_ANGLE = 18.0           # degrees — only major structural creases
EDGE_LINE_WIDTH = 1           # pixel width for edge lines
DEPTH_TOL = 0.8               # depth tolerance for edge visibility test


def _add_lights(scene, camera_pose):
    """3dviewer.net lighting: ambient 0x888888 + directional 0x888888 from camera.
    Both at intensity = PI (~3.14) to match the three.js convention."""
    scene.add(pyrender.DirectionalLight(
        color=[0.533, 0.533, 0.533], intensity=3.14), pose=camera_pose)


def compute_crease_edges(mesh, angle_threshold_deg=CREASE_ANGLE):
    """Compute edges where dihedral angle > threshold.
    Matches THREE.EdgesGeometry from Online3DViewer.
    Also includes boundary edges (edges with only 1 adjacent face)."""
    threshold_cos = np.cos(np.radians(angle_threshold_deg))
    fn = mesh.face_normals

    fadj = mesh.face_adjacency          # (M, 2) adjacent face pairs
    eadj = mesh.face_adjacency_edges    # (M, 2) shared vertex pairs

    dots = np.einsum('ij,ij->i', fn[fadj[:, 0]], fn[fadj[:, 1]])
    crease_mask = dots < threshold_cos
    crease_edges = eadj[crease_mask]

    # Boundary edges (belong to only 1 face)
    all_edges_set = set()
    for face in mesh.faces:
        for i in range(3):
            e = tuple(sorted([face[i], face[(i + 1) % 3]]))
            all_edges_set.add(e)
    adj_set = set(map(tuple, np.sort(eadj, axis=1).tolist()))
    boundary = all_edges_set - adj_set
    if boundary:
        b_arr = np.array(list(boundary))
        crease_edges = np.vstack([crease_edges, b_arr]) if len(crease_edges) else b_arr

    return crease_edges


def _build_projection(yfov, w, h, znear=0.05, zfar=1000.0):
    """Build OpenGL-style perspective projection matrix."""
    aspect = w / h
    f = 1.0 / np.tan(yfov / 2.0)
    P = np.zeros((4, 4))
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (zfar + znear) / (znear - zfar)
    P[2, 3] = 2.0 * zfar * znear / (znear - zfar)
    P[3, 2] = -1.0
    return P


def overlay_crease_edges(color_img, depth_buf, mesh, camera_pose,
                        w, h, yfov, edge_color=EDGE_COLOR,
                        angle_thresh=CREASE_ANGLE, depth_tol=DEPTH_TOL,
                        line_width=EDGE_LINE_WIDTH):
    """Overlay depth-tested crease edges onto the rendered image.
    Vectorised visibility check + PIL ImageDraw for clean lines."""
    crease = compute_crease_edges(mesh, angle_thresh)
    if len(crease) == 0:
        return color_img

    # ---- project every vertex to screen space ----
    view = np.linalg.inv(camera_pose)
    proj = _build_projection(yfov, w, h)

    verts = mesh.vertices
    v_hom = np.hstack([verts, np.ones((len(verts), 1))])  # (N, 4)
    v_cam = (view @ v_hom.T).T
    v_clip = (proj @ v_cam.T).T

    w_clip = v_clip[:, 3]
    valid = w_clip > 0.01
    ndc_x = np.where(valid, v_clip[:, 0] / w_clip, 0)
    ndc_y = np.where(valid, v_clip[:, 1] / w_clip, 0)
    sx = (ndc_x + 1.0) * 0.5 * w
    sy = (1.0 - ndc_y) * 0.5 * h
    cam_z = -v_cam[:, 2]  # positive = in front of camera

    # ---- vectorised edge filtering ----
    v0 = crease[:, 0]
    v1 = crease[:, 1]
    both_ok = valid[v0] & valid[v1] & (cam_z[v0] > 0) & (cam_z[v1] > 0)

    x0s, y0s, z0s = sx[v0], sy[v0], cam_z[v0]
    x1s, y1s, z1s = sx[v1], sy[v1], cam_z[v1]

    in_view = both_ok & ~(
        (np.maximum(x0s, x1s) < 0) | (np.minimum(x0s, x1s) >= w) |
        (np.maximum(y0s, y1s) < 0) | (np.minimum(y0s, y1s) >= h)
    )
    idx = np.where(in_view)[0]
    if len(idx) == 0:
        return color_img

    # Sample 5 points along each edge
    ts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    px = (x0s[idx, None] + ts * (x1s[idx, None] - x0s[idx, None])).astype(int)
    py = (y0s[idx, None] + ts * (y1s[idx, None] - y0s[idx, None])).astype(int)
    pz = z0s[idx, None] + ts * (z1s[idx, None] - z0s[idx, None])

    in_bounds = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    px_c = np.clip(px, 0, w - 1)
    py_c = np.clip(py, 0, h - 1)

    buf_z = depth_buf[py_c, px_c]
    at_surface = in_bounds & (buf_z > 0) & (np.abs(pz - buf_z) < depth_tol)
    n_vis = at_surface.sum(axis=1)
    visible_mask = n_vis >= 2
    vis_idx = idx[visible_mask]

    # Draw visible edges
    img = Image.fromarray(color_img)
    draw = ImageDraw.Draw(img)
    for ei in vis_idx:
        draw.line([(float(x0s[ei]), float(y0s[ei])),
                   (float(x1s[ei]), float(y1s[ei]))],
                  fill=edge_color, width=line_width)
    return np.array(img)


def render_mesh(mesh, camera_pose, w=RENDER_W, h=RENDER_H):
    """
    Render mesh to RGB array — 3dviewer.net style:
    smooth Phong shading + depth-tested crease-edge overlay.
    """
    yfov = np.pi / 4.5

    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene = pyrender.Scene(
        bg_color=BG_COLOR,
        ambient_light=[0.533, 0.533, 0.533],
    )
    scene.add(pr_mesh)
    camera = pyrender.PerspectiveCamera(yfov=yfov)
    scene.add(camera, pose=camera_pose)
    _add_lights(scene, camera_pose)
    renderer = pyrender.OffscreenRenderer(w, h)
    color, depth = renderer.render(scene)
    renderer.delete()

    # Overlay crease edges with depth-buffer occlusion testing
    color = overlay_crease_edges(color, depth, mesh, camera_pose, w, h, yfov)

    return color


def trim_whitespace(img_array, margin=30, bg_thresh=225):
    """Crop near-background borders from rendered image."""
    non_bg = np.where(np.any(img_array < bg_thresh, axis=2))
    if len(non_bg[0]) == 0:
        return img_array
    y_min, y_max = non_bg[0].min(), non_bg[0].max()
    x_min, x_max = non_bg[1].min(), non_bg[1].max()
    y_min = max(0, y_min - margin)
    y_max = min(img_array.shape[0], y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(img_array.shape[1], x_max + margin)
    return img_array[y_min:y_max, x_min:x_max]


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_ref():
    part = np.load(OPT_DIR / "fixed_part.npz")['data']
    occ_orig = np.load(OPT_DIR / "fixed_occ.npz")['data']
    occ_v11 = np.load(OPT_DIR / "optimized_occ_v11.npz")['data']
    occ_v12 = np.load(OPT_DIR / "optimized_occ_v12.npz")['data']
    return part, occ_orig, occ_v11, occ_v12


def load_wireframe(sample_id="00472"):
    p = WIRE_DIR / f"{sample_id}.npz"
    if not p.exists():
        print(f"  WARNING: wireframe not found: {p}")
        return None, None
    d = np.load(p, allow_pickle=True)
    return d['vertices'], d['lines']


def load_sample(sid):
    base_occ_path = DATA_DIR / sid / "occ.npz"
    base_part_path = DATA_DIR / sid / "part.npz"
    if not base_occ_path.exists():
        return None, None, None

    base_occ = np.load(str(base_occ_path))['data']
    base_part = np.load(str(base_part_path))['data'] if base_part_path.exists() else None

    opt_occ = None
    opt_part = base_part
    for d in BATCH_DIR.iterdir():
        sp = d / "optimization_summary.json"
        if sp.exists():
            with open(sp) as f:
                s = json.load(f)
            if s.get('sample_id') == sid:
                op = d / "optimized_occ.npz"
                if op.exists():
                    opt_occ = np.load(str(op))['data']
                pp = d / "fixed_part.npz"
                if pp.exists():
                    opt_part = np.load(str(pp))['data']
                break

    return base_occ, opt_occ, (opt_part if opt_part is not None else base_part)


def build_colored_mesh(occ, part, sigma=BLUR_SIGMA, faces=TARGET_FACES):
    """Build a smooth, part-coloured mesh ready for rendering."""
    mesh = voxels_to_mesh(occ.copy(), blur_sigma=sigma)
    if mesh is None:
        return None
    mesh = decimate(mesh, faces)
    mesh = color_mesh(mesh, part)
    return mesh


def build_cutaway(occ, part, sigma=BLUR_SIGMA, faces=TARGET_FACES, fraction=0.45):
    """Build a cutaway mesh: slice at 45% Y to reveal interior, preserving side walls."""
    mesh = voxels_to_mesh(occ.copy(), blur_sigma=sigma)
    if mesh is None:
        return None
    mesh = decimate(mesh, faces)
    cut = cutaway_mesh(mesh, part, fraction=fraction)
    return cut


def build_transparent_mesh(occ, part, sigma=BLUR_SIGMA, faces=TARGET_FACES, ext_alpha=150):
    """Build mesh with semi-transparent exterior walls to reveal interior.
    Returns (full_mesh, int_mesh, ext_mesh).
    full_mesh — fully coloured (for camera computation / fallback)
    int_mesh  — non-exterior faces, opaque
    ext_mesh  — exterior-wall faces, vertex alpha = ext_alpha
    """
    mesh = voxels_to_mesh(occ.copy(), blur_sigma=sigma)
    if mesh is None:
        return None, None, None
    mesh = decimate(mesh, faces)
    mesh = color_mesh(mesh, part)

    # Determine per-face part label for splitting
    dilated = dilate_part_labels(part)
    centroids = mesh.triangles_center
    shape = np.array(dilated.shape)
    coords = np.clip(np.round(centroids).astype(int), 0, shape - 1)
    labels = dilated[coords[:, 0], coords[:, 1], coords[:, 2]]

    ext_idx = np.where(labels == 1)[0]
    int_idx = np.where(labels != 1)[0]

    ext_mesh = None
    int_mesh = None

    if len(ext_idx) > 0:
        parts = mesh.submesh([ext_idx], only_watertight=False)
        ext_mesh = parts[0] if isinstance(parts, (list, tuple)) else parts
        # Lower alpha for transparency
        vc = np.array(ext_mesh.visual.vertex_colors, dtype=np.uint8).copy()
        vc[:, 3] = ext_alpha
        ext_mesh.visual.vertex_colors = vc

    if len(int_idx) > 0:
        parts = mesh.submesh([int_idx], only_watertight=False)
        int_mesh = parts[0] if isinstance(parts, (list, tuple)) else parts

    return mesh, int_mesh, ext_mesh


def render_mesh_transparent(int_mesh, ext_mesh, camera_pose,
                            w=RENDER_W, h=RENDER_H):
    """Render opaque interior + semi-transparent exterior + crease edges."""
    yfov = np.pi / 4.5

    # Render combined scene for color
    scene = pyrender.Scene(
        bg_color=BG_COLOR,
        ambient_light=[0.533, 0.533, 0.533],
    )
    if int_mesh is not None:
        pr_int = pyrender.Mesh.from_trimesh(int_mesh, smooth=True)
        scene.add(pr_int)
    if ext_mesh is not None:
        pr_ext = pyrender.Mesh.from_trimesh(ext_mesh, smooth=True)
        for prim in pr_ext.primitives:
            prim.material.alphaMode = 'BLEND'
        scene.add(pr_ext)
    camera = pyrender.PerspectiveCamera(yfov=yfov)
    scene.add(camera, pose=camera_pose)
    _add_lights(scene, camera_pose)
    renderer = pyrender.OffscreenRenderer(w, h)
    color, _ = renderer.render(scene)
    renderer.delete()

    # Overlay crease edges from interior mesh (depth-tested against interior depth)
    if int_mesh is not None:
        int_scene = pyrender.Scene(bg_color=BG_COLOR, ambient_light=[0.5, 0.5, 0.5])
        int_scene.add(pyrender.Mesh.from_trimesh(int_mesh, smooth=True))
        int_scene.add(pyrender.PerspectiveCamera(yfov=yfov), pose=camera_pose)
        r2 = pyrender.OffscreenRenderer(w, h)
        _, int_depth = r2.render(int_scene)
        r2.delete()
        color = overlay_crease_edges(color, int_depth, int_mesh, camera_pose,
                                     w, h, yfov)
    return color


# ═══════════════════════════════════════════════════════════════
# Figure generators
# ═══════════════════════════════════════════════════════════════

def add_part_legend(fig):
    legend_elements = [
        mpatches.Patch(facecolor=PART_COLORS[1][:3] / 255, edgecolor='#888',
                       linewidth=0.8, label='Exterior'),
        mpatches.Patch(facecolor=PART_COLORS[2][:3] / 255, edgecolor='#888',
                       linewidth=0.8, label='Interior'),
        mpatches.Patch(facecolor=PART_COLORS[3][:3] / 255, edgecolor='#888',
                       linewidth=0.8, label='Roof'),
        mpatches.Patch(facecolor=PART_COLORS[4][:3] / 255, edgecolor='#888',
                       linewidth=0.8, label='Floor'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=12, framealpha=0.95, bbox_to_anchor=(0.5, 0.002),
               edgecolor='#cccccc', handlelength=1.5, handletextpad=0.5,
               columnspacing=2.0)


def _wireframe_part_color(z_frac):
    """Map normalised Z height to a part colour (RGB 0-1)."""
    if z_frac < 0.15:
        return PART_COLORS[4][:3] / 255.0   # floor — tan
    elif z_frac > 0.65:
        return PART_COLORS[3][:3] / 255.0   # roof — green
    else:
        return PART_COLORS[1][:3] / 255.0   # exterior wall — blue


def render_wireframe_panel(vertices, lines, w=RENDER_W, h=RENDER_H):
    """Render wireframe with vertex dots and colour-coded parts (Z-height).
    Old-school style: dots at every vertex, edges coloured by structural part."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    dpi = 150
    # Use same gray background as pyrender renders for visual consistency
    bg = np.array(BG_COLOR[:3]) / 255.0
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi,
                     facecolor=bg)
    ax = fig.add_subplot(111, projection='3d', facecolor=bg)
    if vertices is not None and lines is not None:
        # Normalise Z for colour assignment
        z_vals = vertices[:, 2]
        z_lo, z_hi = z_vals.min(), z_vals.max()
        z_range = z_hi - z_lo if z_hi > z_lo else 1.0
        z_norm = (z_vals - z_lo) / z_range

        # Draw edges coloured by average Z of endpoints
        for line in lines:
            line = np.asarray(line).flatten()
            if len(line) >= 2:
                pts = vertices[line]
                avg_z = float(np.mean(z_norm[line]))
                color = _wireframe_part_color(avg_z)
                ax.plot3D(pts[:, 0], pts[:, 1], pts[:, 2],
                          color=color, linewidth=1.4, alpha=0.85)

        # Draw dots at every vertex, coloured by Z height — larger size
        v_colors = np.array([_wireframe_part_color(z) for z in z_norm])
        ax.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                     c=v_colors, s=30, alpha=0.95,
                     edgecolors='none', depthshade=True)

    ax.set_axis_off()
    # Match the isometric camera angle used by the 3D renders
    ax.view_init(elev=25, azim=-50)
    # Tighter framing so wireframe fills the panel — same as 3D renders
    if vertices is not None and len(vertices) > 0:
        max_range = np.ptp(vertices, axis=0).max() / 2 * 1.05
        mid = vertices.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    fig.subplots_adjust(left=-0.05, right=1.05, bottom=-0.05, top=1.05)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    data = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return data


# ── fig_model_comparison: 2×4 — solid exteriors + interior cutaways ──

def _pad_to_uniform(img, bg_color=BG_COLOR):
    """Pad a trimmed image to a fixed aspect ratio with the render bg colour
    so every cell in the grid looks identical."""
    bg = np.array(bg_color[:3], dtype=np.uint8)
    h, w = img.shape[:2]
    target_ratio = 4 / 3  # landscape
    cur_ratio = w / h
    if cur_ratio < target_ratio:
        new_w = int(h * target_ratio)
        pad = new_w - w
        left = pad // 2
        right = pad - left
        canvas = np.full((h, new_w, 3), bg, dtype=np.uint8)
        canvas[:, left:left + w] = img
    else:
        new_h = int(w / target_ratio)
        pad = new_h - h
        top = pad // 2
        canvas = np.full((new_h, w, 3), bg, dtype=np.uint8)
        canvas[top:top + h, :] = img
    return canvas


def generate_fig_model_comparison():
    print("\n=== fig_model_comparison.png ===")
    part, occ_orig, occ_v11, occ_v12 = load_ref()
    vertices, lines = load_wireframe()

    orig = build_colored_mesh(occ_orig, part)
    u = build_colored_mesh(occ_v12, part)
    pa = build_colored_mesh(occ_v11, part)

    # Cutaway versions
    orig_cut = build_cutaway(occ_orig, part)
    u_cut = build_cutaway(occ_v12, part)
    pa_cut = build_cutaway(occ_v11, part)

    # Camera from original mesh (consistent framing)
    poses, _ = compute_camera_poses(orig)
    cam_front = poses['front']
    cam_cut = poses['cutaway_front']

    bg_rgb = np.array(BG_COLOR[:3]) / 255.0

    # Layout: col 0 = wireframe (spans 2 rows, vertically centred)
    #         cols 1-3 = b/c/d with solid (row 0) + cutaway (row 1)
    fig = plt.figure(figsize=(24, 11), facecolor=bg_rgb)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.04, wspace=0.04,
                          width_ratios=[1.2, 1, 1, 1])

    # (a) wireframe — spans both rows
    ax_wire = fig.add_subplot(gs[:, 0])
    ax_wire.set_facecolor(bg_rgb); ax_wire.axis('off')
    wire_img = render_wireframe_panel(vertices, lines)
    ax_wire.imshow(trim_whitespace(wire_img))
    ax_wire.set_title('(a) 3DWire Wireframe', fontsize=16, fontweight='bold', pad=8)

    col_titles = [
        '(b) Original Volumetric',
        '(c) SASTO-U (34.3%)',
        '(d) SASTO-PA (45.0%)',
    ]
    meshes_solid = [orig, u, pa]
    meshes_cut = [orig_cut, u_cut, pa_cut]
    labels = ['Original', 'SASTO-U', 'SASTO-PA']

    for ci, (lbl, mesh_s, mesh_c, title) in enumerate(
        zip(labels, meshes_solid, meshes_cut, col_titles)
    ):
        # Row 0: solid exterior
        ax_s = fig.add_subplot(gs[0, ci + 1])
        ax_s.set_facecolor(bg_rgb); ax_s.axis('off')
        print(f"  Rendering {lbl} solid...")
        if mesh_s is not None:
            ax_s.imshow(_pad_to_uniform(trim_whitespace(render_mesh(mesh_s, cam_front))))
        ax_s.set_title(title, fontsize=16, fontweight='bold', pad=8)
        if ci == 0:
            ax_s.set_ylabel('Solid Exterior', fontsize=14, fontweight='bold',
                            rotation=90, labelpad=16)

        # Row 1: cutaway
        ax_c = fig.add_subplot(gs[1, ci + 1])
        ax_c.set_facecolor(bg_rgb); ax_c.axis('off')
        print(f"  Rendering {lbl} cutaway...")
        if mesh_c is not None:
            ax_c.imshow(_pad_to_uniform(trim_whitespace(render_mesh(mesh_c, cam_cut))))
        if ci == 0:
            ax_c.set_ylabel('Interior Cutaway', fontsize=14, fontweight='bold',
                            rotation=90, labelpad=16)

    add_part_legend(fig)
    out = FIG_DIR / "fig_model_comparison.png"
    fig.savefig(str(out), dpi=250, bbox_inches='tight',
                facecolor=bg_rgb, edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── fig12_stl_comparison: original vs SASTO-PA, 3 views ──

def generate_fig12_stl_comparison():
    """Row 1: fully solid exterior. Row 2: interior cutaway."""
    print("\n=== fig12_stl_comparison.png ===")
    part, occ_orig, occ_v11, _ = load_ref()

    orig = build_colored_mesh(occ_orig, part)
    pa = build_colored_mesh(occ_v11, part)
    orig_cut = build_cutaway(occ_orig, part)
    pa_cut = build_cutaway(occ_v11, part)

    poses, _ = compute_camera_poses(orig)
    cam_front = poses['front']
    cam_cut = poses['cutaway_front']

    bg_rgb = np.array(BG_COLOR[:3]) / 255.0
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=bg_rgb)
    for ax in axes.flat:
        ax.set_facecolor(bg_rgb)
        ax.axis('off')

    # Row 0: Solid exterior (front view)
    print("  Rendering Original solid...")
    axes[0, 0].imshow(_pad_to_uniform(trim_whitespace(render_mesh(orig, cam_front))))
    axes[0, 0].set_title('Original', fontsize=16, fontweight='bold', pad=8)
    axes[0, 0].set_ylabel('Solid Exterior', fontsize=14, fontweight='bold',
                          rotation=90, labelpad=16)

    print("  Rendering SASTO-PA solid...")
    axes[0, 1].imshow(_pad_to_uniform(trim_whitespace(render_mesh(pa, cam_front))))
    axes[0, 1].set_title('SASTO-PA Optimized', fontsize=16, fontweight='bold', pad=8)

    # Row 1: Interior cutaway
    print("  Rendering Original cutaway...")
    if orig_cut is not None:
        axes[1, 0].imshow(_pad_to_uniform(trim_whitespace(render_mesh(orig_cut, cam_cut))))
    axes[1, 0].set_ylabel('Interior Cutaway', fontsize=14, fontweight='bold',
                          rotation=90, labelpad=16)

    print("  Rendering SASTO-PA cutaway...")
    if pa_cut is not None:
        axes[1, 1].imshow(_pad_to_uniform(trim_whitespace(render_mesh(pa_cut, cam_cut))))

    add_part_legend(fig)
    plt.subplots_adjust(hspace=0.06, wspace=0.04)
    out = FIG_DIR / "fig12_stl_comparison.png"
    fig.savefig(str(out), dpi=250, bbox_inches='tight',
                facecolor=bg_rgb, edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── fig_wireframe_pipeline: 2×3 — solid + cutaway rows ──

def generate_fig_wireframe_pipeline():
    print("\n=== fig_wireframe_pipeline.png ===")
    part, occ_orig, _, _ = load_ref()
    vertices, lines = load_wireframe()

    orig_solid = build_colored_mesh(occ_orig, part)
    orig_cut = build_cutaway(occ_orig, part)

    vox = voxels_to_mesh(occ_orig.copy(), blur_sigma=0.15)
    vox = decimate(vox, TARGET_FACES)
    vox = color_mesh(vox, part)
    vox_cut = build_cutaway(occ_orig, part, sigma=0.15)

    poses, _ = compute_camera_poses(orig_solid)
    cam_iso = poses['isometric']
    cam_cut = poses['cutaway_front']

    bg_rgb = np.array(BG_COLOR[:3]) / 255.0

    # Layout: col 0 = wireframe (spans 2 rows), cols 1-2 = solid/cutaway rows
    fig = plt.figure(figsize=(22, 11), facecolor=bg_rgb)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.06, wspace=0.08,
                          width_ratios=[1.2, 1, 1])

    # (a) wireframe — spans both rows
    ax_wire = fig.add_subplot(gs[:, 0])
    ax_wire.set_facecolor(bg_rgb); ax_wire.axis('off')
    ax_wire.imshow(trim_whitespace(
        render_wireframe_panel(vertices, lines)))
    ax_wire.set_title('(a) 3DWire Wireframe', fontsize=16, fontweight='bold', pad=8)

    titles_bc = [
        '(b) Volumetric Model\n(4 part types)',
        '(c) Voxelized 128\u00b3 Grid',
    ]
    solid_meshes = [orig_solid, vox]
    cut_meshes = [orig_cut, vox_cut]
    solid_labels = ['(b)', '(c)']

    for ci, (title, ms, mc, slbl) in enumerate(
        zip(titles_bc, solid_meshes, cut_meshes, solid_labels)
    ):
        ax_s = fig.add_subplot(gs[0, ci + 1])
        ax_s.set_facecolor(bg_rgb); ax_s.axis('off')
        print(f"  Rendering {slbl} solid...")
        if ms is not None:
            ax_s.imshow(_pad_to_uniform(trim_whitespace(render_mesh(ms, cam_iso))))
        ax_s.set_title(title, fontsize=16, fontweight='bold', pad=8)
        if ci == 0:
            ax_s.set_ylabel('Solid Exterior', fontsize=14, fontweight='bold',
                            rotation=90, labelpad=16)

        ax_c = fig.add_subplot(gs[1, ci + 1])
        ax_c.set_facecolor(bg_rgb); ax_c.axis('off')
        print(f"  Rendering {slbl} cutaway...")
        if mc is not None:
            ax_c.imshow(_pad_to_uniform(trim_whitespace(render_mesh(mc, cam_cut))))
        if ci == 0:
            ax_c.set_ylabel('Interior Cutaway', fontsize=14, fontweight='bold',
                            rotation=90, labelpad=16)

    add_part_legend(fig)
    out = FIG_DIR / "fig_wireframe_pipeline.png"
    fig.savefig(str(out), dpi=250, bbox_inches='tight',
                facecolor=bg_rgb, edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── Gallery (4 houses × 4 columns: original, orig cutaway, optimized, opt cutaway) ──

def generate_gallery(filename, gallery_ids=None):
    if gallery_ids is None:
        gallery_ids = ["04203", "08018", "05728", "01440"]

    print(f"\n=== {filename} ===")
    rows_data = []
    for sid in gallery_ids:
        print(f"  Loading {sid}...")
        base_occ, opt_occ, part = load_sample(sid)
        if base_occ is None or opt_occ is None or part is None:
            print(f"    SKIP {sid}")
            continue
        orig = build_colored_mesh(base_occ, part)
        opt = build_colored_mesh(opt_occ, part)
        orig_cut = build_cutaway(base_occ, part)
        opt_cut = build_cutaway(opt_occ, part)
        if orig is not None and opt is not None:
            rows_data.append((sid, orig, orig_cut, opt, opt_cut))

    if not rows_data:
        print("  No data!"); return

    nrows = len(rows_data)
    bg_rgb = np.array(BG_COLOR[:3]) / 255.0
    fig, axes = plt.subplots(nrows, 4, figsize=(22, 5 * nrows), facecolor=bg_rgb)
    if nrows == 1:
        axes = axes[np.newaxis, :]
    for ax in axes.flat:
        ax.set_facecolor(bg_rgb)
        ax.axis('off')

    col_titles = ['Original', 'Original Cutaway', 'Optimized (SASTO-PA)', 'Optimized Cutaway']

    for row, (sid, orig, orig_cut, opt, opt_cut) in enumerate(rows_data):
        print(f"  Rendering {sid}...")
        poses, _ = compute_camera_poses(orig)
        cam_iso = poses['isometric']
        cam_cut = poses['cutaway_front']

        axes[row, 0].imshow(_pad_to_uniform(trim_whitespace(
            render_mesh(orig, cam_iso))))
        axes[row, 0].set_ylabel(f'Sample {sid}', fontsize=14, fontweight='bold',
                                rotation=90, labelpad=16)

        if orig_cut is not None:
            axes[row, 1].imshow(_pad_to_uniform(trim_whitespace(
                render_mesh(orig_cut, cam_cut))))

        axes[row, 2].imshow(_pad_to_uniform(trim_whitespace(
            render_mesh(opt, cam_iso))))

        if opt_cut is not None:
            axes[row, 3].imshow(_pad_to_uniform(trim_whitespace(
                render_mesh(opt_cut, cam_cut))))

        if row == 0:
            for c, t in enumerate(col_titles):
                axes[0, c].set_title(t, fontsize=16, fontweight='bold', pad=8)

    add_part_legend(fig)
    plt.subplots_adjust(hspace=0.03, wspace=0.02)
    out = FIG_DIR / filename
    fig.savefig(str(out), dpi=250, bbox_inches='tight',
                facecolor=bg_rgb, edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── fig_type_comparison (original / SASTO-U / SASTO-PA, 4 views) ──

def generate_fig_type_comparison():
    print("\n=== fig_type_comparison.png ===")
    part, occ_orig, occ_v11, occ_v12 = load_ref()

    orig = build_colored_mesh(occ_orig, part)
    u = build_colored_mesh(occ_v12, part)
    pa = build_colored_mesh(occ_v11, part)

    # Cutaway versions for row 4
    orig_cut = build_cutaway(occ_orig, part)
    u_cut = build_cutaway(occ_v12, part)
    pa_cut = build_cutaway(occ_v11, part)

    poses, _ = compute_camera_poses(orig)
    viewpoints = ['front', 'isometric', 'top']
    view_labels = ['Front', 'Isometric', 'Top']

    meshes = [orig, u, pa]
    col_titles = ['Original', 'SASTO-U', 'SASTO-PA']

    bg_rgb = np.array(BG_COLOR[:3]) / 255.0
    fig, axes = plt.subplots(4, 3, figsize=(15, 19), facecolor=bg_rgb)
    for ax in axes.flat:
        ax.set_facecolor(bg_rgb)
        ax.axis('off')

    for col, (mesh, title) in enumerate(zip(meshes, col_titles)):
        for row, (vp, vlabel) in enumerate(zip(viewpoints, view_labels)):
            cam = poses[vp]
            print(f"  {title} {vlabel}...")
            axes[row, col].imshow(_pad_to_uniform(trim_whitespace(
                render_mesh(mesh, cam))))
            if col == 0:
                axes[row, col].set_ylabel(vlabel, fontsize=14, fontweight='bold',
                                          rotation=90, labelpad=16)
        axes[0, col].set_title(title, fontsize=16, fontweight='bold', pad=8)

    # Row 4: interior cutaway
    cam_cut = poses['cutaway_front']
    for col, (cut_mesh, title) in enumerate(zip(
        [orig_cut, u_cut, pa_cut],
        col_titles,
    )):
        if cut_mesh is not None:
            print(f"  {title} cutaway...")
            axes[3, col].imshow(_pad_to_uniform(trim_whitespace(
                render_mesh(cut_mesh, cam_cut))))
        if col == 0:
            axes[3, col].set_ylabel('Interior\nCutaway', fontsize=14,
                                    fontweight='bold', rotation=90, labelpad=16)

    add_part_legend(fig)
    plt.subplots_adjust(hspace=0.04, wspace=0.02)
    out = FIG_DIR / "fig_type_comparison.png"
    fig.savefig(str(out), dpi=250, bbox_inches='tight',
                facecolor=bg_rgb, edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── fig_cross_section_comparison ──

def generate_fig_cross_section_comparison():
    """Single-row cutaway comparison — straight-on view into the cut face."""
    print("\n=== fig_cross_section_comparison.png ===")
    part, occ_orig, occ_v11, occ_v12 = load_ref()

    # Build solid (for camera reference) and cutaway meshes
    orig_solid = build_colored_mesh(occ_orig, part)
    labels = ['Original', 'SASTO-U', 'SASTO-PA']
    occ_list = [occ_orig, occ_v12, occ_v11]
    cuts = [build_cutaway(occ, part) for occ in occ_list]

    poses, _ = compute_camera_poses(orig_solid)
    cam_cut = poses['cutaway_front']

    bg_rgb = np.array(BG_COLOR[:3]) / 255.0
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor=bg_rgb)
    for ax in axes.flat:
        ax.set_facecolor(bg_rgb)
        ax.axis('off')

    for col, (cut_mesh, label_text) in enumerate(zip(cuts, labels)):
        if cut_mesh is None:
            continue
        print(f"  {label_text} cutaway...")
        axes[col].imshow(_pad_to_uniform(trim_whitespace(
            render_mesh(cut_mesh, cam_cut))))
        axes[col].set_title(label_text, fontsize=16, fontweight='bold', pad=8)

    add_part_legend(fig)
    plt.subplots_adjust(wspace=0.02)
    out = FIG_DIR / "fig_cross_section_comparison.png"
    fig.savefig(str(out), dpi=250, bbox_inches='tight',
                facecolor=bg_rgb, edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── fig_failure_gallery: edge cases (low-reduction feasible + high-reduction infeasible) ──

def generate_fig_failure_gallery():
    """Show edge cases with pyrender: 2 low-reduction feasible + 3 high-reduction infeasible.
    Layout: 5 rows × 3 cols (Original, Optimized, Optimized Cutaway)."""
    print("\n=== fig_failure_gallery.png ===")

    # Scan all batch results
    samples = []
    for d in sorted(BATCH_DIR.iterdir()):
        sp = d / "optimization_summary.json"
        op = d / "optimized_occ.npz"
        if sp.exists() and op.exists():
            with open(sp) as f:
                s = json.load(f)
            bp = DATA_DIR / s['sample_id'] / "occ.npz"
            pp = DATA_DIR / s['sample_id'] / "part.npz"
            if bp.exists() and pp.exists():
                samples.append(s)

    feasible = [s for s in samples if s.get('constraints_satisfied')]
    infeasible = [s for s in samples if not s.get('constraints_satisfied')]
    feasible.sort(key=lambda x: x['volume_reduction_pct'])
    infeasible.sort(key=lambda x: x['volume_reduction_pct'], reverse=True)

    low_feasible = feasible[:2]   # only 2 low-reduction feasible rows
    high_infeasible = infeasible[:3]
    all_cases = [(s, 'Low Feasible') for s in low_feasible] + \
                [(s, 'High Infeasible') for s in high_infeasible]

    if not all_cases:
        print("  No edge-case data found!"); return

    nrows = len(all_cases)
    bg_rgb = np.array(BG_COLOR[:3]) / 255.0
    fig, axes = plt.subplots(nrows, 3, figsize=(18, 4.5 * nrows), facecolor=bg_rgb)
    if nrows == 1:
        axes = axes[np.newaxis, :]
    for ax in axes.flat:
        ax.set_facecolor(bg_rgb)
        ax.axis('off')

    col_titles = ['Original', 'Optimized', 'Optimized Cutaway']

    for row, (s, category) in enumerate(all_cases):
        sid = s['sample_id']
        red = s['volume_reduction_pct']
        feas = s.get('constraints_satisfied', False)
        print(f"  {category}: {sid} ({red:.1f}%, feasible={feas})")


        base_occ = np.load(str(DATA_DIR / sid / "occ.npz"))['data']
        part = np.load(str(DATA_DIR / sid / "part.npz"))['data']

        # Find optimized occ
        opt_occ = None
        for d in BATCH_DIR.iterdir():
            sp = d / "optimization_summary.json"
            if sp.exists():
                with open(sp) as f:
                    si = json.load(f)
                if si.get('sample_id') == sid:
                    op = d / "optimized_occ.npz"
                    if op.exists():
                        opt_occ = np.load(str(op))['data']
                    break
        if opt_occ is None:
            continue

        orig = build_colored_mesh(base_occ, part)
        opt = build_colored_mesh(opt_occ, part)
        opt_cut = build_cutaway(opt_occ, part)

        if orig is None or opt is None:
            continue

        poses, _ = compute_camera_poses(orig)
        cam_iso = poses['isometric']
        cam_front = poses['front']
        cam_cut = poses['cutaway_front']

        # Col 1: Original — use front camera so exterior walls are clearly visible
        axes[row, 0].imshow(_pad_to_uniform(trim_whitespace(
            render_mesh(orig, cam_front))))
        status = "\u2713 Feas." if feas else "\u2717 Infeas."
        axes[row, 0].set_ylabel(f'{sid}\n{red:+.1f}%\n{status}',
                                fontsize=14, fontweight='bold',
                                rotation=90, labelpad=18)

        # Col 2: Optimized solid
        axes[row, 1].imshow(_pad_to_uniform(trim_whitespace(
            render_mesh(opt, cam_iso))))

        # Col 3: Optimized cutaway
        if opt_cut is not None:
            axes[row, 2].imshow(_pad_to_uniform(trim_whitespace(
                render_mesh(opt_cut, cam_cut))))

        if row == 0:
            for c, t in enumerate(col_titles):
                axes[0, c].set_title(t, fontsize=16, fontweight='bold', pad=8)

    # Divider annotations
    mid_row = len(low_feasible) - 0.5
    if nrows > 3:
        fig.text(0.5, 1.0 - (mid_row + 0.5) / nrows,
                 '\u2014 Low Reduction Feasible \u2191  |  High Reduction Infeasible \u2193 \u2014',
                 ha='center', fontsize=14, fontweight='bold', color='#666')

    add_part_legend(fig)
    plt.subplots_adjust(hspace=0.06, wspace=0.03)
    out = FIG_DIR / "fig_failure_gallery.png"
    fig.savefig(str(out), dpi=250, bbox_inches='tight',
                facecolor=bg_rgb, edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Rendering Figures (3dviewer.net style)")
    print("=" * 60)

    generate_fig_model_comparison()
    generate_fig12_stl_comparison()
    generate_fig_wireframe_pipeline()
    generate_gallery("fig_optimized_gallery.png")
    generate_gallery("fig_diverse_stl_gallery.png",
                     gallery_ids=["00137", "11357", "06149", "00857"])
    generate_fig_type_comparison()
    generate_fig_cross_section_comparison()
    generate_fig_failure_gallery()

    print("\n" + "=" * 60)
    print(f"Done! Figures in: {FIG_DIR}")
    print("=" * 60)
