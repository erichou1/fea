"""
HYBRID (finished, runnable)

What this does:
- Roof: optional, still built using planar clustering + convex hull (good enough for roof).
- Exterior walls: uses the OLD "correct exterior edge selection" (boundary split -> exterior_edges),
  but meshes exterior walls using polygonize+extrude (so they don't disappear).
  If polygonize fails (open loop), it falls back to convex hull footprint extrusion.
- Interior rooms: uses NEW geometric method (per component Manhattan-frame filter -> polygonize+extrude),
  which removes diagonal stray walls.

Exports per input:
- {base}_roof.(obj/stl)
- {base}_exterior_walls.(obj/stl)
- {base}_interior_rooms.(obj/stl)
- {base}_complete.(ply/stl)
- {base}_cutaway_half.(ply/stl)
- {base}_cutaway_topdown.(ply/stl)  # computed from walls-only (no roof)

Debug (optional):
- {base}_debug_roof.dxf
- {base}_debug_exterior_edges.dxf
- {base}_debug_interior_edges.dxf

Deps:
  pip install numpy scipy trimesh shapely
"""

import numpy as np
from pathlib import Path
import trimesh
from collections import defaultdict
from scipy.spatial import ConvexHull

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union, polygonize_full


SEMANTIC_INFO = {
    1: {'name': 'exterior_wall',  'color': [0.8, 0.6, 0.4]},
    2: {'name': 'interior_room',  'color': [0.7, 0.3, 0.9]},
    3: {'name': 'roof',           'color': [0.95, 0.75, 0.25]},
}


# ---------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------
def _merge_vertices_and_build_edges(line_endpoints, tol=1e-5):
    pts = line_endpoints.reshape(-1, 3)
    q = np.round(pts / tol).astype(np.int64)
    uniq_q, inv = np.unique(q, axis=0, return_inverse=True)
    vertices = (uniq_q.astype(np.float64) * tol).astype(np.float32)
    edges = inv.reshape(-1, 2).astype(np.int32)
    return vertices, edges


def load_wireframe(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    vertices = data.get("vertices", data.get("coords"))
    lines = data.get("lines", data.get("edges"))

    if vertices is None and lines is None:
        raise ValueError(f"Could not load vertices/lines from {npz_path}; keys={list(data.files)}")

    if vertices is not None and lines is not None:
        v = np.asarray(vertices)
        l = np.asarray(lines)

        if v.ndim == 2 and v.shape[1] == 3 and l.ndim == 2 and l.shape[1] == 2:
            return v.astype(np.float32, copy=False), l.astype(np.int32, copy=False)

        if l.ndim == 3 and l.shape[1:] == (2, 3):
            vv, ee = _merge_vertices_and_build_edges(l.astype(np.float32, copy=False), tol=1e-5)
            return vv, ee

        raise ValueError(f"Unsupported NPZ layout: vertices={v.shape} lines={l.shape} keys={list(data.files)}")

    if lines is not None:
        l = np.asarray(lines)
        if l.ndim == 3 and l.shape[1:] == (2, 3):
            vv, ee = _merge_vertices_and_build_edges(l.astype(np.float32, copy=False), tol=1e-5)
            return vv, ee

    raise ValueError(f"Unsupported NPZ layout: keys={list(data.files)}")


# ---------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------
def snap_weld_vertices(vertices, edges, snap_tol=5e-4):
    q = np.round(vertices / snap_tol).astype(np.int64)
    uniq_q, inv = np.unique(q, axis=0, return_inverse=True)
    v_new = (uniq_q.astype(np.float64) * snap_tol).astype(np.float32)
    e_new = inv[edges].astype(np.int32)

    # remove degenerate
    keep = e_new[:, 0] != e_new[:, 1]
    e_new = e_new[keep]

    # remove duplicates (undirected)
    a = np.minimum(e_new[:, 0], e_new[:, 1])
    b = np.maximum(e_new[:, 0], e_new[:, 1])
    key = a.astype(np.int64) * (len(v_new) + 1) + b.astype(np.int64)
    uniq_idx = np.unique(key, return_index=True)[1]
    e_new = e_new[uniq_idx]
    return v_new, e_new


# ---------------------------------------------------------------------
# Debug export
# ---------------------------------------------------------------------
def export_lines_as_dxf(vertices, edges, out_path: Path):
    if edges is None or len(edges) == 0:
        return False
    seg = vertices[edges]
    path = trimesh.load_path(seg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    path.export(str(out_path), file_type="dxf")
    return True


def colorize(mesh, rgb):
    if mesh is None or len(getattr(mesh, "faces", [])) == 0:
        return
    col = (np.array(list(rgb) + [1.0]) * 255).astype(np.uint8)
    mesh.visual.vertex_colors = np.tile(col, (len(mesh.vertices), 1))


# ---------------------------------------------------------------------
# Graph connected components
# ---------------------------------------------------------------------
def edge_connected_components(edges_subset):
    if edges_subset is None or len(edges_subset) == 0:
        return []

    v2e = defaultdict(list)
    for ei, (a, b) in enumerate(edges_subset):
        v2e[int(a)].append(ei)
        v2e[int(b)].append(ei)

    seen = np.zeros(len(edges_subset), dtype=bool)
    comps = []

    for seed in range(len(edges_subset)):
        if seen[seed]:
            continue
        stack = [seed]
        seen[seed] = True
        comp = []
        while stack:
            ei = stack.pop()
            comp.append(ei)
            a, b = edges_subset[ei]
            for v in (int(a), int(b)):
                for nei in v2e[v]:
                    if not seen[nei]:
                        seen[nei] = True
                        stack.append(nei)
        comps.append(np.array(comp, dtype=np.int32))

    return comps


# ---------------------------------------------------------------------
# Roof split heuristic
# ---------------------------------------------------------------------
def split_roof_edges(vertices, edges, roof_z_q=0.85, roof_nonvertical_max=0.65):
    v = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    L = np.linalg.norm(v, axis=1) + 1e-12
    vertical_ratio = np.abs(v[:, 2]) / L
    mids = vertices[edges].mean(axis=1)
    z_thr = float(np.quantile(vertices[:, 2], roof_z_q))
    roof_mask = (mids[:, 2] >= z_thr) & (vertical_ratio <= roof_nonvertical_max)
    return edges[roof_mask], edges[~roof_mask]


# ---------------------------------------------------------------------
# Exterior vs interior split (boundary distance)
# ---------------------------------------------------------------------
def _point_segment_dist_2d(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _point_polygon_boundary_dist_2d(p, poly):
    dmin = 1e18
    K = len(poly)
    for i in range(K):
        a = poly[i]
        b = poly[(i + 1) % K]
        dmin = min(dmin, _point_segment_dist_2d(p, a, b))
    return dmin


def boundary_split(vertices, edges, boundary_quantile=0.10, boundary_eps_min=0.02):
    if len(edges) == 0:
        return np.empty((0, 2), np.int32), np.empty((0, 2), np.int32), 0.0

    pts_xy = vertices[:, :2]
    hull = ConvexHull(pts_xy)
    poly = pts_xy[hull.vertices]

    mids = vertices[edges].mean(axis=1)
    dists = np.array([_point_polygon_boundary_dist_2d(p, poly) for p in mids[:, :2]], dtype=np.float32)

    thr = max(boundary_eps_min, float(np.quantile(dists, boundary_quantile)))
    ext = edges[dists <= thr]
    intr = edges[dists > thr]
    return ext, intr, thr


def choose_best_boundary_split(vertices, edges,
                              boundary_quantiles=(0.03, 0.05, 0.08, 0.10, 0.15, 0.20),
                              boundary_eps_min=0.02,
                              min_edges_per_room=6):
    best = None
    best_score = (-1, -1, -1)
    for q in boundary_quantiles:
        ext, intr, thr = boundary_split(vertices, edges, boundary_quantile=q, boundary_eps_min=boundary_eps_min)
        comps = edge_connected_components(intr)
        rooms = sum(1 for c in comps if len(c) >= min_edges_per_room)
        score = (rooms, len(intr), len(ext))
        if score > best_score:
            best_score = score
            best = (ext, intr, thr, q, rooms)

    ext, intr, thr, q, rooms = best
    print(f"Boundary picked q={q:.2f} thr={thr:.4f} exterior={len(ext)} interior={len(intr)} room_components={rooms}")
    return ext, intr


# ---------------------------------------------------------------------
# Roof meshing (planar clustering + convex hull)
# ---------------------------------------------------------------------
def cluster_by_planarity(vertices, edges, z_tol=0.10, vertical_cos=0.55):
    if len(edges) < 3:
        return []

    midpoints = vertices[edges].mean(axis=1)
    edge_vecs = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    edge_len = np.linalg.norm(edge_vecs, axis=1) + 1e-12
    vertical_mask = (np.abs(edge_vecs[:, 2]) / edge_len) > vertical_cos

    clusters = []

    if np.any(~vertical_mask):
        horiz_edges = np.where(~vertical_mask)[0]
        horiz_z = midpoints[horiz_edges, 2]
        order = np.argsort(horiz_z)

        current = [horiz_edges[order[0]]]
        for i in range(1, len(order)):
            eidx = horiz_edges[order[i]]
            zprev = horiz_z[order[i - 1]]
            zcur = horiz_z[order[i]]
            if abs(float(zcur - zprev)) < z_tol:
                current.append(eidx)
            else:
                if len(current) >= 3:
                    clusters.append(np.array(current, dtype=np.int32))
                current = [eidx]
        if len(current) >= 3:
            clusters.append(np.array(current, dtype=np.int32))

    if np.any(vertical_mask):
        vert_edges = np.where(vertical_mask)[0]
        if len(vert_edges) >= 3:
            clusters.append(np.array(vert_edges, dtype=np.int32))

    return clusters


def edges_to_convex_hull_mesh(vertices, edges_subset):
    vids = np.unique(edges_subset.reshape(-1))
    pts = vertices[vids]
    if len(pts) < 4:
        return None
    try:
        hull = ConvexHull(pts)
        return trimesh.Trimesh(vertices=pts, faces=hull.simplices, process=True)
    except Exception:
        return None


def planar_clustering_mesh(vertices, edges, z_tol=0.10, vertical_cos=0.55):
    clusters = cluster_by_planarity(vertices, edges, z_tol=z_tol, vertical_cos=vertical_cos)
    if not clusters:
        return None
    meshes = []
    for c in clusters:
        m = edges_to_convex_hull_mesh(vertices, edges[c])
        if m is not None and len(m.faces) > 0:
            meshes.append(m)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


# ---------------------------------------------------------------------
# Polygonize + extrude walls
# ---------------------------------------------------------------------
def estimate_z0_z1(vertices, edges_subset):
    vids = np.unique(edges_subset.reshape(-1)) if edges_subset is not None and len(edges_subset) else np.arange(len(vertices))
    z = vertices[vids, 2]
    return float(np.quantile(z, 0.10)), float(np.quantile(z, 0.90))


def bottom_edges(vertices, edges_subset, z0, z_snap=0.03):
    if edges_subset is None or len(edges_subset) == 0:
        return np.empty((0, 2), np.int32)
    za = vertices[edges_subset[:, 0], 2]
    zb = vertices[edges_subset[:, 1], 2]
    m = (np.abs(za - z0) <= z_snap) & (np.abs(zb - z0) <= z_snap)
    return edges_subset[m]


def rotate_xy(xy, theta):
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (xy @ R.T).astype(np.float64)


def estimate_manhattan_theta(vertices, edges_subset, z0, z_snap=0.03, min_len=1e-4, bins=90):
    be = bottom_edges(vertices, edges_subset, z0=z0, z_snap=z_snap)
    if len(be) < 5:
        return 0.0

    v = vertices[be[:, 1]] - vertices[be[:, 0]]
    dx = v[:, 0].astype(np.float64)
    dy = v[:, 1].astype(np.float64)
    L = np.hypot(dx, dy)

    keep = L > min_len
    dx, dy, L = dx[keep], dy[keep], L[keep]
    if len(L) < 5:
        return 0.0

    ang = np.arctan2(dy, dx)
    ang = np.mod(ang, np.pi / 2.0)
    hist, edges = np.histogram(ang, bins=bins, range=(0.0, np.pi / 2.0))
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


def filter_edges_axis_aligned_in_frame(vertices, edges_subset, theta, angle_deg=8.0, min_xy=1e-6):
    if edges_subset is None or len(edges_subset) == 0:
        return edges_subset

    v = vertices[edges_subset[:, 1]] - vertices[edges_subset[:, 0]]
    vxy = v[:, :2].astype(np.float64)
    vxy_r = rotate_xy(vxy, -theta)
    norm = np.linalg.norm(vxy_r, axis=1)

    vertical_xy = norm < min_xy

    ux = np.abs(vxy_r[:, 0]) / (norm + 1e-12)
    uy = np.abs(vxy_r[:, 1]) / (norm + 1e-12)
    c = float(np.cos(np.deg2rad(angle_deg)))
    axis = (ux >= c) | (uy >= c)

    return edges_subset[vertical_xy | axis]


def filter_edges_two_z_levels(vertices, edges_subset, z0, z1, z_snap=0.03):
    if edges_subset is None or len(edges_subset) == 0:
        return edges_subset
    za = vertices[edges_subset[:, 0], 2]
    zb = vertices[edges_subset[:, 1], 2]
    a_ok = (np.abs(za - z0) <= z_snap) | (np.abs(za - z1) <= z_snap)
    b_ok = (np.abs(zb - z0) <= z_snap) | (np.abs(zb - z1) <= z_snap)
    return edges_subset[a_ok & b_ok]


def polygonize_xy_lines(xy_vertices, edges_local):
    lines = []
    for a, b in edges_local:
        pa = xy_vertices[int(a)]
        pb = xy_vertices[int(b)]
        if np.linalg.norm(pa - pb) < 1e-9:
            continue
        lines.append(LineString([tuple(pa), tuple(pb)]))
    if len(lines) < 3:
        return []
    merged = unary_union(lines)
    polys_gc, cuts_gc, dangles_gc, invalid_gc = polygonize_full(merged)
    polys = list(getattr(polys_gc, "geoms", []))
    polys = [p for p in polys if isinstance(p, Polygon) and p.is_valid and p.area > 1e-6]
    return polys


def polygonize_from_edges(vertices, edges_subset, z0, z_snap=0.03, theta=0.0):
    be = bottom_edges(vertices, edges_subset, z0=z0, z_snap=z_snap)
    if len(be) < 3:
        return []

    vids = np.unique(be.reshape(-1))
    vid_to_local = {int(v): i for i, v in enumerate(vids)}
    xy = vertices[vids][:, :2].astype(np.float64)
    xy_r = rotate_xy(xy, -theta)

    e2 = np.array([[vid_to_local[int(a)], vid_to_local[int(b)]] for a, b in be], dtype=np.int64)
    polys = polygonize_xy_lines(xy_r, e2)

    if abs(theta) > 1e-12 and polys:
        out = []
        for p in polys:
            ext = rotate_xy(np.asarray(p.exterior.coords, dtype=np.float64), theta)
            holes = [rotate_xy(np.asarray(r.coords, dtype=np.float64), theta) for r in p.interiors]
            out.append(Polygon(ext, holes=holes))
        polys = out

    return polys


def extrude_polygon_walls(poly: Polygon, z0: float, z1: float):
    vertices = []
    faces = []

    def add_ring(coords):
        pts = np.asarray(coords, dtype=np.float64)
        if len(pts) < 4:
            return
        pts = pts[:-1]
        n = len(pts)
        base = len(vertices)

        for i in range(n):
            x, y = pts[i]
            vertices.append([x, y, z0])
            vertices.append([x, y, z1])

        for i in range(n):
            j = (i + 1) % n
            b0 = base + 2 * i
            t0 = base + 2 * i + 1
            b1 = base + 2 * j
            t1 = base + 2 * j + 1
            faces.append([b0, b1, t1])
            faces.append([b0, t1, t0])

    add_ring(poly.exterior.coords)
    for interior in poly.interiors:
        add_ring(interior.coords)

    if len(faces) == 0:
        return None

    return trimesh.Trimesh(np.asarray(vertices, np.float32),
                           np.asarray(faces, np.int64),
                           process=True)


def extrude_polygons(polys, z0, z1, color_rgb):
    meshes = []
    for p in polys:
        m = extrude_polygon_walls(p, z0=z0, z1=z1)
        if m is not None and len(m.faces) > 0:
            meshes.append(m)
    if not meshes:
        return None
    out = trimesh.util.concatenate(meshes)
    colorize(out, color_rgb)
    return out


def exterior_walls_from_exterior_edges(vertices, exterior_edges, z_snap=0.03):
    """
    Exterior (robust):
    - polygonize from bottom edges of exterior_edges
    - if polygonize fails, convex hull of bottom vertices
    - extrude largest polygon to z0..z1
    """
    if exterior_edges is None or len(exterior_edges) == 0:
        return None

    z0, z1 = estimate_z0_z1(vertices, exterior_edges)

    polys = polygonize_from_edges(vertices, exterior_edges, z0=z0, z_snap=z_snap, theta=0.0)

    if not polys:
        be = bottom_edges(vertices, exterior_edges, z0=z0, z_snap=z_snap)
        vids = np.unique(be.reshape(-1)) if len(be) else np.unique(exterior_edges.reshape(-1))
        xy = vertices[vids][:, :2].astype(np.float64)
        if len(xy) >= 3:
            hull = ConvexHull(xy)
            ring = xy[hull.vertices]
            ring = np.vstack([ring, ring[:1]])
            polys = [Polygon(ring)]
        else:
            polys = []

    if not polys:
        return None

    poly = max(polys, key=lambda p: p.area)
    return extrude_polygons([poly], z0=z0, z1=z1, color_rgb=SEMANTIC_INFO[1]["color"])


def interior_room_mesh_from_component(vertices, ecomp, angle_deg=8.0, z_snap=0.03):
    if ecomp is None or len(ecomp) == 0:
        return None

    z0, z1 = estimate_z0_z1(vertices, ecomp)
    theta = estimate_manhattan_theta(vertices, ecomp, z0=z0, z_snap=z_snap)

    e = filter_edges_axis_aligned_in_frame(vertices, ecomp, theta=theta, angle_deg=angle_deg)
    e = filter_edges_two_z_levels(vertices, e, z0=z0, z1=z1, z_snap=z_snap)

    polys = polygonize_from_edges(vertices, e, z0=z0, z_snap=z_snap, theta=theta)
    if not polys:
        return None

    poly = max(polys, key=lambda p: p.area)
    return extrude_polygons([poly], z0=z0, z1=z1, color_rgb=SEMANTIC_INFO[2]["color"])


# ---------------------------------------------------------------------
# Cutaways (slice keeps positive-normal side) [web:88]
# ---------------------------------------------------------------------
def make_cutaway_half_x(mesh, cap=False):
    if mesh is None or len(mesh.faces) == 0:
        return None
    from trimesh.intersections import slice_mesh_plane
    c = mesh.bounds.mean(axis=0)
    return slice_mesh_plane(mesh,
                            plane_normal=np.array([-1.0, 0.0, 0.0], dtype=float),
                            plane_origin=np.array([c[0], 0.0, 0.0], dtype=float),
                            cap=cap)


def make_cutaway_topdown(mesh, z_frac=0.60, keep_lower=True, cap=False):
    if mesh is None or len(mesh.faces) == 0:
        return None
    from trimesh.intersections import slice_mesh_plane
    zmin = float(mesh.bounds[0, 2])
    zmax = float(mesh.bounds[1, 2])
    zcut = zmin + float(z_frac) * (zmax - zmin)
    normal = np.array([0.0, 0.0, -1.0 if keep_lower else 1.0], dtype=float)
    return slice_mesh_plane(mesh,
                            plane_normal=normal,
                            plane_origin=np.array([0.0, 0.0, zcut], dtype=float),
                            cap=cap)


# ---------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------
def export_mesh(mesh, path: Path):
    if mesh is None or len(getattr(mesh, "faces", [])) == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return True


# ---------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------
def process_file(npz_path: Path,
                 output_dir: str,
                 snap_tol=5e-4,
                 roof_z_q=0.85,
                 roof_nonvertical_max=0.65,
                 roof_z_tol=0.10,
                 roof_vertical_cos=0.55,
                 boundary_quantiles=(0.03, 0.05, 0.08, 0.10, 0.15, 0.20),
                 boundary_eps_min=0.02,
                 min_edges_per_room=6,
                 interior_angle_deg=8.0,
                 wall_z_snap=0.03,
                 cutaway_cap=False,
                 cutaway_topdown_z_frac=0.60,
                 debug_export_lines=True):
    print(f"\n{'='*70}\nProcessing: {npz_path.name}\n{'='*70}")

    vertices, edges = load_wireframe(npz_path)
    print(f"Loaded: vertices={len(vertices)} edges={len(edges)}")

    vertices, edges = snap_weld_vertices(vertices, edges, snap_tol=snap_tol)
    print(f"After snap_weld (tol={snap_tol}): vertices={len(vertices)} edges={len(edges)}")

    out_dir = Path(output_dir)
    base = npz_path.stem

    # 1) roof split
    roof_edges, remaining_edges = split_roof_edges(
        vertices, edges,
        roof_z_q=roof_z_q,
        roof_nonvertical_max=roof_nonvertical_max
    )

    # 2) old exterior selection + interior edges
    exterior_edges, interior_edges = choose_best_boundary_split(
        vertices, remaining_edges,
        boundary_quantiles=boundary_quantiles,
        boundary_eps_min=boundary_eps_min,
        min_edges_per_room=min_edges_per_room
    )

    if debug_export_lines:
        export_lines_as_dxf(vertices, roof_edges, out_dir / f"{base}_debug_roof.dxf")
        export_lines_as_dxf(vertices, exterior_edges, out_dir / f"{base}_debug_exterior_edges.dxf")
        export_lines_as_dxf(vertices, interior_edges, out_dir / f"{base}_debug_interior_edges.dxf")

    # Roof mesh (optional)
    roof_mesh = planar_clustering_mesh(vertices, roof_edges, z_tol=roof_z_tol, vertical_cos=roof_vertical_cos)
    colorize(roof_mesh, SEMANTIC_INFO[3]["color"])

    # Exterior mesh (fixed)
    exterior_mesh = exterior_walls_from_exterior_edges(vertices, exterior_edges, z_snap=wall_z_snap)
    if exterior_mesh is None:
        print("⚠️ Exterior mesh failed. Try snap_tol=1e-3 and/or wall_z_snap=0.05.")

    # Interior rooms
    comps = edge_connected_components(interior_edges)
    print(f"Interior components (raw): {len(comps)}")

    room_meshes = []
    for comp in comps:
        ecomp = interior_edges[comp]
        if len(ecomp) < min_edges_per_room:
            continue
        m = interior_room_mesh_from_component(vertices, ecomp, angle_deg=interior_angle_deg, z_snap=wall_z_snap)
        if m is not None and len(m.faces) > 0:
            room_meshes.append(m)

    interior_mesh = trimesh.util.concatenate(room_meshes) if room_meshes else None
    if interior_mesh is None:
        print("⚠️ No interior rooms created.")

    # Complete
    parts = [m for m in (roof_mesh, exterior_mesh, interior_mesh) if m is not None and len(m.faces) > 0]
    complete = trimesh.util.concatenate(parts) if parts else None

    # Cutaways
    cutaway_half = make_cutaway_half_x(complete, cap=cutaway_cap)

    walls_only = trimesh.util.concatenate([m for m in (exterior_mesh, interior_mesh) if m is not None and len(m.faces) > 0]) \
        if (exterior_mesh is not None or interior_mesh is not None) else None

    cand_a = make_cutaway_topdown(walls_only, z_frac=cutaway_topdown_z_frac, keep_lower=True, cap=cutaway_cap)
    cand_b = make_cutaway_topdown(walls_only, z_frac=cutaway_topdown_z_frac, keep_lower=False, cap=cutaway_cap)
    fa = 0 if cand_a is None else len(cand_a.faces)
    fb = 0 if cand_b is None else len(cand_b.faces)
    cutaway_topdown = cand_a if fa >= fb else cand_b

    # Export
    export_mesh(roof_mesh, out_dir / f"{base}_roof.obj")
    export_mesh(roof_mesh, out_dir / f"{base}_roof.stl")

    export_mesh(exterior_mesh, out_dir / f"{base}_exterior_walls.obj")
    export_mesh(exterior_mesh, out_dir / f"{base}_exterior_walls.stl")

    export_mesh(interior_mesh, out_dir / f"{base}_interior_rooms.obj")
    export_mesh(interior_mesh, out_dir / f"{base}_interior_rooms.stl")

    export_mesh(complete, out_dir / f"{base}_complete.ply")
    export_mesh(complete, out_dir / f"{base}_complete.stl")

    export_mesh(cutaway_half, out_dir / f"{base}_cutaway_half.ply")
    export_mesh(cutaway_half, out_dir / f"{base}_cutaway_half.stl")

    export_mesh(cutaway_topdown, out_dir / f"{base}_cutaway_topdown.ply")
    export_mesh(cutaway_topdown, out_dir / f"{base}_cutaway_topdown.stl")

    print("\n✅ Exported:", base)
    return True


def process_batch(input_dir="data/3dwire_raw",
                  output_dir="data/3dwire_planar_parts",
                  max_files=5,
                  **kwargs):
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files in {input_dir}")

    if max_files:
        files = files[:max_files]

    print(f"\nFiles: {len(files)} | Output: {output_dir}")

    ok = 0
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}]")
        try:
            if process_file(f, output_dir, **kwargs):
                ok += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ Done: {ok}/{len(files)} succeeded")
    print(f"📁 {Path(output_dir).absolute()}")


if __name__ == "__main__":
    process_batch(
        input_dir="data/3dwire_raw",
        output_dir="data/3dwire_planar_parts",
        max_files=5,

        # If exterior still fails to close, raise this first:
        snap_tol=5e-4,            # try 1e-3

        # Roof split
        roof_z_q=0.85,
        roof_nonvertical_max=0.65,

        # Roof meshing
        roof_z_tol=0.10,
        roof_vertical_cos=0.55,

        # Interior edge selection
        boundary_quantiles=(0.03, 0.05, 0.08, 0.10, 0.15, 0.20),
        boundary_eps_min=0.02,
        min_edges_per_room=6,

        # Interior diagonal killing (tighten if needed)
        interior_angle_deg=8.0,   # try 6.0 if any diagonal survives
        wall_z_snap=0.03,         # try 0.05 if Z noise is higher

        # Cutaways
        cutaway_cap=False,
        cutaway_topdown_z_frac=0.60,

        debug_export_lines=True
    )
