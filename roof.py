import numpy as np
from pathlib import Path
import trimesh
from collections import defaultdict
from scipy.spatial import ConvexHull

import shapely
from shapely.geometry import LineString, Polygon, MultiPolygon, GeometryCollection
from shapely.geometry.polygon import orient
from shapely.ops import unary_union, polygonize_full


SEMANTIC_INFO = {
    1: {"name": "exterior_wall", "color": [0.8, 0.6, 0.4]},
    2: {"name": "interior_room", "color": [0.7, 0.3, 0.9]},
    3: {"name": "roof", "color": [0.95, 0.75, 0.25]},
    4: {"name": "floor", "color": [0.6, 0.6, 0.6]},
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

    keep = e_new[:, 0] != e_new[:, 1]
    e_new = e_new[keep]

    a = np.minimum(e_new[:, 0], e_new[:, 1])
    b = np.maximum(e_new[:, 0], e_new[:, 1])
    key = a.astype(np.int64) * (len(v_new) + 1) + b.astype(np.int64)
    uniq_idx = np.unique(key, return_index=True)[1]
    e_new = e_new[uniq_idx]
    return v_new, e_new


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
# Roof split heuristic (more permissive, to avoid “flat plane roof”)
# ---------------------------------------------------------------------
def split_roof_edges(
    vertices,
    edges,
    roof_z_q=0.65,
    roof_nonvertical_max=0.98,
    roof_keep_vertical=True,
    roof_vertical_z_q=0.88,
    roof_vertical_ratio_min=0.85,
):
    v = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    L = np.linalg.norm(v, axis=1) + 1e-12
    vertical_ratio = np.abs(v[:, 2]) / L
    mids = vertices[edges].mean(axis=1)

    z_thr = float(np.quantile(vertices[:, 2], roof_z_q))
    roof_mask = (mids[:, 2] >= z_thr) & (vertical_ratio <= roof_nonvertical_max)

    if roof_keep_vertical:
        z_thr_v = float(np.quantile(vertices[:, 2], roof_vertical_z_q))
        roof_mask = roof_mask | ((mids[:, 2] >= z_thr_v) & (vertical_ratio >= roof_vertical_ratio_min))

    return edges[roof_mask], edges[~roof_mask]


# ---------------------------------------------------------------------
# Roof meshing (planar clustering + convex hull)  (UNCHANGED ROOF BUILDER)
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
# Exterior inference: choose largest vertical component as "exterior walls"
# ---------------------------------------------------------------------
def vertical_edge_mask(vertices, edges, vertical_cos=0.7):
    v = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    L = np.linalg.norm(v, axis=1) + 1e-12
    return (np.abs(v[:, 2]) / L) > vertical_cos


def component_xy_area(vertices, edges_subset, edge_ids):
    vids = np.unique(edges_subset[edge_ids].reshape(-1))
    pts = vertices[vids][:, :2]
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return float(hull.volume)  # 2D hull "volume" == area
    except Exception:
        return 0.0


def infer_exterior_interior_wall_edges(vertices, edges, vertical_cos=0.7):
    vmask = vertical_edge_mask(vertices, edges, vertical_cos=vertical_cos)
    wall_edges = edges[vmask]
    other_edges = edges[~vmask]

    if len(wall_edges) == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32), other_edges

    comps = edge_connected_components(wall_edges)
    if len(comps) == 1:
        return wall_edges, np.empty((0, 2), dtype=np.int32), other_edges

    areas = np.array([component_xy_area(vertices, wall_edges, c) for c in comps], dtype=np.float32)
    ext_i = int(np.argmax(areas))

    exterior_edges = wall_edges[comps[ext_i]]
    rest = [c for i, c in enumerate(comps) if i != ext_i]
    interior_edges = wall_edges[np.concatenate(rest)] if rest else np.empty((0, 2), dtype=np.int32)

    return exterior_edges, interior_edges, other_edges


# ---------------------------------------------------------------------
# Polygonize helpers
# ---------------------------------------------------------------------
def estimate_z0_z1(vertices, edges_subset):
    vids = (
        np.unique(edges_subset.reshape(-1))
        if edges_subset is not None and len(edges_subset)
        else np.arange(len(vertices))
    )
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
    hist, edgesh = np.histogram(ang, bins=bins, range=(0.0, np.pi / 2.0))
    k = int(np.argmax(hist))
    return float(0.5 * (edgesh[k] + edgesh[k + 1]))


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

    return trimesh.Trimesh(
        np.asarray(vertices, np.float32),
        np.asarray(faces, np.int64),
        process=True,
    )


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


# ---------------------------------------------------------------------
# Footprint edges from bottom edges
# ---------------------------------------------------------------------
def infer_exterior_footprint_edges(vertices, edges, z0, z_snap=0.03):
    if edges is None or len(edges) == 0:
        return np.empty((0, 2), dtype=np.int32)

    za = vertices[edges[:, 0], 2]
    zb = vertices[edges[:, 1], 2]
    bottom_mask = (np.abs(za - z0) <= z_snap) & (np.abs(zb - z0) <= z_snap)
    bottom = edges[bottom_mask]

    if len(bottom) == 0:
        return np.empty((0, 2), dtype=np.int32)

    comps = edge_connected_components(bottom)
    if len(comps) == 1:
        return bottom

    areas = np.array([component_xy_area(vertices, bottom, c) for c in comps], dtype=np.float32)
    return bottom[comps[int(np.argmax(areas))]]


# ---------------------------------------------------------------------
# Robust polygon repair
# ---------------------------------------------------------------------
def _largest_polygon(geom):
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
        return max(polys, key=lambda p: p.area, default=None)
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, Polygon)]
        return max(polys, key=lambda p: p.area, default=None)
    return None


def _repair_polygon(poly: Polygon):
    if poly is None:
        return None

    if not poly.is_valid:
        try:
            fixed = shapely.make_valid(poly, method="structure", keep_collapsed=False)
        except Exception:
            fixed = shapely.make_valid(poly)
        poly = _largest_polygon(fixed)
        if poly is None:
            return None

    try:
        poly = poly.buffer(0)
    except Exception:
        pass
    try:
        poly = orient(poly, sign=1.0)
    except Exception:
        pass

    if (not isinstance(poly, Polygon)) or (poly.area <= 1e-8):
        return None

    return poly


# ---------------------------------------------------------------------
# Floor slab (robust)
# ---------------------------------------------------------------------
def floor_slab_from_polygon(poly: Polygon, z0: float, thickness: float = 0.05, color_rgb=None):
    if color_rgb is None:
        color_rgb = SEMANTIC_INFO[4]["color"]

    poly = _repair_polygon(poly)
    if poly is None:
        return None

    try:
        slab = trimesh.creation.extrude_polygon(
            poly,
            height=float(thickness),
            engine="earcut",
            force_vertices=False,
        )
    except Exception as e:
        print(f"⚠️ Floor extrude_polygon failed: {type(e).__name__}: {e}")
        return None

    slab.apply_translation([0.0, 0.0, float(z0 - thickness)])
    colorize(slab, color_rgb)
    return slab


# ---------------------------------------------------------------------
# Roof floor cap (flat cap at wall-top z1)
# ---------------------------------------------------------------------
def roof_floor_cap_from_polygon(poly: Polygon, z1: float, engine="earcut", color_rgb=None):
    if color_rgb is None:
        color_rgb = SEMANTIC_INFO[3]["color"]

    poly = _repair_polygon(poly)
    if poly is None:
        return None

    try:
        v2, f = trimesh.creation.triangulate_polygon(poly, engine=engine, force_vertices=False)
    except Exception as e:
        print(f"⚠️ Roof-floor triangulate_polygon failed: {type(e).__name__}: {e}")
        return None

    if v2 is None or len(f) == 0:
        return None

    v3 = np.column_stack([v2.astype(np.float64), np.full((len(v2),), float(z1), dtype=np.float64)]).astype(np.float32)
    m = trimesh.Trimesh(vertices=v3, faces=f.astype(np.int64), process=True)
    colorize(m, color_rgb)
    return m


# ---------------------------------------------------------------------
# FIXED: Conform wall tops to roof (no IndexError)
# ---------------------------------------------------------------------
def conform_wall_tops_to_roof(
    exterior_mesh: trimesh.Trimesh,
    roof_mesh: trimesh.Trimesh,
    z_eps=1e-3,
    ray_start_offset=0.02,
    max_raise=10.0,
):
    """
    Raise wall-top vertices to the first roof hit along +Z.

    Uses:
      locations, index_ray, index_tri = mesh.ray.intersects_location(..., multiple_hits=False)
    where index_ray indexes into the input ray arrays. [trimesh docs]
    """
    if exterior_mesh is None or roof_mesh is None:
        return exterior_mesh
    if len(getattr(exterior_mesh, "vertices", [])) == 0 or len(getattr(roof_mesh, "faces", [])) == 0:
        return exterior_mesh

    v = exterior_mesh.vertices.copy()
    z_top = float(np.max(v[:, 2]))

    top_mask = np.abs(v[:, 2] - z_top) <= float(z_eps)
    top_vids = np.where(top_mask)[0]
    if len(top_vids) == 0:
        return exterior_mesh

    origins = v[top_vids].astype(np.float64, copy=True)
    origins[:, 2] -= float(ray_start_offset)
    directions = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float64), (len(origins), 1))

    try:
        locations, index_ray, index_tri = roof_mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False,
        )
    except Exception as e:
        print(f"⚠️ Wall-top conform skipped (ray backend missing?): {type(e).__name__}: {e}")
        return exterior_mesh

    if locations is None or len(locations) == 0:
        return exterior_mesh

    locations = np.asarray(locations, dtype=np.float64)
    index_ray = np.asarray(index_ray, dtype=np.int64)

    ok = (index_ray >= 0) & (index_ray < len(top_vids))
    if not np.any(ok):
        return exterior_mesh

    locations = locations[ok]
    index_ray = index_ray[ok]

    hit_vids = top_vids[index_ray]
    z_new = locations[:, 2].astype(np.float64)

    dz = z_new - v[hit_vids, 2].astype(np.float64)
    ok_raise = (dz > 0.0) & (dz <= float(max_raise))
    if not np.any(ok_raise):
        return exterior_mesh

    hit_vids = hit_vids[ok_raise]
    z_new = z_new[ok_raise]

    v[hit_vids, 2] = z_new.astype(v.dtype)

    exterior_mesh.vertices = v
    exterior_mesh.process(validate=True)
    return exterior_mesh


# ---------------------------------------------------------------------
# Walls + floor; also return footprint polygon + z0/z1
# ---------------------------------------------------------------------
def exterior_walls_and_floor_from_edges(vertices, non_roof_edges, ext_wall_edges, z_snap=0.03, floor_thickness=0.05):
    if non_roof_edges is None or len(non_roof_edges) == 0:
        return None, None, None, None, None

    if ext_wall_edges is not None and len(ext_wall_edges):
        z0, z1 = estimate_z0_z1(vertices, ext_wall_edges)
    else:
        z0, z1 = estimate_z0_z1(vertices, non_roof_edges)

    footprint_edges = infer_exterior_footprint_edges(vertices, non_roof_edges, z0=z0, z_snap=z_snap)
    polys = polygonize_from_edges(vertices, footprint_edges, z0=z0, z_snap=z_snap, theta=0.0)

    if not polys:
        vids = np.unique(footprint_edges.reshape(-1)) if len(footprint_edges) else np.unique(non_roof_edges.reshape(-1))
        xy = vertices[vids][:, :2].astype(np.float64)
        if len(xy) >= 3:
            hull = ConvexHull(xy)
            ring = xy[hull.vertices]
            ring = np.vstack([ring, ring[:1]])
            polys = [Polygon(ring)]

    if not polys:
        return None, None, None, z0, z1

    poly = max(polys, key=lambda p: p.area)
    poly = _repair_polygon(poly)
    if poly is None:
        return None, None, None, z0, z1

    walls = extrude_polygons([poly], z0=z0, z1=z1, color_rgb=SEMANTIC_INFO[1]["color"])
    floor = floor_slab_from_polygon(poly, z0=z0, thickness=float(floor_thickness), color_rgb=SEMANTIC_INFO[4]["color"])
    return walls, floor, poly, z0, z1


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
def process_file(
    npz_path: Path,
    output_dir: str,
    snap_tol=5e-4,

    # roof split (defaults tuned to avoid flat roofs)
    roof_z_q=0.65,
    roof_nonvertical_max=0.98,
    roof_keep_vertical=True,
    roof_vertical_z_q=0.88,
    roof_vertical_ratio_min=0.85,

    # roof meshing
    roof_z_tol=0.10,
    roof_vertical_cos=0.55,

    # exterior inference
    wall_vertical_cos=0.7,

    # polygonize/extrude tolerances
    wall_z_snap=0.03,

    # floor thickness
    floor_thickness=0.05,

    # roof floor cap + gap fix
    roof_floor_engine="earcut",
    add_roof_floor=True,
    conform_walls_to_roof=True,

    # interior
    min_edges_per_room=6,
    interior_angle_deg=8.0,
):
    print(f"\n{'='*70}\nProcessing: {npz_path.name}\n{'='*70}")

    vertices, edges = load_wireframe(npz_path)
    print(f"Loaded: vertices={len(vertices)} edges={len(edges)}")

    vertices, edges = snap_weld_vertices(vertices, edges, snap_tol=snap_tol)
    print(f"After snap_weld (tol={snap_tol}): vertices={len(vertices)} edges={len(edges)}")

    out_dir = Path(output_dir)
    base = npz_path.stem

    # 1) roof edges
    roof_edges, non_roof_edges = split_roof_edges(
        vertices,
        edges,
        roof_z_q=roof_z_q,
        roof_nonvertical_max=roof_nonvertical_max,
        roof_keep_vertical=roof_keep_vertical,
        roof_vertical_z_q=roof_vertical_z_q,
        roof_vertical_ratio_min=roof_vertical_ratio_min,
    )

    # 2) roof mesh (convex hull per planarity cluster)
    roof_mesh = planar_clustering_mesh(vertices, roof_edges, z_tol=roof_z_tol, vertical_cos=roof_vertical_cos)
    colorize(roof_mesh, SEMANTIC_INFO[3]["color"])
    if roof_mesh is None:
        print("⚠️ Roof mesh failed. Try snap_tol=1e-3, roof_z_q=0.55, roof_nonvertical_max=0.99.")

    # 3) exterior + floor + footprint polygon
    ext_wall_edges, int_wall_edges, other_edges = infer_exterior_interior_wall_edges(
        vertices, non_roof_edges, vertical_cos=wall_vertical_cos
    )

    exterior_mesh, floor_mesh, footprint_poly, z0, z1 = exterior_walls_and_floor_from_edges(
        vertices,
        non_roof_edges=non_roof_edges,
        ext_wall_edges=ext_wall_edges,
        z_snap=wall_z_snap,
        floor_thickness=floor_thickness,
    )
    if exterior_mesh is None:
        print("⚠️ Exterior mesh failed (footprint polygonize). Try snap_tol=1e-3 and/or wall_z_snap=0.05.")
    if floor_mesh is None:
        print("⚠️ Floor mesh failed (extrude_polygon). See error above; try pip install triangle and set engine='triangle' in floor_slab_from_polygon.")

    # 4) roof floor cap at wall-top z1 (fills the “roof floor vs wall top” gap)
    roof_floor_mesh = None
    if add_roof_floor and footprint_poly is not None and z1 is not None:
        roof_floor_mesh = roof_floor_cap_from_polygon(footprint_poly, z1=float(z1), engine=roof_floor_engine)
        if roof_floor_mesh is None:
            print("⚠️ Roof floor cap failed (triangulate footprint).")

    # 5) optional: conform wall tops to sloped roof surface
    if conform_walls_to_roof and exterior_mesh is not None and roof_mesh is not None:
        exterior_mesh = conform_wall_tops_to_roof(exterior_mesh, roof_mesh)

    # 6) interior rooms
    candidate_interior_edges = (
        np.concatenate([int_wall_edges, other_edges], axis=0)
        if len(int_wall_edges) or len(other_edges)
        else np.empty((0, 2), np.int32)
    )

    comps = edge_connected_components(candidate_interior_edges)
    print(f"Interior components (raw): {len(comps)}")

    room_meshes = []
    for comp in comps:
        ecomp = candidate_interior_edges[comp]
        if len(ecomp) < min_edges_per_room:
            continue
        m = interior_room_mesh_from_component(vertices, ecomp, angle_deg=interior_angle_deg, z_snap=wall_z_snap)
        if m is not None and len(m.faces) > 0:
            room_meshes.append(m)

    interior_mesh = trimesh.util.concatenate(room_meshes) if room_meshes else None
    if interior_mesh is None:
        print("⚠️ No interior rooms created.")

    # 7) complete
    parts = [m for m in (roof_mesh, roof_floor_mesh, exterior_mesh, interior_mesh, floor_mesh)
             if m is not None and len(getattr(m, "faces", [])) > 0]
    complete = trimesh.util.concatenate(parts) if parts else None

    # Export outputs
    export_mesh(roof_mesh, out_dir / f"{base}_roof.stl")
    export_mesh(roof_floor_mesh, out_dir / f"{base}_roof_floor.stl")
    export_mesh(exterior_mesh, out_dir / f"{base}_exterior_walls.stl")
    export_mesh(interior_mesh, out_dir / f"{base}_interior_rooms.stl")
    export_mesh(floor_mesh, out_dir / f"{base}_floor.stl")

    export_mesh(complete, out_dir / f"{base}_complete.stl")
    export_mesh(complete, out_dir / f"{base}_complete.ply")

    print("\n✅ Exported:", base)
    return True


def process_batch(
    input_dir="data/3dwire_raw",
    output_dir="data/3dwire_parts_combined",
    max_files=5,
    **kwargs,
):
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
    # pip install numpy scipy trimesh shapely mapbox_earcut
    # For ray casting acceleration: pip install rtree and/or embreex

    process_batch(
        input_dir="data/3dwire_raw",
        output_dir="data/3dwire_parts_combined",
        max_files=5,

        snap_tol=5e-4,          # try 1e-3 if loops are slightly open
        wall_z_snap=0.03,       # try 0.05 if Z noise higher
        floor_thickness=0.05,

        # roof selection (tuned to avoid flat roofs)
        roof_z_q=0.65,
        roof_nonvertical_max=0.98,
        roof_keep_vertical=True,
        roof_vertical_z_q=0.88,
        roof_vertical_ratio_min=0.85,

        roof_z_tol=0.10,
        roof_vertical_cos=0.55,

        wall_vertical_cos=0.7,

        # gap fixes
        add_roof_floor=True,
        roof_floor_engine="earcut",
        conform_walls_to_roof=True,

        min_edges_per_room=6,
        interior_angle_deg=8.0,
    )
