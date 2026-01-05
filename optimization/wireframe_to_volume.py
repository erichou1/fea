"""
3DWire: ConvexHull pitched roof + wireframe-derived eave outline + walls/floor/rooms

STRICT FEA EXPORT POLICY (Gmsh-friendly) + ROOF THICKNESS + ATTIC FIX

Key behaviors:

- Roof thickness: if roof is a closed watertight solid, try to hollow it to thickness using boolean (outer-inner).
  If that hollowing fails strictness, fall back to solid roof.

- Interior rooms are clipped to stop below wall_z1 by (attic_thickness + small gap), but clamped to keep a minimum room height.

- Attic floor exported as its own STL and NOT unioned into *_complete by default.

- *_complete.stl is exported ONLY when strict manifold-volume checks pass.

Deps:
  pip install numpy scipy trimesh shapely mapbox_earcut
Booleans (recommended):
  pip install manifold3d
  or boolean_engine="blender" with Blender installed
"""

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

# =========================
# SCALE-SAFE THICKNESS
# =========================
EXT_FRAC_MAX = 0.030
INT_FRAC_MAX = 0.020
ROOF_FRAC_MAX = 0.015

MIN_EXT = 0.002
MIN_INT = 0.0015
MIN_ROOF = 0.001

# Roof sits slightly INTO wall (usually 0 is fine for booleans; overlap handled by fuse eps)
ROOF_SIT_OVERLAP = 0.0

# Tiny overlap used ONLY for boolean fusion to avoid coplanar/tangent contacts.
FUSE_EPS_FRAC = 2.0e-4
FUSE_EPS_MIN = 1.0e-6

# New: fuse eps also tied to thickness to avoid nearly-coplanar unions
FUSE_EPS_THICKNESS_FRAC = 0.05  # 5% of min thickness (clamped below)

# Coverage rules
FLOOR_PAD_EXT_FACTOR = 1.0
ROOF_OVERHANG_EXT_FACTOR = 1.75

# Shapely buffer parameters
WALL_JOIN_STYLE = "bevel"
WALL_MITRE_LIMIT = 2.0

# Roof thickness minimum (prevents "paper thin" appearance)
ROOF_THICKNESS_MIN = 0.01

# Keep interior below attic region
ATTIC_GAP_MULT = 4.0  # multiplied by fuse_eps

# Attic slab export and whether to union into complete
EXPORT_ATTIC_FLOOR = True
INCLUDE_ATTIC_IN_COMPLETE = False

# --- Thickness scaling request ---
THICKNESS_SCALE = 1.8  # applies to roof+attic thickness (scale BEFORE clamp)

# Prevent interiors from vanishing due to over-clipping
MIN_INTERIOR_HEIGHT = 0.25  # adjust to your units if needed (meters)


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
# Thickness sampling (per house) - raw values (will be clamped)
# ---------------------------------------------------------------------
def sample_house_thicknesses(
    rng,
    ext_range_m=(0.165, 0.210),
    int_range_m=(0.105, 0.130),

    # roof: keep discrete options by default
    roof_choices_m=(0.0111125, 0.0127, 0.015875),
    roof_range_m=None,  # if set, overrides roof_choices_m

    # NEW: independent per-house thicknesses
    floor_range_m=(0.04, 0.07),
    attic_range_m=(0.02, 0.05),
):
    t_ext = float(rng.uniform(*ext_range_m))
    t_int = float(rng.uniform(*int_range_m))

    if roof_range_m is not None:
        t_roof = float(rng.uniform(*roof_range_m))
    else:
        t_roof = float(rng.choice(np.array(roof_choices_m, dtype=np.float64)))

    t_floor = float(rng.uniform(*floor_range_m))
    t_attic = float(rng.uniform(*attic_range_m))
    return t_ext, t_int, t_roof, t_floor, t_attic


def clamp_thickness_to_footprint(footprint_poly: Polygon, t_ext, t_int, t_roof):
    if footprint_poly is None:
        return t_ext, t_int, t_roof

    minx, miny, maxx, maxy = footprint_poly.bounds
    s = max(1e-9, min(float(maxx - minx), float(maxy - miny)))

    t_ext_c = float(np.clip(t_ext, MIN_EXT, EXT_FRAC_MAX * s))
    t_int_c = float(np.clip(t_int, MIN_INT, INT_FRAC_MAX * s))
    t_roof_c = float(np.clip(t_roof, MIN_ROOF, ROOF_FRAC_MAX * s))
    return t_ext_c, t_int_c, t_roof_c


# ---------------------------------------------------------------------
# Geometry cleanup helpers
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


def safe_buffer(poly: Polygon, dist: float):
    if poly is None:
        return None

    if not getattr(poly, "is_valid", True):
        try:
            poly = shapely.make_valid(poly, method="structure", keep_collapsed=False)
        except Exception:
            try:
                poly = shapely.make_valid(poly)
            except Exception:
                pass
        poly = _largest_polygon(poly)
        if poly is None:
            return None

    try:
        out = poly.buffer(float(dist), join_style=WALL_JOIN_STYLE, mitre_limit=float(WALL_MITRE_LIMIT))
    except TypeError:
        out = poly.buffer(float(dist))

    try:
        out = out.buffer(0)
    except Exception:
        pass

    return _largest_polygon(out)


def outward_floor_polygon(footprint_poly: Polygon, t_ext: float):
    return safe_buffer(footprint_poly, float(t_ext) * float(FLOOR_PAD_EXT_FACTOR))


def roof_clip_polygon_from_footprint(footprint_poly: Polygon, t_ext: float):
    fp_outer = safe_buffer(footprint_poly, float(t_ext))
    if fp_outer is None:
        return None
    overhang = float(t_ext) * float(ROOF_OVERHANG_EXT_FACTOR)
    return safe_buffer(fp_outer, overhang)


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
# Exterior inference (largest vertical component = exterior)
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
        return float(hull.volume)
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
# Polygonize/extrude helpers
# ---------------------------------------------------------------------
def estimate_z0_z1_minmax(vertices, edges_subset):
    vids = np.unique(edges_subset.reshape(-1)) if edges_subset is not None and len(edges_subset) else np.arange(len(vertices))
    z = vertices[vids, 2]
    return float(np.min(z)), float(np.max(z))


def estimate_z0_z1_quantile(vertices, edges_subset):
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


# ---------------------------------------------------------------------
# Floor slab
# ---------------------------------------------------------------------
def floor_slab_from_polygon(poly: Polygon, z0: float, thickness: float = 0.05, color_rgb=None):
    if color_rgb is None:
        color_rgb = SEMANTIC_INFO[4]["color"]

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
# Exterior walls: OUTWARD-ONLY thickness
# ---------------------------------------------------------------------
def exterior_wall_outward_only(fp: Polygon, z0: float, z1: float, thickness: float, engine="earcut", boolean_engine="manifold"):
    if fp is None or thickness <= 1e-9:
        return None
    height = float(z1 - z0)
    if height <= 1e-8:
        return None

    fp = _largest_polygon(fp)
    if fp is None or fp.area <= 1e-8:
        return None

    if not getattr(fp, "is_valid", True):
        try:
            fpv = shapely.make_valid(fp, method="structure", keep_collapsed=False)
        except Exception:
            try:
                fpv = shapely.make_valid(fp)
            except Exception:
                fpv = fp
        fp = _largest_polygon(fpv)
        if fp is None:
            return None

    try:
        fp = fp.buffer(0)
    except Exception:
        pass
    try:
        fp = orient(fp, sign=1.0)
    except Exception:
        pass

    outer = safe_buffer(fp, float(thickness))
    if outer is None or outer.area <= 1e-8:
        return None
    try:
        outer = orient(outer, sign=1.0)
    except Exception:
        pass

    wall = None
    try:
        outer_prism = trimesh.creation.extrude_polygon(outer, height=height, engine=engine, force_vertices=False)
        inner_prism = trimesh.creation.extrude_polygon(fp, height=height, engine=engine, force_vertices=False)
        outer_prism.apply_translation([0.0, 0.0, float(z0)])
        inner_prism.apply_translation([0.0, 0.0, float(z0)])
        outer_prism.process(validate=True)
        inner_prism.process(validate=True)
        diff = trimesh.boolean.difference([outer_prism, inner_prism], engine=boolean_engine, check_volume=False)
        if isinstance(diff, (list, tuple)):
            diff = trimesh.util.concatenate(diff)
        if diff is not None and len(getattr(diff, "faces", [])) > 0:
            wall = diff
    except Exception:
        wall = None

    if wall is None:
        ring = outer.difference(fp)
        try:
            ring = ring.buffer(0)
        except Exception:
            pass
        ring_poly = _largest_polygon(ring)
        if ring_poly is None or ring_poly.area <= 1e-8:
            return None
        try:
            ring_poly = orient(ring_poly, sign=1.0)
        except Exception:
            pass
        try:
            wall = trimesh.creation.extrude_polygon(ring_poly, height=height, engine=engine, force_vertices=False)
        except Exception as e:
            print(f"Exterior wall extrude failed: {type(e).__name__} {e}")
            return None
        wall.apply_translation([0.0, 0.0, float(z0)])

    try:
        wall.process(validate=True)
    except Exception:
        pass

    wall = repair_mesh_for_fea(wall)
    return wall


def thick_wall_solid_symmetric(
    footprint_poly: Polygon,
    z0: float,
    z1: float,
    thickness: float,
    engine="earcut",
    join_style=2,
    cap_style=2,
):
    if footprint_poly is None or thickness <= 1e-9:
        return None

    p = footprint_poly
    try:
        p = p.buffer(0)
    except Exception:
        pass
    try:
        p = orient(p, sign=1.0)
    except Exception:
        pass
    if (not isinstance(p, Polygon)) or (p.area <= 1e-8):
        return None

    height = float(z1 - z0)
    if height <= 1e-8:
        return None

    half = 0.5 * float(thickness)
    outer = p.buffer(half, join_style=join_style, cap_style=cap_style)
    inner = p.buffer(-half, join_style=join_style, cap_style=cap_style)
    ring = outer if inner.is_empty else outer.difference(inner)

    try:
        ring = ring.buffer(0)
    except Exception:
        pass

    ring_poly = _largest_polygon(ring)
    if ring_poly is None or ring_poly.area <= 1e-8:
        return None

    try:
        wall = trimesh.creation.extrude_polygon(ring_poly, height=height, engine=engine, force_vertices=False)
    except Exception as e:
        print(f"⚠️ Interior wall extrude failed: {type(e).__name__}: {e}")
        return None

    wall.apply_translation([0.0, 0.0, float(z0)])
    wall.process(validate=True)
    return wall


# ---------------------------------------------------------------------
# Footprint inference (walls/floor)
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


def exterior_walls_and_floor_from_edges(
    vertices,
    non_roof_edges,
    ext_wall_edges,
    z_snap=0.03,
    floor_thickness=0.05,
    wall_thickness=0.18,
    prism_engine="earcut",
    boolean_engine="manifold",
):
    if non_roof_edges is None or len(non_roof_edges) == 0:
        return None, None, None, (None, None)

    if ext_wall_edges is not None and len(ext_wall_edges):
        z0, z1 = estimate_z0_z1_minmax(vertices, ext_wall_edges)
    else:
        z0, z1 = estimate_z0_z1_minmax(vertices, non_roof_edges)

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
        return None, None, None, (z0, z1)

    footprint_poly = max(polys, key=lambda p: p.area)

    walls = exterior_wall_outward_only(
        footprint_poly, z0=z0, z1=z1, thickness=float(wall_thickness),
        engine=prism_engine, boolean_engine=boolean_engine
    )
    colorize(walls, SEMANTIC_INFO[1]["color"])

    fp_floor = outward_floor_polygon(footprint_poly, t_ext=float(wall_thickness))
    floor = floor_slab_from_polygon(fp_floor, z0=z0, thickness=float(floor_thickness), color_rgb=SEMANTIC_INFO[4]["color"])

    return walls, floor, footprint_poly, (z0, z1)


# ---------------------------------------------------------------------
# Interior rooms (CLIPPED to avoid attic, but clamped to keep some height)
# ---------------------------------------------------------------------
def interior_room_mesh_from_component(
    vertices,
    ecomp,
    wall_thickness=0.115,
    angle_deg=8.0,
    z_snap=0.03,
    prism_engine="earcut",
    z0_clip=None,
    z1_clip=None,
):
    if ecomp is None or len(ecomp) == 0:
        return None

    z0, z1 = estimate_z0_z1_quantile(vertices, ecomp)

    if z0_clip is not None:
        z0 = max(float(z0), float(z0_clip))
    if z1_clip is not None:
        z1 = min(float(z1), float(z1_clip))
    if z1 <= z0 + 1e-6:
        return None

    theta = estimate_manhattan_theta(vertices, ecomp, z0=z0, z_snap=z_snap)

    e = filter_edges_axis_aligned_in_frame(vertices, ecomp, theta=theta, angle_deg=angle_deg)
    e = filter_edges_two_z_levels(vertices, e, z0=z0, z1=z1, z_snap=z_snap)

    polys = polygonize_from_edges(vertices, e, z0=z0, z_snap=z_snap, theta=theta)
    if not polys:
        return None

    poly = max(polys, key=lambda p: p.area)
    m = thick_wall_solid_symmetric(poly, z0=z0, z1=z1, thickness=float(wall_thickness), engine=prism_engine)
    colorize(m, SEMANTIC_INFO[2]["color"])
    return m


# ---------------------------------------------------------------------
# Roof: convex hull -> clip
# ---------------------------------------------------------------------
def eave_polygons_from_wireframe(vertices, edges_pool, z_center, z_snap=0.10):
    if edges_pool is None or len(edges_pool) == 0:
        return []
    if z_center is None or z_snap <= 0:
        return []

    za = vertices[edges_pool[:, 0], 2]
    zb = vertices[edges_pool[:, 1], 2]
    mask = (np.abs(za - z_center) <= z_snap) & (np.abs(zb - z_center) <= z_snap)
    e = edges_pool[mask]
    if len(e) < 3:
        return []

    vids = np.unique(e.reshape(-1))
    vid_to_local = {int(v): i for i, v in enumerate(vids)}
    xy = vertices[vids][:, :2].astype(np.float64)
    e_local = np.array([[vid_to_local[int(a)], vid_to_local[int(b)]] for a, b in e], dtype=np.int64)

    return polygonize_xy_lines(xy, e_local)


def _bbox_area(p: Polygon):
    minx, miny, maxx, maxy = p.bounds
    return float((maxx - minx) * (maxy - miny))


def choose_true_eave_polygon(polys, footprint_poly, min_area=1e-6):
    if footprint_poly is None or not isinstance(footprint_poly, Polygon):
        return None

    fp = footprint_poly
    try:
        fp = fp.buffer(0)
    except Exception:
        pass

    good = []
    for p in polys:
        if (not isinstance(p, Polygon)) or (not p.is_valid) or (p.area <= min_area):
            continue
        pp = p
        try:
            pp = pp.buffer(0)
        except Exception:
            pass
        try:
            if pp.contains(fp):
                good.append(pp)
        except Exception:
            continue

    if not good:
        return None
    return max(good, key=_bbox_area)


def find_eave_polygon_by_search(vertices, edges_pool, footprint_poly, z_center, z_span=0.30, steps=10, z_snap=0.10):
    if z_center is None:
        return None
    zs = np.linspace(float(z_center - z_span), float(z_center + z_span), int(steps))

    best = None
    best_score = -1.0
    for z0 in zs:
        polys = eave_polygons_from_wireframe(vertices, edges_pool, z_center=z0, z_snap=z_snap)
        p = choose_true_eave_polygon(polys, footprint_poly)
        if p is None:
            continue
        score = _bbox_area(p)
        if score > best_score:
            best = p
            best_score = score
    return best


def _convex_hull_mesh_3d(points3d, qhull_options="QJ"):
    pts = np.asarray(points3d, dtype=np.float64)
    if len(pts) < 4:
        return None
    try:
        hull = ConvexHull(pts, qhull_options=qhull_options)
    except Exception:
        return None
    m = trimesh.Trimesh(vertices=pts.astype(np.float32), faces=hull.simplices.astype(np.int64), process=True)
    if len(m.faces) == 0:
        return None
    m.process(validate=True)
    return m


def _make_footprint_prism(footprint_poly, zmin, zmax, prism_engine="earcut"):
    if footprint_poly is None or footprint_poly.area <= 1e-8:
        return None
    height = float(zmax - zmin)
    if height <= 1e-8:
        return None
    prism = trimesh.creation.extrude_polygon(
        footprint_poly,
        height=height,
        engine=prism_engine,
        force_vertices=False,
    )
    prism.apply_translation([0.0, 0.0, float(zmin)])
    prism.process(validate=True)
    return prism


def trim_mesh_to_prism(mesh: trimesh.Trimesh, clip_poly: Polygon, zmin: float, zmax: float,
                       prism_engine="earcut", boolean_engine="manifold"):
    if mesh is None or len(getattr(mesh, "faces", [])) == 0 or clip_poly is None:
        return mesh

    prism = _make_footprint_prism(clip_poly, zmin=float(zmin), zmax=float(zmax), prism_engine=prism_engine)
    if prism is None or len(getattr(prism, "faces", [])) == 0:
        return mesh

    try:
        out = trimesh.boolean.intersection([mesh, prism], engine=boolean_engine, check_volume=False)
    except Exception:
        return mesh

    if out is None:
        return mesh
    if isinstance(out, (list, tuple)):
        out = trimesh.util.concatenate(out)

    try:
        out.process(validate=True)
    except Exception:
        pass
    return out


def roof_convexhull_pitch_clip_to_true_eaves(
    vertices,
    edges,
    roof_edges,
    non_roof_edges,
    footprint_poly,
    wall_z1,
    roof_clip_zpad=5.0,
    eave_z_snap=0.10,
    eave_search_span=0.35,
    eave_search_steps=12,
    prism_engine="earcut",
    boolean_engine="manifold",
    boolean_check_volume=True,
    clip_poly_override=None,
    debug=True,
):
    if roof_edges is None or len(roof_edges) < 3:
        return None
    if wall_z1 is None:
        return None

    roof_vids = np.unique(roof_edges.reshape(-1))
    roof_pts = vertices[roof_vids]
    if len(roof_pts) < 4:
        return None

    hull_mesh = _convex_hull_mesh_3d(roof_pts, qhull_options="QJ")
    if hull_mesh is None:
        return None

    polys_nr = eave_polygons_from_wireframe(vertices, non_roof_edges, z_center=wall_z1, z_snap=eave_z_snap)
    polys_r = eave_polygons_from_wireframe(vertices, roof_edges, z_center=wall_z1, z_snap=eave_z_snap)
    polys_all = eave_polygons_from_wireframe(vertices, edges, z_center=wall_z1, z_snap=eave_z_snap)

    cand = choose_true_eave_polygon(polys_nr + polys_r + polys_all, footprint_poly)
    if cand is None:
        cand = find_eave_polygon_by_search(
            vertices=vertices,
            edges_pool=edges,
            footprint_poly=footprint_poly,
            z_center=wall_z1,
            z_span=eave_search_span,
            steps=eave_search_steps,
            z_snap=eave_z_snap,
        )

    clip_poly = clip_poly_override if clip_poly_override is not None else (cand if cand is not None else footprint_poly)
    if clip_poly is None:
        return hull_mesh

    zmin = float(wall_z1)
    zmax = float(np.max(roof_pts[:, 2]) + roof_clip_zpad)
    prism = _make_footprint_prism(clip_poly, zmin=zmin, zmax=zmax, prism_engine=prism_engine)
    if prism is None:
        return hull_mesh

    try:
        clipped = trimesh.boolean.intersection(
            [hull_mesh, prism],
            engine=boolean_engine,
            check_volume=boolean_check_volume,
        )
    except Exception as e:
        print(f"⚠️ Roof boolean intersection failed ({boolean_engine}): {type(e).__name__}: {e}")
        return hull_mesh

    if clipped is None or len(clipped.faces) == 0:
        return hull_mesh

    clipped.process(validate=True)
    return clipped


def solidify_roof_mesh(roof_mesh: trimesh.Trimesh, thickness: float):
    if roof_mesh is None or len(getattr(roof_mesh, "faces", [])) == 0 or thickness <= 1e-9:
        return roof_mesh

    m = roof_mesh.copy()
    m.remove_unreferenced_vertices()
    m.process(validate=True)

    v0 = m.vertices.astype(np.float64)
    n = m.vertex_normals.astype(np.float64)
    v1 = v0 - n * float(thickness)

    f0 = m.faces.astype(np.int64)
    f1 = (f0[:, ::-1] + len(v0)).astype(np.int64)

    try:
        idx = trimesh.grouping.group_rows(m.edges_sorted, require_count=1)
        boundary = m.edges[idx]
    except Exception:
        boundary = None

    faces_side_list = []
    if boundary is not None and len(boundary) > 0:
        for a, b in boundary:
            a = int(a)
            b = int(b)
            a2 = a + len(v0)
            b2 = b + len(v0)
            faces_side_list.append([a, b, b2])
            faces_side_list.append([a, b2, a2])

    faces_side = (
        np.asarray(faces_side_list, dtype=np.int64).reshape(-1, 3)
        if faces_side_list
        else np.empty((0, 3), dtype=np.int64)
    )

    out = trimesh.Trimesh(
        vertices=np.vstack([v0, v1]).astype(np.float32),
        faces=np.vstack([f0, f1, faces_side]).astype(np.int64),
        process=True,
    )
    out.process(validate=True)
    return out


def hollow_solid_mesh(mesh: trimesh.Trimesh, thickness: float, boolean_engine="manifold"):
    """Convert a closed watertight solid into a shell of approximately thickness using boolean outer-inner."""
    if mesh is None or len(getattr(mesh, "faces", [])) == 0 or thickness <= 1e-9:
        return mesh

    m0 = mesh.copy()
    try:
        m0.process(validate=True)
    except Exception:
        pass

    if not getattr(m0, "is_watertight", False):
        return m0

    try:
        v0 = m0.vertices.astype(np.float64)
        n = m0.vertex_normals.astype(np.float64)
        v1 = v0 - n * float(thickness)
        m1 = trimesh.Trimesh(vertices=v1.astype(np.float32), faces=m0.faces.copy(), process=True)
        m1.process(validate=True)
    except Exception:
        return m0

    try:
        out = trimesh.boolean.difference([m0, m1], engine=boolean_engine, check_volume=False)
        if isinstance(out, (list, tuple)):
            out = trimesh.util.concatenate(out)
        if out is None or len(getattr(out, "faces", [])) == 0:
            return m0
        out.process(validate=True)
        return out
    except Exception:
        return m0


def snap_mesh_min_z_to(mesh: trimesh.Trimesh, target_min_z: float):
    if mesh is None or len(getattr(mesh, "faces", [])) == 0:
        return mesh
    b = mesh.bounds
    if b is None:
        return mesh
    dz = float(target_min_z - b[0, 2])
    mesh.apply_translation([0.0, 0.0, dz])
    return mesh


# ---------------------------------------------------------------------
# FEA validity helpers
# ---------------------------------------------------------------------
def compute_fuse_epsilon(footprint_poly: Polygon, meshes, frac: float = FUSE_EPS_FRAC, min_eps: float = FUSE_EPS_MIN):
    eps = float(min_eps)

    # scale by footprint span
    try:
        if footprint_poly is not None and getattr(footprint_poly, "area", 0.0) > 0:
            minx, miny, maxx, maxy = footprint_poly.bounds
            span = max(1e-12, min(float(maxx - minx), float(maxy - miny)))
            eps = max(eps, float(frac) * span)
    except Exception:
        pass

    # scale by mesh bbox
    if meshes:
        try:
            bb = [m.bounds for m in meshes if m is not None and len(getattr(m, "faces", [])) > 0 and m.bounds is not None]
            if bb:
                mins = np.min(np.stack([b[0] for b in bb], axis=0), axis=0)
                maxs = np.max(np.stack([b[1] for b in bb], axis=0), axis=0)
                span = max(1e-12, float(np.min(maxs[:2] - mins[:2])))
                eps = max(eps, float(frac) * span)
        except Exception:
            pass

    return float(eps)


def repair_mesh_for_fea(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if mesh is None or len(getattr(mesh, "faces", [])) == 0:
        return mesh
    m = mesh.copy()

    try:
        m.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        m.remove_duplicate_faces()
    except Exception:
        pass
    try:
        m.remove_degenerate_faces()
    except Exception:
        pass
    try:
        m.merge_vertices()
    except Exception:
        pass

    try:
        trimesh.repair.fix_normals(m)
    except Exception:
        pass
    try:
        trimesh.repair.fix_inversion(m)
    except Exception:
        pass
    try:
        trimesh.repair.fill_holes(m)
    except Exception:
        pass

    try:
        m.process(validate=True)
    except Exception:
        pass
    return m


def _edge_stats(mesh: trimesh.Trimesh):
    boundary_edges = 0
    nonmanifold_edges = 0
    try:
        counts = np.bincount(mesh.edges_unique_inverse)
        boundary_edges = int(np.sum(counts == 1))
        nonmanifold_edges = int(np.sum(counts > 2))
    except Exception:
        pass
    return boundary_edges, nonmanifold_edges


def strict_fea_ok(mesh: trimesh.Trimesh) -> bool:
    if mesh is None or len(getattr(mesh, "faces", [])) == 0:
        return False
    try:
        mesh.process(validate=True)
    except Exception:
        pass

    watertight = bool(getattr(mesh, "is_watertight", False))
    is_volume = bool(getattr(mesh, "is_volume", False))
    if not (watertight and is_volume):
        return False

    boundary_edges, nonmanifold_edges = _edge_stats(mesh)
    return (boundary_edges == 0) and (nonmanifold_edges == 0)


def fea_quality_report(mesh: trimesh.Trimesh, name: str = "mesh"):
    print(f"\n--- FEA check: {name} ---")
    if mesh is None or len(getattr(mesh, "faces", [])) == 0:
        print("Empty mesh")
        return

    try:
        mesh.process(validate=True)
    except Exception:
        pass

    watertight = bool(getattr(mesh, "is_watertight", False))
    is_volume = bool(getattr(mesh, "is_volume", False))
    winding = bool(getattr(mesh, "is_winding_consistent", False))

    boundary_edges, nonmanifold_edges = _edge_stats(mesh)

    print(f"Vertices: {len(mesh.vertices)} | Faces: {len(mesh.faces)}")
    print(f"Watertight: {watertight}")
    print(f"Volume solid (is_volume): {is_volume}")
    print(f"Winding consistent: {winding}")
    print(f"Boundary edges (count==1): {boundary_edges}")
    print(f"Non-manifold edges (count>2): {nonmanifold_edges}")
    print(f"FEA-ready (strict): {strict_fea_ok(mesh)}")


# ---------- Robust boolean union ----------
def _bool_union(meshes, engine="manifold", check_volume=True):
    meshes = [m for m in meshes if m is not None and len(getattr(m, "faces", [])) > 0]
    if not meshes:
        return None
    meshes = [repair_mesh_for_fea(m) for m in meshes]

    try:
        out = trimesh.boolean.union(meshes, engine=engine, check_volume=check_volume)
    except Exception:
        return None

    if out is None:
        return None
    if isinstance(out, (list, tuple)):
        out = trimesh.util.concatenate(out)

    out = repair_mesh_for_fea(out)
    return out


def build_complete_mesh(
    roof_mesh,
    exterior_mesh,
    floor_mesh,
    interior_mesh=None,
    attic_mesh=None,
    footprint_poly=None,
    fuse_eps_span=None,
    t_min_thickness=None,
    boolean_engine="manifold",
):
    parts = [roof_mesh, exterior_mesh, floor_mesh]
    if INCLUDE_ATTIC_IN_COMPLETE:
        parts.append(attic_mesh)

    # keep interior out of complete by default (often you want separate solids)
    # If you want to include: set INCLUDE_INTERIOR=True and append it here.
    INCLUDE_INTERIOR = False
    if INCLUDE_INTERIOR:
        parts.append(interior_mesh)

    parts = [p for p in parts if p is not None and len(getattr(p, "faces", [])) > 0]
    if not parts:
        return None

    # compute a stronger fuse epsilon: span-based AND thickness-based
    fuse_eps = float(fuse_eps_span if fuse_eps_span is not None else compute_fuse_epsilon(footprint_poly, parts))
    if t_min_thickness is not None:
        fuse_eps = max(fuse_eps, float(FUSE_EPS_THICKNESS_FRAC) * float(t_min_thickness))
        fuse_eps = min(fuse_eps, 0.25 * float(t_min_thickness))

    # Attempt unions in a few robust passes
    engines = [boolean_engine]
    if boolean_engine != "blender":
        engines.append("blender")  # will no-op if blender isn't installed; exceptions are caught

    # Try with overlap, then with larger overlap, then without overlap
    overlap_trials = [fuse_eps, 3.0 * fuse_eps, 0.0]

    for eng in engines:
        for ov in overlap_trials:
            roof_fuse = roof_mesh.copy() if roof_mesh is not None and len(getattr(roof_mesh, "faces", [])) > 0 else None
            floor_fuse = floor_mesh.copy() if floor_mesh is not None and len(getattr(floor_mesh, "faces", [])) > 0 else None
            exterior_fuse = exterior_mesh.copy() if exterior_mesh is not None and len(getattr(exterior_mesh, "faces", [])) > 0 else None
            attic_fuse = attic_mesh.copy() if (INCLUDE_ATTIC_IN_COMPLETE and attic_mesh is not None and len(getattr(attic_mesh, "faces", [])) > 0) else None
            interior_fuse = interior_mesh.copy() if (INCLUDE_INTERIOR and interior_mesh is not None and len(getattr(interior_mesh, "faces", [])) > 0) else None

            if ov > 0.0:
                # create a clear overlap so boolean isn't tangent
                if roof_fuse is not None:
                    roof_fuse.apply_translation([0.0, 0.0, -float(ov)])
                if floor_fuse is not None:
                    floor_fuse.apply_translation([0.0, 0.0, float(ov)])

            meshes_try = [exterior_fuse, floor_fuse, roof_fuse]
            if INCLUDE_ATTIC_IN_COMPLETE:
                meshes_try.append(attic_fuse)
            if INCLUDE_INTERIOR:
                meshes_try.append(interior_fuse)

            # Pass A: check_volume=True (more robust correctness)
            out = _bool_union(meshes_try, engine=eng, check_volume=True)
            if out is not None and strict_fea_ok(out):
                return out

            # Pass B: check_volume=False (sometimes avoids engine refusal)
            out = _bool_union(meshes_try, engine=eng, check_volume=False)
            if out is not None and strict_fea_ok(out):
                return out

    return None


# ---------------------------------------------------------------------
# Main per-file pipeline
# ---------------------------------------------------------------------
def process_file(
    npz_path: Path,
    output_dir: str,
    snap_tol=5e-4,
    roof_z_q=0.85,
    roof_nonvertical_max=0.65,
    wall_vertical_cos=0.7,
    wall_z_snap=0.03,

    # NOTE: still supported as a fallback if floor_range_m is None
    floor_thickness=0.05,

    min_edges_per_room=6,
    interior_angle_deg=8.0,
    roof_clip_zpad=5.0,
    eave_z_snap=0.10,
    eave_search_span=0.35,
    eave_search_steps=12,
    prism_engine="earcut",
    boolean_engine="manifold",
    boolean_check_volume=True,
    debug_roof=True,
    per_house_seed_mode="stem_hash",
    fixed_seed=12345,

    ext_range_m=(0.165, 0.210),
    int_range_m=(0.105, 0.130),
    roof_choices_m=(0.0111125, 0.0127, 0.015875),

    # NEW:
    roof_range_m=None,
    floor_range_m=(0.04, 0.07),
    attic_range_m=(0.02, 0.05),

    export_complete_ply=True,
):
    if eave_z_snap <= 0:
        raise ValueError("eave_z_snap must be > 0 (try 0.10-0.25)")

    print("=" * 70)
    print(npz_path.name)
    print("=" * 70)

    vertices, edges = load_wireframe(npz_path)
    print(f"Loaded: vertices={len(vertices)} edges={len(edges)}")

    vertices, edges = snap_weld_vertices(vertices, edges, snap_tol=snap_tol)
    print(f"After snap_weld (tol={snap_tol}): vertices={len(vertices)} edges={len(edges)}")

    outdir = Path(output_dir)
    base = npz_path.stem

    seed = int(fixed_seed) if per_house_seed_mode == "fixed" else int(abs(hash(npz_path.stem)) % (2**32))
    rng = np.random.default_rng(seed)

    # Sample (per-house)
    t_ext_raw, t_int_raw, t_roof_raw, t_floor_raw, t_attic_raw = sample_house_thicknesses(
        rng,
        ext_range_m=ext_range_m,
        int_range_m=int_range_m,
        roof_choices_m=roof_choices_m,
        roof_range_m=roof_range_m,
        floor_range_m=floor_range_m,
        attic_range_m=attic_range_m,
    )

    # Decide actual floor thickness used for this house:
    # - If floor_range_m is provided -> randomized via t_floor_raw
    # - If caller passes floor_range_m=None -> fall back to floor_thickness param
    floor_thickness_house = float(t_floor_raw) if (floor_range_m is not None) else float(floor_thickness)

    # Apply requested roof scaling BEFORE clamping-to-footprint
    t_roof_raw_scaled = float(THICKNESS_SCALE) * float(t_roof_raw)

    roof_edges, non_roof_edges = split_roof_edges(vertices, edges, roof_z_q=roof_z_q, roof_nonvertical_max=roof_nonvertical_max)

    ext_wall_edges, int_wall_edges, other_edges = infer_exterior_interior_wall_edges(
        vertices, non_roof_edges, vertical_cos=wall_vertical_cos
    )

    if ext_wall_edges is not None and len(ext_wall_edges):
        wall_z0, wall_z1 = estimate_z0_z1_minmax(vertices, ext_wall_edges)
    else:
        wall_z0, wall_z1 = estimate_z0_z1_minmax(vertices, non_roof_edges) if len(non_roof_edges) else (None, None)

    # provisional walls/floor to infer footprint (use ext thickness raw)
    exterior_mesh, floor_mesh, footprint_poly, (wall_z0b, wall_z1b) = exterior_walls_and_floor_from_edges(
        vertices,
        non_roof_edges=non_roof_edges,
        ext_wall_edges=ext_wall_edges,
        z_snap=wall_z_snap,
        floor_thickness=floor_thickness_house,
        wall_thickness=t_ext_raw,
        prism_engine=prism_engine,
        boolean_engine=boolean_engine,
    )

    if footprint_poly is None:
        print(f"[FAIL] {base}: could not infer footprint -> skipping")
        return False

    # clamp thickness to footprint scale (roof already scaled)
    t_ext, t_int, t_roof = clamp_thickness_to_footprint(footprint_poly, t_ext_raw, t_int_raw, t_roof_raw_scaled)

    # final roof/attic thicknesses (attic is independent)
    t_roof_eff = float(max(t_roof, ROOF_THICKNESS_MIN))
    t_attic_eff = float(t_attic_raw)

    print(f"Thickness raw: ext={t_ext_raw:.4f} int={t_int_raw:.4f} roof={t_roof_raw:.4f} floor={t_floor_raw:.4f} attic={t_attic_raw:.4f} seed={seed}")
    print(f"Thickness raw*{THICKNESS_SCALE:.2f}: roof={t_roof_raw_scaled:.4f}")
    print(f"Thickness clamp: ext={t_ext:.4f} int={t_int:.4f} roof={t_roof:.4f} (eff={t_roof_eff:.4f}) | floor_used={floor_thickness_house:.4f} | attic_eff={t_attic_eff:.4f}")

    # compute fuse eps early so interior cap uses it
    fuse_eps0 = compute_fuse_epsilon(footprint_poly, meshes=None)
    attic_gap = float(ATTIC_GAP_MULT) * float(fuse_eps0)

    # interior clip cap: keep interiors below attic slab, but don't delete them
    interior_z1_cap = None
    if wall_z1 is not None and wall_z0 is not None:
        interior_z1_cap = float(wall_z1) - float(t_attic_eff) - float(attic_gap)
        interior_z1_cap = max(interior_z1_cap, float(wall_z0) + float(MIN_INTERIOR_HEIGHT))

    # rebuild floor to cover outer wall face
    fp_floor = outward_floor_polygon(footprint_poly, t_ext=t_ext)
    if fp_floor is not None and wall_z0 is not None:
        floor_mesh = floor_slab_from_polygon(
            fp_floor,
            z0=wall_z0,
            thickness=float(floor_thickness_house),
            color_rgb=SEMANTIC_INFO[4]["color"],
        )

    # rebuild exterior walls with clamped thickness
    if footprint_poly is not None and wall_z0 is not None and wall_z1 is not None:
        exterior_mesh = exterior_wall_outward_only(
            footprint_poly, z0=wall_z0, z1=wall_z1, thickness=t_ext,
            engine=prism_engine, boolean_engine=boolean_engine
        )
        colorize(exterior_mesh, SEMANTIC_INFO[1]["color"])

    # interior rooms mesh
    candidate_interior_edges = (
        np.concatenate([int_wall_edges, other_edges], axis=0)
        if (len(int_wall_edges) or len(other_edges))
        else np.empty((0, 2), np.int32)
    )

    comps = edge_connected_components(candidate_interior_edges)
    print(f"Interior components raw: {len(comps)}")

    room_meshes = []
    for comp in comps:
        ecomp = candidate_interior_edges[comp]
        if len(ecomp) < min_edges_per_room:
            continue

        m = interior_room_mesh_from_component(
            vertices,
            ecomp,
            wall_thickness=t_int,
            angle_deg=interior_angle_deg,
            z_snap=wall_z_snap,
            prism_engine=prism_engine,
            z0_clip=wall_z0,
            z1_clip=interior_z1_cap,
        )
        if m is not None and len(m.faces) > 0:
            room_meshes.append(m)

    interior_mesh = trimesh.util.concatenate(room_meshes) if room_meshes else None
    if interior_mesh is None:
        print("No interior rooms created.")

    # roof
    clip_poly = roof_clip_polygon_from_footprint(footprint_poly, t_ext=t_ext)
    roof_mesh = roof_convexhull_pitch_clip_to_true_eaves(
        vertices=vertices,
        edges=edges,
        roof_edges=roof_edges,
        non_roof_edges=non_roof_edges,
        footprint_poly=footprint_poly,
        wall_z1=wall_z1,
        roof_clip_zpad=roof_clip_zpad,
        eave_z_snap=eave_z_snap,
        eave_search_span=eave_search_span,
        eave_search_steps=eave_search_steps,
        prism_engine=prism_engine,
        boolean_engine=boolean_engine,
        boolean_check_volume=boolean_check_volume,
        clip_poly_override=clip_poly,
        debug=debug_roof,
    )

    # enforce roof bottom >= wall_z1
    if wall_z1 is not None and clip_poly is not None and roof_mesh is not None and len(getattr(roof_mesh, "faces", [])) > 0:
        zmax = float(np.max(roof_mesh.vertices[:, 2]) + float(roof_clip_zpad))
        roof_mesh = trim_mesh_to_prism(
            roof_mesh, clip_poly=clip_poly, zmin=float(wall_z1), zmax=zmax,
            prism_engine=prism_engine, boolean_engine=boolean_engine
        )
        roof_mesh = repair_mesh_for_fea(roof_mesh)
        roof_mesh = snap_mesh_min_z_to(roof_mesh, target_min_z=float(wall_z1) - float(ROOF_SIT_OVERLAP))

    # roof thickness (shell if possible; fallback to solid)
    if roof_mesh is not None and len(getattr(roof_mesh, "faces", [])) > 0:
        roof_mesh = repair_mesh_for_fea(roof_mesh)
        try:
            roof_mesh.process(validate=True)
        except Exception:
            pass

        if getattr(roof_mesh, "is_volume", False) and getattr(roof_mesh, "is_watertight", False):
            shell = hollow_solid_mesh(roof_mesh, thickness=t_roof_eff, boolean_engine=boolean_engine)
            shell = repair_mesh_for_fea(shell)
            if strict_fea_ok(shell):
                roof_mesh = shell
            else:
                roof_mesh = solidify_roof_mesh(roof_mesh, thickness=t_roof_eff)
                roof_mesh = repair_mesh_for_fea(roof_mesh)

        colorize(roof_mesh, SEMANTIC_INFO[3]["color"])

    # attic floor (export only by default)
    attic_floor_mesh = None
    if EXPORT_ATTIC_FLOOR and (footprint_poly is not None) and (wall_z1 is not None):
        attic_fp = footprint_poly
        try:
            attic_fp = attic_fp.buffer(float(t_ext) * float(FLOOR_PAD_EXT_FACTOR), join_style="round", mitre_limit=float(WALL_MITRE_LIMIT))
        except TypeError:
            attic_fp = attic_fp.buffer(float(t_ext) * float(FLOOR_PAD_EXT_FACTOR))
        try:
            attic_fp = attic_fp.buffer(0)
        except Exception:
            pass
        attic_fp = _largest_polygon(attic_fp)
        attic_floor_mesh = floor_slab_from_polygon(
            attic_fp,
            z0=float(wall_z1),
            thickness=float(t_attic_eff),
            color_rgb=SEMANTIC_INFO[4]["color"],
        )

    # repairs
    roof_mesh = repair_mesh_for_fea(roof_mesh)
    exterior_mesh = repair_mesh_for_fea(exterior_mesh)
    interior_mesh = repair_mesh_for_fea(interior_mesh)
    floor_mesh = repair_mesh_for_fea(floor_mesh)
    attic_floor_mesh = repair_mesh_for_fea(attic_floor_mesh)

    # Compute a stable fuse eps span
    parts_for_eps = [m for m in (roof_mesh, exterior_mesh, interior_mesh, floor_mesh) if m is not None and len(getattr(m, "faces", [])) > 0]
    fuse_eps_span = compute_fuse_epsilon(footprint_poly, parts_for_eps)
    print(f"Fuse eps (span-based): {fuse_eps_span:.6g}")

    # Build strict complete with robust union retries
    t_min_for_fuse = float(min(
        float(t_ext) if t_ext is not None else 1e9,
        float(floor_thickness_house),
        float(t_roof_eff),
        float(t_attic_eff),
    ))

    complete = build_complete_mesh(
        roof_mesh=roof_mesh,
        exterior_mesh=exterior_mesh,
        floor_mesh=floor_mesh,
        interior_mesh=interior_mesh,
        attic_mesh=attic_floor_mesh,
        footprint_poly=footprint_poly,
        fuse_eps_span=fuse_eps_span,
        t_min_thickness=t_min_for_fuse,
        boolean_engine=boolean_engine,
    )

    complete = repair_mesh_for_fea(complete)

    # Reports
    fea_quality_report(roof_mesh, name=f"{base}_roof")
    fea_quality_report(exterior_mesh, name=f"{base}_exterior_walls")
    fea_quality_report(interior_mesh, name=f"{base}_interior_rooms")
    fea_quality_report(floor_mesh, name=f"{base}_floor")
    fea_quality_report(attic_floor_mesh, name=f"{base}_attic_floor")
    fea_quality_report(complete, name=f"{base}_complete")

    # Export per-part
    outdir.mkdir(parents=True, exist_ok=True)

    if roof_mesh is not None and len(getattr(roof_mesh, "faces", [])) > 0:
        roof_mesh.export(outdir / f"{base}_roof.stl")

    if exterior_mesh is not None and len(getattr(exterior_mesh, "faces", [])) > 0:
        exterior_mesh.export(outdir / f"{base}_exterior_walls.stl")

    if interior_mesh is not None and len(getattr(interior_mesh, "faces", [])) > 0:
        interior_mesh.export(outdir / f"{base}_interior_rooms.stl")

    if floor_mesh is not None and len(getattr(floor_mesh, "faces", [])) > 0:
        floor_mesh.export(outdir / f"{base}_floor.stl")

    if EXPORT_ATTIC_FLOOR and attic_floor_mesh is not None and len(getattr(attic_floor_mesh, "faces", [])) > 0:
        attic_floor_mesh.export(outdir / f"{base}_attic_floor.stl")
    print(
    f"[THICKNESS FINAL] {base} | seed={seed} | "
    f"ext_wall={t_ext:.6f} m | int_wall={t_int:.6f} m | "
    f"roof_raw={t_roof_raw:.6f} m | roof_raw_scaled={t_roof_raw_scaled:.6f} m | "
    f"roof_clamped={t_roof:.6f} m | roof_eff={t_roof_eff:.6f} m | "
    f"floor={floor_thickness_house:.6f} m | attic_floor={t_attic_eff:.6f} m"
    )
    # STRICT: only export complete if it is truly FEA-ready
    if strict_fea_ok(complete):
        complete.export(outdir / f"{base}_complete.stl")
        if export_complete_ply:
            complete.export(outdir / f"{base}_complete.ply")
        print(f"[OK] Exported strict complete: {base}_complete.stl")
        return True
    else:
        print(f"[FAIL] {base}: complete is not a strict watertight manifold volume -> NOT exporting *_complete.stl")
        return False


def process_batch(input_dir="data/3dwire_raw", output_dir="data/3dwire_parts_combined", max_files=None, **kwargs):
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files in {input_dir}")

    if max_files:
        files = files[:max_files]

    print(f"Files: {len(files)} | Output: {output_dir}")

    ok = 0
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {f.name}")
        try:
            if process_file(f, output_dir, **kwargs):
                ok += 1
        except Exception as e:
            print(f"[FAILED] {f.name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone: {ok}/{len(files)} succeeded (exported *_complete.stl)")
    print(f"Output dir: {Path(output_dir).absolute()}")


if __name__ == "__main__":
    process_batch(
        input_dir="data/3dwire_raw",
        output_dir="data/3dwire_parts_combined",
        max_files=None,

        snap_tol=5e-4,
        wall_z_snap=0.03,

        # If you set floor_range_m=None, then this fixed thickness is used instead:
        floor_thickness=0.05,

        roof_z_q=0.85,
        roof_nonvertical_max=0.65,
        wall_vertical_cos=0.7,

        min_edges_per_room=6,
        interior_angle_deg=8.0,

        roof_clip_zpad=5.0,
        eave_z_snap=0.10,
        eave_search_span=0.35,
        eave_search_steps=12,

        prism_engine="earcut",
        boolean_engine="manifold",  # try "blender" if you have Blender installed
        boolean_check_volume=True,

        per_house_seed_mode="stem_hash",

        ext_range_m=(0.165, 0.210),
        int_range_m=(0.105, 0.130),

        # roof: keep discrete choices OR set roof_range_m=(min,max)
        roof_choices_m=(0.0111125, 0.0127, 0.015875),
        roof_range_m=None,

        # NEW:
        floor_range_m=(0.04, 0.07),
        attic_range_m=(0.02, 0.05),

        debug_roof=True,
        export_complete_ply=True,
    )
