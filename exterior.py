"""
3DWire wireframe (.npz) -> meshes using ONLY planar clustering.

Outputs per input:
- {base}_exterior_walls.(stl/obj)
- {base}_interior_walls.(stl/obj)
- {base}_complete.(stl/ply)

NPZ supported:
A) vertices (N,3) + lines/edges (M,2) int indices
B) lines/edges (M,2,3) float endpoints (vertices optional)

No semantics required:
- Infer exterior vs interior walls by vertical-edge connected components:
  - Largest XY footprint component => exterior
  - Others => interior
"""

import numpy as np
from pathlib import Path
import trimesh
from collections import defaultdict
from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------
# Material presets (thickness unused by planar hull, but kept for parity)
# ---------------------------------------------------------------------
SEMANTIC_INFO = {
    0: {'name': 'unknown',        'color': [0.5, 0.5, 0.5], 'thickness': 0.10},
    1: {'name': 'exterior_wall',  'color': [0.8, 0.6, 0.4], 'thickness': 0.20},
    2: {'name': 'interior_wall',  'color': [0.9, 0.9, 0.9], 'thickness': 0.10},
}


# ---------------------------------------------------------------------
# Robust NPZ loading
# ---------------------------------------------------------------------
def _merge_vertices_and_build_edges(line_endpoints, tol=1e-5):
    """
    line_endpoints: (M,2,3) float
    Returns:
      vertices: (N,3) float32
      edges: (M,2) int32
    """
    pts = line_endpoints.reshape(-1, 3)
    q = np.round(pts / tol).astype(np.int64)
    uniq_q, inv = np.unique(q, axis=0, return_inverse=True)
    vertices = (uniq_q.astype(np.float64) * tol).astype(np.float32)
    edges = inv.reshape(-1, 2).astype(np.int32)
    return vertices, edges


def load_wireframe(npz_path: Path):
    """
    Returns:
      vertices: (N,3) float32
      edges:    (M,2) int32
      edge_semantics: None (kept for signature compatibility)
    """
    data = np.load(npz_path, allow_pickle=True)

    vertices = data.get("vertices", data.get("coords"))
    lines = data.get("lines", data.get("edges"))

    if vertices is None and lines is None:
        raise ValueError(f"Could not load vertices/lines from {npz_path}; keys={list(data.files)}")

    # Standard: vertices (N,3) + lines (M,2)
    if vertices is not None and lines is not None:
        v = np.asarray(vertices)
        l = np.asarray(lines)

        if v.ndim == 2 and v.shape[1] == 3 and l.ndim == 2 and l.shape[1] == 2:
            vertices = v.astype(np.float32, copy=False)
            edges = l.astype(np.int32, copy=False)
            return vertices, edges, None

        if l.ndim == 3 and l.shape[1:] == (2, 3):
            line_endpoints = l.astype(np.float32, copy=False)
            vertices, edges = _merge_vertices_and_build_edges(line_endpoints, tol=1e-5)
            return vertices, edges, None

        raise ValueError(
            f"Unsupported NPZ layout in {npz_path}; keys={list(data.files)}; "
            f"vertices shape={v.shape}; lines shape={l.shape}"
        )

    # Only endpoints exist
    if lines is not None:
        l = np.asarray(lines)
        if l.ndim == 3 and l.shape[1:] == (2, 3):
            line_endpoints = l.astype(np.float32, copy=False)
            vertices, edges = _merge_vertices_and_build_edges(line_endpoints, tol=1e-5)
            return vertices, edges, None

    raise ValueError(f"Unsupported NPZ layout in {npz_path}; keys={list(data.files)}")


# ============================================================================
# ONLY STRATEGY: Planar clustering + convex hull surface
# ============================================================================

def cluster_by_planarity(vertices, edges, z_tol=0.10, vertical_cos=0.7):
    """
    Simple planar clustering:
    - Horizontal-ish edges grouped by Z height bands (z_tol)
    - Vertical-ish edges grouped together (caller should pre-split walls!)
    """
    if len(edges) < 3:
        return []

    midpoints = vertices[edges].mean(axis=1)
    edge_vecs = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    edge_len = np.linalg.norm(edge_vecs, axis=1) + 1e-12
    vertical_mask = (np.abs(edge_vecs[:, 2]) / edge_len) > vertical_cos

    clusters = []

    # Horizontal edges by Z
    if np.any(~vertical_mask):
        horiz_edges = np.where(~vertical_mask)[0]
        horiz_z = midpoints[horiz_edges, 2]
        order = np.argsort(horiz_z)

        current = [horiz_edges[order[0]]]
        for i in range(1, len(order)):
            eidx = horiz_edges[order[i]]
            zprev = horiz_z[order[i - 1]]
            zcur = horiz_z[order[i]]
            if abs(zcur - zprev) < z_tol:
                current.append(eidx)
            else:
                if len(current) >= 3:
                    clusters.append(np.array(current, dtype=np.int32))
                current = [eidx]
        if len(current) >= 3:
            clusters.append(np.array(current, dtype=np.int32))

    # Vertical edges: one cluster
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


def planar_clustering_mesh(vertices, edges, z_tol=0.10, vertical_cos=0.7):
    """
    ONLY meshing method: planar clustering -> convex hull per cluster -> concatenate.
    """
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


def process_component(vertices, edges, info, z_tol=0.10, vertical_cos=0.7):
    name = info["name"]
    print(f"\n  Processing {name}: edges={len(edges)}")

    if len(edges) < 3:
        print("    ⚠️ Too few edges")
        return None

    mesh = planar_clustering_mesh(vertices, edges, z_tol=z_tol, vertical_cos=vertical_cos)
    if mesh is None or len(mesh.faces) == 0:
        print("    ❌ Planar clustering produced no mesh")
        return None

    color = (np.array(info["color"] + [1.0]) * 255).astype(np.uint8)
    mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))

    print(f"    ✅ Mesh: V={len(mesh.vertices)} F={len(mesh.faces)}")
    return mesh


# ============================================================================
# No-label inference: exterior vs interior walls
# ============================================================================

def vertical_edge_mask(vertices, edges, vertical_cos=0.7):
    v = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    L = np.linalg.norm(v, axis=1) + 1e-12
    return (np.abs(v[:, 2]) / L) > vertical_cos


def edge_connected_components(edges_subset):
    """
    Connected components by shared endpoints (vertex indices).
    edges_subset: (M,2) int32
    returns list of arrays of edge indices into edges_subset
    """
    v2e = defaultdict(list)
    for ei, (a, b) in enumerate(edges_subset):
        a = int(a); b = int(b)
        v2e[a].append(ei)
        v2e[b].append(ei)

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


# ============================================================================
# Export / main
# ============================================================================

def export_required_outputs(base_name, output_dir: Path, meshes: dict):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Required
    ext = meshes.get("exterior_wall")
    inter = meshes.get("interior_wall")

    if ext is not None:
        ext.export(output_dir / f"{base_name}_exterior_walls.stl")
        ext.export(output_dir / f"{base_name}_exterior_walls.obj")
    else:
        print("⚠️  No exterior wall mesh produced.")

    if inter is not None:
        inter.export(output_dir / f"{base_name}_interior_walls.stl")
        inter.export(output_dir / f"{base_name}_interior_walls.obj")
    else:
        print("⚠️  No interior wall mesh produced.")

    complete = trimesh.util.concatenate([m for m in meshes.values() if m is not None])
    complete.export(output_dir / f"{base_name}_complete.stl")
    complete.export(output_dir / f"{base_name}_complete.ply")

    print(f"\n✅ Exported: {base_name}_exterior_walls, {base_name}_interior_walls, {base_name}_complete")


def process_file(npz_path, output_dir, vertical_cos=0.7, z_tol=0.10):
    print(f"\n{'='*70}\nProcessing: {npz_path.name}\n{'='*70}")

    vertices, edges, _ = load_wireframe(npz_path)
    print(f"Loaded: vertices={len(vertices)} edges={len(edges)}")

    ext_edges, int_edges, other_edges = infer_exterior_interior_wall_edges(
        vertices, edges, vertical_cos=vertical_cos
    )

    meshes = {}

    # Walls (pre-split into components first, then planar clustering)
    ext_mesh = process_component(vertices, ext_edges, SEMANTIC_INFO[1], z_tol=z_tol, vertical_cos=vertical_cos)
    if ext_mesh is not None:
        meshes["exterior_wall"] = ext_mesh

    int_mesh = process_component(vertices, int_edges, SEMANTIC_INFO[2], z_tol=z_tol, vertical_cos=vertical_cos)
    if int_mesh is not None:
        meshes["interior_wall"] = int_mesh

    # Optional: keep the rest so _complete has more than just walls
    other_mesh = process_component(vertices, other_edges, SEMANTIC_INFO[0], z_tol=z_tol, vertical_cos=vertical_cos)
    if other_mesh is not None:
        meshes["other"] = other_mesh

    if not meshes:
        print("❌ No meshes created")
        return None

    export_required_outputs(npz_path.stem, Path(output_dir), meshes)
    return True


def process_batch(input_dir="data/3dwire_raw",
                  output_dir="data/3dwire_planar_meshes",
                  max_files=5,
                  vertical_cos=0.7,
                  z_tol=0.10):
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files in {input_dir}")

    if max_files:
        files = files[:max_files]

    print(f"\nPlanar-only conversion\nFiles: {len(files)}\nOutput: {output_dir}")

    ok = 0
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}]")
        try:
            if process_file(f, output_dir, vertical_cos=vertical_cos, z_tol=z_tol):
                ok += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ Done: {ok}/{len(files)} succeeded")
    print(f"📁 {Path(output_dir).absolute()}")


if __name__ == "__main__":
    # pip install numpy scipy trimesh
    process_batch(
        input_dir="data/3dwire_raw",
        output_dir="data/3dwire_planar_meshes",
        max_files=5,
        vertical_cos=0.7,
        z_tol=0.10
    )
