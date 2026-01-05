"""
Extract 2D footprint and room polygons from 3D house wireframe NPZ.

- Projects vertices to XY.
- Builds a 2D graph from lines.
- Finds simple cycles.
- Chooses the largest cycle as building footprint.
- Treats other cycles inside the footprint as candidate rooms.

This is Step 1 of the pipeline: recover interior structure in 2D.
"""

import numpy as np
from pathlib import Path
from collections import defaultdict, deque
import trimesh


def load_wireframe_2d(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    vertices = data.get("vertices", data.get("coords"))
    edges = data.get("lines", data.get("edges"))
    if vertices is None or edges is None:
        raise ValueError(f"Could not load vertices/lines from {npz_path}")
    # Project to XY plane
    verts_2d = vertices[:, :2]
    return verts_2d, vertices, edges


def build_adjacency(edges):
    adj = defaultdict(set)
    for i, (u, v) in enumerate(edges):
        adj[u].add((v, i))
        adj[v].add((u, i))
    return adj


def find_simple_cycles_undirected(edges, max_len=40, max_cycles=5000):
    """
    Very simple undirected cycle finder (not minimum / unique, but useful).
    Returns cycles as lists of vertex indices (closed, no repeated last).
    """
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    cycles = set()

    def dfs(start, current, visited, parent):
        if len(visited) > max_len:
            return
        for nxt in adj[current]:
            if nxt == parent:
                continue
            if nxt == start and len(visited) >= 3:
                cyc = tuple(sorted(visited))
                cycles.add(cyc)
            elif nxt not in visited:
                dfs(start, nxt, visited + [nxt], current)

    for v in range(len(adj)):
        dfs(v, v, [v], -1)
        if len(cycles) >= max_cycles:
            break

    # Convert to list of ordered cycles (not sorted)
    unique_cycles = []
    seen = set()
    for cyc in cycles:
        # Normalize by rotation to canonical form
        cyc_list = list(cyc)
        n = len(cyc_list)
        # Rotate so smallest index first
        min_idx = min(range(n), key=lambda i: cyc_list[i])
        rot = cyc_list[min_idx:] + cyc_list[:min_idx]
        key = tuple(rot)
        if key not in seen:
            seen.add(key)
            unique_cycles.append(rot)

    return unique_cycles


def polygon_area_2d(points):
    """
    Signed area of 2D polygon (x, y).
    Positive if CCW, negative if CW.
    """
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def point_in_polygon(point, poly):
    """
    Ray casting in 2D.
    point: (2,)
    poly: (N, 2)
    """
    x, y = point
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and \
           (x < (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-9) + x1):
            inside = not inside
    return inside


def classify_cycles_as_footprint_and_rooms(verts_2d, cycles):
    """
    Given all cycles, pick:
    - footprint: largest-area cycle
    - rooms: cycles entirely inside footprint (smaller area)
    """
    if not cycles:
        return None, []

    polys = []
    for cyc in cycles:
        pts = verts_2d[np.array(cyc)]
        area = abs(polygon_area_2d(pts))
        polys.append((cyc, pts, area))

    # footprint = largest area polygon
    polys.sort(key=lambda x: x[2], reverse=True)
    footprint_cyc, footprint_pts, _ = polys[0]

    rooms = []
    for cyc, pts, area in polys[1:]:
        if area < 1e-6:
            continue
        # test centroid inside footprint
        centroid = pts.mean(axis=0)
        if point_in_polygon(centroid, footprint_pts):
            rooms.append((cyc, pts, area))

    return (footprint_cyc, footprint_pts), rooms


def cycles_to_lines_mesh(verts_2d, cycles, out_path):
    """
    Save 2D cycles as a 3D line mesh at z=0 for debugging in a 3D viewer.
    """
    if not cycles:
        print(f"No cycles to save for {out_path.name}")
        return

    # Build vertices in 3D (z=0)
    verts_3d = np.c_[verts_2d, np.zeros(len(verts_2d))]
    line_segments = []

    for cyc in cycles:
        idxs = list(cyc) + [cyc[0]]
        for i in range(len(idxs) - 1):
            line_segments.append([idxs[i], idxs[i + 1]])

    line_segments = np.array(line_segments, dtype=np.int64)

    # Represent lines as very thin cylinders for visualization
    seg_meshes = []
    radius = 0.002
    for u, v in line_segments:
        p1 = verts_3d[u]
        p2 = verts_3d[v]
        if np.allclose(p1, p2):
            continue
        seg = trimesh.creation.cylinder(radius=radius, segment=(p1, p2))
        seg_meshes.append(seg)

    if not seg_meshes:
        print(f"No valid segments to save for {out_path.name}")
        return

    combined = trimesh.util.concatenate(seg_meshes)
    combined.export(out_path)
    print(f"Saved cycles as line mesh: {out_path}")


def extract_and_save_layout(npz_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = npz_path.stem

    verts_2d, verts_3d, edges = load_wireframe_2d(npz_path)
    print(f"Loaded: {len(verts_3d)} verts, {len(edges)} edges from {npz_path.name}")

    # Find cycles in 2D
    cycles = find_simple_cycles_undirected(edges)
    print(f"Found {len(cycles)} raw cycles")

    if not cycles:
        return

    # Classify into footprint + rooms
    footprint, rooms = classify_cycles_as_footprint_and_rooms(verts_2d, cycles)
    if footprint is None:
        print("No footprint found")
        return

    footprint_cyc, footprint_pts = footprint
    print(f"Footprint: {len(footprint_cyc)} vertices, {len(rooms)} room candidates")

    # Save footprint-only mesh
    cycles_to_lines_mesh(verts_2d, [footprint_cyc], output_dir / f"{base}_footprint_2d_lines.stl")

    # Save all rooms-only mesh
    room_cycles = [cyc for (cyc, pts, area) in rooms]
    if room_cycles:
        cycles_to_lines_mesh(verts_2d, room_cycles, output_dir / f"{base}_rooms_2d_lines.stl")

    # Save all cycles for debugging
    cycles_to_lines_mesh(verts_2d, [c for c in cycles], output_dir / f"{base}_allcycles_2d_lines.stl")


def process_batch_layout(input_dir="data/3dwire_raw",
                         output_dir="data/3dwire_layout_2d",
                         max_files=5):
    input_dir = Path(input_dir)
    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files in {input_dir}")

    if max_files:
        npz_files = npz_files[:max_files]

    print(f"Extracting 2D layout (footprint + rooms) from {len(npz_files)} files")
    for i, f in enumerate(npz_files, 1):
        print(f"\n[{i}/{len(npz_files)}] {f.name}")
        try:
            extract_and_save_layout(f, output_dir)
        except Exception as e:
            print(f"  FAILED on {f.name}: {e}")


if __name__ == "__main__":
    process_batch_layout(
        input_dir="data/3dwire_raw",
        output_dir="data/3dwire_layout_2d",
        max_files=5
    )
