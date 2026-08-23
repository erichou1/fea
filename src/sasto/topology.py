"""Canonical local digital topology: 6-foreground / 26-background."""

from __future__ import annotations

from collections import deque
from itertools import product
from typing import Iterable, Sequence, Tuple


Coordinate = Tuple[int, int, int]

FACE_NEIGHBORS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
FULL_NEIGHBORS = tuple(
    (dx, dy, dz)
    for dx, dy, dz in product((-1, 0, 1), repeat=3)
    if (dx, dy, dz) != (0, 0, 0)
)


def _shape(volume: Sequence[Sequence[Sequence[bool]]]) -> Coordinate:
    if not volume or not volume[0] or not volume[0][0]:
        raise ValueError("volume must be a non-empty rectangular 3D grid")
    z = len(volume)
    y = len(volume[0])
    x = len(volume[0][0])
    if any(len(plane) != y or any(len(row) != x for row in plane) for plane in volume):
        raise ValueError("volume must be rectangular")
    return z, y, x


def _component_count(cells: set, neighbors: Iterable[Coordinate]) -> int:
    neighbors = tuple(neighbors)
    count = 0
    unseen = set(cells)
    while unseen:
        count += 1
        queue = deque([unseen.pop()])
        while queue:
            z, y, x = queue.popleft()
            for dz, dy, dx in neighbors:
                next_cell = (z + dz, y + dy, x + dx)
                if next_cell in unseen:
                    unseen.remove(next_cell)
                    queue.append(next_cell)
    return count


def is_simple_point_6_26(volume: Sequence[Sequence[Sequence[bool]]], point: Coordinate) -> bool:
    """Return whether deleting foreground `point` preserves local 6/26 topology.

    The foreground after deletion must be one 6-connected component and local
    background (including the deleted point) must be one 26-connected component.
    Isolated final-voxel deletion is rejected to retain one foreground component.
    """
    z_size, y_size, x_size = _shape(volume)
    z, y, x = point
    if not (0 <= z < z_size and 0 <= y < y_size and 0 <= x < x_size):
        raise ValueError("point is outside volume")
    if not volume[z][y][x]:
        return False
    foreground = set()
    background = set()
    for dz, dy, dx in product((-1, 0, 1), repeat=3):
        nz, ny, nx = z + dz, y + dy, x + dx
        if not (0 <= nz < z_size and 0 <= ny < y_size and 0 <= nx < x_size):
            continue
        local = (nz, ny, nx)
        if local == point or not volume[nz][ny][nx]:
            background.add(local)
        else:
            foreground.add(local)
    return _component_count(foreground, FACE_NEIGHBORS) == 1 and _component_count(
        background, FULL_NEIGHBORS
    ) == 1
