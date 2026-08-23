"""Canonical local digital topology: 6-foreground / 26-background."""

from __future__ import annotations

from collections import deque
from itertools import product
from typing import Iterable, Sequence, Tuple

import numpy as np


Coordinate = Tuple[int, int, int]

FACE_NEIGHBORS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
FULL_NEIGHBORS = tuple(
    (dx, dy, dz)
    for dx, dy, dz in product((-1, 0, 1), repeat=3)
    if (dx, dy, dz) != (0, 0, 0)
)


def _shape(volume: object) -> Coordinate:
    """Validate a nested Boolean grid or a three-dimensional bool ndarray."""
    if isinstance(volume, np.ndarray):
        if volume.ndim != 3 or volume.dtype != np.dtype(bool) or any(size == 0 for size in volume.shape):
            raise ValueError("volume must be a non-empty three-dimensional bool ndarray")
        return tuple(int(size) for size in volume.shape)
    if not isinstance(volume, Sequence) or isinstance(volume, (str, bytes)) or not volume:
        raise ValueError("volume must be a non-empty rectangular 3D Boolean grid")
    if not all(isinstance(plane, Sequence) and not isinstance(plane, (str, bytes)) and plane for plane in volume):
        raise ValueError("volume must be a non-empty rectangular 3D Boolean grid")
    z, y = len(volume), len(volume[0])
    if not all(len(plane) == y for plane in volume):
        raise ValueError("volume must be rectangular")
    if not all(
        isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and row
        for plane in volume
        for row in plane
    ):
        raise ValueError("volume must be a non-empty rectangular 3D Boolean grid")
    x = len(volume[0][0])
    if not all(len(row) == x for plane in volume for row in plane):
        raise ValueError("volume must be rectangular")
    if not all(isinstance(cell, (bool, np.bool_)) for plane in volume for row in plane for cell in row):
        raise ValueError("volume cells must be Boolean")
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


def _is_volume_boundary(cell: Coordinate, shape: Coordinate) -> bool:
    return any(index in (0, size - 1) for index, size in zip(cell, shape))


def _background_component_count(background: set[Coordinate], shape: Coordinate) -> int:
    """Count global 26-background components, including a connected exterior node."""
    exterior = object()
    unseen: set[object] = set(background)
    unseen.add(exterior)
    count = 0
    while unseen:
        count += 1
        queue = deque([unseen.pop()])
        while queue:
            cell = queue.popleft()
            neighbors: list[object]
            if cell is exterior:
                neighbors = [candidate for candidate in background if _is_volume_boundary(candidate, shape)]
            else:
                assert isinstance(cell, tuple)
                z, y, x = cell
                neighbors = [
                    (z + dz, y + dy, x + dx)
                    for dz, dy, dx in FULL_NEIGHBORS
                    if (z + dz, y + dy, x + dx) in background
                ]
                if _is_volume_boundary(cell, shape):
                    neighbors.append(exterior)
            for neighbor in neighbors:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    return count


def _is_simple_point_6_26(volume: object, point: Coordinate) -> bool:
    """Return whether deleting `point` preserves global 6/26 component counts.

    This deliberately scans the complete volume before and after the candidate
    deletion.  The global comparison rejects disconnected foreground cases that
    a 3x3x3 neighborhood cannot see, and includes an explicit exterior node in
    the 26-background count.  It is therefore correctness-first rather than a
    constant-time local predicate; performance optimization requires a future
    proof that preserves these exact global invariants.
    """
    z_size, y_size, x_size = _shape(volume)
    z, y, x = point
    if not (0 <= z < z_size and 0 <= y < y_size and 0 <= x < x_size):
        raise ValueError("point is outside volume")
    if not volume[z][y][x]:
        return False
    all_cells = {(nz, ny, nx) for nz in range(z_size) for ny in range(y_size) for nx in range(x_size)}
    foreground = {(nz, ny, nx) for nz, ny, nx in all_cells if volume[nz][ny][nx]}
    background = all_cells - foreground
    after_foreground = foreground - {point}
    after_background = background | {point}
    shape = (z_size, y_size, x_size)
    return _component_count(foreground, FACE_NEIGHBORS) == _component_count(
        after_foreground, FACE_NEIGHBORS
    ) and _background_component_count(background, shape) == _background_component_count(after_background, shape)


def is_simple_point_6_26(volume: object, point: Coordinate) -> bool:
    """Fail closed unless deletion preserves exact global 6/26 invariants."""
    try:
        return _is_simple_point_6_26(volume, point)
    except (IndexError, TypeError, ValueError):
        return False
