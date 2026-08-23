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


def _is_volume_boundary(cell: Coordinate, shape: Coordinate) -> bool:
    return any(index in (0, size - 1) for index, size in zip(cell, shape))


def _background_component_count(
    background: set[Coordinate], shape: Coordinate, exterior_touched: bool
) -> int:
    """Count local pre-deletion 26-background components with an exterior node."""
    exterior = object()
    unseen: set[object] = set(background)
    if exterior_touched:
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
                if exterior_touched and _is_volume_boundary(cell, shape):
                    neighbors.append(exterior)
            for neighbor in neighbors:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    return count


def is_simple_point_6_26(volume: Sequence[Sequence[Sequence[bool]]], point: Coordinate) -> bool:
    """Return whether deleting foreground `point` preserves local 6/26 topology.

    Foreground neighbors after deletion must form one 6-component.  Background
    is evaluated *before* deletion: its 26-components must already be one,
    with an explicit exterior component when the 3x3x3 stencil reaches outside
    the volume.  This prevents a boundary deletion from joining an enclosed
    cavity to exterior background through the deleted point.
    """
    z_size, y_size, x_size = _shape(volume)
    z, y, x = point
    if not (0 <= z < z_size and 0 <= y < y_size and 0 <= x < x_size):
        raise ValueError("point is outside volume")
    if not volume[z][y][x]:
        return False
    foreground = set()
    background = set()
    exterior_touched = False
    for dz, dy, dx in product((-1, 0, 1), repeat=3):
        nz, ny, nx = z + dz, y + dy, x + dx
        if not (0 <= nz < z_size and 0 <= ny < y_size and 0 <= nx < x_size):
            exterior_touched = True
            continue
        local = (nz, ny, nx)
        if local == point:
            continue
        if volume[nz][ny][nx]:
            foreground.add(local)
        else:
            background.add(local)
    return _component_count(foreground, FACE_NEIGHBORS) == 1 and _background_component_count(
        background, (z_size, y_size, x_size), exterior_touched
    ) == 1
