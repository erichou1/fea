"""6-foreground / exterior-aware-26-background topology contracts.

``exact_global_6_26`` is the authoritative, deliberately slow, full-volume
component-count oracle.  ``conservative_local_6_26`` is a separately named,
proof-backed *subset* for production admission: it can false-reject an exact
admissible deletion, but an accepted deletion preserves the oracle's two
component counts.  It is not an exact local replacement and does not claim
full digital homotopy, mesh-component, printability, or physical safety.
"""
from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable, Sequence, Tuple

import numpy as np

Coordinate = Tuple[int, int, int]
FACE_NEIGHBORS: tuple[Coordinate, ...] = (
    (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)
)
FULL_NEIGHBORS: tuple[Coordinate, ...] = tuple(
    (dz, dy, dx)
    for dz, dy, dx in product((-1, 0, 1), repeat=3)
    if (dz, dy, dx) != (0, 0, 0)
)
_LOCAL_INDEX = {offset: index for index, offset in enumerate(FULL_NEIGHBORS)}
_LOCAL_FACE_ADJACENCY: tuple[int, ...] = tuple(
    sum(
        1 << other_index
        for other_index, other in enumerate(FULL_NEIGHBORS)
        if abs(offset[0] - other[0]) + abs(offset[1] - other[1]) + abs(offset[2] - other[2]) == 1
    )
    for offset in FULL_NEIGHBORS
)
_LOCAL_FULL_ADJACENCY: tuple[int, ...] = tuple(
    sum(
        1 << other_index
        for other_index, other in enumerate(FULL_NEIGHBORS)
        if max(abs(offset[axis] - other[axis]) for axis in range(3)) == 1
    )
    for offset in FULL_NEIGHBORS
)
_LOCAL_FACE_BITS = sum(1 << _LOCAL_INDEX[offset] for offset in FACE_NEIGHBORS)


@dataclass(frozen=True)
class ExactTopologyPreflight:
    """Exact global input facts; caller policy, rather than this probe, decides rejection."""

    foreground_6_components: int
    background_26_components_with_exterior: int
    has_cavities: bool
    shape: tuple[int, int, int]
    occupied_count: int
    input_sha256: str
    boundary_semantics: str = "explicit_exterior_node_connected_to_boundary_background"
    topology_mode: str = "exact_global_6_26"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SequentialDeletionResult:
    """Result of rechecking each proposed deletion against the current volume."""

    volume: np.ndarray
    accepted_points: tuple[Coordinate, ...]
    rejected_points: tuple[Coordinate, ...]
    sequential_recheck: bool = True
    topology_mode: str = "conservative_local_6_26"


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
    z_size, y_size = len(volume), len(volume[0])
    if not all(len(plane) == y_size for plane in volume):
        raise ValueError("volume must be rectangular")
    if not all(
        isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and row
        for plane in volume
        for row in plane
    ):
        raise ValueError("volume must be a non-empty rectangular 3D Boolean grid")
    x_size = len(volume[0][0])
    if not all(len(row) == x_size for plane in volume for row in plane):
        raise ValueError("volume must be rectangular")
    if not all(isinstance(cell, (bool, np.bool_)) for plane in volume for row in plane for cell in row):
        raise ValueError("volume cells must be Boolean")
    return z_size, y_size, x_size


def _as_bool_volume(volume: object) -> np.ndarray:
    _shape(volume)
    return volume if isinstance(volume, np.ndarray) else np.asarray(volume, dtype=bool)


def _point_in(shape: Coordinate, point: object) -> bool:
    return (
        isinstance(point, tuple)
        and len(point) == 3
        and all(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) for value in point)
        and all(0 <= int(value) < size for value, size in zip(point, shape))
    )


def _component_count(mask: np.ndarray, neighbors: Iterable[Coordinate]) -> int:
    shape = tuple(int(size) for size in mask.shape)
    seen = np.zeros(shape, dtype=bool)
    total = 0
    offsets = tuple(neighbors)
    for start_raw in np.argwhere(mask):
        start = tuple(int(value) for value in start_raw)
        if seen[start]:
            continue
        total += 1
        seen[start] = True
        queue: deque[Coordinate] = deque([start])
        while queue:
            point = queue.popleft()
            for offset in offsets:
                neighbor = tuple(point[axis] + offset[axis] for axis in range(3))
                if _point_in(shape, neighbor) and mask[neighbor] and not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
    return total


def _is_volume_boundary(point: Coordinate, shape: Coordinate) -> bool:
    return any(index in (0, size - 1) for index, size in zip(point, shape))


def _background_component_count_with_exterior(foreground: np.ndarray) -> int:
    """Count 26-background components after adding one explicit exterior node."""
    background = ~foreground
    shape = tuple(int(size) for size in foreground.shape)
    exterior_reachable = np.zeros(shape, dtype=bool)
    queue: deque[Coordinate] = deque()
    for point_raw in np.argwhere(background):
        point = tuple(int(value) for value in point_raw)
        if _is_volume_boundary(point, shape):
            exterior_reachable[point] = True
            queue.append(point)
    while queue:
        point = queue.popleft()
        for offset in FULL_NEIGHBORS:
            neighbor = tuple(point[axis] + offset[axis] for axis in range(3))
            if _point_in(shape, neighbor) and background[neighbor] and not exterior_reachable[neighbor]:
                exterior_reachable[neighbor] = True
                queue.append(neighbor)
    # Exterior is always one component, including a fully solid finite volume.
    return 1 + _component_count(background & ~exterior_reachable, FULL_NEIGHBORS)


def _exact_global_6_26(volume: object, point: Coordinate) -> bool:
    foreground = _as_bool_volume(volume)
    shape = tuple(int(size) for size in foreground.shape)
    if not _point_in(shape, point) or not foreground[point]:
        return False
    before = (_component_count(foreground, FACE_NEIGHBORS), _background_component_count_with_exterior(foreground))
    after = foreground.copy()
    after[point] = False
    return before == (_component_count(after, FACE_NEIGHBORS), _background_component_count_with_exterior(after))


def exact_global_6_26(volume: object, point: Coordinate) -> bool:
    """Authoritative slow full-volume 6/26 component-count oracle; fail closed.

    This exact offline/preflight/reference API scans before and after the
    candidate deletion.  It accepts nested Boolean lists and 3-D bool NumPy
    arrays, preserves them, and returns ``False`` for malformed input/points.
    """
    try:
        return _exact_global_6_26(volume, point)
    except (IndexError, TypeError, ValueError):
        return False


def is_simple_point_6_26(volume: object, point: Coordinate) -> bool:
    """Backward-compatible alias for the exact, intentionally slow global oracle."""
    return exact_global_6_26(volume, point)


def _components_bits(mask: int, adjacency: tuple[int, ...]) -> int:
    components = 0
    unseen = mask
    while unseen:
        components += 1
        seed = unseen & -unseen
        unseen ^= seed
        stack = seed
        while stack:
            bit = stack & -stack
            stack ^= bit
            index = bit.bit_length() - 1
            additions = adjacency[index] & unseen
            unseen ^= additions
            stack |= additions
    return components


def _local_masks(foreground: np.ndarray, point: Coordinate) -> tuple[int, int, int]:
    shape = tuple(int(size) for size in foreground.shape)
    foreground_mask = background_mask = exterior_touch_mask = 0
    for index, offset in enumerate(FULL_NEIGHBORS):
        neighbor = tuple(point[axis] + offset[axis] for axis in range(3))
        bit = 1 << index
        if not _point_in(shape, neighbor):
            continue
        if foreground[neighbor]:
            foreground_mask |= bit
        else:
            background_mask |= bit
            if _is_volume_boundary(neighbor, shape):
                exterior_touch_mask |= bit
    return foreground_mask, background_mask, exterior_touch_mask


def _conservative_local_6_26(foreground: np.ndarray, point: Coordinate) -> bool:
    shape = tuple(int(size) for size in foreground.shape)
    if not _point_in(shape, point) or not foreground[point]:
        return False
    foreground_mask, background_mask, exterior_touch_mask = _local_masks(foreground, point)
    # Avoid deleting a singleton foreground component.  All locally incident
    # foreground components must be joined without the candidate under 6-adjacency.
    if not foreground_mask & _LOCAL_FACE_BITS or _components_bits(foreground_mask, _LOCAL_FACE_ADJACENCY) != 1:
        return False
    # Every local background component incident to the new background point must
    # already be joined under 26-adjacency.  At a boundary it must also be tied
    # to a boundary background witness, i.e. the explicit exterior node.
    if _components_bits(background_mask, _LOCAL_FULL_ADJACENCY) != 1:
        return False
    if _is_volume_boundary(point, shape) and not exterior_touch_mask:
        return False
    return True


def conservative_local_6_26(volume: object, point: Coordinate) -> bool:
    """Fast conservative local 6/26 admissibility predicate (may false-reject).

    Acceptance is sufficient, not necessary: the 26-bit neighbor proof joins
    every incident foreground component locally under 6-adjacency and every
    incident background component locally under 26-adjacency, with an exterior
    witness for boundary deletions.  It never claims equivalence to
    :func:`exact_global_6_26`.
    """
    try:
        return _conservative_local_6_26(_as_bool_volume(volume), point)
    except (IndexError, TypeError, ValueError):
        return False


def _mask_for_volume(mask: object, shape: Coordinate, name: str) -> np.ndarray:
    checked = _as_bool_volume(mask)
    if tuple(int(size) for size in checked.shape) != shape:
        raise ValueError("{} must be a bool volume with the same shape".format(name))
    return checked


def apply_conservative_deletions_sequentially(
    volume: object,
    points: Iterable[Coordinate],
    *,
    protected_mask: object | None = None,
    edit_mask: object | None = None,
) -> SequentialDeletionResult:
    """Copy ``volume`` and recheck the conservative gate after *each* proposal.

    ``protected_mask=True`` forbids deletion; ``edit_mask=False`` forbids it.
    The masks are intentionally separate policy inputs.  Proposed deletions are
    not validated as a set, because each accepted deletion changes the next
    candidate's local neighborhood.
    """
    source = _as_bool_volume(volume)
    current = source.copy()
    shape = tuple(int(size) for size in current.shape)
    protected = np.zeros(shape, dtype=bool) if protected_mask is None else _mask_for_volume(protected_mask, shape, "protected_mask")
    editable = np.ones(shape, dtype=bool) if edit_mask is None else _mask_for_volume(edit_mask, shape, "edit_mask")
    accepted: list[Coordinate] = []
    rejected: list[Coordinate] = []
    for point in points:
        normalized = tuple(int(value) for value in point) if isinstance(point, tuple) and len(point) == 3 else point
        if not _point_in(shape, normalized) or protected[normalized] or not editable[normalized] or not _conservative_local_6_26(current, normalized):
            rejected.append(point)  # type: ignore[arg-type]
            continue
        current[normalized] = False
        accepted.append(normalized)
    return SequentialDeletionResult(current, tuple(accepted), tuple(rejected))


def _input_digest(foreground: np.ndarray) -> str:
    shape_text = ",".join(str(int(size)) for size in foreground.shape)
    packed = np.packbits(foreground.reshape(-1), bitorder="little").tobytes()
    return hashlib.sha256(b"sasto-topology-6-26-v1\x00" + shape_text.encode("ascii") + b"\x00" + packed).hexdigest()


def exact_topology_preflight_6_26(volume: object) -> ExactTopologyPreflight:
    """Return exact topology facts without imposing connectedness/cavity policy."""
    foreground = _as_bool_volume(volume)
    shape = tuple(int(size) for size in foreground.shape)
    background_components = _background_component_count_with_exterior(foreground)
    return ExactTopologyPreflight(
        foreground_6_components=_component_count(foreground, FACE_NEIGHBORS),
        background_26_components_with_exterior=background_components,
        has_cavities=background_components > 1,
        shape=shape,
        occupied_count=int(np.count_nonzero(foreground)),
        input_sha256=_input_digest(foreground),
    )


def topology_artifact_record(
    preflight: ExactTopologyPreflight,
    *,
    campaign_hash: str,
    sequential_recheck: bool,
    topology_mode: str = "conservative_local_6_26",
) -> dict[str, object]:
    """Build explicit artifact fields for a conservative topology trajectory.

    The record retains the exact preflight facts and campaign identity.  It does
    not infer a connectedness/cavity policy and rejects malformed hashes rather
    than emitting an ambiguous evidence record.
    """
    if topology_mode != "conservative_local_6_26":
        raise ValueError("topology_mode must explicitly name conservative_local_6_26")
    if not isinstance(campaign_hash, str) or len(campaign_hash) != 64 or any(character not in "0123456789abcdef" for character in campaign_hash):
        raise ValueError("campaign_hash must be a lowercase SHA-256 digest")
    if not isinstance(sequential_recheck, bool):
        raise ValueError("sequential_recheck must be Boolean")
    return {
        "topology_mode": topology_mode,
        "exact_preflight": preflight.as_dict(),
        "campaign_hash": campaign_hash,
        "sequential_recheck": sequential_recheck,
    }


def make_foreground_remote_pair() -> tuple[np.ndarray, np.ndarray, Coordinate]:
    """Regression pair with the same 3³ window and different exact outcomes."""
    point = (3, 3, 3)
    split = np.zeros((7, 7, 7), dtype=bool)
    for cell in ((3, 3, 2), (3, 2, 2), (3, 1, 2), (3, 1, 4), (3, 2, 4), (3, 3, 4), point):
        split[cell] = True
    joined = split.copy()
    joined[3, 1, 3] = True
    return joined, split, point


def make_background_remote_pair() -> tuple[np.ndarray, np.ndarray, Coordinate]:
    """Complementary background regression pair with the same 3³ window."""
    point = (3, 3, 3)
    split = np.ones((7, 7, 7), dtype=bool)
    for cell in ((3, 3, 2), (3, 2, 2), (3, 1, 2), (3, 1, 4), (3, 2, 4), (3, 3, 4)):
        split[cell] = False
    joined = split.copy()
    joined[3, 1, 3] = False
    return joined, split, point
