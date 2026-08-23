"""Deterministic differential campaign for ``conservative_local_6_26``.

Run ``python -m sasto.topology_campaign --neighborhoods 1000000`` for the
promotion-scale local campaign.  The random campaign uses an independent exact
3-cubed bit reference; this is the exact global oracle for that finite volume,
not a claim that the production local predicate is globally complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from .topology import (
    FACE_NEIGHBORS,
    FULL_NEIGHBORS,
    apply_conservative_deletions_sequentially,
    conservative_local_6_26,
    exact_global_6_26,
    make_background_remote_pair,
    make_foreground_remote_pair,
)

_CENTER = (1, 1, 1)
_CUBE = tuple(product(range(3), repeat=3))
_CENTER_INDEX = _CUBE.index(_CENTER)
_CUBE_FACE_ADJACENCY = tuple(
    sum(1 << other_index for other_index, other in enumerate(_CUBE) if sum(abs(cell[i] - other[i]) for i in range(3)) == 1)
    for cell in _CUBE
)
_CUBE_FULL_ADJACENCY = tuple(
    sum(1 << other_index for other_index, other in enumerate(_CUBE) if max(abs(cell[i] - other[i]) for i in range(3)) == 1)
    for cell in _CUBE
)
_CUBE_BOUNDARY_MASK = sum(1 << index for index, cell in enumerate(_CUBE) if any(value in (0, 2) for value in cell))
_CUBE_ALL_MASK = (1 << 27) - 1
_OFFSETS = FULL_NEIGHBORS
_FACE_BITS = sum(1 << _OFFSETS.index(offset) for offset in FACE_NEIGHBORS)
_LOCAL_FACE_ADJACENCY = tuple(
    sum(1 << other_index for other_index, other in enumerate(_OFFSETS) if sum(abs(offset[i] - other[i]) for i in range(3)) == 1)
    for offset in _OFFSETS
)
_LOCAL_FULL_ADJACENCY = tuple(
    sum(1 << other_index for other_index, other in enumerate(_OFFSETS) if max(abs(offset[i] - other[i]) for i in range(3)) == 1)
    for offset in _OFFSETS
)


def _components(mask: int, adjacency: tuple[int, ...]) -> int:
    count = 0
    unseen = mask
    while unseen:
        count += 1
        bit = unseen & -unseen
        unseen ^= bit
        stack = bit
        while stack:
            bit = stack & -stack
            stack ^= bit
            additions = adjacency[bit.bit_length() - 1] & unseen
            unseen ^= additions
            stack |= additions
    return count


def _background_count(mask: int) -> int:
    background = _CUBE_ALL_MASK ^ mask
    reached = 0
    stack = background & _CUBE_BOUNDARY_MASK
    while stack:
        bit = stack & -stack
        stack ^= bit
        if reached & bit:
            continue
        reached |= bit
        stack |= _CUBE_FULL_ADJACENCY[bit.bit_length() - 1] & background & ~reached
    return 1 + _components(background & ~reached, _CUBE_FULL_ADJACENCY)


def _exact_3cube_pattern(pattern: int) -> bool:
    mask = 1 << _CENTER_INDEX
    for index, offset in enumerate(_OFFSETS):
        if pattern & (1 << index):
            mask |= 1 << _CUBE.index(tuple(1 + value for value in offset))
    before = (_components(mask, _CUBE_FACE_ADJACENCY), _background_count(mask))
    after = mask ^ (1 << _CENTER_INDEX)
    return before == (_components(after, _CUBE_FACE_ADJACENCY), _background_count(after))


def _conservative_pattern(pattern: int) -> bool:
    return bool(pattern & _FACE_BITS) and _components(pattern, _LOCAL_FACE_ADJACENCY) == 1 and _components(((1 << 26) - 1) ^ pattern, _LOCAL_FULL_ADJACENCY) == 1


def _array_from_pattern(pattern: int) -> np.ndarray:
    volume = np.zeros((3, 3, 3), dtype=bool)
    volume[_CENTER] = True
    for index, offset in enumerate(_OFFSETS):
        if pattern & (1 << index):
            volume[tuple(1 + value for value in offset)] = True
    return volume


def _exhaustive_two_cube() -> dict[str, int]:
    result = {"cases": 0, "false_accepts": 0, "exact_only_false_rejects": 0}
    for bits in range(1 << 8):
        volume = np.array([(bits >> index) & 1 for index in range(8)], dtype=bool).reshape(2, 2, 2)
        for point in product(range(2), repeat=3):
            exact = exact_global_6_26(volume, point)
            fast = conservative_local_6_26(volume, point)
            result["cases"] += 1
            result["false_accepts"] += int(fast and not exact)
            result["exact_only_false_rejects"] += int(exact and not fast)
    return result


def _witnesses() -> dict[str, object]:
    result: dict[str, object] = {}
    for name, maker in (("foreground_remote", make_foreground_remote_pair), ("background_remote", make_background_remote_pair)):
        joined, split, point = maker()
        result[name] = {
            "identical_local_window": bool(np.array_equal(joined[2:5, 2:5, 2:5], split[2:5, 2:5, 2:5])),
            "joined_exact": exact_global_6_26(joined, point),
            "split_exact": exact_global_6_26(split, point),
            "joined_conservative": conservative_local_6_26(joined, point),
            "split_conservative": conservative_local_6_26(split, point),
        }
    cavity = np.ones((3, 3, 3), dtype=bool); cavity[1, 1, 1] = False
    isolated = np.array([[[False, False], [False, True]], [[True, True], [False, False]]], dtype=bool)
    result["historical"] = {
        "cavity": {"exact": exact_global_6_26(cavity, (0, 0, 0)), "conservative": conservative_local_6_26(cavity, (0, 0, 0))},
        "isolated": {"exact": exact_global_6_26(isolated, (0, 1, 1)), "conservative": conservative_local_6_26(isolated, (0, 1, 1))},
    }
    return result


def _sequential_check() -> dict[str, object]:
    volume = np.zeros((5, 5, 5), dtype=bool)
    volume[2, 2, 1:4] = True
    points = ((2, 2, 1), (2, 2, 3))
    helper = apply_conservative_deletions_sequentially(volume, points)
    current = volume.copy()
    false_accepts = exact_only_false_rejects = accepted = 0
    for point in points:
        fast = conservative_local_6_26(current, point)
        exact = exact_global_6_26(current, point)
        false_accepts += int(fast and not exact)
        exact_only_false_rejects += int(exact and not fast)
        if fast:
            current[point] = False
            accepted += 1
    if helper.accepted_points != tuple(point for point in points if point in helper.accepted_points) or accepted != len(helper.accepted_points):
        raise AssertionError("sequential helper did not recheck the current volume")
    return {"sequential_recheck": helper.sequential_recheck, "cases": len(points), "accepted": accepted, "rejected": len(helper.rejected_points), "false_accepts": false_accepts, "exact_only_false_rejects": exact_only_false_rejects}


def _benchmark(data_root: Path) -> dict[str, object]:
    candidates = sorted(data_root.rglob("occ.npz"))[:10]
    if len(candidates) != 10:
        raise ValueError("data root must contain at least ten occ.npz files: {}".format(data_root))
    samples: list[dict[str, object]] = []
    for index, path in enumerate(candidates):
        with np.load(path, allow_pickle=False) as payload:
            if payload.files != ["data"]:
                raise ValueError("unexpected NPZ schema at {}".format(path))
            raw = payload["data"]
        if raw.shape != (64, 64, 64) or raw.dtype != np.dtype(np.uint8) or not np.isin(raw, (0, 1)).all():
            raise ValueError("expected uint8 {{0,1}} 64-cubed occupancy at {}".format(path))
        volume = raw.astype(bool, copy=True)
        occupied = np.argwhere(volume)
        if not len(occupied):
            raise ValueError("empty occupancy cannot benchmark production gate: {}".format(path))
        selected = occupied[: min(20_000, len(occupied))]
        points = [tuple(int(value) for value in row) for row in selected]
        for point in points[: min(1_000, len(points))]:
            conservative_local_6_26(volume, point)
        start = time.perf_counter()
        accepted = sum(conservative_local_6_26(volume, point) for point in points)
        elapsed = time.perf_counter() - start
        samples.append({"sample_id": path.parent.name, "relative_path": str(path.relative_to(data_root)), "candidates": len(points), "accepted": accepted, "tests_per_s": len(points) / elapsed})
    rates = [float(sample["tests_per_s"]) for sample in samples]
    return {"input_status": "EXPLICIT_DATA_ROOT_OCC_NPZ", "data_root": str(data_root), "samples": samples, "tests_per_s_min": min(rates), "tests_per_s_median": statistics.median(rates), "tests_per_s_max": max(rates), "meets_1000_tests_per_s": min(rates) >= 1000.0}


def _campaign_hash(identity: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def run_campaign(*, neighborhoods: int = 1_000_000, data_root: Path | None = None, seed: int = 1_103_515_245) -> dict[str, object]:
    """Run deterministic source-reference differential evidence and optional real benchmark."""
    if neighborhoods < 1:
        raise ValueError("neighborhoods must be positive")
    started = time.perf_counter()
    exhaustive = _exhaustive_two_cube()
    rng = np.random.default_rng(seed)
    false_accepts = exact_only_false_rejects = exact_accepts = conservative_accepts = 0
    for _ in range(neighborhoods):
        pattern = int(rng.integers(0, 1 << 26, dtype=np.uint64))
        exact = _exact_3cube_pattern(pattern)
        fast = _conservative_pattern(pattern)
        false_accepts += int(fast and not exact)
        exact_only_false_rejects += int(exact and not fast)
        exact_accepts += int(exact)
        conservative_accepts += int(fast)
    witnesses = _witnesses()
    sequential = _sequential_check()
    identity = {"schema_version": "1.0.0", "topology_mode": "conservative_local_6_26", "oracle": "exact_global_6_26", "neighborhoods": neighborhoods, "seed": seed, "exhaustive": exhaustive, "random": {"false_accepts": false_accepts, "exact_only_false_rejects": exact_only_false_rejects, "exact_accepts": exact_accepts, "conservative_accepts": conservative_accepts}, "witnesses": witnesses, "sequential_recheck": True, "sequential_batch": sequential}
    benchmark = {"input_status": "SKIPPED_NO_EXPLICIT_DATA_ROOT"} if data_root is None else _benchmark(Path(data_root))
    total_false_accepts = exhaustive["false_accepts"] + false_accepts
    total_false_rejects = exhaustive["exact_only_false_rejects"] + exact_only_false_rejects
    random_local_neighborhoods = {"cases": neighborhoods, "false_accepts": false_accepts, "exact_only_false_rejects": exact_only_false_rejects, "exact_accepts": exact_accepts, "conservative_accepts": conservative_accepts, "reference": "independent_exact_global_3cubed_bit_oracle"}
    return {**identity, "campaign_hash": _campaign_hash(identity), "random_local_neighborhoods": random_local_neighborhoods, "false_accepts": total_false_accepts, "exact_only_false_rejects": total_false_rejects, "recall": conservative_accepts / exact_accepts if exact_accepts else 1.0, "benchmark_64cubed": benchmark, "campaign_elapsed_s": time.perf_counter() - started, "promotion_verdict": "PASS_SOUNDNESS" if total_false_accepts == 0 else "FAIL_FALSE_ACCEPT"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic conservative 6/26 topology campaign")
    parser.add_argument("--neighborhoods", type=int, default=1_000_000)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_campaign(neighborhoods=args.neighborhoods, data_root=args.data_root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["promotion_verdict"] == "PASS_SOUNDNESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
