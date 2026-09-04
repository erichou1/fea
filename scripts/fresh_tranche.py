"""Fresh-tranche pipeline: never-used 3DWire wireframes -> K6-format trajectory records.

Amendment 15 support. Converts the paper's re-analysis into an out-of-sample
test by running the whole chain on wireframes with IDs above 16008, which no
stage of this project has touched.

Chain, per wireframe:
  1. wireframe NPZ -> semantic STL parts     (frozen wireframe_to_volume.py,
                                              deterministic seed from SHA-256)
  2. STL parts -> 64^3 occupancy + labels    (frozen prepare_real_data.py)
  3. baseline solve                          (canonical Hex8, G3 activity config)
  4. baseline prediction                     (certified G2 ensemble)
  5. geometric trajectory to full depth      (geometric_trajectory, zero solves)
  6. one state per depth bin, frozen rule    (select_state_index)
  7. solve + predict each selected state

Output records use the G3 trajectory schema plus a `fresh_tranche` block, and
carry `role: "fresh"` so no existing K6 reader will pool them with any
certified role. They are written to a NEW root and never into any existing
artifact root.

Determinism: the legacy generator seeds thickness sampling from Python's
salted `hash(stem)`. We bypass that with `per_house_seed_mode="fixed"` and a
per-house seed derived from SHA-256 of the raw NPZ bytes, so a re-run on any
host reproduces the same house.

Depth bins here are the amendment-13 closed 5-point bins from 0 to 50%, not
the pre-registered K6 bins, because amendment 11 retracted every result built
on an open-ended tail bin. The K6 bins can be recovered from `fraction_removed`.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FROZEN_OPT = Path("/Users/eric/workspace/sasto-modernization-control/frozen-20260822/optimization")
LEGACY_ROOT = Path("/Users/eric/workspace/sasto-data/legacy-fea-20260217/fea_ml")
LEGACY_SCRIPTS = LEGACY_ROOT / "fea_ml/scripts"
WIRE_ARCHIVE = Path("/Users/eric/workspace/sasto-modernization-control/archives/3dwire_npz.zip")
WIRE_ARCHIVE_SHA = "af82d8560a7ef4ed328420fc864a1ef4028a51997f80a6de7a977788a8645f8e"
ENSEMBLE = REPO / "artifacts/g2/ensemble-v1"
FIRST_UNUSED_ID = 16009
RESOLUTION = 64
BATCH_CAP = 40  # frozen ceiling in activity_campaign.geometric_trajectory; identical to G3
FRESH_BINS = tuple((a / 100, (a + 5) / 100) for a in range(0, 50, 5))
FRESH_BIN_LABELS = tuple(f"({a}%,{a+5}%]" for a in range(0, 50, 5))
SCHEMA = "fresh-tranche-1.0.0"

# Generator kwargs: the frozen __main__ block of wireframe_to_volume.py, with
# the seed mode switched to fixed. Everything else verbatim.
GENERATOR_KWARGS = dict(
    snap_tol=5e-4, wall_z_snap=0.03, floor_thickness=0.05,
    roof_z_q=0.85, roof_nonvertical_max=0.65, wall_vertical_cos=0.7,
    min_edges_per_room=6, interior_angle_deg=8.0,
    roof_clip_zpad=5.0, eave_z_snap=0.10, eave_search_span=0.35, eave_search_steps=12,
    prism_engine="earcut", boolean_engine="manifold", boolean_check_volume=True,
    per_house_seed_mode="fixed",
    ext_range_m=(0.165, 0.210), int_range_m=(0.105, 0.130),
    roof_choices_m=(0.0111125, 0.0127, 0.015875), roof_range_m=None,
    floor_range_m=(0.04, 0.07), attic_range_m=(0.02, 0.05),
    debug_roof=False, export_complete_ply=False,
)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fresh_bin_index(fraction: float) -> int | None:
    for index, (a, b) in enumerate(FRESH_BINS):
        if a < fraction <= b:
            return index
    return None


def select_fresh_state(family_id: str, bin_index: int, state_indices: list[int]) -> int:
    """Same construction as the frozen G3 rule S (identifier-only SHA-256 argmin),
    under a fresh namespace because S validates against the six K6 bins and this
    tranche uses ten closed bins. Response independent by construction."""
    if not 0 <= bin_index < len(FRESH_BINS) or not state_indices or len(set(state_indices)) != len(state_indices):
        raise ValueError("invalid fresh bin or state indices")
    def key(state_index: int) -> str:
        return hashlib.sha256("\0".join(("sasto-fresh-sampling-v1", family_id, str(bin_index), str(state_index))).encode()).hexdigest()
    return min(state_indices, key=key)


def house_seed(npz_bytes: bytes) -> int:
    return int.from_bytes(hashlib.sha256(b"sasto-fresh-tranche-v1\0" + npz_bytes).digest()[:4], "little")


def list_unused_ids() -> list[str]:
    with zipfile.ZipFile(WIRE_ARCHIVE) as archive:
        names = [n for n in archive.namelist() if n.endswith(".npz")]
    ids = sorted(Path(n).stem for n in names)
    return [i for i in ids if i.isdigit() and int(i) >= FIRST_UNUSED_ID]


def _import_frozen():
    """Import the two legacy modules without letting them shadow anything."""
    sys.path.insert(0, str(FROZEN_OPT))
    sys.path.insert(0, str(LEGACY_ROOT))
    sys.path.insert(0, str(LEGACY_SCRIPTS))
    import wireframe_to_volume as w2v  # noqa: E402
    import prepare_real_data as prd  # noqa: E402
    for _ in range(3):
        sys.path.pop(0)
    return w2v, prd


def wireframe_to_voxels(sample_id: str, work: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Steps 1 and 2. Returns (occupancy bool, parts uint8, provenance)."""
    w2v, prd = _import_frozen()
    with zipfile.ZipFile(WIRE_ARCHIVE) as archive:
        npz_bytes = archive.read(f"{sample_id}.npz")
    raw_dir = work / "raw"
    parts_dir = work / "parts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    npz_path = raw_dir / f"{sample_id}.npz"
    npz_path.write_bytes(npz_bytes)
    seed = house_seed(npz_bytes)

    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        ok = w2v.process_file(npz_path, parts_dir, fixed_seed=seed, **GENERATOR_KWARGS)
    if not ok:
        raise RuntimeError("generator rejected wireframe")

    occupancy, labels, meta = prd.process_sample_parts(sample_id, parts_dir, RESOLUTION)
    if occupancy is None:
        raise RuntimeError("voxelizer produced nothing")
    occupancy = occupancy.astype(bool)
    labels = labels.astype(np.uint8)
    if not np.array_equal(labels > 0, occupancy):
        raise RuntimeError("label/occupancy mismatch after voxelization")
    part_files = sorted(p.name for p in parts_dir.glob(f"{sample_id}_*.stl"))
    provenance = {
        "wireframe_sha256": hashlib.sha256(npz_bytes).hexdigest(),
        "house_seed": seed, "seed_namespace": "sasto-fresh-tranche-v1",
        "generator_sha256": hashlib.sha256((FROZEN_OPT / "wireframe_to_volume.py").read_bytes()).hexdigest(),
        "voxelizer_sha256": hashlib.sha256((LEGACY_SCRIPTS / "prepare_real_data.py").read_bytes()).hexdigest(),
        "generator_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in GENERATOR_KWARGS.items()},
        "part_files": part_files,
        "voxel_meta": meta,
        "occupancy_sha256": hashlib.sha256(np.ascontiguousarray(occupancy).tobytes()).hexdigest(),
        "parts_sha256": hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest(),
        "occupied_voxels": int(occupancy.sum()),
    }
    return occupancy, labels, provenance


def solver_config(meta: dict, occupancy: np.ndarray):
    from sasto.activity_campaign import _activity_config
    from sasto.voxel_fea import VoxelFEAConfig
    v = float(meta["voxel_size"])
    base = VoxelFEAConfig(voxel_size=(v, v, v), youngs_modulus_pa=25.0e9, poisson_ratio=0.20,
                          density_kg_m3=2400.0, include_self_weight=True, fixed_total_force_n=(0.0, 0.0, -100.0))
    return _activity_config(occupancy, base)


def _predictor():
    import torch
    torch.set_num_threads(1)
    from sasto.g3_trajectory_calibration import EnsemblePredictor
    return EnsemblePredictor(ensemble_root=ENSEMBLE, normalization_path=ENSEMBLE / "normalization-stats.json", device="cpu")


def _scientific(solver: dict) -> dict:
    return {k: v for k, v in solver.items() if k != "timing"}


def run_family(sample_id: str, out_root: Path, predictor=None) -> dict:
    """Full chain for one wireframe. Writes one record; returns a summary."""
    from sasto.activity_campaign import geometric_trajectory
    from sasto.g3_trajectory_calibration import _channels, family_seed
    from sasto.voxel_fea import solve_voxels

    dest = out_root / f"trajectory-fresh-{sample_id}.json"
    if dest.exists():
        return {"sample_id": sample_id, "status": "exists"}
    started = time.perf_counter()
    predictor = predictor or _predictor()
    family_id = f"fresh-{sample_id}"

    with tempfile.TemporaryDirectory(prefix=f"fresh-{sample_id}-") as tmp:
        work = Path(tmp)
        try:
            occupancy, parts, provenance = wireframe_to_voxels(sample_id, work)
        except Exception as error:
            record = {"schema_version": SCHEMA, "role": "fresh", "sample_id": sample_id, "family_id": family_id,
                      "status": "geometry_failure", "reason": f"{type(error).__name__}: {error}"}
            _write(dest, record)
            return {"sample_id": sample_id, "status": "geometry_failure", "reason": record["reason"]}

    config = solver_config(provenance["voxel_meta"], occupancy)
    baseline_solver = _scientific(solve_voxels(occupancy, config))
    if baseline_solver.get("status") != "success":
        record = {"schema_version": SCHEMA, "role": "fresh", "sample_id": sample_id, "family_id": family_id,
                  "status": "baseline_unsolvable", "fresh_tranche": provenance, "baseline_solver": baseline_solver}
        _write(dest, record)
        return {"sample_id": sample_id, "status": "baseline_unsolvable", "reason": baseline_solver.get("reason")}
    baseline_prediction = predictor.predict(_channels(occupancy, parts))

    trajectory, volumes = geometric_trajectory(sample_id=sample_id, volume=occupancy, batch_cap=BATCH_CAP,
                                               ranking_seed=family_seed(family_id))
    if trajectory.get("solver_call_count") != 0:
        raise RuntimeError("geometry consulted the solver")
    per_bin: dict[int, list[int]] = {i: [] for i in range(len(FRESH_BINS))}
    by_index = {}
    for batch in trajectory["batches"]:
        idx = int(batch["batch_index"])
        b = fresh_bin_index(float(batch["proposed_material_reduction"]))
        if b is not None:
            per_bin[b].append(idx)
        by_index[idx] = batch

    selected, unsolved, solve_calls = [], [], 1
    for b, indices in per_bin.items():
        if not indices:
            continue
        state_index = select_fresh_state(family_id, b, indices)
        state = volumes[state_index]
        batch = by_index[state_index]
        solver = _scientific(solve_voxels(state, config))
        solve_calls += 1
        entry = {"bin_index": b, "bin_label": FRESH_BIN_LABELS[b], "state_index": state_index,
                 "fraction_removed": batch["proposed_material_reduction"],
                 "state_occupancy_sha256": hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()}
        if solver.get("status") != "success":
            if solver.get("reason") == "preconditioner_unavailable":
                raise RuntimeError("preconditioner unavailable; environment fault")
            unsolved.append({**entry, "solver": solver})
            continue
        entry["prediction"] = predictor.predict(_channels(state, parts))
        entry["solver"] = solver
        selected.append(entry)

    record = {
        "schema_version": SCHEMA, "role": "fresh", "sample_id": sample_id, "family_id": family_id,
        "family_seed": family_seed(family_id), "status": "complete",
        "fresh_tranche": provenance,
        "solver_config": {"fixed_total_force_n": [0.0, 0.0, -100.0], "include_self_weight": False,
                          "relative_tolerance": 2e-8, "voxel_size_m": float(provenance["voxel_meta"]["voxel_size"]),
                          "expected_loaded_node_count": config.expected_loaded_node_count},
        "baseline": {"prediction": baseline_prediction, "solver": baseline_solver},
        "trajectory": trajectory,
        "depth_bins": list(FRESH_BIN_LABELS),
        "selected_states": sorted(selected, key=lambda s: s["bin_index"]),
        "unsolved_states": unsolved,
        "intermediate_solver_call_count": 0,
        "selected_solver_call_count": solve_calls,
        "seconds": round(time.perf_counter() - started, 2),
    }
    _write(dest, record)
    return {"sample_id": sample_id, "status": "complete", "selected": len(selected), "unsolved": len(unsolved),
            "max_fraction": max((s["fraction_removed"] for s in selected), default=0.0),
            "seconds": record["seconds"]}


def _write(dest: Path, record: dict) -> None:
    record.pop("trajectory_digest", None)
    record["trajectory_digest"] = _digest(record)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, dest)


def _worker(args):
    sample_id, out_root = args
    return run_family(sample_id, Path(out_root))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--ids", nargs="*", default=None, help="explicit IDs (overrides count/offset)")
    args = ap.parse_args()

    if hashlib.sha256(WIRE_ARCHIVE.read_bytes()).hexdigest() != WIRE_ARCHIVE_SHA:
        raise SystemExit("3dwire archive digest does not match the frozen pin")
    for name in ("wireframe_to_volume.py",):
        if not (FROZEN_OPT / name).exists():
            raise SystemExit(f"frozen generator missing: {name}")
    if not (LEGACY_SCRIPTS / "prepare_real_data.py").exists():
        raise SystemExit("frozen voxelizer missing")
    if "sasto-g3-gb200-inbound" in str(args.out_root) or "artifacts/g3" in str(args.out_root):
        raise SystemExit("refusing to write fresh records into a certified root")

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    ids = args.ids if args.ids else list_unused_ids()[args.offset:args.offset + args.count]
    for i in ids:
        if int(i) < FIRST_UNUSED_ID:
            raise SystemExit(f"{i} is below the first never-used ID {FIRST_UNUSED_ID}")

    identity = {"schema": SCHEMA, "wire_archive_sha256": WIRE_ARCHIVE_SHA, "first_unused_id": FIRST_UNUSED_ID,
                "batch_cap": BATCH_CAP, "bins": list(FRESH_BIN_LABELS), "resolution": RESOLUTION,
                "generator_sha256": hashlib.sha256((FROZEN_OPT / "wireframe_to_volume.py").read_bytes()).hexdigest(),
                "voxelizer_sha256": hashlib.sha256((LEGACY_SCRIPTS / "prepare_real_data.py").read_bytes()).hexdigest(),
                "ensemble_root": str(ENSEMBLE),
                "g3_sha256": hashlib.sha256((REPO / "src/sasto/g3_trajectory_calibration.py").read_bytes()).hexdigest(),
                "activity_sha256": hashlib.sha256((REPO / "src/sasto/activity_campaign.py").read_bytes()).hexdigest(),
                "voxel_fea_sha256": hashlib.sha256((REPO / "src/sasto/voxel_fea.py").read_bytes()).hexdigest()}
    ident_path = out_root / "fresh-tranche-identity.json"
    if ident_path.exists():
        if json.loads(ident_path.read_text()) != identity:
            raise SystemExit("identity mismatch; refusing to resume into a root built by different code")
    else:
        ident_path.write_text(json.dumps(identity, indent=1, sort_keys=True))

    log = open(out_root / "fresh-tranche-log.jsonl", "a")
    started = time.perf_counter()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, (i, str(out_root))) for i in ids]
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            log.write(json.dumps(r) + "\n")
            log.flush()
            if done % 10 == 0 or done == len(ids):
                el = time.perf_counter() - started
                print(f"  {done}/{len(ids)} | {done/el*60:.1f} fam/min | ~{(len(ids)-done)/(done/el)/60:.0f} min left", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
