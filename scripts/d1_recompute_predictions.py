"""G3-D1 correction: recompute every trajectory-state prediction on RAW part labels.

The GB200 trajectory records (source bundle af2ce9f7..., g3 file hash 2587d40b...)
were produced with `_channels` masking part labels by current occupancy. The G2
ensemble was trained on raw labels (surrogate.py:225, :250). Commit 00856c4 fixes
`_channels`. This script replays each record's geometry, verifies every selected
state against its frozen `state_occupancy_sha256`, and re-predicts with the fixed
representation. Geometry and solver fields are copied unchanged. No solver call
is made anywhere in this script.

Output is a NEW root. The inbound GB200 root is never written (invariant 5).
Each corrected record keeps the original prediction and original digest under
`d1_correction`, and carries a fresh `trajectory_digest` so the frozen
`_verified_json` reader accepts it.

Resumable: a record whose corrected file already exists and verifies is skipped.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ARCHIVE = Path("/Users/eric/workspace/sasto-modernization-control/archives/fea_ml.zip")
ARCHIVE_SHA = "79640406e1e0921c0ccfdc1df7ce51e05a8ecfece2ceccb7dec42c981736beda"
GB200 = Path("/Users/eric/workspace/sasto-g3-gb200-inbound/trajectory-calibration-gb200")
OUT = Path("/Users/eric/workspace/sasto-g3-gb200-d1corrected/trajectory-calibration-gb200-d1")
ENSEMBLE = REPO / "artifacts/g2/ensemble-v1"
FIX_COMMIT = "00856c4b1d5fb555ef586686de3c0e7dade007cb"
WORKERS = int(os.environ.get("D1_WORKERS", "8"))
ROLES = ("development", "calibration")

_PREDICTOR = None


def _predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        import torch
        torch.set_num_threads(1)
        from sasto.g3_trajectory_calibration import EnsemblePredictor
        _PREDICTOR = EnsemblePredictor(ensemble_root=ENSEMBLE,
                                       normalization_path=ENSEMBLE / "normalization-stats.json",
                                       device="cpu")
    return _PREDICTOR


def correct_family(record_path: str) -> dict:
    from sasto.activity_campaign import geometric_trajectory
    from sasto.g3_trajectory_calibration import _channels, family_seed

    record = json.loads(Path(record_path).read_text())
    sample_id, family_id = record["sample_id"], record["family_id"]
    selected = record.get("selected_states") or []
    if not selected:
        return {"path": record_path, "sample_id": sample_id, "predictions": [], "mismatched": 0, "seconds": 0.0}

    started = time.perf_counter()
    with zipfile.ZipFile(ARCHIVE) as archive:
        with archive.open(f"fea_ml/data/runs_real/{sample_id}/occ.npz") as handle:
            occupancy = np.load(io.BytesIO(handle.read()), allow_pickle=False)["data"]
        with archive.open(f"fea_ml/data/runs_real/{sample_id}/part.npz") as handle:
            parts = np.load(io.BytesIO(handle.read()), allow_pickle=False)["data"]

    _, volumes = geometric_trajectory(
        sample_id=sample_id, volume=occupancy.astype(bool),
        batch_cap=40, ranking_seed=family_seed(family_id),
    )

    predictor = _predictor()
    out, mismatched = [], 0
    for state in selected:
        volume = volumes.get(state["state_index"])
        if volume is None:
            mismatched += 1
            continue
        contiguous = np.ascontiguousarray(volume)
        digest = hashlib.sha256(contiguous.tobytes()).hexdigest()
        if digest != state["state_occupancy_sha256"]:
            mismatched += 1
            continue
        prediction = predictor.predict(_channels(contiguous.astype(np.bool_), parts))
        out.append({"state_index": state["state_index"], "state_occupancy_sha256": digest,
                    "prediction": prediction})
    return {"path": record_path, "sample_id": sample_id, "predictions": out,
            "mismatched": mismatched, "seconds": time.perf_counter() - started}


def write_corrected(record_path: Path, result: dict, g3_sha: str) -> Path:
    from sasto.g3_trajectory_calibration import _digest

    original = json.loads(record_path.read_text())
    corrected = copy.deepcopy(original)
    by_index = {p["state_index"]: p for p in result["predictions"]}
    originals = []
    for state in corrected["selected_states"]:
        new = by_index[state["state_index"]]
        assert new["state_occupancy_sha256"] == state["state_occupancy_sha256"]
        originals.append({"state_index": state["state_index"], "prediction": state["prediction"]})
        state["prediction"] = new["prediction"]
    corrected["d1_correction"] = {
        "defect": "G3-D1 train/inference channel mismatch",
        "fix_commit": FIX_COMMIT,
        "g3_trajectory_calibration_sha256": g3_sha,
        "original_trajectory_digest": original["trajectory_digest"],
        "original_predictions_masked_parts": originals,
        "predictor_device": "cpu",
        "solver_records_changed": False,
        "geometry_changed": False,
    }
    corrected.pop("trajectory_digest", None)
    corrected["trajectory_digest"] = _digest(corrected)
    dest = OUT / record_path.name
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(corrected, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, dest)
    return dest


def main() -> None:
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise SystemExit("source archive digest does not match the frozen pin")
    g3_file = REPO / "src/sasto/g3_trajectory_calibration.py"
    g3_sha = hashlib.sha256(g3_file.read_bytes()).hexdigest()
    if "parts * current" in g3_file.read_text():
        raise SystemExit("g3_trajectory_calibration.py still masks parts; refusing to run")
    OUT.mkdir(parents=True, exist_ok=True)

    identity = {"fix_commit": FIX_COMMIT, "g3_trajectory_calibration_sha256": g3_sha,
                "archive_sha256": ARCHIVE_SHA, "source_root": str(GB200), "batch_cap": 40}
    identity_path = OUT / "d1-correction-identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise SystemExit("correction identity mismatch; refusing to resume")
    else:
        identity_path.write_text(json.dumps(identity, indent=1, sort_keys=True))

    # Constants are copied by value so k6_coverage can read the same root, and so
    # the amendment-02 sensitivity check against the producing host's constants
    # still runs. They are untouched by D1 (computed on raw baselines).
    for name in ("kappa-development-evidence.json", "baseline-calibration.json", "campaign-manifest.json"):
        src, dst = GB200 / name, OUT / name
        if src.exists() and not dst.exists():
            dst.write_bytes(src.read_bytes())

    from sasto.g3_trajectory_calibration import _verified_json
    todo = []
    for role in ROLES:
        for path in sorted(GB200.glob(f"trajectory-{role}-*.json")):
            dest = OUT / path.name
            if dest.exists():
                try:
                    _verified_json(dest, "corrected case", "trajectory_digest")
                    continue
                except Exception:
                    dest.unlink()
            todo.append(path)
    print(f"{len(todo)} records to correct, {WORKERS} workers", flush=True)

    log = open(OUT / "d1-correction-log.jsonl", "a")
    total_mismatch = 0
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(correct_family, str(p)): p for p in todo}
        for i, future in enumerate(as_completed(futures)):
            path = futures[future]
            result = future.result()
            if result["mismatched"]:
                total_mismatch += result["mismatched"]
                log.write(json.dumps({"sample_id": result["sample_id"], "mismatched": result["mismatched"]}) + "\n")
                log.flush()
                continue
            dest = write_corrected(path, result, g3_sha)
            log.write(json.dumps({"sample_id": result["sample_id"], "written": dest.name,
                                  "seconds": round(result["seconds"], 2)}) + "\n")
            log.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                elapsed = time.perf_counter() - started
                rate = (i + 1) / elapsed
                print(f"  {i+1}/{len(todo)} | {rate*60:.1f} fam/min | ~{(len(todo)-i-1)/rate/60:.0f} min left",
                      flush=True)
    print(f"done; mismatched states: {total_mismatch}", flush=True)
    if total_mismatch:
        raise SystemExit("digest mismatches present; corrected root is incomplete")


if __name__ == "__main__":
    main()
