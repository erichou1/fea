"""Parallel replay of trajectory geometry into a digest-verified cache.

Amendment 07/08 support. Replays the deterministic erosion for every
development family in the frozen GB200 records across worker processes, and
writes each selected state's occupancy as packed bits together with its
verified state_occupancy_sha256. Inference then runs from the cache in one
pass with no replay in the loop.

Determinism note: geometric_trajectory is deterministic per family given
(sample_id, ranking_seed), so parallelism changes wall-clock and nothing else.
Every volume is digest-verified against its frozen record before it is
written, and the digest rides with the cache row for re-verification at read
time.
"""
from __future__ import annotations

import hashlib
import io
import json
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
OUT = REPO / "artifacts/g2b/replay-cache"
WORKERS = 6


def replay_family(record_path: str) -> dict:
    """Replay one family; return packed verified volumes for its selected states."""
    from sasto.activity_campaign import geometric_trajectory
    from sasto.g3_trajectory_calibration import family_seed

    record = json.loads(Path(record_path).read_text())
    sample_id, family_id = record["sample_id"], record["family_id"]
    selected = record.get("selected_states") or []
    if not selected:
        return {"sample_id": sample_id, "states": [], "mismatched": 0}

    with zipfile.ZipFile(ARCHIVE) as archive:
        with archive.open(f"fea_ml/data/runs_real/{sample_id}/occ.npz") as handle:
            occupancy = np.load(io.BytesIO(handle.read()), allow_pickle=False)["data"]
        with archive.open(f"fea_ml/data/runs_real/{sample_id}/part.npz") as handle:
            parts = np.load(io.BytesIO(handle.read()), allow_pickle=False)["data"]

    _, volumes = geometric_trajectory(
        sample_id=sample_id, volume=occupancy.astype(bool),
        batch_cap=40, ranking_seed=family_seed(family_id),
    )

    states, mismatched = [], 0
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
        states.append({
            "family_id": family_id, "sample_id": sample_id,
            "state_index": state["state_index"], "bin_label": state["bin_label"],
            "fraction_removed": state["fraction_removed"],
            "state_occupancy_sha256": digest,
            "occupancy_packed": np.packbits(contiguous.reshape(-1)).tobytes().hex(),
            "solver": state["solver"],
        })
    return {"sample_id": sample_id, "states": states, "mismatched": mismatched,
            "parts_packed": {str(level): np.packbits((parts == level).reshape(-1)).tobytes().hex()
                             for level in range(1, 6)} if states else {}}


def main() -> None:
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise SystemExit("source archive digest does not match the frozen pin")
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "replay-cache.jsonl"
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["sample_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"resuming: {len(done_ids)} families already cached", flush=True)

    records = [str(p) for p in sorted(GB200.glob("trajectory-development-*.json"))
               if json.loads(p.read_text())["sample_id"] not in done_ids] \
        if done_ids else [str(p) for p in sorted(GB200.glob("trajectory-development-*.json"))]

    total_states = total_mismatched = 0
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool, open(out_path, "a") as out:
        futures = {pool.submit(replay_family, p): p for p in records}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            total_states += len(result["states"])
            total_mismatched += result["mismatched"]
            if result["states"]:
                out.write(json.dumps(result) + "\n")
                out.flush()
            if (i + 1) % 50 == 0:
                rate = (i + 1) / (time.perf_counter() - started)
                remaining = (len(records) - i - 1) / rate / 60
                print(f"  {i+1}/{len(records)} | {total_states} states | "
                      f"{rate*60:.0f} fam/min | ~{remaining:.0f} min left", flush=True)

    print(f"cached {total_states} verified states, {total_mismatched} mismatched", flush=True)
    if total_mismatched:
        raise SystemExit("digest mismatches present; cache is not usable")


if __name__ == "__main__":
    main()
