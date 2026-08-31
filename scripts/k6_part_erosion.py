"""Frozen record of where erosion removes material, by semantic part.

Regenerates the trajectory for each family, verifies every selected state
against its frozen state_occupancy_sha256, and then measures which semantic
parts the erosion consumed. Nothing here is recomputed from a prediction, so
the figures and the Setup paragraph both cite this record.

Part identity comes from the archive's part.npz. It was verified geometrically
before being named: part 1 sits at the footprint perimeter (77% of its voxels
within 3 of the edge), part 2 is interior, part 3 is the top z band, part 4 the
thin base band. Note the campaign computes its OWN protected set from the
min/max occupied x layers and never reads protected_mask.npz, so protection is
not part of the naming evidence.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/eric/workspace/fea-sasto-v/src")
from sasto.activity_campaign import geometric_trajectory
from sasto.g3_trajectory_calibration import family_seed

ARCHIVE = Path("/Users/eric/workspace/sasto-modernization-control/archives/fea_ml.zip")
INBOUND = Path("/Users/eric/workspace/sasto-g3-gb200-inbound/trajectory-calibration-gb200")
OUT = Path("/Users/eric/workspace/sasto-modernization-control/v2/g3"
           "/k6-part-erosion.json")

SAMPLES = ["00001", "00005", "00010", "00023"]
NAMES = {1: "exterior_walls", 2: "interior_partitions", 3: "roof",
         4: "floor_slabs"}


def main() -> int:
    z = zipfile.ZipFile(ARCHIVE)

    def member(sid: str, leaf: str) -> np.ndarray:
        with np.load(io.BytesIO(z.read(f"fea_ml/data/runs_real/{sid}/{leaf}")),
                     allow_pickle=False) as loaded:
            return loaded["data"]

    out: dict = {
        "fea_ml_zip_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "part_semantics": NAMES,
        "note": ("part identity verified geometrically; the campaign computes "
                 "its own protected set from min/max occupied x layers and "
                 "does not read protected_mask.npz"),
        "per_sample": {},
    }
    frac_of_part: dict[int, list[float]] = {k: [] for k in NAMES}
    share_of_removed: dict[int, list[float]] = {k: [] for k in NAMES}

    for sid in SAMPLES:
        rec = json.loads((INBOUND / f"trajectory-development-{sid}.json").read_text())
        base = member(sid, "occ.npz").astype(bool)
        parts = member(sid, "part.npz")
        _, states = geometric_trajectory(
            sample_id=sid, volume=base, batch_cap=40,
            ranking_seed=family_seed(rec["family_id"]))
        for s in rec["selected_states"]:
            got = hashlib.sha256(states[s["state_index"]].tobytes()).hexdigest()
            if got != s["state_occupancy_sha256"]:
                raise SystemExit(f"digest mismatch {sid} state {s['state_index']}")

        deep = max(rec["selected_states"], key=lambda s: s["fraction_removed"])
        v = states[deep["state_index"]]
        gone = base & ~v

        entry = {
            "state_index": deep["state_index"],
            "state_occupancy_sha256": deep["state_occupancy_sha256"],
            "fraction_removed": deep["fraction_removed"],
            "parts": {},
        }
        for pid, name in NAMES.items():
            m = (parts == pid) & base
            f = float((gone & m).sum() / max(1, m.sum()))
            sh = float((gone & m).sum() / max(1, gone.sum()))
            entry["parts"][name] = {"fraction_of_part_removed": f,
                                    "share_of_all_removal": sh}
            frac_of_part[pid].append(f)
            share_of_removed[pid].append(sh)
        out["per_sample"][sid] = entry

    out["mean_fraction_of_part_removed"] = {
        NAMES[p]: float(np.mean(v)) for p, v in frac_of_part.items()}
    out["mean_share_of_all_removal"] = {
        NAMES[p]: float(np.mean(v)) for p, v in share_of_removed.items()}
    out["interior_share"] = float(
        np.mean(share_of_removed[2]) + np.mean(share_of_removed[3]))

    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT.name}")
    for name, v in out["mean_fraction_of_part_removed"].items():
        print(f"  {name:22s} {v * 100:5.1f}% of part removed")
    print(f"  partitions+roof share of all removal: "
          f"{out['interior_share'] * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
