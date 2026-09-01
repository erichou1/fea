"""Amendment 06: where the optimizer removes material.

Frozen at K6_AMENDMENT_06_SPATIAL_PATTERN.md, SHA-256
eb7dc9ad21bbecc5c3b96c6e2af675238418dee420e38cc581f0cb3bba277d93, before any
statistic here was computed.

EXPLORATORY. These states already produced the primary K6 result and five
amendments, so nothing here can be confirmatory. The purpose is descriptive: to
anchor the paper's geometric language to a measurement.

Profiles computed, all over each family's deepest verified state:
  (a) removal against normalized height
  (b) removal against normalized distance from the footprint boundary
  (c) removal against baseline 6-neighbour count
  (d) Gini of removal across height bands, per family
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
           "/k6-spatial-pattern.json")

AMENDMENT = "eb7dc9ad21bbecc5c3b96c6e2af675238418dee420e38cc581f0cb3bba277d93"
N_HEIGHT = 10
N_PERIM = 6
N_FAMILIES = 120     # sampled deterministically; full 2,096 replay is hours


def neighbour_count(v: np.ndarray) -> np.ndarray:
    """6-neighbour occupancy count for every voxel."""
    c = np.zeros(v.shape, dtype=np.int8)
    c[1:, :, :] += v[:-1, :, :]
    c[:-1, :, :] += v[1:, :, :]
    c[:, 1:, :] += v[:, :-1, :]
    c[:, :-1, :] += v[:, 1:, :]
    c[:, :, 1:] += v[:, :, :-1]
    c[:, :, :-1] += v[:, :, 1:]
    return c


def perimeter_distance(footprint: np.ndarray) -> np.ndarray:
    """Chebyshev distance from the footprint edge, computed by erosion peeling."""
    dist = np.zeros(footprint.shape, dtype=np.int16)
    cur = footprint.copy()
    d = 0
    while cur.any():
        inner = cur.copy()
        inner[0, :] = False; inner[-1, :] = False
        inner[:, 0] = False; inner[:, -1] = False
        shrunk = (inner
                  & np.roll(cur, 1, 0) & np.roll(cur, -1, 0)
                  & np.roll(cur, 1, 1) & np.roll(cur, -1, 1))
        dist[cur & ~shrunk] = d
        cur = shrunk
        d += 1
        if d > 64:
            break
    return dist


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() <= 0:
        return 0.0
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def main() -> int:
    z = zipfile.ZipFile(ARCHIVE)
    files = sorted(INBOUND.glob("trajectory-development-*.json"))
    step = max(1, len(files) // N_FAMILIES)
    picked = files[::step][:N_FAMILIES]

    h_removed = np.zeros(N_HEIGHT); h_total = np.zeros(N_HEIGHT)
    p_removed = np.zeros(N_PERIM); p_total = np.zeros(N_PERIM)
    n_removed = np.zeros(7); n_total = np.zeros(7)
    ginis: list[float] = []
    used = 0

    for f in picked:
        rec = json.loads(f.read_text())
        sid = rec["sample_id"]
        try:
            raw = z.read(f"fea_ml/data/runs_real/{sid}/occ.npz")
        except KeyError:
            continue
        with np.load(io.BytesIO(raw), allow_pickle=False) as loaded:
            base = loaded["data"].astype(bool)
        _, states = geometric_trajectory(
            sample_id=sid, volume=base, batch_cap=40,
            ranking_seed=family_seed(rec["family_id"]))
        ok = True
        for s in rec["selected_states"]:
            got = hashlib.sha256(states[s["state_index"]].tobytes()).hexdigest()
            if got != s["state_occupancy_sha256"]:
                ok = False
                break
        if not ok:
            raise SystemExit(f"digest mismatch for {sid}")

        deep = max(rec["selected_states"], key=lambda s: s["fraction_removed"])
        v = states[deep["state_index"]]
        gone = base & ~v
        xs, ys, zs = np.where(base)

        # (a) height
        z0, z1 = zs.min(), zs.max()
        znorm = (zs - z0) / max(1, z1 - z0)
        hb = np.clip((znorm * N_HEIGHT).astype(int), 0, N_HEIGHT - 1)
        gvals = gone[xs, ys, zs]
        per_band = np.zeros(N_HEIGHT)
        for b in range(N_HEIGHT):
            m = hb == b
            h_total[b] += m.sum()
            r = gvals[m].sum()
            h_removed[b] += r
            per_band[b] = r
        ginis.append(gini(per_band))

        # (b) distance from footprint boundary
        foot = base.any(axis=2)
        pd = perimeter_distance(foot)
        pv = np.clip(pd[xs, ys], 0, N_PERIM - 1)
        for b in range(N_PERIM):
            m = pv == b
            p_total[b] += m.sum()
            p_removed[b] += gvals[m].sum()

        # (c) neighbour count
        nc = neighbour_count(base)[xs, ys, zs]
        for b in range(7):
            m = nc == b
            n_total[b] += m.sum()
            n_removed[b] += gvals[m].sum()

        used += 1

    def frac(a, b):
        return [float(x / y) if y else None for x, y in zip(a, b)]

    out = {
        "amendment_06_sha256": AMENDMENT,
        "status": "EXPLORATORY",
        "fea_ml_zip_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "families_used": used,
        "height_bands": N_HEIGHT,
        "removal_by_height": frac(h_removed, h_total),
        "removal_by_perimeter_distance": frac(p_removed, p_total),
        "removal_by_neighbour_count": frac(n_removed, n_total),
        "voxels_by_neighbour_count": [int(x) for x in n_total],
        "gini_across_height_bands": {
            "median": float(np.median(ginis)),
            "p10": float(np.percentile(ginis, 10)),
            "p90": float(np.percentile(ginis, 90)),
        },
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT.name}  ({used} families)")
    print("height   :", [f"{x:.2f}" if x else "-" for x in out["removal_by_height"]])
    print("perimeter:", [f"{x:.2f}" if x else "-" for x in out["removal_by_perimeter_distance"]])
    print("neighbour:", [f"{x:.2f}" if x else "-" for x in out["removal_by_neighbour_count"]])
    print("gini     :", out["gini_across_height_bands"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
