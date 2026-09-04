"""Replay trajectory geometry and run architecture-B inference. Amendment 07.

Governed by K6_AMENDMENT_07_SECOND_ARCHITECTURE.md sha256
b6d27643c6a01fa19f37502d17ee5adf7a96aa19d8f8cbc9472d7baf8f030ce4.

For every selected state in the frozen GB200 trajectory records, this replays
the deterministic erosion to recover the voxel geometry, verifies it against the
recorded state_occupancy_sha256, and evaluates the architecture-B ensemble on
it. FEA targets are read from the records and never recomputed. No solver call
is made.

Writes one row per state: family, bin, fraction removed, the frozen FEA truth,
and B's (mu, sigma) per target, in the same normalized log space as A.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from sasto.activity_campaign import geometric_trajectory  # noqa: E402
from sasto.g3_trajectory_calibration import family_seed  # noqa: E402
from sasto.surrogate_b import TARGET_NAMES, ResidualSurrogateCNN  # noqa: E402

AMENDMENT_SHA = "b6d27643c6a01fa19f37502d17ee5adf7a96aa19d8f8cbc9472d7baf8f030ce4"
ARCHIVE = Path("/Users/eric/workspace/sasto-modernization-control/archives/fea_ml.zip")
ARCHIVE_SHA = "79640406e1e0921c0ccfdc1df7ce51e05a8ecfece2ceccb7dec42c981736beda"
GB200 = Path("/Users/eric/workspace/sasto-g3-gb200-inbound/trajectory-calibration-gb200")
ENSEMBLE = REPO / "artifacts/g2b/ensemble-v1"
OUT = REPO / "artifacts/g2b/inference"

# solver key -> target name, as recorded in the frozen records
SOLVER_KEY = {
    "compliance": "compliance_j",
    "max_displacement": "max_displacement_m",
    "max_von_mises": "max_gauss_von_mises_pa",
}


def load_ensemble(device):
    summary = json.loads((ENSEMBLE / "ensemble-summary.json").read_text())
    if summary.get("amendment_sha256") != AMENDMENT_SHA:
        raise SystemExit("ensemble was not trained under amendment 07")
    models = []
    for member in summary["members"]:
        path = ENSEMBLE / member["checkpoint"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != member["checkpoint_sha256"]:
            raise SystemExit(f"checkpoint digest mismatch: {path.name}")
        model = ResidualSurrogateCNN().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
    return models, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="families to process, 0 = all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise SystemExit("source archive digest does not match the frozen pin")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models, summary = load_ensemble(device)
    norm = json.loads((REPO / "artifacts/g2/ensemble-v1/normalization-stats.json").read_text())
    means, scales = norm["means"], norm["scales"]

    records = sorted(GB200.glob("trajectory-*.json"))
    records = [p for i, p in enumerate(records) if i % args.shards == args.shard]
    if args.limit:
        records = records[: args.limit]

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"arch-b-states-{args.shard:02d}.jsonl"
    rows_written = verified = mismatched = 0
    started = time.perf_counter()

    with zipfile.ZipFile(ARCHIVE) as archive, open(out_path, "w") as out:
        for n, path in enumerate(records):
            record = json.loads(path.read_text())
            sample_id, family_id = record["sample_id"], record["family_id"]
            selected = record.get("selected_states") or []
            if not selected:
                continue
            member = f"fea_ml/data/runs_real/{sample_id}/occ.npz"
            with archive.open(member) as handle:
                occupancy = np.load(io.BytesIO(handle.read()), allow_pickle=False)["data"]
            trajectory, volumes = geometric_trajectory(
                sample_id=sample_id, volume=occupancy.astype(bool),
                batch_cap=40, ranking_seed=family_seed(family_id),
            )
            parts_member = f"fea_ml/data/runs_real/{sample_id}/part.npz"
            with archive.open(parts_member) as handle:
                parts = np.load(io.BytesIO(handle.read()), allow_pickle=False)["data"]

            batch_channels, batch_meta = [], []
            for state in selected:
                index = state["state_index"]
                volume = volumes.get(index)
                if volume is None:
                    mismatched += 1
                    continue
                digest = hashlib.sha256(np.ascontiguousarray(volume).tobytes()).hexdigest()
                if digest != state["state_occupancy_sha256"]:
                    mismatched += 1
                    continue
                verified += 1
                # G3-D1: train_arch_b.py (line 96) and surrogate.py train on RAW
                # part labels. Masking by occupancy here fed an unseen
                # representation. Parts stay raw; occupancy encodes removal.
                channels = np.stack((volume.astype(np.float32),
                                     parts.astype(np.float32)), axis=0)
                batch_channels.append(channels)
                batch_meta.append(state)

            if not batch_channels:
                continue
            tensor = torch.from_numpy(np.stack(batch_channels)).to(device)
            with torch.no_grad():
                member_mu, member_var = [], []
                for m in models:
                    mu_k, dispersion_k = m(tensor)
                    member_mu.append(mu_k)
                    member_var.append(dispersion_k.square())
            stacked_mu = torch.stack(member_mu)
            # Total predictive variance, identical composition to the frozen
            # G3 EnsemblePredictor: aleatoric member variance plus epistemic
            # disagreement of member means.
            total_var = torch.stack(member_var).mean(dim=0) + stacked_mu.var(dim=0, unbiased=False)
            mu = stacked_mu.mean(dim=0).cpu().numpy()
            sigma = total_var.sqrt().cpu().numpy()

            for k, state in enumerate(batch_meta):
                solver = state["solver"]
                truth = {}
                for name in TARGET_NAMES:
                    raw = float(solver[SOLVER_KEY[name]])
                    truth[name] = (float(np.log(raw)) - means[name]) / scales[name]
                out.write(json.dumps({
                    "family_id": family_id, "sample_id": sample_id,
                    "state_index": state["state_index"], "bin_label": state["bin_label"],
                    "fraction_removed": state["fraction_removed"],
                    "y": truth,
                    "mu": {n: float(mu[k, j]) for j, n in enumerate(TARGET_NAMES)},
                    "sigma": {n: float(sigma[k, j]) for j, n in enumerate(TARGET_NAMES)},
                }) + "\n")
                rows_written += 1

            if (n + 1) % 25 == 0:
                rate = (n + 1) / (time.perf_counter() - started)
                print(f"  {n+1}/{len(records)} families | {rows_written} states "
                      f"| {rate*60:.0f} fam/min", flush=True)

    print(f"wrote {rows_written} states to {out_path}", flush=True)
    print(f"digest verified {verified}, mismatched {mismatched}", flush=True)
    if mismatched:
        raise SystemExit(f"{mismatched} states failed digest verification")


if __name__ == "__main__":
    main()
