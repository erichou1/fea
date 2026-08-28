"""Compute held-out development ensemble mean and uncertainty evidence for G2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from sasto.surrogate import (
    DenseSurrogateCNN,
    _normalized_batches,
    build_packed_ingest_cache,
    open_role_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROL = Path("/Users/eric/workspace/sasto-modernization-control")
SPLIT = CONTROL / "v2/g1/split-manifest.json"
ARCHIVE = CONTROL / "archives/fea_ml.zip"
G1B = ROOT / "artifacts/g1b/relabel-v3"
SPLIT_SHA = "ca526a068137308ca4bb05325d62bab5a7ad45c81d54566d5fa8e3ef62a91650"
ARCHIVE_SHA = "79640406e1e0921c0ccfdc1df7ce51e05a8ecfece2ceccb7dec42c981736beda"
COHORT_SHA = "b7066e14c6713eb69e1555f7ccacae4d82bb1fd092eef61ab113bf3ee540b8d8"
CLUSTER_SHA = "9c3691f523b681b0bffaa26f9559b7a4008c096cea52c32d2b74ab1c20394227"
CACHE_ROOT = ROOT / "artifacts/g2/ingest-cache-v1/79640406e1e0921c-b7066e14c6713eb6"
ENSEMBLE_ROOT = ROOT / "artifacts/g2/ensemble-v1"
TARGETS = ("compliance", "max_von_mises", "max_displacement")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main() -> None:
    import torch

    opened = {role: open_role_dataset(role=role, split_manifest=SPLIT, expected_split_sha256=SPLIT_SHA,
        archive=ARCHIVE, expected_archive_sha256=ARCHIVE_SHA, g1b_root=G1B,
        expected_cohort_manifest_sha256=COHORT_SHA, expected_cluster_role_manifest_sha256=CLUSTER_SHA)
        for role in ("fit", "development")}
    cached = build_packed_ingest_cache(cache_root=CACHE_ROOT, datasets=[opened["fit"], opened["development"]])
    normalization = json.loads((ENSEMBLE_ROOT / "normalization-stats.json").read_text())
    members = [json.loads((ENSEMBLE_ROOT / "members" / f"member-{index:02d}.json").read_text()) for index in range(5)]
    models = []
    for member in members:
        model = DenseSurrogateCNN(base_channels=int(member["base_channels"])).to("mps")
        state = torch.load(ENSEMBLE_ROOT / "members" / member["checkpoint"]["path"], map_location="cpu", weights_only=True)
        model.load_state_dict(state["state_dict"]); model.eval(); models.append(model)
    means = np.array([normalization["means"][name] for name in TARGETS]); scales = np.array([normalization["scales"][name] for name in TARGETS])
    raw_absolute = np.zeros(3); normalized_absolute = np.zeros(3); epistemic = np.zeros(3); aleatoric = np.zeros(3); total = np.zeros(3); count = 0
    with torch.no_grad():
        for batch in _normalized_batches(cached["development"], normalization, batch_size=4):
            channels = torch.stack([row[1] for row in batch]).to("mps")
            expected = torch.stack([row[2] for row in batch]).numpy()
            outputs = [model(channels) for model in models]
            member_means = np.stack([output["mean"].detach().cpu().numpy() for output in outputs], axis=1)
            member_scales = np.stack([output["dispersion"].detach().cpu().numpy() for output in outputs], axis=1)
            ensemble_mean = member_means.mean(axis=1)
            raw_absolute += np.abs(np.exp(ensemble_mean * scales + means) - np.exp(expected * scales + means)).sum(axis=0)
            normalized_absolute += np.abs(ensemble_mean - expected).sum(axis=0)
            epistemic += member_means.std(axis=1, ddof=0).sum(axis=0)
            aleatoric += np.sqrt(np.mean(member_scales ** 2, axis=1)).sum(axis=0)
            total += np.sqrt(member_means.var(axis=1) + np.mean(member_scales ** 2, axis=1)).sum(axis=0)
            count += len(batch)
    result = {"schema_version": "1.0.0", "label": "CERTIFIED_G2_ENSEMBLE_DEVELOPMENT_EVIDENCE", "role": "development",
        "member_checkpoint_sha256": [member["checkpoint"]["sha256"] for member in members], "sample_count": count,
        "ensemble_mean_development_mae": {name: float(raw_absolute[index] / count) for index, name in enumerate(TARGETS)},
        "ensemble_mean_development_normalized_log_mae": {name: float(normalized_absolute[index] / count) for index, name in enumerate(TARGETS)},
        "mean_epistemic_normalized_std": {name: float(epistemic[index] / count) for index, name in enumerate(TARGETS)},
        "mean_aleatoric_normalized_std": {name: float(aleatoric[index] / count) for index, name in enumerate(TARGETS)},
        "mean_total_normalized_std": {name: float(total[index] / count) for index, name in enumerate(TARGETS)}, "k5_not_adjudicated": True}
    result["evidence_digest"] = digest(result)
    path = ENSEMBLE_ROOT / "ensemble-development-evidence.json"; payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    if path.exists() and path.read_text() != payload:
        raise RuntimeError("append-only ensemble evidence mismatch")
    if not path.exists(): path.write_text(payload)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
