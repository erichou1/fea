"""Run the certified G2 cache, at-scale capacity study, and full ensemble."""
from __future__ import annotations

import json
from pathlib import Path

from sasto.surrogate import (
    build_packed_ingest_cache,
    capacity_study,
    compute_fit_normalization,
    open_role_dataset,
    packed_role_subset,
    surrogate_source_bundle,
    train_certified_ensemble,
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
CAPACITY_ROOT = ROOT / "artifacts/g2/capacity-v1"
ENSEMBLE_ROOT = ROOT / "artifacts/g2/ensemble-v1"


def write_new(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise RuntimeError(f"append-only artifact differs: {path}")
        return
    path.write_text(encoded)


def main() -> None:
    datasets = {}
    for role in ("fit", "development"):
        datasets[role] = open_role_dataset(role=role, split_manifest=SPLIT, expected_split_sha256=SPLIT_SHA,
            archive=ARCHIVE, expected_archive_sha256=ARCHIVE_SHA, g1b_root=G1B,
            expected_cohort_manifest_sha256=COHORT_SHA, expected_cluster_role_manifest_sha256=CLUSTER_SHA)
    cached = build_packed_ingest_cache(cache_root=CACHE_ROOT, datasets=[datasets["fit"], datasets["development"]])
    cache_manifest_path = CACHE_ROOT / "cache-manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text())
    normalization = compute_fit_normalization(cached["fit"])
    files, bundle_sha = surrogate_source_bundle(ROOT)
    write_new(CAPACITY_ROOT / "normalization-stats.json", normalization)
    capacity = capacity_study(fit_examples=packed_role_subset(cached["fit"], sample_count=1000),
        development_examples=packed_role_subset(cached["development"], sample_count=512), normalization=normalization,
        widths=(4, 16, 32), epochs=12, device="mps", campaign_seed=20260828,
        provenance={**cached["fit"].provenance, "source_bundle_sha256": bundle_sha})
    write_new(CAPACITY_ROOT / "capacity-study.json", capacity)
    result = train_certified_ensemble(output_root=ENSEMBLE_ROOT, fit=cached["fit"], development=cached["development"],
        normalization=normalization, source_bundle_sha256=bundle_sha,
        cache_manifest_sha256=__import__("hashlib").sha256(cache_manifest_path.read_bytes()).hexdigest(),
        member_count=5, max_epochs=20, patience=4, base_channels=int(capacity["recommended_base_channels"]), device="mps",
        campaign_seed=20260828, ingest_wall_seconds=float(cache_manifest["input_wall_seconds"]))
    write_new(ENSEMBLE_ROOT / "normalization-stats.json", normalization)
    write_new(ENSEMBLE_ROOT / "run-result.json", {"cache_manifest_sha256": __import__("hashlib").sha256(cache_manifest_path.read_bytes()).hexdigest(),
        "cache_digest": cache_manifest["cache_digest"], "capacity_study_digest": capacity["study_digest"],
        "capacity_study_sha256": __import__("hashlib").sha256((CAPACITY_ROOT / "capacity-study.json").read_bytes()).hexdigest(),
        "source_bundle_files": files, "source_bundle_sha256": bundle_sha, "result": result})
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
