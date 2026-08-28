"""Focused G2 contracts: role isolation, anchored data, and fit-only statistics."""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from sasto.splits import build_family_split_manifest, split_sha256


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _npz_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.savez(output, data=array)
    return output.getvalue()


def _g2_inputs(tmp_path: Path) -> dict[str, object]:
    """Create four role-disjoint G1b-style samples without a confirmation read."""
    ids = ["fit-000", "development-000", "calibration-000", "confirmation-000"]
    split = build_family_split_manifest([{"sample_id": sample_id, "family_id": sample_id} for sample_id in ids], seed=7)
    split_path = tmp_path / "split.json"; split_path.write_text(json.dumps(split, sort_keys=True))
    roles = {sample_id: role for role, partition in split["partitions"].items() for sample_id in partition["sample_ids"]}
    root = tmp_path / "g1b"; cases = root / "cases"; cases.mkdir(parents=True)
    records = []
    for index, sample_id in enumerate(ids, start=1):
        case = {"sample_id": sample_id, "role": roles[sample_id], "exclusion_reasons": [], "solver": {"status": "success", "compliance_j": float(index), "max_von_mises_pa": float(index * 10), "max_displacement_m": float(index) / 10,
        }}
        case["case_digest"] = _canonical_digest(case)
        (cases / f"{sample_id}.json").write_text(json.dumps(case, sort_keys=True))
        records.append({"sample_id": sample_id, "role": roles[sample_id], "exclusion_reasons": [], "case_digest": case["case_digest"]})
    cohort = {"schema_version": "1.0.0", "population_count": 4, "records": sorted(records, key=lambda row: row["sample_id"]),
              "records_digest": _canonical_digest(sorted(records, key=lambda row: row["sample_id"])), "eligible_ids": sorted(ids),
              "eligible_count": 4, "excluded_count": 0, "exclusion_counts": {}}
    cluster_rows = [{"cluster_id": f"cluster:{index:05d}", "members": [sample_id], "role": roles[sample_id]} for index, sample_id in enumerate(sorted(ids))]
    cluster = {"schema_version": "1.0.0", "algorithm": "family-cluster-v1", "base_algorithm": "family-id-v1", "seed_lineage": 7,
               "duplicate_tolerance": "medium", "cluster_count": 4, "role_counts": {role: 1 for role in split["partitions"]}, "clusters": cluster_rows,
               "source_split_manifest_sha256": split_sha256(split), "near_duplicate_summary_sha256": "0" * 64, "near_duplicate_verified_pairs_sha256": "1" * 64}
    cohort_path = root / "cohort-manifest.json"; cohort_path.write_text(json.dumps(cohort, sort_keys=True))
    cluster_path = root / "cluster-role-manifest.json"; cluster_path.write_text(json.dumps(cluster, sort_keys=True))
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as opened:
        for index, sample_id in enumerate(ids, start=1):
            occupancy = np.zeros((64, 64, 64), dtype=np.uint8); occupancy[index, 1, 1] = 1
            part = np.zeros((64, 64, 64), dtype=np.uint8); part[index, 1, 1] = index
            opened.writestr(f"fea_ml/data/runs_real/{sample_id}/occ.npz", _npz_bytes(occupancy))
            opened.writestr(f"fea_ml/data/runs_real/{sample_id}/part.npz", _npz_bytes(part))
    return {"split": split_path, "archive": archive, "root": root, "roles": roles,
            "split_sha": hashlib.sha256(split_path.read_bytes()).hexdigest(), "archive_sha": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "cohort_sha": hashlib.sha256(cohort_path.read_bytes()).hexdigest(), "cluster_sha": hashlib.sha256(cluster_path.read_bytes()).hexdigest()}


def test_confirmation_role_is_unconditionally_denied_before_any_input_open() -> None:
    """G2 must never offer an API path that could open confirmation payloads."""
    from sasto.surrogate import SurrogateRoleError, open_role_dataset

    missing = Path("/definitely/not/a/g2-input")
    with pytest.raises(SurrogateRoleError, match="confirmation.*sealed"):
        open_role_dataset(
            role="confirmation",
            split_manifest=missing,
            expected_split_sha256="0" * 64,
            archive=missing,
            expected_archive_sha256="0" * 64,
            g1b_root=missing,
            expected_cohort_manifest_sha256="0" * 64,
            expected_cluster_role_manifest_sha256="0" * 64,
        )


def test_fit_loader_reads_only_eligible_canonical_case_targets_and_packs_occupancy(tmp_path: Path) -> None:
    from sasto.surrogate import open_role_dataset

    inputs = _g2_inputs(tmp_path)
    dataset = open_role_dataset(
        role="fit", split_manifest=inputs["split"], expected_split_sha256=inputs["split_sha"],
        archive=inputs["archive"], expected_archive_sha256=inputs["archive_sha"], g1b_root=inputs["root"],
        expected_cohort_manifest_sha256=inputs["cohort_sha"], expected_cluster_role_manifest_sha256=inputs["cluster_sha"],
    )
    example = next(iter(dataset))
    assert example.sample_id == next(sample_id for sample_id, role in inputs["roles"].items() if role == "fit")
    assert example.targets == {"compliance": 1.0, "max_von_mises": 10.0, "max_displacement": 0.1}
    assert example.packed_occupancy_nbytes == 64 ** 3 // 8
    assert example.channels.shape == (2, 64, 64, 64)
    assert dataset.provenance["archive_sha256"] == inputs["archive_sha"]


def test_normalization_record_is_fit_only_log_transformed_and_self_digesting(tmp_path: Path) -> None:
    from sasto.surrogate import compute_fit_normalization, open_role_dataset

    inputs = _g2_inputs(tmp_path)
    fit = open_role_dataset(
        role="fit", split_manifest=inputs["split"], expected_split_sha256=inputs["split_sha"],
        archive=inputs["archive"], expected_archive_sha256=inputs["archive_sha"], g1b_root=inputs["root"],
        expected_cohort_manifest_sha256=inputs["cohort_sha"], expected_cluster_role_manifest_sha256=inputs["cluster_sha"],
    )
    stats = compute_fit_normalization(fit)
    assert stats["source_sample_ids"] == list(fit.sample_ids)
    assert stats["target_names"] == ["compliance", "max_von_mises", "max_displacement"]
    assert stats["transform"] == {"name": "natural_log", "domain": "strictly_positive", "clipping": "none"}
    assert stats["means"]["compliance"] == 0.0
    assert stats["stats_digest"] == _canonical_digest({key: value for key, value in stats.items() if key != "stats_digest"})


def test_dense_cnn_returns_named_means_and_positive_dispersion() -> None:
    torch = pytest.importorskip("torch")
    from sasto.surrogate import DenseSurrogateCNN

    model = DenseSurrogateCNN(target_names=("compliance", "max_von_mises", "max_displacement"), base_channels=4)
    prediction = model(torch.zeros((2, 2, 64, 64, 64), dtype=torch.float32))
    assert set(prediction) == {"mean", "dispersion"}
    assert tuple(prediction["mean"].shape) == (2, 3)
    assert tuple(prediction["dispersion"].shape) == (2, 3)
    assert bool(torch.all(prediction["dispersion"] > 0))
    assert model.parameter_count > 0


def test_nonpromotable_smoke_writes_digest_bound_member_checkpoint_and_ledger(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from sasto.surrogate import train_smoke_ensemble

    examples = [("synthetic-000", torch.zeros((2, 64, 64, 64)), torch.zeros(3)), ("synthetic-001", torch.ones((2, 64, 64, 64), dtype=torch.float32), torch.ones(3))]
    result = train_smoke_ensemble(
        output_root=tmp_path / "smoke", examples=examples, target_names=("compliance", "max_von_mises", "max_displacement"),
        normalization_stats_digest="2" * 64, source_bundle_sha256="3" * 64, split_sha256="4" * 64, archive_sha256="5" * 64,
        cohort_manifest_sha256="6" * 64, member_count=1, epochs=1, base_channels=2, device="cpu",
    )
    member = result["members"][0]
    manifest = json.loads((tmp_path / "smoke" / "members" / "member-00.json").read_text())
    assert result["label"] == "SMOKE_ONLY_NONPROMOTABLE"
    assert manifest["seed"] == member["seed"]
    assert manifest["normalization_stats_digest"] == "2" * 64
    assert manifest["compute_ledger"]["epochs"] == 1
    assert (tmp_path / "smoke" / "members" / "member-00.pt").is_file()


def test_smoke_records_exact_campaign_seed_derivation_for_every_member(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from sasto.surrogate import deterministic_seed, train_smoke_ensemble

    campaign_seed = 424242
    namespace = "sasto-v-g2-dense-ensemble-v1"
    result = train_smoke_ensemble(
        output_root=tmp_path / "smoke", examples=[("development-000", torch.zeros((2, 64, 64, 64)), torch.zeros(3))],
        target_names=("compliance", "max_von_mises", "max_displacement"), normalization_stats_digest="2" * 64,
        source_bundle_sha256="3" * 64, split_sha256="4" * 64, archive_sha256="5" * 64, cohort_manifest_sha256="6" * 64,
        member_count=3, epochs=1, base_channels=2, device="cpu", campaign_seed=campaign_seed,
    )
    summary = json.loads((tmp_path / "smoke" / "smoke-summary.json").read_text())
    assert summary["campaign_seed"] == campaign_seed
    for member_index, member in enumerate(result["members"]):
        manifest = json.loads((tmp_path / "smoke" / "members" / f"member-{member_index:02d}.json").read_text())
        expected = deterministic_seed(namespace, campaign_seed, member_index)
        assert manifest["campaign_seed"] == campaign_seed
        assert manifest["seed"] == member["seed"] == expected


def test_same_campaign_seed_produces_byte_identical_checkpoints(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from sasto.surrogate import train_smoke_ensemble

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    kwargs = dict(
        examples=[("development-000", torch.zeros((2, 64, 64, 64)), torch.zeros(3))],
        target_names=("compliance", "max_von_mises", "max_displacement"), normalization_stats_digest="2" * 64,
        source_bundle_sha256="3" * 64, split_sha256="4" * 64, archive_sha256="5" * 64, cohort_manifest_sha256="6" * 64,
        member_count=1, epochs=1, base_channels=2, device=device, campaign_seed=424242,
    )
    first = train_smoke_ensemble(output_root=tmp_path / "first", **kwargs)
    second = train_smoke_ensemble(output_root=tmp_path / "second", **kwargs)
    assert first["members"][0]["checkpoint_sha256"] == second["members"][0]["checkpoint_sha256"]


def test_anchored_smoke_opens_real_role_datasets_and_records_nonzero_source_digests(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from sasto.surrogate import run_anchored_smoke

    inputs = _g2_inputs(tmp_path)
    result = run_anchored_smoke(
        output_root=tmp_path / "anchored-smoke", split_manifest=inputs["split"], expected_split_sha256=inputs["split_sha"],
        archive=inputs["archive"], expected_archive_sha256=inputs["archive_sha"], g1b_root=inputs["root"],
        expected_cohort_manifest_sha256=inputs["cohort_sha"], expected_cluster_role_manifest_sha256=inputs["cluster_sha"],
        sample_count=1, member_count=1, epochs=1, base_channels=2, device="cpu",
    )
    summary = json.loads((tmp_path / "anchored-smoke" / "smoke-summary.json").read_text())
    assert result["label"] == "SMOKE_ONLY_NONPROMOTABLE"
    assert summary["data_role"] == "development"
    assert summary["split_sha256"] == inputs["split_sha"]
    assert summary["archive_sha256"] == inputs["archive_sha"]
    assert summary["cohort_manifest_sha256"] == inputs["cohort_sha"]
    assert {summary["split_sha256"], summary["archive_sha256"], summary["cohort_manifest_sha256"]} != {"0" * 64}
    study = json.loads((tmp_path / "anchored-smoke" / "capacity-study.json").read_text())
    assert study["selection_role"] == "development"
    assert study["not_k5_adjudication"] is True
    assert [row["base_channels"] for row in study["rows"]] == [4, 16, 32]
    assert all(set(row["development_mae"]) == {"compliance", "max_von_mises", "max_displacement"} for row in study["rows"])


def test_source_bundle_matches_ast_transitive_local_import_closure() -> None:
    from sasto.surrogate import surrogate_source_bundle

    files, bundle_sha = surrogate_source_bundle()
    assert bundle_sha == _canonical_digest([{"path": path, "sha256": files[path]} for path in sorted(files)])
    assert {path for path in files if path.startswith("src/sasto/")} == {
        "src/sasto/surrogate.py", "src/sasto/manifest.py", "src/sasto/splits.py", "src/sasto/targets.py",
    }


def test_cli_refuses_promotable_training_until_g1b_certification() -> None:
    completed = subprocess.run([sys.executable, "-m", "sasto.surrogate", "--mode", "train", "--output", "/tmp/no-g2-train"], text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "SMOKE_ONLY_NONPROMOTABLE" in completed.stderr


def test_development_early_stopping_selects_best_epoch_without_calibration_or_confirmation_inputs() -> None:
    from sasto.surrogate import development_early_stopping_epoch

    assert development_early_stopping_epoch([1.0, 0.4, 0.5, 0.6], patience=2) == 2
