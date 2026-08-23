from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sasto.manifest import ManifestVerificationError, build_run_manifest, verify_run_manifest
from sasto.splits import (
    FamilyLeakageError,
    build_family_split,
    build_family_split_manifest,
    validate_family_split,
)
from sasto.sasto_pa import evaluate_erosion_candidate
from sasto.statistics import minimum_zero_failure_sample_size, one_sided_zero_failure_upper_bound
from sasto.targets import TargetRegistry, TargetSpec
from sasto.topology import is_simple_point_6_26


def test_target_registry_is_semantic_under_order_permutation() -> None:
    original = TargetRegistry(
        (
            TargetSpec("compliance", "J", "upper", 1.15),
            TargetSpec("max_von_mises", "Pa", "upper", 5_000_000.0),
            TargetSpec("max_displacement", "m", "upper", 0.028),
        )
    )
    permuted = TargetRegistry(tuple(reversed(tuple(original))))

    responses = {
        "compliance": 1.14,
        "max_von_mises": 4_900_000.0,
        "max_displacement": 0.027,
    }
    assert original.evaluate(responses) == permuted.evaluate(responses)
    assert original.evaluate(responses)["compliance"].passed


def test_target_contract_rejects_unitless_compliance_without_ratio_semantics() -> None:
    with pytest.raises(ValueError, match="absolute targets cannot use unit '1'"):
        TargetSpec("compliance", "1", "upper", 1.15)
    with pytest.raises(ValueError, match="_ratio"):
        TargetSpec("compliance", "1", "upper", 1.15, normalization="baseline_ratio", base_target="baseline_compliance")
    with pytest.raises(ValueError, match="base_target"):
        TargetSpec("compliance_ratio", "1", "upper", 1.15, normalization="baseline_ratio")

    ratio = TargetSpec(
        "compliance_ratio",
        "1",
        "upper",
        1.15,
        normalization="baseline_ratio",
        base_target="baseline_compliance",
    )
    assert ratio.as_dict()["normalization"] == "baseline_ratio"


def test_ratio_base_target_is_external_provenance_not_a_registry_member() -> None:
    ratio = TargetSpec(
        "compliance_ratio",
        "1",
        "upper",
        1.15,
        normalization="baseline_ratio",
        base_target="compliance",
    )

    assert TargetRegistry((ratio,)).names == ("compliance_ratio",)


def test_family_split_rejects_leakage_and_is_deterministic() -> None:
    samples = [
        {"sample_id": f"{family}-v{variant}", "family_id": family}
        for family in ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10")
        for variant in ("a", "b")
    ]
    first = build_family_split(samples, seed=42)
    second = build_family_split(samples, seed=42)
    assert first == second
    assert validate_family_split(samples, first) is None
    detailed = build_family_split_manifest(samples, seed=42)
    assert detailed["algorithm"] == "family-id-v1"
    assert set(detailed["partitions"]["fit"]["family_ids"]) <= {sample["family_id"] for sample in samples}

    leaked = {name: list(ids) for name, ids in first.items()}
    leaked["confirmation"].append(leaked["fit"][0])
    with pytest.raises(FamilyLeakageError, match="appears in multiple partitions"):
        validate_family_split(samples, leaked)


def test_family_split_requires_a_family_in_each_functional_role() -> None:
    too_small = [{"sample_id": "only/base", "family_id": "only"}]
    with pytest.raises(FamilyLeakageError, match="at least 4 families"):
        build_family_split(too_small, seed=42)

    samples = [{"sample_id": f"f{index}/base", "family_id": f"f{index}"} for index in range(1, 5)]
    split = build_family_split(samples, seed=42)
    assert {role: len(sample_ids) for role, sample_ids in split.items()} == {
        "fit": 1,
        "development": 1,
        "calibration": 1,
        "confirmation": 1,
    }

    empty_confirmation = {
        "fit": ["f1/base", "f2/base"],
        "development": ["f3/base"],
        "calibration": ["f4/base"],
        "confirmation": [],
    }
    with pytest.raises(FamilyLeakageError, match="confirmation.*at least one family"):
        validate_family_split(samples, empty_confirmation)


def test_default_family_split_allocation_matches_requested_fractions() -> None:
    samples = [{"sample_id": f"f{index}/base", "family_id": f"f{index}"} for index in range(1, 11)]

    split = build_family_split(samples, seed=42)

    assert {role: len(sample_ids) for role, sample_ids in split.items()} == {
        "fit": 6,
        "development": 2,
        "calibration": 1,
        "confirmation": 1,
    }


def test_hash_mutation_rejects_complete_manifest(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    split = tmp_path / "split.json"
    source.write_text('{"fixture": true}\n', encoding="utf-8")
    output.write_text('{"status": "ok"}\n', encoding="utf-8")
    split.write_text('{"algorithm":"family-id-v1","partitions":{}}\n', encoding="utf-8")
    split_payload = json.loads(split.read_text(encoding="utf-8"))
    manifest = build_run_manifest(
        run_id="unit-run",
        inputs={"fixture/input": source},
        outputs={"result": output, "split": split},
        targets=TargetRegistry((TargetSpec("compliance", "J", "upper", 1.15),)),
        split_sha256=hashlib.sha256(json.dumps(split_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        split_artifact="split",
        artifact_root=tmp_path,
    )
    manifest_path = tmp_path / "run-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    assert verify_run_manifest(manifest_path) is None

    output.write_text('{"status": "mutated"}\n', encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="sha256 mismatch"):
        verify_run_manifest(manifest_path)


def test_manifest_builder_rejects_external_and_symlinked_artifact_records(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    external = tmp_path / "external.json"
    external.write_text('{"external": true}\n', encoding="utf-8")
    linked = bundle / "linked.json"
    linked.symlink_to(external)
    registry = TargetRegistry((TargetSpec("compliance", "J", "upper", 1.15),))

    with pytest.raises(ManifestVerificationError, match="beneath artifact root"):
        build_run_manifest(
            run_id="external-file",
            inputs={"fixture": external},
            outputs={"result": external},
            targets=registry,
            split_sha256="0" * 64,
            split_artifact="result",
            artifact_root=bundle,
        )
    with pytest.raises(ManifestVerificationError, match="symlink"):
        build_run_manifest(
            run_id="linked-file",
            inputs={"fixture": linked},
            outputs={"result": linked},
            targets=registry,
            split_sha256="0" * 64,
            split_artifact="result",
            artifact_root=bundle,
        )


def test_manifest_builder_rejects_symlinked_artifact_root_before_resolving(tmp_path: Path) -> None:
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ManifestVerificationError, match="artifact_root must be a real directory"):
        build_run_manifest(
            run_id="linked-root",
            inputs={"fixture": real_root / "input.json"},
            outputs={"result": real_root / "result.json"},
            targets=TargetRegistry((TargetSpec("compliance", "J", "upper", 1.15),)),
            split_sha256="0" * 64,
            split_artifact="result",
            artifact_root=linked_root,
        )


def test_simple_point_6_26_rejects_bridge_and_accepts_endpoint() -> None:
    bridge = [[[False for _ in range(3)] for _ in range(3)] for _ in range(3)]
    bridge[1][1][0] = True
    bridge[1][1][1] = True
    bridge[1][1][2] = True
    assert not is_simple_point_6_26(bridge, (1, 1, 1))
    assert is_simple_point_6_26(bridge, (1, 1, 0))


def test_simple_point_6_26_rejects_boundary_cavity_opening_and_keeps_safe_boundary_erosion() -> None:
    solid_with_cavity = [[[True for _ in range(3)] for _ in range(3)] for _ in range(3)]
    solid_with_cavity[1][1][1] = False
    assert not is_simple_point_6_26(solid_with_cavity, (0, 0, 0))

    solid_without_cavity = [[[True for _ in range(3)] for _ in range(3)] for _ in range(3)]
    assert is_simple_point_6_26(solid_without_cavity, (0, 0, 0))


def test_canonical_sasto_pa_rejects_a_topology_break_before_proxy_admission() -> None:
    volume = [[[False for _ in range(3)] for _ in range(3)] for _ in range(3)]
    volume[1][1][0] = volume[1][1][1] = volume[1][1][2] = True
    registry = TargetRegistry((TargetSpec("compliance", "J", "upper", 1.15),))
    decision = evaluate_erosion_candidate(volume, (1, 1, 1), {"compliance": 1.0}, registry)
    assert not decision.accepted
    assert decision.reason == "topology-not-simple-6-26"


def test_zero_failure_upper_bound_uses_exact_one_sided_formula() -> None:
    assert one_sided_zero_failure_upper_bound(100, alpha=0.05) == pytest.approx(
        1.0 - 0.05 ** (1.0 / 100.0), rel=0, abs=1e-15
    )
    assert minimum_zero_failure_sample_size(0.02, alpha=0.05) == 149
    with pytest.raises(ValueError):
        one_sided_zero_failure_upper_bound(0)
