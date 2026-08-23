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


def test_hash_mutation_rejects_complete_manifest(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    source.write_text('{"fixture": true}\n', encoding="utf-8")
    output.write_text('{"status": "ok"}\n', encoding="utf-8")
    manifest = build_run_manifest(
        run_id="unit-run",
        inputs={"fixture/input": source},
        outputs={"result": output},
        targets=TargetRegistry((TargetSpec("compliance", "J", "upper", 1.15),)),
        split_sha256="0" * 64,
    )
    manifest_path = tmp_path / "run-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    assert verify_run_manifest(manifest_path) is None

    output.write_text('{"status": "mutated"}\n', encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="sha256 mismatch"):
        verify_run_manifest(manifest_path)


def test_simple_point_6_26_rejects_bridge_and_accepts_endpoint() -> None:
    bridge = [[[False for _ in range(3)] for _ in range(3)] for _ in range(3)]
    bridge[1][1][0] = True
    bridge[1][1][1] = True
    bridge[1][1][2] = True
    assert not is_simple_point_6_26(bridge, (1, 1, 1))
    assert is_simple_point_6_26(bridge, (1, 1, 0))


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
