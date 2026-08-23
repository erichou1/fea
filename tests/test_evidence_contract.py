from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sasto.manifest import ManifestVerificationError, build_run_manifest, verify_run_manifest
from sasto.splits import (
    FamilyLeakageError,
    build_family_split,
    build_family_split_manifest,
    validate_family_split,
)
from sasto.sasto_pa import evaluate_erosion_candidate
from sasto.statistics import format_zero_failure_evidence, minimum_zero_failure_sample_size, one_sided_zero_failure_upper_bound
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


def test_canonical_locked_environment_contract() -> None:
    repository = Path(__file__).parents[1]
    pyproject = (repository / "pyproject.toml").read_text(encoding="utf-8")
    makefile = (repository / "Makefile").read_text(encoding="utf-8")
    gitignore = (repository / ".gitignore").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.11,<3.12"' in pyproject
    assert '"numpy>=2.0,<3"' in pyproject
    assert '[dependency-groups]' in pyproject
    assert '"pytest>=8,<9"' in pyproject
    assert (repository / ".python-version").read_text(encoding="utf-8") == "3.11.15\n"
    assert (repository / "uv.lock").is_file()
    assert "*.egg-info/" in gitignore
    assert "test-locked:" in makefile
    assert "uv sync --frozen" in makefile
    assert "uv run --frozen --group test python -m pytest -q" in makefile
    assert "EXPECTED_MANIFEST_SHA256 is required from an external trust anchor" in makefile
    assert '--expected-manifest-sha256 "$$EXPECTED_MANIFEST_SHA256"' in makefile
    assert "export ARTIFACT_DIR" in makefile
    assert '"$$ARTIFACT_DIR/run-manifest.json"' in makefile
    assert "$(MAKE) verify-artifact" not in makefile
    assert "awk" not in makefile
    assert "sha256_file" in makefile
    assert "read_bytes" not in makefile


@pytest.mark.parametrize(
    "kwargs",
    (
        {"name": 7},
        {"unit": 7},
        {"threshold": True},
        {"threshold": "not-a-number"},
        {"threshold": float("nan")},
        {"threshold": float("inf")},
        {"normalization": 7},
        {"base_target": 7},
    ),
)
def test_target_spec_rejects_non_string_and_nonfinite_runtime_contract_values(kwargs: dict) -> None:
    values = {"name": "compliance", "unit": "J", "direction": "upper", "threshold": 1.15}
    values.update(kwargs)
    with pytest.raises(ValueError):
        TargetSpec(**values)


@pytest.mark.parametrize("response", (True, "1.0", float("nan"), float("inf"), -float("inf")))
def test_target_registry_rejects_nonfinite_or_nonnumeric_responses(response: object) -> None:
    registry = TargetRegistry((TargetSpec("compliance", "J", "upper", 1.15),))
    with pytest.raises(ValueError):
        registry.evaluate({"compliance": response})


def test_target_contract_accepts_finite_numpy_real_scalars_and_rejects_invalid_numpy_scalars() -> None:
    import numpy as np

    registry = TargetRegistry((TargetSpec("compliance", "J", "upper", np.float32(1.15)),))
    assert registry.evaluate({"compliance": np.int64(1)})["compliance"].passed
    assert not registry.evaluate({"compliance": np.float32(1.2)})["compliance"].passed

    for invalid in (np.bool_(True), np.complex64(1 + 0j), np.float32("nan"), np.float64("inf")):
        with pytest.raises(ValueError, match="finite number"):
            registry.evaluate({"compliance": invalid})
    for invalid_threshold in (np.bool_(True), np.complex128(1 + 0j), np.float32("nan"), np.float64("inf")):
        with pytest.raises(ValueError, match="finite number"):
            TargetSpec("compliance", "J", "upper", invalid_threshold)


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
    assert detailed["sample_to_family"] == sorted(
        ({"sample_id": sample["sample_id"], "family_id": sample["family_id"]} for sample in samples),
        key=lambda row: row["sample_id"],
    )
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
    split_payload = build_family_split_manifest(
        [{"sample_id": "f{}/base".format(index), "family_id": "f{}".format(index)} for index in range(1, 5)],
        seed=42,
    )
    split.write_text(json.dumps(split_payload, sort_keys=True) + "\n", encoding="utf-8")
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
    expected_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert verify_run_manifest(manifest_path, expected_manifest_sha256)["run_id"] == "unit-run"

    output.write_text('{"status": "mutated"}\n', encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="sha256 mismatch"):
        verify_run_manifest(manifest_path, expected_manifest_sha256)


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

    with pytest.raises(ManifestVerificationError, match="artifact_root must not contain symlink components"):
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


def test_simple_point_6_26_rejects_review_false_accept_that_merges_foreground_components() -> None:
    volume = [
        [[False, False], [False, True]],
        [[True, True], [False, False]],
    ]

    assert not is_simple_point_6_26(volume, (0, 1, 1))


def test_simple_point_6_26_fails_closed_for_malformed_or_out_of_range_input() -> None:
    assert not is_simple_point_6_26([[[True], []]], (0, 0, 0))
    assert not is_simple_point_6_26([[[True]]], (2, 0, 0))


def test_simple_point_6_26_true_preserves_global_component_counts_for_all_2x2x2_volumes() -> None:
    from collections import deque

    face_neighbors = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
    full_neighbors = tuple(
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) != (0, 0, 0)
    )

    def count(cells, neighbors):
        unseen = set(cells)
        components = 0
        while unseen:
            components += 1
            queue = deque([unseen.pop()])
            while queue:
                z, y, x = queue.popleft()
                for dz, dy, dx in neighbors:
                    candidate = (z + dz, y + dy, x + dx)
                    if candidate in unseen:
                        unseen.remove(candidate)
                        queue.append(candidate)
        return components

    def counts(grid):
        foreground = {(z, y, x) for z in range(2) for y in range(2) for x in range(2) if grid[z][y][x]}
        background = set((z, y, x) for z in range(2) for y in range(2) for x in range(2)) - foreground
        # The exterior node is a distinct 26-background component until a boundary
        # background voxel connects it to the grid.
        exterior = object()
        augmented = set(background) | {exterior}
        unseen = set(augmented)
        bg_components = 0
        while unseen:
            bg_components += 1
            queue = deque([unseen.pop()])
            while queue:
                cell = queue.popleft()
                if cell is exterior:
                    neighbors = [candidate for candidate in background if 0 in candidate or 1 in candidate]
                else:
                    z, y, x = cell
                    neighbors = [
                        (z + dz, y + dy, x + dx)
                        for dz, dy, dx in full_neighbors
                        if (z + dz, y + dy, x + dx) in background
                    ]
                    if 0 in cell or 1 in cell:
                        neighbors.append(exterior)
                for neighbor in neighbors:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        queue.append(neighbor)
        return count(foreground, face_neighbors), bg_components

    for mask in range(1 << 8):
        volume = [[[bool(mask & (1 << (4 * z + 2 * y + x))) for x in range(2)] for y in range(2)] for z in range(2)]
        before = counts(volume)
        for point in ((z, y, x) for z in range(2) for y in range(2) for x in range(2)):
            if is_simple_point_6_26(volume, point):
                candidate = [[list(row) for row in plane] for plane in volume]
                candidate[point[0]][point[1]][point[2]] = False
                assert counts(candidate) == before, (mask, point, before, counts(candidate))


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


@pytest.mark.parametrize("invalid_n", (True, np.bool_(True)))
def test_zero_failure_sample_count_rejects_python_and_numpy_booleans(invalid_n: object) -> None:
    import numpy as np

    with pytest.raises(ValueError, match="positive integer"):
        one_sided_zero_failure_upper_bound(invalid_n)
    with pytest.raises(ValueError, match="positive integer"):
        format_zero_failure_evidence(invalid_n)
