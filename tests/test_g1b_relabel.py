"""Focused contracts for G1b canonical baseline relabeling."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest


def _success(*, residual: float = 1e-10, loaded: int = 4) -> dict[str, object]:
    return {
        "status": "success", "relative_residual": residual, "loaded_node_count": loaded,
        "compliance_j": 1.0, "max_displacement_m": 2.0, "max_gauss_von_mises_pa": 3.0,
    }


def test_solver_validity_uses_only_the_five_preregistered_predicates() -> None:
    from sasto.g1b_relabel import cohort_reasons

    connected = np.zeros((3, 1, 1), dtype=bool); connected[:, 0, 0] = True
    assert cohort_reasons(connected, _success(), expected_loaded_nodes=4) == []

    disconnected = connected.copy(); disconnected[1, 0, 0] = False
    assert cohort_reasons(disconnected, _success(), expected_loaded_nodes=4) == ["occupancy_not_face_connected"]
    assert cohort_reasons(connected, {"status": "failure", "reason": "iterative_nonconvergence"}, expected_loaded_nodes=4) == [
        "solver_status_failure", "relative_residual_exceeds_2e-8", "nonfinite_or_nonpositive_outputs", "unstable_loaded_node_set",
    ]
    assert cohort_reasons(connected, _success(residual=2.1e-8), expected_loaded_nodes=4) == ["relative_residual_exceeds_2e-8"]
    bad_metrics = _success(); bad_metrics["compliance_j"] = float("nan")
    assert cohort_reasons(connected, bad_metrics, expected_loaded_nodes=4) == ["nonfinite_or_nonpositive_outputs"]
    assert cohort_reasons(connected, _success(loaded=3), expected_loaded_nodes=4) == ["unstable_loaded_node_set"]


def test_id_hash_shards_are_total_disjoint_and_stable() -> None:
    from sasto.g1b_relabel import shard_for_id

    ids = ["00001", "00002", "00003", "00004", "00005"]
    assignment = {sample_id: shard_for_id(sample_id, 3) for sample_id in ids}
    assert set(assignment.values()) <= {0, 1, 2}
    assert assignment == {sample_id: shard_for_id(sample_id, 3) for sample_id in reversed(ids)}
    assert sum(sample_id in [item for item in ids if shard_for_id(item, 3) == shard] for shard in range(3) for sample_id in ids) == len(ids)
    with pytest.raises(ValueError, match="shard"):
        shard_for_id("00001", 0)


def test_cluster_roles_inherit_base_role_and_stop_on_cross_role_cluster() -> None:
    from sasto.g1b_relabel import RelabelError, build_cluster_table

    roles = {"a": "fit", "b": "fit", "c": "development"}
    table = build_cluster_table(sample_roles=roles, duplicate_clusters=[["a", "b"]])
    assert table == [
        {"cluster_id": "cluster:00000", "members": ["a", "b"], "role": "fit"},
        {"cluster_id": "cluster:00001", "members": ["c"], "role": "development"},
    ]
    with pytest.raises(RelabelError, match="cross-role"):
        build_cluster_table(sample_roles=roles, duplicate_clusters=[["a", "c"]])


def test_append_only_records_resume_by_verified_digest_and_merge_is_byte_stable(tmp_path: Path) -> None:
    from sasto.g1b_relabel import (
        RelabelError, load_verified_case, merge_completed_records, write_case_record,
    )

    root = tmp_path / "relabel"; (root / "cases").mkdir(parents=True)
    alpha = {"sample_id": "a", "role": "fit", "exclusion_reasons": [], "solver": _success()}
    beta = {"sample_id": "b", "role": "development", "exclusion_reasons": ["solver_status_failure"], "solver": {"status": "failure"}}
    write_case_record(root, alpha)
    write_case_record(root, beta)
    assert load_verified_case(root, "a")["case_digest"] == load_verified_case(root, "a")["case_digest"]
    merged = merge_completed_records(root=root, selected_ids=["b", "a"], total_population=2)
    assert merged["eligible_ids"] == ["a"]
    assert merged["exclusion_counts"] == {"solver_status_failure": 1}
    assert merged["records_digest"] == hashlib.sha256(
        __import__("json").dumps(merged["records"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(RelabelError, match="incomplete"):
        merge_completed_records(root=root, selected_ids=["a", "missing"], total_population=2)
