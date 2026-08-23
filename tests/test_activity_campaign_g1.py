"""Focused contracts for the frozen fit-only constraint-activity campaign."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def _manifest(fit: list[str]) -> dict[str, object]:
    return {"partitions": {
        "fit": {"sample_ids": fit},
        "development": {"sample_ids": ["dev-only"]},
        "calibration": {"sample_ids": ["cal-only"]},
        "confirmation": {"sample_ids": ["confirm-only"]},
    }}


def _success(*, compliance: float, stress: float, displacement: float, loaded: int = 4) -> dict[str, object]:
    return {"status": "success", "reason": None, "compliance_j": compliance,
            "p99_gauss_von_mises_pa": stress, "max_displacement_m": displacement,
            "relative_residual": 1e-10, "loaded_node_count": loaded,
            "scientific_digest": hashlib.sha256(repr((compliance, stress, displacement, loaded)).encode()).hexdigest()}


def test_fit_subsets_are_hash_ranked_disjoint_and_role_guarded() -> None:
    from sasto.activity_campaign import FitOnlyAccessError, ranked_fit_subsets

    fit = ["fit-c", "fit-a", "fit-b", "fit-d"]
    subsets = ranked_fit_subsets(_manifest(fit), threshold_count=2, audit_count=2)
    expected = sorted(fit, key=lambda item: hashlib.sha256(b"sasto-v-benchmark-activity-v1\0" + item.encode()).hexdigest())
    assert subsets["threshold_design"] == expected[:2]
    assert subsets["activity_audit"] == expected[2:]
    assert not (set(subsets["threshold_design"]) & set(subsets["activity_audit"]))
    leaking = _manifest(fit)
    leaking["partitions"]["fit"]["sample_ids"].append("dev-only")  # type: ignore[index]
    with pytest.raises(FitOnlyAccessError, match="overlap"):
        ranked_fit_subsets(leaking, threshold_count=2, audit_count=2)


def test_candidate_ranking_is_deterministic_and_protected_x_layers_are_never_editable() -> None:
    from sasto.activity_campaign import editable_mask, rank_candidate_coordinates

    volume = np.ones((3, 2, 2), dtype=bool)
    editable, protected = editable_mask(volume)
    assert not editable[0].any() and not editable[2].any()
    assert protected[0].all() and protected[2].all()
    rank_a = rank_candidate_coordinates(volume, editable, sample_id="fit-a", batch_index=1)
    rank_b = rank_candidate_coordinates(volume, editable, sample_id="fit-a", batch_index=1)
    assert rank_a == rank_b
    assert all(point[0] == 1 for point in rank_a)


def test_trajectory_rechecks_sequential_gate_records_ratios_and_stable_load_nodes() -> None:
    from sasto.activity_campaign import run_trajectory

    volume = np.zeros((4, 5, 5), dtype=bool)
    volume[:, 1:4, 1:4] = True
    calls: list[int] = []
    def solver(current: np.ndarray, _: object) -> dict[str, object]:
        calls.append(int(current.sum()))
        # A real fixture would use canonical V; injection keeps the trajectory test tiny.
        return _success(compliance=100.0 / current.sum(), stress=10.0 / current.sum(), displacement=1.0 / current.sum())

    result = run_trajectory(sample_id="fit-a", volume=volume, config=object(), solver=solver, batch_cap=2)
    assert result["stopping_reason"] in {"candidate_exhaustion", "smoke_batch_cap"}
    assert result["baseline"]["loaded_node_count"] == result["batches"][0]["candidate"]["loaded_node_count"]
    assert result["batches"][0]["compliance_ratio"] > 1.0
    assert result["topology"]["topology_mode"] == "conservative_local_6_26"
    assert result["accepted_material_reduction"] > 0
    assert calls[0] == int(volume.sum()) and len(calls) >= 2
    assert json.loads(json.dumps(result, allow_nan=False))["sample_id"] == "fit-a"


def test_threshold_selection_uses_same_batch_normalized_exceedance_and_compliance_tie() -> None:
    from sasto.activity_campaign import choose_binding_reason, select_thresholds

    assert choose_binding_reason(1.20, 1.20, 1.10, 1.10) == "compliance"
    assert choose_binding_reason(1.20, 1.25, 1.10, 1.10) == "stress"
    trajectories = [
        {"sample_id": "fit-a", "stopping_reason": "candidate_exhaustion", "batches": [{"batch_index": 1, "compliance_ratio": 1.10, "p99_gauss_stress_ratio": 1.03}]},
        {"sample_id": "fit-b", "stopping_reason": "candidate_exhaustion", "batches": [{"batch_index": 1, "compliance_ratio": 1.01, "p99_gauss_stress_ratio": 1.11}]},
    ]
    chosen = select_thresholds(trajectories, beta_grid=(1.02, 1.10))
    assert chosen["selected"]["beta_compliance"] == 1.02
    assert chosen["selected"]["beta_stress"] == 1.10
    assert chosen["selection"]["case_reasons"] == {"fit-a": "compliance", "fit-b": "stress"}


def test_threshold_replay_checks_earlier_success_before_terminal_solver_failure_or_cap() -> None:
    from sasto.activity_campaign import _case_for_threshold

    crossed_then_failure = {
        "stopping_reason": "solver_failure",
        "batches": [
            {"batch_index": 1, "candidate": {"status": "success"}, "compliance_ratio": 1.11, "p99_gauss_stress_ratio": 1.01},
            {"batch_index": 2, "candidate": {"status": "failure"}},
        ],
    }
    crossed_then_cap = {
        "stopping_reason": "defensive_cap",
        "batches": [{"batch_index": 1, "candidate": {"status": "success"}, "compliance_ratio": 1.01, "p99_gauss_stress_ratio": 1.11}],
    }
    assert _case_for_threshold(crossed_then_failure, 1.05, 1.05) == ("compliance", False)
    assert _case_for_threshold(crossed_then_cap, 1.05, 1.05) == ("stress", False)
    assert _case_for_threshold({"stopping_reason": "solver_failure", "batches": [{"candidate": {"status": "failure"}}]}, 1.05, 1.05) == ("solver_failure", False)
    assert _case_for_threshold({"stopping_reason": "solver_failure", "batches": [{"candidate": {"status": "failure"}, "compliance_ratio": 1.11, "p99_gauss_stress_ratio": 1.01}]}, 1.05, 1.05) == ("solver_failure", False)
    assert _case_for_threshold({"stopping_reason": "defensive_cap", "batches": []}, 1.05, 1.05) == ("defensive_cap", False)


def test_threshold_selection_targets_larger_individual_named_fraction_not_combined_activity() -> None:
    from sasto.activity_campaign import select_thresholds

    trajectories = [
        {"sample_id": "fit-a", "stopping_reason": "candidate_exhaustion", "batches": [{"compliance_ratio": 1.11, "p99_gauss_stress_ratio": 1.01}]},
        {"sample_id": "fit-b", "stopping_reason": "candidate_exhaustion", "batches": [{"compliance_ratio": 1.11, "p99_gauss_stress_ratio": 1.01}]},
        {"sample_id": "fit-c", "stopping_reason": "candidate_exhaustion", "batches": [{"compliance_ratio": 1.11, "p99_gauss_stress_ratio": 1.01}]},
        {"sample_id": "fit-d", "stopping_reason": "candidate_exhaustion", "batches": [{"compliance_ratio": 1.01, "p99_gauss_stress_ratio": 1.11}]},
        {"sample_id": "fit-e", "stopping_reason": "candidate_exhaustion", "batches": [{"compliance_ratio": 1.01, "p99_gauss_stress_ratio": 1.11}]},
        {"sample_id": "fit-f", "stopping_reason": "candidate_exhaustion", "batches": []},
        {"sample_id": "fit-g", "stopping_reason": "candidate_exhaustion", "batches": []},
        {"sample_id": "fit-h", "stopping_reason": "candidate_exhaustion", "batches": []},
        {"sample_id": "fit-i", "stopping_reason": "candidate_exhaustion", "batches": []},
        {"sample_id": "fit-j", "stopping_reason": "candidate_exhaustion", "batches": []},
    ]
    chosen = select_thresholds(trajectories, beta_grid=(1.05,))
    selection = chosen["selection"]
    assert selection["constraint_activity"] == {"compliance": 0.3, "stress": 0.2}
    assert selection["combined_named_constraint_fraction"] == 0.5
    assert selection["individual_named_constraint_fraction"] == 0.3
    assert selection["score_fields"] == ["has_defensive_cap", "individual_distance_from_half", "individual_range_penalty", "solver_failure_fraction", "co_crossing_count", "beta_compliance", "beta_stress"]
    assert selection["score"][1:3] == (0.2, True)


def test_threshold_selection_prioritizes_distance_before_individual_range_penalty() -> None:
    from sasto.activity_campaign import _case_for_threshold, select_thresholds

    # This trajectory set makes range-first choose 1.05/1.05 (44%/43%, both
    # individual fractions in range), whereas distance-first chooses 1.05/1.10
    # (54%/16%, closer to 50% but individually out of range).
    counts = {
        (1.00, 1.00): 13, (1.00, 1.06): 17, (1.00, 1.11): 16,
        (1.06, 1.00): 8, (1.06, 1.06): 10, (1.06, 1.11): 10,
        (1.11, 1.00): 11, (1.11, 1.06): 8, (1.11, 1.11): 7,
    }
    trajectories = [
        {"sample_id": f"fit-{compliance}-{stress}-{index}", "stopping_reason": "candidate_exhaustion",
         "batches": [{"candidate": {"status": "success"}, "compliance_ratio": compliance, "p99_gauss_stress_ratio": stress}]}
        for (compliance, stress), count in counts.items() for index in range(count)
    ]
    chosen = select_thresholds(trajectories, beta_grid=(1.05, 1.10))
    assert chosen["selected"] == {"beta_compliance": 1.05, "beta_stress": 1.10}
    assert chosen["selection"]["score"][0] is False
    assert chosen["selection"]["score"][1] == pytest.approx(0.04)
    assert chosen["selection"]["score"][2] is True

    # Deliberately reconstruct the preregistered counterfactual range-first
    # comparator: its answer must differ from the implementation's distance-first.
    range_first: list[tuple[tuple[object, ...], tuple[float, float]]] = []
    for beta_compliance in (1.05, 1.10):
        for beta_stress in (1.05, 1.10):
            reasons = [_case_for_threshold(case, beta_compliance, beta_stress)[0] for case in trajectories]
            compliance_fraction = reasons.count("compliance") / len(reasons)
            stress_fraction = reasons.count("stress") / len(reasons)
            larger = max(compliance_fraction, stress_fraction)
            penalty = not (0.40 <= compliance_fraction <= 0.60 and 0.40 <= stress_fraction <= 0.60)
            range_first.append(((False, penalty, abs(larger - 0.50), 0, beta_compliance, beta_stress),
                                (beta_compliance, beta_stress)))
    assert min(range_first)[1] == (1.05, 1.05)


def test_trajectory_retains_last_admitted_volume_when_threshold_candidate_violates() -> None:
    from sasto.activity_campaign import run_trajectory

    volume = np.zeros((4, 5, 5), dtype=bool)
    volume[:, 1:4, 1:4] = True

    def solver(current: np.ndarray, _: object) -> dict[str, object]:
        return _success(compliance=100.0 / current.sum(), stress=10.0 / current.sum(), displacement=1.0 / current.sum())

    result = run_trajectory(sample_id="fit-a", volume=volume, config=object(), solver=solver, beta_compliance=1.01, beta_stress=2.0, batch_cap=2)
    batch = result["batches"][0]
    assert result["stopping_reason"] == "compliance"
    assert result["accepted_material_reduction"] == 0.0
    assert result["proposed_material_reduction"] > 0.0
    assert result["last_admitted_batch_index"] is None
    assert batch["admitted_under_policy"] is False
    assert batch["binding_reason"] == "compliance"


def test_trajectory_retains_last_admitted_volume_when_candidate_solver_fails() -> None:
    from sasto.activity_campaign import run_trajectory

    volume = np.zeros((4, 5, 5), dtype=bool)
    volume[:, 1:4, 1:4] = True
    calls = 0

    def solver(current: np.ndarray, _: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _success(compliance=100.0 / current.sum(), stress=10.0 / current.sum(), displacement=1.0 / current.sum())
        return {"status": "failure", "reason": "injected_failure"}

    result = run_trajectory(sample_id="fit-a", volume=volume, config=object(), solver=solver, batch_cap=2)
    batch = result["batches"][0]
    assert result["stopping_reason"] == "solver_failure"
    assert result["accepted_material_reduction"] == 0.0
    assert result["proposed_material_reduction"] > 0.0
    assert result["last_admitted_batch_index"] is None
    assert batch["admitted_under_policy"] is False
    assert batch["binding_reason"] is None


def test_audit_gate_distinguishes_pass_repose_and_hold() -> None:
    from sasto.activity_campaign import audit_gate

    reasons = [{"sample_id": str(index), "stopping_reason": "compliance" if index < 80 else "candidate_exhaustion"} for index in range(200)]
    assert audit_gate(reasons)["verdict"] == "PASS"
    low = [{"sample_id": str(index), "stopping_reason": "candidate_exhaustion"} for index in range(200)]
    assert audit_gate(low)["verdict"] == "STOP_REPOSE"
    hold = [{"sample_id": str(index), "stopping_reason": "compliance" if index < 50 else "candidate_exhaustion"} for index in range(200)]
    assert audit_gate(hold)["verdict"] == "HOLD_OUTSIDE_TARGET"


def test_generation_report_resume_requires_digest_and_exact_completed_case_set(tmp_path: Path) -> None:
    from sasto.activity_campaign import CampaignError, _generation_report, _verify_generation_report, write_case_record

    root = tmp_path / "campaign"
    root.mkdir()
    (root / "cases").mkdir()
    for sample_id in ("fit-a", "fit-b"):
        write_case_record(root, {"sample_id": sample_id, "batches": [], "stopping_reason": "candidate_exhaustion"})
    complete = _generation_report(root=root, mode="smoke", label="SMOKE_ONLY_NONPROMOTABLE", ids=["fit-a", "fit-b"])
    assert complete["cumulative_expected_payload_count"] == 4
    assert complete["cumulative_expected_payload_members"] == [
        "fea_ml/data/runs_real/fit-a/occ.npz", "fea_ml/data/runs_real/fit-a/meta.json",
        "fea_ml/data/runs_real/fit-b/occ.npz", "fea_ml/data/runs_real/fit-b/meta.json",
    ]
    assert "fit_payload_access_count" not in complete
    _verify_generation_report(root=root, mode="smoke", report=complete)
    _verify_generation_report(root=root, mode="smoke", report=complete)

    report_path = root / "generation-smoke.json"
    tampered = json.loads(report_path.read_text())
    tampered["case_count"] = 99
    report_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(CampaignError, match="digest"):
        _verify_generation_report(root=root, mode="smoke", report=complete)


def test_generation_report_resume_rejects_valid_digest_partial_report(tmp_path: Path) -> None:
    from sasto.activity_campaign import CampaignError, _generation_report, _verify_generation_report, write_case_record

    root = tmp_path / "campaign"
    root.mkdir()
    (root / "cases").mkdir()
    for sample_id in ("fit-a", "fit-b"):
        write_case_record(root, {"sample_id": sample_id, "batches": [], "stopping_reason": "candidate_exhaustion"})
    complete = _generation_report(root=root, mode="smoke", label="SMOKE_ONLY_NONPROMOTABLE", ids=["fit-a", "fit-b"])
    partial = _generation_report(root=root, mode="smoke", label="SMOKE_ONLY_NONPROMOTABLE", ids=["fit-a"])
    _verify_generation_report(root=root, mode="smoke", report=partial)
    with pytest.raises(CampaignError, match="completed cases"):
        _verify_generation_report(root=root, mode="smoke", report=complete)


def test_generation_invocation_receipts_keep_measured_accesses_separate_on_resume(tmp_path: Path) -> None:
    from sasto.activity_campaign import _generation_report, _verify_generation_report, _write_generation_invocation_receipt, write_case_record

    root = tmp_path / "campaign"; root.mkdir(); (root / "cases").mkdir(); (root / "invocations").mkdir()
    for sample_id in ("fit-a", "fit-b"):
        write_case_record(root, {"sample_id": sample_id, "batches": [], "stopping_reason": "candidate_exhaustion"})
    stable = _generation_report(root=root, mode="smoke", label="SMOKE_ONLY_NONPROMOTABLE", ids=["fit-a", "fit-b"])
    _verify_generation_report(root=root, mode="smoke", report=stable)
    members = stable["cumulative_expected_payload_members"]
    fresh = _write_generation_invocation_receipt(root=root, mode="smoke", members=members, fit_accesses=len(members), nonfit_accesses=0)
    resumed = _write_generation_invocation_receipt(root=root, mode="smoke", members=[], fit_accesses=0, nonfit_accesses=0)
    assert fresh["invocation_measured_payload_count"] == 4
    assert resumed["invocation_measured_payload_count"] == 0
    assert _verify_generation_report(root=root, mode="smoke", report=stable) == stable


def test_frozen_selection_embeds_only_threshold_protocol_not_manifest_digest(tmp_path: Path) -> None:
    from sasto.activity_campaign import _digest, select_thresholds_from_root, write_case_record

    root = tmp_path / "campaign"; root.mkdir(); (root / "cases").mkdir()
    ids = [f"fit-{index}" for index in range(50)]
    protocol = {"schema_version": "1.0.0", "namespace": "sasto-v-benchmark-activity-v1", "role": "fit", "mode": "threshold_design",
                "threshold_design_ids": ids, "activity_audit_ids": [f"audit-{index}" for index in range(200)], "selected_ids": ids,
                "split_manifest_sha256": "a" * 64, "archive_sha256": "b" * 64, "code_sha256": "c" * 64,
                "verifier": {"fixed_total_benchmark_force_n": [0.0, 0.0, -100.0], "include_self_weight": False,
                             "support": "minimum_physical_element_x_face", "load": "maximum_physical_element_x_face", "admission_relative_tolerance": 2e-8},
                "topology_mode": "conservative_local_6_26", "max_batches": 40, "frozen_thresholds": None, "label": "FROZEN_FIT_ONLY_PROTOCOL"}
    (root / "campaign-manifest.json").write_text(json.dumps({**protocol, "manifest_digest": _digest(protocol)}), encoding="utf-8")
    for sample_id in ids:
        write_case_record(root, {"sample_id": sample_id, "batches": [], "stopping_reason": "candidate_exhaustion"})
    frozen = select_thresholds_from_root(root)
    assert frozen["protocol"] == protocol
    assert "manifest_digest" not in frozen["protocol"]


def test_audit_compatibility_rejects_each_load_bearing_threshold_protocol_field() -> None:
    from copy import deepcopy
    from sasto.activity_campaign import CampaignError, validate_threshold_audit_compatibility

    digest = "a" * 64
    protocol = {
        "schema_version": "1.0.0", "namespace": "sasto-v-benchmark-activity-v1", "role": "fit", "mode": "threshold_design",
        "threshold_design_ids": [f"threshold-{index}" for index in range(50)],
        "activity_audit_ids": [f"audit-{index}" for index in range(200)],
        "selected_ids": [f"threshold-{index}" for index in range(50)],
        "split_manifest_sha256": digest, "archive_sha256": digest, "code_sha256": digest,
        "verifier": {"fixed_total_benchmark_force_n": [0.0, 0.0, -100.0], "include_self_weight": False,
                     "support": "minimum_physical_element_x_face", "load": "maximum_physical_element_x_face", "admission_relative_tolerance": 2e-8},
        "topology_mode": "conservative_local_6_26", "max_batches": 40, "frozen_thresholds": None,
        "label": "FROZEN_FIT_ONLY_PROTOCOL",
    }
    assert validate_threshold_audit_compatibility({"protocol": protocol}, protocol) is None

    mutations = [
        ("schema_version", "2.0.0"), ("namespace", "wrong"), ("role", "audit"), ("mode", "activity_audit"),
        ("threshold_design_ids", ["wrong"] * 50), ("activity_audit_ids", ["wrong"] * 200),
        ("selected_ids", ["wrong"] * 50), ("split_manifest_sha256", "b" * 64), ("archive_sha256", "b" * 64),
        ("code_sha256", "b" * 64), ("topology_mode", "wrong"), ("max_batches", 2),
        ("frozen_thresholds", {"beta_compliance": 1.05}), ("label", "wrong"),
    ]
    for field, replacement in mutations:
        frozen = {"protocol": deepcopy(protocol)}
        frozen["protocol"][field] = replacement
        with pytest.raises(CampaignError, match="incompatible"):
            validate_threshold_audit_compatibility(frozen, protocol)
    for verifier_field, replacement in (("fixed_total_benchmark_force_n", [1.0, 0.0, 0.0]), ("include_self_weight", True),
                                         ("support", "wrong"), ("load", "wrong"), ("admission_relative_tolerance", 1e-3)):
        frozen = {"protocol": deepcopy(protocol)}
        frozen["protocol"]["verifier"][verifier_field] = replacement
        with pytest.raises(CampaignError, match="incompatible"):
            validate_threshold_audit_compatibility(frozen, protocol)


def test_audit_summary_is_digest_verified_and_exactly_recomputed(tmp_path: Path) -> None:
    from sasto.activity_campaign import CampaignError, _audit_summary, _generation_report, _verify_audit_summary, write_case_record

    root = tmp_path / "campaign"; root.mkdir(); (root / "cases").mkdir()
    write_case_record(root, {"sample_id": "fit-a", "batches": [], "stopping_reason": "candidate_exhaustion"})
    generation = _generation_report(root=root, mode="smoke", label="SMOKE_ONLY_NONPROMOTABLE", ids=["fit-a"])
    expected = _audit_summary(generation=generation, audit={"case_count": 200, "verdict": "PASS"}, threshold_protocol_hash="a" * 64)
    _verify_audit_summary(root=root, summary=expected)
    _verify_audit_summary(root=root, summary=expected)

    path = root / "audit-summary.json"
    tampered = json.loads(path.read_text())
    tampered["audit"]["verdict"] = "HOLD_OUTSIDE_TARGET"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(CampaignError, match="digest"):
        _verify_audit_summary(root=root, summary=expected)

    stale = _audit_summary(generation=generation, audit={"case_count": 199, "verdict": "PASS"}, threshold_protocol_hash="a" * 64)
    path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(CampaignError, match="recomputed"):
        _verify_audit_summary(root=root, summary=expected)


def test_campaign_resume_accepts_verified_case_and_rejects_tampering(tmp_path: Path) -> None:
    from sasto.activity_campaign import CampaignError, load_verified_case, write_case_record

    root = tmp_path / "campaign"; root.mkdir(); (root / "cases").mkdir()
    record = {"sample_id": "fit-a", "batches": [], "stopping_reason": "candidate_exhaustion"}
    write_case_record(root, record)
    assert load_verified_case(root, "fit-a")["sample_id"] == "fit-a"
    path = root / "cases" / "fit-a.json"
    payload = json.loads(path.read_text())
    payload["stopping_reason"] = "defensive_cap"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CampaignError, match="digest"):
        load_verified_case(root, "fit-a")
