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
    assert chosen["selected"]["beta_compliance"] == 1.10
    assert chosen["selected"]["beta_stress"] == 1.10
    assert chosen["selection"]["case_reasons"] == {"fit-a": "candidate_exhaustion", "fit-b": "stress"}


def test_audit_gate_distinguishes_pass_repose_and_hold() -> None:
    from sasto.activity_campaign import audit_gate

    reasons = [{"sample_id": str(index), "stopping_reason": "compliance" if index < 80 else "candidate_exhaustion"} for index in range(200)]
    assert audit_gate(reasons)["verdict"] == "PASS"
    low = [{"sample_id": str(index), "stopping_reason": "candidate_exhaustion"} for index in range(200)]
    assert audit_gate(low)["verdict"] == "STOP_REPOSE"
    hold = [{"sample_id": str(index), "stopping_reason": "compliance" if index < 50 else "candidate_exhaustion"} for index in range(200)]
    assert audit_gate(hold)["verdict"] == "HOLD_OUTSIDE_TARGET"


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
