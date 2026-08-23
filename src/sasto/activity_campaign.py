"""Frozen, resumable, fit-only constraint-activity campaign.

The module deliberately separates source admission, deterministic trajectory generation,
threshold replay, and the 200-design audit.  It never lists archive members and it
never accepts a non-fit sample identifier.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import math
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .fit_probe import FitOnlyAccessError, _PayloadAccessLedger, _configuration, _load_occupancy, select_fit_sample_ids
from .manifest import ManifestVerificationError, open_new_artifact_root, read_regular_path_snapshot, write_new_regular_path
from .topology import conservative_local_6_26, exact_topology_preflight_6_26
from .voxel_fea import VoxelFEAConfig, assemble_voxel_system, solve_voxels

NAMESPACE = "sasto-v-benchmark-activity-v1"
BETA_GRID = (1.02, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50, 2.00)
SCHEMA_VERSION = "1.0.0"


class CampaignError(ValueError):
    """Fail-closed campaign configuration, source, or artifact error."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_text(*parts: object) -> str:
    return hashlib.sha256("\0".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _lower_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise CampaignError("{} must be an exact lowercase SHA-256 digest".format(label))
    return value


def ranked_fit_subsets(manifest: Mapping[str, object], *, threshold_count: int = 50, audit_count: int = 200) -> dict[str, list[str]]:
    """Validate the complete split and assign frozen fit-only selections."""
    if any(not isinstance(count, int) or isinstance(count, bool) or count < 1 for count in (threshold_count, audit_count)):
        raise CampaignError("subset counts must be positive integers")
    try:
        # This shared guard validates every role and proves all are disjoint before selection.
        fit = select_fit_sample_ids(manifest, None, limit=len(manifest["partitions"]["fit"]["sample_ids"]))  # type: ignore[index]
    except (FitOnlyAccessError, KeyError, TypeError) as error:
        raise FitOnlyAccessError(str(error)) from error
    ordered = sorted(fit, key=lambda sample_id: _sha_text(NAMESPACE, sample_id))
    if len(ordered) < threshold_count + audit_count:
        raise CampaignError("fit population is insufficient for frozen threshold and audit subsets")
    return {"threshold_design": ordered[:threshold_count], "activity_audit": ordered[threshold_count:threshold_count + audit_count]}


def editable_mask(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Protect baseline occupied minimum/maximum physical element-x layers."""
    if not isinstance(volume, np.ndarray) or volume.dtype != np.bool_ or volume.ndim != 3 or not volume.any():
        raise CampaignError("volume must be a nonempty three-dimensional bool array")
    occupied_x = np.flatnonzero(np.any(volume, axis=(1, 2)))
    protected = np.zeros_like(volume, dtype=bool)
    protected[occupied_x.min()] = volume[occupied_x.min()]
    protected[occupied_x.max()] = volume[occupied_x.max()]
    return volume & ~protected, protected


def rank_candidate_coordinates(volume: np.ndarray, editable: np.ndarray, *, sample_id: str, batch_index: int) -> list[tuple[int, int, int]]:
    """Rank *remaining* editable occupied voxels cryptographically per batch."""
    if not isinstance(batch_index, int) or isinstance(batch_index, bool) or batch_index < 1:
        raise CampaignError("batch index must be a positive integer")
    points = [tuple(int(value) for value in row) for row in np.argwhere(volume & editable)]
    return sorted(points, key=lambda point: _sha_text(NAMESPACE, sample_id, batch_index, *point))


def _metrics(record: Mapping[str, object]) -> tuple[float, float, float]:
    try:
        values = (float(record["compliance_j"]), float(record["p99_gauss_von_mises_pa"]), float(record["max_displacement_m"]))
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError("canonical success record lacks activity metrics") from error
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise CampaignError("canonical activity metrics must be finite and positive")
    return values


def _scientific_record(record: Mapping[str, object]) -> dict[str, object]:
    # Timing is explicitly excluded from every activity scientific hash.
    return {key: value for key, value in record.items() if key != "timing"}


def _load_coordinates(volume: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    assembled = assemble_voxel_system(volume, VoxelFEAConfig(include_self_weight=False, fixed_total_force_n=(0.0, 0.0, -100.0)))
    max_x = int(np.argwhere(volume)[:, 0].max()) + 1
    return tuple(tuple(int(value) for value in point) for point in assembled.node_coordinates if int(point[0]) == max_x)


def _activity_config(volume: np.ndarray, base: VoxelFEAConfig) -> VoxelFEAConfig:
    coordinates = _load_coordinates(volume)
    return dataclasses.replace(
        base, include_self_weight=False, fixed_total_force_n=(0.0, 0.0, -100.0), relative_tolerance=2e-8,
        expected_loaded_node_count=len(coordinates), expected_loaded_node_coordinates=coordinates,
    )


def _failure_reason(record: Mapping[str, object]) -> str:
    reason = record.get("reason")
    return reason if isinstance(reason, str) and reason else "solver_failure"


def choose_binding_reason(compliance_ratio: float, stress_ratio: float, beta_compliance: float, beta_stress: float) -> str | None:
    """Apply the frozen same-batch ratio rule, favoring compliance on exact ties."""
    compliance = compliance_ratio > beta_compliance
    stress = stress_ratio > beta_stress
    if not compliance and not stress:
        return None
    if compliance and not stress:
        return "compliance"
    if stress and not compliance:
        return "stress"
    return "compliance" if compliance_ratio / beta_compliance >= stress_ratio / beta_stress else "stress"


def run_trajectory(
    *, sample_id: str, volume: np.ndarray, config: object, solver: Callable[[np.ndarray, object], dict[str, object]] = solve_voxels,
    beta_compliance: float | None = None, beta_stress: float | None = None, batch_cap: int = 40,
) -> dict[str, object]:
    """Run sequential conservative deletions and canonical V after every accepted batch."""
    if not isinstance(sample_id, str) or not sample_id:
        raise CampaignError("sample ID must be nonempty")
    if (beta_compliance is None) != (beta_stress is None):
        raise CampaignError("both thresholds or neither threshold must be supplied")
    if any(beta is not None and (not np.isfinite(beta) or beta <= 1.0) for beta in (beta_compliance, beta_stress)):
        raise CampaignError("thresholds must be finite ratios greater than one")
    if not isinstance(batch_cap, int) or isinstance(batch_cap, bool) or not 1 <= batch_cap <= 40:
        raise CampaignError("batch cap must be an integer from one through forty")
    if not isinstance(volume, np.ndarray) or volume.dtype != np.bool_:
        raise CampaignError("volume must be boolean")
    source = volume.copy()
    editable, protected = editable_mask(source)
    topology = exact_topology_preflight_6_26(source).as_dict()
    baseline = solver(source.copy(), config)
    if baseline.get("status") != "success":
        return {"schema_version": SCHEMA_VERSION, "sample_id": sample_id, "baseline": _scientific_record(baseline), "batches": [],
                "stopping_reason": "solver_failure", "solver_failure_reason": _failure_reason(baseline),
                "accepted_material_reduction": 0.0, "proposed_material_reduction": 0.0,
                "last_admitted_batch_index": None,
                "topology": {"topology_mode": "conservative_local_6_26", "exact_preflight": topology, "sequential_recheck": True}}
    c0, s0, d0 = _metrics(baseline)
    expected_loaded = baseline.get("loaded_node_count")
    if not isinstance(expected_loaded, int) or expected_loaded < 1:
        raise CampaignError("baseline missing stable loaded-node evidence")
    current = source.copy()
    last_admitted = source.copy()
    last_admitted_batch_index: int | None = None
    batches: list[dict[str, object]] = []
    per_batch_limit = max(1, math.ceil(0.05 * int(source.sum())))
    stopping = "candidate_exhaustion"
    solver_failure_reason: str | None = None
    for batch_index in range(1, batch_cap + 1):
        accepted: list[tuple[int, int, int]] = []
        rejected = 0
        for point in rank_candidate_coordinates(current, editable, sample_id=sample_id, batch_index=batch_index):
            if len(accepted) >= per_batch_limit:
                break
            if conservative_local_6_26(current, point):
                current[point] = False
                accepted.append(point)
            else:
                rejected += 1
        if not accepted:
            stopping = "candidate_exhaustion"
            break
        candidate = solver(current.copy(), config)
        batch: dict[str, object] = {"batch_index": batch_index, "accepted_coordinates": [list(point) for point in accepted],
            "accepted_count": len(accepted), "rejected_before_limit": rejected, "candidate": _scientific_record(candidate),
            "volume_ratio": float(current.sum() / source.sum()), "proposed_volume_ratio": float(current.sum() / source.sum()),
            "proposed_material_reduction": 1.0 - (float(current.sum()) / float(source.sum())),
            "admitted_under_policy": False, "binding_reason": None, "sequential_recheck": True}
        if candidate.get("status") != "success":
            batches.append(batch)
            stopping = "solver_failure"; solver_failure_reason = _failure_reason(candidate)
            break
        c, s, d = _metrics(candidate)
        loaded = candidate.get("loaded_node_count")
        if loaded != expected_loaded:
            batches.append(batch)
            stopping = "solver_failure"; solver_failure_reason = "unstable_load_node_set"
            break
        batch.update({"compliance_ratio": c / c0, "p99_gauss_stress_ratio": s / s0, "max_displacement_ratio": d / d0,
                     "relative_residual": candidate.get("relative_residual"), "load_node_stable": True})
        binding = choose_binding_reason(c / c0, s / s0, float(beta_compliance), float(beta_stress)) if beta_compliance is not None else None
        batch["binding_reason"] = binding
        if binding is not None:
            batches.append(batch)
            stopping = binding
            break
        batch["admitted_under_policy"] = True
        last_admitted = current.copy()
        last_admitted_batch_index = batch_index
        batches.append(batch)
    else:
        stopping = "defensive_cap" if batch_cap == 40 else "smoke_batch_cap"
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION, "sample_id": sample_id, "baseline": _scientific_record(baseline), "batches": batches,
        "stopping_reason": stopping,
        "accepted_material_reduction": 1.0 - (float(last_admitted.sum()) / float(source.sum())),
        "proposed_material_reduction": 1.0 - (float(current.sum()) / float(source.sum())),
        "last_admitted_batch_index": last_admitted_batch_index,
        "topology": {"topology_mode": "conservative_local_6_26", "exact_preflight": topology, "sequential_recheck": True,
                     "protected_minimum_and_maximum_occupied_element_x_layers": True, "protected_voxel_count": int(protected.sum())},
        "per_batch_acceptance_limit": per_batch_limit, "solver_call_count": 1 + len(batches),
    }
    if solver_failure_reason is not None:
        result["solver_failure_reason"] = solver_failure_reason
    identity = {key: value for key, value in result.items() if key != "case_digest"}
    result["case_digest"] = _digest(identity)
    return result


def write_case_record(root: Path, record: Mapping[str, object]) -> str:
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id or any(char in sample_id for char in "/\\\x00"):
        raise CampaignError("case record sample ID is unsafe")
    path = Path(root) / "cases" / "{}.json".format(sample_id)
    payload = dict(record)
    payload["case_digest"] = _digest({key: value for key, value in payload.items() if key != "case_digest"})
    try:
        return write_new_regular_path(path, _canonical_bytes(payload) + b"\n", "activity case")
    except ManifestVerificationError as error:
        raise CampaignError(str(error)) from error


def load_verified_case(root: Path, sample_id: str) -> dict[str, object]:
    try:
        snapshot = read_regular_path_snapshot(Path(root) / "cases" / "{}.json".format(sample_id), "activity case")
        value = json.loads(snapshot.bytes)
    except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError("cannot safely load activity case") from error
    if not isinstance(value, dict) or value.get("sample_id") != sample_id:
        raise CampaignError("activity case identity is invalid")
    observed = value.get("case_digest")
    if not isinstance(observed, str) or observed != _digest({key: item for key, item in value.items() if key != "case_digest"}):
        raise CampaignError("activity case digest mismatch")
    return value


def _expected_payload_members(ids: Sequence[str]) -> list[str]:
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in ids) or len(set(ids)) != len(ids):
        raise CampaignError("generation report sample IDs are invalid")
    return ["fea_ml/data/runs_real/{}/{}".format(sample_id, leaf) for sample_id in ids for leaf in ("occ.npz", "meta.json")]


def _generation_report(*, root: Path, mode: str, label: object, ids: Sequence[str]) -> dict[str, object]:
    """Derive the immutable cumulative report solely from completed verified cases."""
    members = _expected_payload_members(ids)
    report = {"mode": mode, "label": label, "case_count": len(ids),
              "results": [{"sample_id": sample_id, "case_digest": load_verified_case(root, sample_id)["case_digest"]} for sample_id in ids],
              "cumulative_expected_payload_count": len(members),
              "cumulative_expected_payload_members": members}
    return {**report, "report_digest": _digest(report)}


def _write_generation_invocation_receipt(*, root: Path, mode: str, members: Sequence[str], fit_accesses: int, nonfit_accesses: int) -> dict[str, object]:
    """Write an immutable per-invocation ledger receipt outside the stable report."""
    if fit_accesses != len(members) or nonfit_accesses < 0:
        raise CampaignError("generation invocation ledger is inconsistent")
    receipt = {"mode": mode, "invocation_measured_payload_members": list(members),
               "invocation_measured_payload_count": fit_accesses,
               "invocation_measured_nonfit_payload_count": nonfit_accesses}
    receipt["invocation_receipt_digest"] = _digest(receipt)
    path = Path(root) / "invocations" / "generation-{}-{}.json".format(mode, receipt["invocation_receipt_digest"])
    try:
        if path.exists():
            snapshot = read_regular_path_snapshot(path, "generation invocation receipt")
            existing = json.loads(snapshot.bytes)
            if existing != receipt:
                raise CampaignError("existing generation invocation receipt is inconsistent")
        else:
            write_new_regular_path(path, _canonical_bytes(receipt) + b"\n", "generation invocation receipt")
    except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError("cannot safely write generation invocation receipt") from error
    return receipt


def _load_verified_generation_report(root: Path, mode: str) -> dict[str, object]:
    try:
        snapshot = read_regular_path_snapshot(Path(root) / "generation-{}.json".format(mode), "generation report")
        value = json.loads(snapshot.bytes)
    except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError("generation report is unavailable") from error
    if not isinstance(value, dict):
        raise CampaignError("generation report is malformed")
    observed = value.get("report_digest")
    if not isinstance(observed, str) or observed != _digest({key: item for key, item in value.items() if key != "report_digest"}):
        raise CampaignError("generation report digest mismatch")
    return value


def _verify_generation_report(*, root: Path, mode: str, report: Mapping[str, object]) -> dict[str, object]:
    """Write once or reject a resume whose report is not an exact recomputation."""
    report_path = Path(root) / "generation-{}.json".format(mode)
    expected = dict(report)
    if report_path.exists():
        existing = _load_verified_generation_report(root, mode)
        if existing != expected:
            raise CampaignError("existing generation report does not match completed cases")
        return existing
    try:
        write_new_regular_path(report_path, _canonical_bytes(expected) + b"\n", "generation report")
    except ManifestVerificationError as error:
        raise CampaignError(str(error)) from error
    return expected


def _audit_summary(*, generation: Mapping[str, object], audit: Mapping[str, object], threshold_protocol_hash: str) -> dict[str, object]:
    _lower_digest(threshold_protocol_hash, "threshold protocol hash")
    summary = {"generation": dict(generation), "audit": dict(audit), "threshold_protocol_hash": threshold_protocol_hash}
    return {**summary, "audit_summary_digest": _digest(summary)}


def _load_verified_audit_summary(root: Path) -> dict[str, object]:
    try:
        snapshot = read_regular_path_snapshot(Path(root) / "audit-summary.json", "audit summary")
        value = json.loads(snapshot.bytes)
    except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError("audit summary is unavailable") from error
    if not isinstance(value, dict):
        raise CampaignError("audit summary is malformed")
    observed = value.get("audit_summary_digest")
    if not isinstance(observed, str) or observed != _digest({key: item for key, item in value.items() if key != "audit_summary_digest"}):
        raise CampaignError("audit summary digest mismatch")
    return value


def _verify_audit_summary(*, root: Path, summary: Mapping[str, object]) -> dict[str, object]:
    """Append-only audit summary: accept only the exact current recomputation."""
    expected = dict(summary)
    path = Path(root) / "audit-summary.json"
    if path.exists():
        existing = _load_verified_audit_summary(root)
        if existing != expected:
            raise CampaignError("existing audit summary does not match recomputed generation, audit, and threshold hash")
        return existing
    try:
        write_new_regular_path(path, _canonical_bytes(expected) + b"\n", "audit summary")
    except ManifestVerificationError as error:
        raise CampaignError(str(error)) from error
    return expected


def _case_for_threshold(case: Mapping[str, object], beta_compliance: float, beta_stress: float) -> tuple[str, bool]:
    """Replay every successful ratio-bearing batch before honoring terminal state."""
    batches = case.get("batches")
    if not isinstance(batches, list):
        raise CampaignError("trajectory batches are invalid")
    for batch in batches:
        if not isinstance(batch, dict):
            raise CampaignError("trajectory batch is invalid")
        candidate = batch.get("candidate")
        if isinstance(candidate, Mapping) and candidate.get("status") != "success":
            continue
        if "compliance_ratio" not in batch or "p99_gauss_stress_ratio" not in batch:
            # A solver-failed terminal candidate has no ratios and cannot cross a threshold.
            continue
        try:
            compliance_ratio = float(batch["compliance_ratio"])
            stress_ratio = float(batch["p99_gauss_stress_ratio"])
        except (TypeError, ValueError) as error:
            raise CampaignError("threshold trajectory has invalid replay ratios") from error
        if not all(np.isfinite(value) and value > 0.0 for value in (compliance_ratio, stress_ratio)):
            raise CampaignError("threshold trajectory has invalid replay ratios")
        reason = choose_binding_reason(compliance_ratio, stress_ratio, beta_compliance, beta_stress)
        if reason:
            return reason, ((reason == "compliance" and stress_ratio > beta_stress) or
                            (reason == "stress" and compliance_ratio > beta_compliance))
    stopping_reason = case.get("stopping_reason", "candidate_exhaustion")
    if not isinstance(stopping_reason, str):
        raise CampaignError("trajectory stopping reason is invalid")
    return stopping_reason, False


_SCORE_FIELDS = [
    "has_defensive_cap",
    "individual_distance_from_half",
    "individual_range_penalty",
    "solver_failure_fraction",
    "co_crossing_count",
    "beta_compliance",
    "beta_stress",
]


def select_thresholds(trajectories: Sequence[Mapping[str, object]], *, beta_grid: Sequence[float] = BETA_GRID) -> dict[str, object]:
    """Replay complete threshold trajectories only; no solver calls occur here."""
    if not trajectories or not beta_grid:
        raise CampaignError("threshold selection requires nonempty trajectories and grid")
    candidates = []
    for beta_c in beta_grid:
        for beta_s in beta_grid:
            reasons: dict[str, str] = {}; co_crossings = 0
            for case in trajectories:
                sample_id = case.get("sample_id")
                if not isinstance(sample_id, str) or sample_id in reasons:
                    raise CampaignError("threshold trajectory sample IDs must be unique")
                reason, co_crossed = _case_for_threshold(case, float(beta_c), float(beta_s))
                reasons[sample_id] = reason; co_crossings += int(co_crossed)
            n = len(reasons)
            comp = sum(reason == "compliance" for reason in reasons.values()) / n
            stress = sum(reason == "stress" for reason in reasons.values()) / n
            failures = sum(reason == "solver_failure" for reason in reasons.values()) / n
            caps = sum(reason == "defensive_cap" for reason in reasons.values())
            combined_named = comp + stress
            individual_named = max(comp, stress)
            individual_range_penalty = not (0.40 <= individual_named <= 0.60)
            score = (caps != 0, abs(individual_named - 0.50), individual_range_penalty,
                     failures, co_crossings, float(beta_c), float(beta_s))
            candidates.append((score, {"beta_compliance": float(beta_c), "beta_stress": float(beta_s), "case_reasons": reasons,
                                       "constraint_activity": {"compliance": comp, "stress": stress},
                                       "combined_named_constraint_fraction": combined_named,
                                       "individual_named_constraint_fraction": individual_named,
                                       "solver_failure_fraction": failures, "defensive_cap_count": caps,
                                       "co_crossing_count": co_crossings, "score_fields": _SCORE_FIELDS, "score": score}))
    winning_score, selected = min(candidates, key=lambda item: item[0])
    if selected["score"] != winning_score:
        raise CampaignError("threshold selection score is inconsistent")
    return {"selected": {"beta_compliance": selected["beta_compliance"], "beta_stress": selected["beta_stress"]}, "selection": selected}


def audit_gate(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(cases) != 200:
        raise CampaignError("audit gate requires exactly 200 cases")
    reasons = [case.get("stopping_reason") for case in cases]
    if any(not isinstance(reason, str) for reason in reasons):
        raise CampaignError("audit cases lack stopping reasons")
    table = {reason: reasons.count(reason) for reason in sorted(set(reasons))}
    activity = {name: table.get(name, 0) / 200.0 for name in ("compliance", "stress")}
    if any(0.40 <= fraction <= 0.60 for fraction in activity.values()):
        verdict = "PASS"
    elif max(activity.values()) < 0.20:
        verdict = "STOP_REPOSE"
    else:
        verdict = "HOLD_OUTSIDE_TARGET"
    return {"case_count": 200, "constraint_activity": activity, "stopping_reason_counts": table, "verdict": verdict,
            "solver_failure_count": table.get("solver_failure", 0), "candidate_exhaustion_count": table.get("candidate_exhaustion", 0),
            "defensive_cap_count": table.get("defensive_cap", 0), "accepted_material_reduction_mean": float(np.mean([float(case.get("accepted_material_reduction", 0.0)) for case in cases]))}


def _source_snapshot(split_manifest: Path, archive: Path, split_sha: str, archive_sha: str) -> tuple[Mapping[str, object], bytes, str, str]:
    _lower_digest(split_sha, "split anchor"); _lower_digest(archive_sha, "archive anchor")
    try:
        split = read_regular_path_snapshot(split_manifest, "split manifest")
    except ManifestVerificationError as error:
        raise CampaignError(str(error)) from error
    if split.sha256 != split_sha:
        raise CampaignError("split manifest sha256 mismatch")
    try:
        manifest = json.loads(split.bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError("split manifest is malformed") from error
    # Complete role validation before the archive descriptor is opened.
    ranked_fit_subsets(manifest)
    try:
        archive_snapshot = read_regular_path_snapshot(archive, "archive")
    except ManifestVerificationError as error:
        raise CampaignError(str(error)) from error
    if archive_snapshot.sha256 != archive_sha:
        raise CampaignError("archive sha256 mismatch")
    return manifest, archive_snapshot.bytes, split.sha256, archive_snapshot.sha256


def _code_hash() -> str:
    path = Path(__file__)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _threshold_design_protocol(*, subsets: Mapping[str, Sequence[str]], split_sha256: str, archive_sha256: str) -> dict[str, object]:
    """Build the complete frozen threshold-design protocol from current inputs."""
    threshold_ids = list(subsets["threshold_design"])
    audit_ids = list(subsets["activity_audit"])
    return {"schema_version": SCHEMA_VERSION, "namespace": NAMESPACE, "role": "fit", "mode": "threshold_design",
            "threshold_design_ids": threshold_ids, "activity_audit_ids": audit_ids, "selected_ids": threshold_ids,
            "split_manifest_sha256": split_sha256, "archive_sha256": archive_sha256, "code_sha256": _code_hash(),
            "verifier": {"fixed_total_benchmark_force_n": [0.0, 0.0, -100.0], "include_self_weight": False,
                         "support": "minimum_physical_element_x_face", "load": "maximum_physical_element_x_face", "admission_relative_tolerance": 2e-8},
            "topology_mode": "conservative_local_6_26", "max_batches": 40, "frozen_thresholds": None,
            "label": "FROZEN_FIT_ONLY_PROTOCOL"}


def _audit_compatibility_inputs(*, split_manifest: Path, expected_split_sha256: str, expected_archive_sha256: str) -> dict[str, object]:
    """Read only the split descriptor to form audit compatibility inputs before archive access."""
    _lower_digest(expected_split_sha256, "split anchor")
    _lower_digest(expected_archive_sha256, "archive anchor")
    try:
        split = read_regular_path_snapshot(split_manifest, "split manifest")
        sources = json.loads(split.bytes.decode("utf-8"))
    except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError("split manifest is unavailable for audit compatibility") from error
    if split.sha256 != expected_split_sha256:
        raise CampaignError("split manifest sha256 mismatch")
    if not isinstance(sources, Mapping):
        raise CampaignError("split manifest is malformed")
    return _threshold_design_protocol(subsets=ranked_fit_subsets(sources), split_sha256=split.sha256, archive_sha256=expected_archive_sha256)


def validate_threshold_audit_compatibility(frozen_selection: Mapping[str, object], audit_inputs: Mapping[str, object]) -> None:
    """Pure fail-closed equivalence check of frozen design and current audit inputs."""
    protocol = frozen_selection.get("protocol")
    if not isinstance(protocol, Mapping) or dict(protocol) != dict(audit_inputs):
        raise CampaignError("frozen threshold protocol is incompatible with current audit inputs")


def _campaign_manifest(root: Path) -> dict[str, object]:
    try:
        snapshot = read_regular_path_snapshot(root / "campaign-manifest.json", "campaign manifest")
        value = json.loads(snapshot.bytes)
    except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError("campaign manifest is unavailable") from error
    if not isinstance(value, dict) or value.get("manifest_digest") != _digest({key: item for key, item in value.items() if key != "manifest_digest"}):
        raise CampaignError("campaign manifest digest mismatch")
    return value


def _new_campaign(root: Path, manifest: Mapping[str, object]) -> None:
    try:
        with open_new_artifact_root(root) as root_fd:
            os.mkdir("cases", dir_fd=root_fd)
            os.mkdir("invocations", dir_fd=root_fd)
        write_new_regular_path(root / "campaign-manifest.json", _canonical_bytes({**manifest, "manifest_digest": _digest(manifest)}) + b"\n", "campaign manifest")
    except (ManifestVerificationError, OSError) as error:
        raise CampaignError("cannot create campaign root") from error


def generate_trajectories(*, root: Path, split_manifest: Path, archive: Path, expected_split_sha256: str, expected_archive_sha256: str,
                          mode: str, smoke_batch_cap: int | None = None, beta_compliance: float | None = None,
                          beta_stress: float | None = None, threshold_protocol_hash: str | None = None) -> dict[str, object]:
    if mode not in {"threshold_design", "activity_audit", "smoke"}:
        raise CampaignError("mode must be threshold_design, activity_audit, or smoke")
    if mode == "activity_audit":
        if beta_compliance is None or beta_stress is None:
            raise CampaignError("activity audit requires frozen compliance and stress thresholds")
        _lower_digest(threshold_protocol_hash, "threshold protocol hash")
    elif beta_compliance is not None or beta_stress is not None or threshold_protocol_hash is not None:
        raise CampaignError("only activity audit may supply frozen thresholds")
    sources, archive_bytes, split_sha, archive_sha = _source_snapshot(split_manifest, archive, expected_split_sha256, expected_archive_sha256)
    subsets = ranked_fit_subsets(sources)
    if mode == "activity_audit":
        ids = subsets["activity_audit"]
    else:
        ids = subsets["threshold_design"]
    if mode == "smoke":
        ids = ids[:3]
        if smoke_batch_cap is None or smoke_batch_cap != 2:
            raise CampaignError("SMOKE_ONLY_NONPROMOTABLE requires --smoke-batch-cap 2")
    if mode == "threshold_design":
        protocol = _threshold_design_protocol(subsets=subsets, split_sha256=split_sha, archive_sha256=archive_sha)
    else:
        protocol = {"schema_version": SCHEMA_VERSION, "namespace": NAMESPACE, "role": "fit", "mode": mode,
                    "threshold_design_ids": subsets["threshold_design"], "activity_audit_ids": subsets["activity_audit"],
                    "selected_ids": ids, "split_manifest_sha256": split_sha, "archive_sha256": archive_sha, "code_sha256": _code_hash(),
                    "verifier": {"fixed_total_benchmark_force_n": [0.0, 0.0, -100.0], "include_self_weight": False,
                                 "support": "minimum_physical_element_x_face", "load": "maximum_physical_element_x_face", "admission_relative_tolerance": 2e-8},
                    "topology_mode": "conservative_local_6_26", "max_batches": 40,
                    "frozen_thresholds": None if mode == "smoke" else {"beta_compliance": beta_compliance, "beta_stress": beta_stress},
                    "label": "SMOKE_ONLY_NONPROMOTABLE" if mode == "smoke" else "FROZEN_FIT_ONLY_PROTOCOL"}
        if mode == "activity_audit":
            protocol["threshold_protocol_hash"] = threshold_protocol_hash
    if root.exists():
        existing = _campaign_manifest(root)
        if {key: existing.get(key) for key in protocol} != protocol:
            raise CampaignError("existing campaign manifest does not match frozen protocol")
    else:
        _new_campaign(root, protocol)
    archive_ledger = _PayloadAccessLedger(ids)
    generated_ids: list[str] = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as opened:
        for sample_id in ids:
            try:
                load_verified_case(root, sample_id)
                continue
            except CampaignError as error:
                if (root / "cases" / "{}.json".format(sample_id)).exists():
                    raise error
            volume = _load_occupancy(opened, archive_ledger, sample_id)
            base = _configuration(opened, archive_ledger, sample_id, (0.0, 0.0, -100.0))
            config = _activity_config(volume, base)
            case = run_trajectory(sample_id=sample_id, volume=volume, config=config, batch_cap=smoke_batch_cap or 40,
                                  beta_compliance=beta_compliance, beta_stress=beta_stress)
            write_case_record(root, case)
            generated_ids.append(sample_id)
    members, fit_accesses, nonfit_accesses = archive_ledger.evidence()
    expected_members = _expected_payload_members(generated_ids)
    if members != expected_members or fit_accesses != len(expected_members) or nonfit_accesses != 0:
        raise CampaignError("activity generation accessed an unexpected payload set")
    invocation = _write_generation_invocation_receipt(root=root, mode=mode, members=members, fit_accesses=fit_accesses,
                                                      nonfit_accesses=nonfit_accesses)
    report = _generation_report(root=root, mode=mode, label=protocol["label"], ids=ids)
    return {"generation": _verify_generation_report(root=root, mode=mode, report=report), "invocation": invocation}


def select_thresholds_from_root(root: Path) -> dict[str, object]:
    manifest = _campaign_manifest(root)
    if manifest.get("mode") == "activity_audit":
        raise CampaignError("audit campaign cannot select thresholds")
    ids = manifest.get("threshold_design_ids")
    if not isinstance(ids, list) or len(ids) != 50:
        raise CampaignError("threshold selection requires all 50 completed threshold designs")
    trajectories = [load_verified_case(root, sample_id) for sample_id in ids if isinstance(sample_id, str)]
    if len(trajectories) != 50:
        raise CampaignError("threshold design trajectories are incomplete")
    selected = select_thresholds(trajectories)
    hashes = {case["sample_id"]: case["case_digest"] for case in trajectories}
    protocol = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    frozen = {"schema_version": SCHEMA_VERSION, "protocol": protocol, "threshold_trajectory_hashes": hashes, **selected}
    frozen["protocol_hash"] = _digest(frozen)
    write_new_regular_path(root / "threshold-selection.json", _canonical_bytes(frozen) + b"\n", "threshold selection")
    return frozen


def _frozen_threshold_selection(path: Path) -> dict[str, object]:
    try:
        snapshot = read_regular_path_snapshot(path, "threshold selection")
        value = json.loads(snapshot.bytes)
    except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError("frozen threshold selection is unavailable") from error
    if not isinstance(value, dict) or value.get("protocol_hash") != _digest({key: item for key, item in value.items() if key != "protocol_hash"}):
        raise CampaignError("frozen threshold selection digest mismatch")
    selected = value.get("selected")
    if not isinstance(selected, dict):
        raise CampaignError("frozen threshold selection is malformed")
    try:
        beta_c, beta_s = float(selected["beta_compliance"]), float(selected["beta_stress"])
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError("frozen threshold selection lacks betas") from error
    if (beta_c, beta_s) not in {(left, right) for left in BETA_GRID for right in BETA_GRID}:
        raise CampaignError("frozen threshold selection uses an unregistered beta")
    return value


def summarize(root: Path) -> dict[str, object]:
    manifest = _campaign_manifest(root)
    ids = manifest.get("selected_ids", [])
    cases = [load_verified_case(root, sample_id) for sample_id in ids if isinstance(sample_id, str)]
    return {"label": manifest.get("label"), "case_count": len(cases), "cases": [{"sample_id": case["sample_id"], "stopping_reason": case["stopping_reason"], "case_digest": case["case_digest"]} for case in cases]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen resumable fit-only constraint-activity campaign")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("generate-trajectories", "select-thresholds", "run-audit", "summarize"), required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-split-manifest-sha256")
    parser.add_argument("--expected-fea-archive-sha256")
    parser.add_argument("--smoke-batch-cap", type=int)
    parser.add_argument("--threshold-selection", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "generate-trajectories":
            if not all((args.split_manifest, args.archive, args.expected_split_manifest_sha256, args.expected_fea_archive_sha256)):
                raise CampaignError("generate-trajectories requires exact external split and archive anchors")
            result = generate_trajectories(root=args.root, split_manifest=args.split_manifest, archive=args.archive,
                expected_split_sha256=args.expected_split_manifest_sha256, expected_archive_sha256=args.expected_fea_archive_sha256,
                mode="smoke" if args.smoke_batch_cap is not None else "threshold_design", smoke_batch_cap=args.smoke_batch_cap)
        elif args.mode == "select-thresholds":
            result = select_thresholds_from_root(args.root)
        elif args.mode == "run-audit":
            if not all((args.split_manifest, args.archive, args.expected_split_manifest_sha256, args.expected_fea_archive_sha256, args.threshold_selection)):
                raise CampaignError("run-audit requires frozen selection and exact external split and archive anchors")
            frozen = _frozen_threshold_selection(args.threshold_selection)
            audit_inputs = _audit_compatibility_inputs(split_manifest=args.split_manifest,
                expected_split_sha256=args.expected_split_manifest_sha256, expected_archive_sha256=args.expected_fea_archive_sha256)
            # This is deliberately before generate_trajectories opens archive bytes or payloads.
            validate_threshold_audit_compatibility(frozen, audit_inputs)
            selected = frozen["selected"]
            generation_run = generate_trajectories(root=args.root, split_manifest=args.split_manifest, archive=args.archive,
                expected_split_sha256=args.expected_split_manifest_sha256, expected_archive_sha256=args.expected_fea_archive_sha256,
                mode="activity_audit", beta_compliance=float(selected["beta_compliance"]), beta_stress=float(selected["beta_stress"]),
                threshold_protocol_hash=str(frozen["protocol_hash"]))
            generation = generation_run.get("generation")
            invocation = generation_run.get("invocation")
            if not isinstance(generation, Mapping) or not isinstance(invocation, Mapping):
                raise CampaignError("activity generation result is malformed")
            audit_manifest = _campaign_manifest(args.root)
            audit_ids = audit_manifest["activity_audit_ids"]
            cases = [load_verified_case(args.root, sample_id) for sample_id in audit_ids]
            summary = _audit_summary(generation=generation, audit=audit_gate(cases),
                                     threshold_protocol_hash=str(frozen["protocol_hash"]))
            result = {**_verify_audit_summary(root=args.root, summary=summary), "invocation": invocation}
        else:
            result = summarize(args.root)
    except (CampaignError, FitOnlyAccessError, OSError, zipfile.BadZipFile) as error:
        print("REJECTED: {}".format(error)); return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
