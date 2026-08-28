"""K6 pre-coverage preparation: frozen trajectories and calibration policy.

This module intentionally has no coverage calculation.  It produces only the
append-only trajectory records, baseline/trajectory conformal quantiles, and a
policy hash that must exist before K6 evidence may be evaluated.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import math
import os
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .activity_campaign import _activity_config, run_trajectory
from .fit_probe import _PayloadAccessLedger, _configuration, _load_occupancy
from .g1b_relabel import load_verified_case
from .manifest import ManifestVerificationError, open_new_artifact_root, read_regular_path_snapshot, sha256_file, write_new_regular_path
from .splits import validate_family_split_manifest
from .surrogate import DenseSurrogateCNN, RoleDataset, SurrogateError, TARGET_NAMES, open_role_dataset
from .voxel_fea import solve_voxels

SCHEMA_VERSION = "1.0.0"
CAMPAIGN_SEED = 20260828
FAMILY_SEED_NAMESPACE = "sasto-v-k6-family-seed-v1"
SAMPLING_NAMESPACE = "sasto-v-k6-sampling-v1"
POLICY_NAMESPACE = "sasto-v-k6-policy-v1"
ALPHA = 0.05
J = 3
ALPHA_J = ALPHA / J
KAPPA_GRID = tuple(index / 4.0 for index in range(17))
KAPPA_TARGET_COVERAGE = 1.0 - ALPHA_J
DEPTH_BINS = ("(0,5%]", "(5,10%]", "(10,15%]", "(15,20%]", "(20,25%]", ">25%")
SOURCE_BUNDLE_PATHS = (
    ".python-version", "pyproject.toml", "uv.lock", "src/sasto/g3_trajectory_calibration.py",
    "src/sasto/activity_campaign.py", "src/sasto/fit_probe.py", "src/sasto/g1b_relabel.py",
    "src/sasto/manifest.py", "src/sasto/splits.py", "src/sasto/surrogate.py", "src/sasto/topology.py",
    "src/sasto/voxel_fea.py",
)


class G3Error(ValueError):
    """A frozen K6 preparation invariant was violated."""


class G3RoleError(G3Error):
    """A caller requested a sealed or otherwise unavailable role."""


@dataclass(frozen=True)
class G3Role:
    role: str
    dataset: RoleDataset
    family_by_sample: Mapping[str, str]
    provenance: Mapping[str, str]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_text(*parts: object) -> str:
    return hashlib.sha256("\0".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _lower_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise G3Error("{} must be an exact lowercase SHA-256 digest".format(label))
    return value


def family_seed(family_id: str) -> int:
    """Return the frozen non-Python-hash seed for one family trajectory."""
    if not isinstance(family_id, str) or not family_id:
        raise G3Error("family ID must be nonempty")
    return int.from_bytes(
        hashlib.sha256((FAMILY_SEED_NAMESPACE + "\0" + str(CAMPAIGN_SEED) + "\0" + family_id).encode("utf-8")).digest()[:8],
        "big",
    ) % (2 ** 31 - 1)


def depth_bin_index(fraction_removed: float) -> int | None:
    """Assign a state to the pre-registered open/closed material-depth bin."""
    if not isinstance(fraction_removed, (int, float)) or isinstance(fraction_removed, bool) or not math.isfinite(float(fraction_removed)):
        raise G3Error("fraction removed must be finite")
    value = float(fraction_removed)
    if value <= 0.0:
        return None
    for index, upper in enumerate((0.05, 0.10, 0.15, 0.20, 0.25)):
        if value <= upper:
            return index
    return 5


def select_state_index(family_id: str, bin_index: int, state_indices: Sequence[int]) -> int:
    """Frozen S: choose by identifier-only SHA-256, never an outcome or prediction."""
    if not isinstance(family_id, str) or not family_id or not isinstance(bin_index, int) or isinstance(bin_index, bool) or not 0 <= bin_index < len(DEPTH_BINS):
        raise G3Error("family ID or depth bin is invalid")
    indices = list(state_indices)
    if not indices or any(not isinstance(index, int) or isinstance(index, bool) or index < 1 for index in indices) or len(set(indices)) != len(indices):
        raise G3Error("state indices must be unique positive integers")
    return min(indices, key=lambda state_index: _sha_text(SAMPLING_NAMESPACE, family_id, bin_index, state_index))


def split_conformal_quantile(scores: Iterable[float], *, alpha: float) -> float:
    """Conservative split-conformal order statistic; +inf for insufficient n."""
    values = sorted(float(value) for value in scores)
    if not values or not 0.0 < alpha < 1.0 or not all(math.isfinite(value) for value in values):
        raise G3Error("split conformal scores and alpha are invalid")
    order = math.ceil((len(values) + 1) * (1.0 - alpha))
    return float("inf") if order > len(values) else values[order - 1]


def _snapshot_json(path: Path, expected_sha: str, label: str) -> tuple[dict[str, object], str]:
    _lower_digest(expected_sha, label + " anchor")
    try:
        snapshot = read_regular_path_snapshot(path, label)
        value = json.loads(snapshot.bytes.decode("utf-8"))
    except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise G3Error("{} is unavailable or malformed".format(label)) from error
    if snapshot.sha256 != expected_sha or not isinstance(value, dict):
        raise G3Error("{} anchor or schema is invalid".format(label))
    return value, snapshot.sha256


def _family_map(split: Mapping[str, object], role: str, sample_ids: Sequence[str]) -> dict[str, str]:
    try:
        validate_family_split_manifest(split)
        ids = split["partitions"][role]["sample_ids"]  # type: ignore[index]
        families = split["partitions"][role]["family_ids"]  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as error:
        raise G3Error("frozen split is invalid") from error
    if not isinstance(ids, list) or not isinstance(families, list) or len(ids) != len(families):
        raise G3Error("split family membership is invalid")
    mapping = dict(zip(ids, families))
    if set(mapping) != set(sample_ids) or len(set(mapping.values())) != len(mapping):
        raise G3Error("eligible role must contain exactly one sample per family")
    if any(not isinstance(item, str) or not item for item in mapping.values()):
        raise G3Error("family ID is invalid")
    return {sample_id: str(mapping[sample_id]) for sample_id in sample_ids}


def open_g3_role(
    *, role: str, split_manifest: Path, expected_split_sha256: str, archive: Path, expected_archive_sha256: str,
    g1b_root: Path, expected_cohort_manifest_sha256: str, expected_cluster_role_manifest_sha256: str,
) -> G3Role:
    """Open an anchored development/calibration handle; confirmation is sealed first."""
    if role == "confirmation":
        raise G3RoleError("confirmation is sealed and cannot be opened by G3")
    if role not in {"development", "calibration"}:
        raise G3RoleError("G3 role must be development or calibration")
    # The G2 loader validates source anchors and admission before any archive payload.
    try:
        dataset = open_role_dataset(
            role=role, split_manifest=split_manifest, expected_split_sha256=expected_split_sha256,
            archive=archive, expected_archive_sha256=expected_archive_sha256, g1b_root=g1b_root,
            expected_cohort_manifest_sha256=expected_cohort_manifest_sha256,
            expected_cluster_role_manifest_sha256=expected_cluster_role_manifest_sha256,
            calibration_pass=(role == "calibration"),
        )
    except SurrogateError as error:
        raise G3Error(str(error)) from error
    split, _ = _snapshot_json(split_manifest, expected_split_sha256, "split manifest")
    family_by_sample = _family_map(split, role, dataset.sample_ids)
    return G3Role(role=role, dataset=dataset, family_by_sample=family_by_sample, provenance=dataset.provenance)


class EnsemblePredictor:
    """Frozen G2 ensemble prediction on the normalized natural-log target scale."""

    def __init__(self, *, ensemble_root: Path, normalization_path: Path, device: str = "mps") -> None:
        try:
            import torch
        except ModuleNotFoundError as error:
            raise G3Error("PyTorch is required for G3 prediction") from error
        self._torch = torch
        if device not in {"cpu", "mps"} or (device == "mps" and not torch.backends.mps.is_available()):
            raise G3Error("requested G3 prediction device is unavailable")
        self.device = device
        ensemble_root = Path(ensemble_root)
        campaign_path = ensemble_root / "campaign-manifest.json"
        try:
            campaign = json.loads(read_regular_path_snapshot(campaign_path, "G2 campaign manifest").bytes.decode("utf-8"))
            normalization = json.loads(read_regular_path_snapshot(normalization_path, "G2 normalization statistics").bytes.decode("utf-8"))
        except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise G3Error("G2 ensemble or normalization record is unavailable") from error
        if not isinstance(campaign, dict) or campaign.get("label") != "CERTIFIED_G2_ENSEMBLE" or not isinstance(normalization, dict):
            raise G3Error("G2 ensemble or normalization record is invalid")
        if normalization.get("target_names") != list(TARGET_NAMES) or normalization.get("stats_digest") != campaign.get("normalization_stats_digest"):
            raise G3Error("G2 normalization binding is invalid")
        self.normalization = normalization
        self.campaign = campaign
        self.models = []
        checkpoints: list[dict[str, object]] = []
        member_count = campaign.get("member_count")
        base_channels = campaign.get("base_channels")
        if not isinstance(member_count, int) or not isinstance(base_channels, int) or member_count < 1:
            raise G3Error("G2 campaign model configuration is invalid")
        for index in range(member_count):
            manifest_path = ensemble_root / "members" / "member-{:02d}.json".format(index)
            try:
                member = json.loads(read_regular_path_snapshot(manifest_path, "G2 member manifest").bytes.decode("utf-8"))
            except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise G3Error("G2 member manifest is unavailable") from error
            if not isinstance(member, dict) or member.get("campaign_digest") != campaign.get("campaign_digest"):
                raise G3Error("G2 member does not bind the supplied campaign")
            checkpoint = member.get("checkpoint")
            if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("path"), str) or not isinstance(checkpoint.get("sha256"), str):
                raise G3Error("G2 member checkpoint record is invalid")
            checkpoint_path = manifest_path.parent / checkpoint["path"]
            if sha256_file(checkpoint_path) != checkpoint["sha256"]:
                raise G3Error("G2 checkpoint digest mismatch")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping) or payload.get("target_names") != TARGET_NAMES or payload.get("base_channels") != base_channels:
                raise G3Error("G2 checkpoint payload is invalid")
            model = DenseSurrogateCNN(base_channels=base_channels).to(device)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            self.models.append(model)
            checkpoints.append({"member_index": index, "member_manifest_sha256": sha256_file(manifest_path), "checkpoint_sha256": checkpoint["sha256"]})
        self.checkpoints = checkpoints

    def predict(self, channels: np.ndarray) -> dict[str, object]:
        if not isinstance(channels, np.ndarray) or channels.shape != (2, 64, 64, 64) or not np.all(np.isfinite(channels)):
            raise G3Error("surrogate channels are invalid")
        torch = self._torch
        tensor = torch.from_numpy(channels.astype(np.float32, copy=False)).unsqueeze(0).to(self.device)
        means = []
        variances = []
        with torch.no_grad():
            for model in self.models:
                result = model(tensor)
                means.append(result["mean"].detach().cpu().numpy()[0])
                variances.append(result["dispersion"].square().detach().cpu().numpy()[0])
        member_means = np.stack(means)
        mu = member_means.mean(axis=0)
        # Total predictive variance: aleatoric member variance plus epistemic disagreement.
        total_variance = np.stack(variances).mean(axis=0) + member_means.var(axis=0, ddof=0)
        sigma = np.sqrt(total_variance)
        return {"scale": "normalized_natural_log", "target_names": list(TARGET_NAMES),
                "mu": {name: float(mu[index]) for index, name in enumerate(TARGET_NAMES)},
                "sigma": {name: float(sigma[index]) for index, name in enumerate(TARGET_NAMES)}}


def _normalized_targets(targets: Mapping[str, float], normalization: Mapping[str, object]) -> dict[str, float]:
    means = normalization.get("means")
    scales = normalization.get("scales")
    if not isinstance(means, Mapping) or not isinstance(scales, Mapping):
        raise G3Error("normalization record is malformed")
    result: dict[str, float] = {}
    for name in TARGET_NAMES:
        try:
            value, mean, scale = float(targets[name]), float(means[name]), float(scales[name])
        except (KeyError, TypeError, ValueError) as error:
            raise G3Error("target or normalization value is invalid") from error
        if not value > 0.0 or not math.isfinite(value) or not math.isfinite(mean) or not scale > 0.0 or not math.isfinite(scale):
            raise G3Error("target or normalization value is nonfinite")
        result[name] = (math.log(value) - mean) / scale
    return result


def _baseline_rows(role: G3Role, predictor: EnsemblePredictor) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for example in role.dataset:
        prediction = predictor.predict(example.channels)
        rows.append({"sample_id": example.sample_id, "family_id": role.family_by_sample[example.sample_id],
                     "y": _normalized_targets(example.targets, predictor.normalization), "prediction": prediction})
    if len(rows) != len(role.dataset):
        raise G3Error("baseline role did not materialize exactly once")
    return rows


def select_kappas(development_baselines: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Fix each kappa by the declared development-only baseline grid procedure."""
    if not development_baselines:
        raise G3Error("development baseline evidence is empty")
    evidence: dict[str, object] = {}
    values: dict[str, float] = {}
    for target in TARGET_NAMES:
        rows = []
        for row in development_baselines:
            try:
                y = float(row["y"][target])  # type: ignore[index]
                mu = float(row["prediction"]["mu"][target])  # type: ignore[index]
                sigma = float(row["prediction"]["sigma"][target])  # type: ignore[index]
            except (KeyError, TypeError, ValueError) as error:
                raise G3Error("development prediction evidence is malformed") from error
            if not all(math.isfinite(value) for value in (y, mu, sigma)) or sigma <= 0.0:
                raise G3Error("development prediction evidence is nonfinite")
            rows.append((y, mu, sigma))
        grid = []
        for kappa in KAPPA_GRID:
            covered = sum(y <= mu + kappa * sigma for y, mu, sigma in rows)
            grid.append({"kappa": kappa, "covered_count": covered, "sample_count": len(rows), "coverage": covered / len(rows)})
        selected = next((row for row in grid if row["coverage"] >= KAPPA_TARGET_COVERAGE), None)
        if selected is None:
            raise G3Error("development evidence does not reach the frozen kappa coverage target")
        values[target] = float(selected["kappa"])
        evidence[target] = {"selection": "smallest_kappa_on_fixed_grid_reaching_development_baseline_marginal_coverage",
                            "target_coverage": KAPPA_TARGET_COVERAGE, "grid": grid, "selected": selected}
    return {"schema_version": SCHEMA_VERSION, "role": "development", "alpha": ALPHA, "J": J, "alpha_j": ALPHA_J,
            "prediction_scale": "normalized_natural_log", "kappa_grid": list(KAPPA_GRID), "kappa": values,
            "development_evidence": evidence}


def _case_path(root: Path, role: str, sample_id: str) -> Path:
    if role not in {"development", "calibration"} or not isinstance(sample_id, str) or not sample_id or any(char in sample_id for char in "/\\\x00"):
        raise G3Error("trajectory case identity is unsafe")
    # Flat, role-prefixed leaf names preserve append-only writes without creating
    # unverified nested directory state during resumptions.
    return root / "trajectory-{}-{}.json".format(role, sample_id)


def _verified_json(path: Path, label: str, digest_field: str) -> dict[str, object]:
    try:
        snapshot = read_regular_path_snapshot(path, label)
        value = json.loads(snapshot.bytes.decode("utf-8"))
    except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise G3Error("{} is unavailable or malformed".format(label)) from error
    if not isinstance(value, dict) or value.get(digest_field) != _digest({key: item for key, item in value.items() if key != digest_field}):
        raise G3Error("{} digest mismatch".format(label))
    return value


def _write_new_or_match(path: Path, payload: Mapping[str, object], label: str, digest_field: str) -> str:
    value = dict(payload)
    value.pop(digest_field, None)
    value[digest_field] = _digest(value)
    encoded = _canonical_bytes(value) + b"\n"
    if path.exists():
        existing = _verified_json(path, label, digest_field)
        if existing != value:
            raise G3Error("existing {} does not match deterministic recomputation".format(label))
        return sha256_file(path)
    try:
        return write_new_regular_path(path, encoded, label)
    except ManifestVerificationError as error:
        raise G3Error(str(error)) from error


def _ensure_output_root(root: Path) -> None:
    """Create a new descriptor-anchored root or admit only a regular resume root."""
    root = Path(root)
    if root.exists():
        if not root.is_dir() or root.is_symlink():
            raise G3Error("G3 output root must be a non-symlink directory")
        return
    try:
        with open_new_artifact_root(root):
            pass
    except ManifestVerificationError as error:
        raise G3Error("cannot safely create G3 output root") from error


def _channels(current: np.ndarray, parts: np.ndarray) -> np.ndarray:
    if current.dtype != np.bool_ or parts.shape != current.shape or current.shape != (64, 64, 64):
        raise G3Error("trajectory occupancy or parts shape is invalid")
    return np.stack((current.astype(np.float32), (parts * current).astype(np.float32)), axis=0)


def _trajectory_case(*, role: G3Role, sample_id: str, archive_open: zipfile.ZipFile, ledger: _PayloadAccessLedger,
                     predictor: EnsemblePredictor) -> dict[str, object]:
    family_id = role.family_by_sample[sample_id]
    occupancy = _load_occupancy(archive_open, ledger, sample_id)
    base = _configuration(archive_open, ledger, sample_id, (0.0, 0.0, -100.0))
    # Parts are admitted only for the role-scoped sample; no confirmation route exists.
    _packed, parts = role.dataset._payload(sample_id)  # pylint: disable=protected-access
    if occupancy.shape != (64, 64, 64) or parts.shape != occupancy.shape:
        raise G3Error("trajectory source payload has invalid shape")
    config = _activity_config(occupancy, base)

    def canonical_with_prediction(current: np.ndarray, solver_config: object) -> dict[str, object]:
        record = solve_voxels(current, solver_config)
        record["surrogate_prediction"] = predictor.predict(_channels(current, parts))
        return record

    trajectory = run_trajectory(sample_id=sample_id, volume=occupancy, config=config, solver=canonical_with_prediction,
                                batch_cap=40, ranking_seed=family_seed(family_id))
    if trajectory.get("topology", {}).get("topology_mode") != "conservative_local_6_26" or trajectory.get("topology", {}).get("sequential_recheck") is not True:
        raise G3Error("trajectory invariant layer is not certified conservative sequential topology")
    baseline = trajectory.get("baseline")
    batches = trajectory.get("batches")
    if not isinstance(baseline, Mapping) or not isinstance(batches, list) or "surrogate_prediction" not in baseline:
        raise G3Error("trajectory lacks baseline prediction evidence")
    for batch in batches:
        if not isinstance(batch, Mapping) or not isinstance(batch.get("candidate"), Mapping) or "surrogate_prediction" not in batch["candidate"]:
            raise G3Error("trajectory lacks batch prediction evidence")
    result: dict[str, object] = {"schema_version": SCHEMA_VERSION, "sample_id": sample_id, "family_id": family_id,
        "role": role.role, "family_seed_namespace": FAMILY_SEED_NAMESPACE, "campaign_seed": CAMPAIGN_SEED,
        "family_seed": family_seed(family_id), "trajectory": trajectory}
    result["trajectory_digest"] = _digest(result)
    return result


def _load_or_generate_trajectories(*, root: Path, role: G3Role, predictor: EnsemblePredictor) -> tuple[list[dict[str, object]], dict[str, object]]:
    cases: list[dict[str, object]] = []
    generated = 0
    resumed = 0
    expected_members = ["fea_ml/data/runs_real/{}/{}".format(sample_id, leaf) for sample_id in role.dataset.sample_ids for leaf in ("occ.npz", "meta.json", "part.npz")]
    ledger = _PayloadAccessLedger(list(role.dataset.sample_ids))
    with zipfile.ZipFile(role.dataset._archive, "r") as archive_open:  # pylint: disable=protected-access
        for sample_id in role.dataset.sample_ids:
            path = _case_path(root, role.role, sample_id)
            if path.exists():
                case = _verified_json(path, "G3 trajectory case", "trajectory_digest")
                if case.get("sample_id") != sample_id or case.get("family_id") != role.family_by_sample[sample_id] or case.get("role") != role.role:
                    raise G3Error("resumed trajectory identity is invalid")
                cases.append(case); resumed += 1
                continue
            case = _trajectory_case(role=role, sample_id=sample_id, archive_open=archive_open, ledger=ledger, predictor=predictor)
            _write_new_or_match(path, case, "G3 trajectory case", "trajectory_digest")
            cases.append(case); generated += 1
    members, accesses, nonfit = ledger.evidence()
    if nonfit or accesses != len(members):
        raise G3Error("trajectory payload ledger is inconsistent")
    if len(cases) != len(role.dataset):
        raise G3Error("trajectory cases are incomplete")
    return cases, {"role": role.role, "case_count": len(cases), "generated_count": generated, "resumed_count": resumed,
                   "payload_access_count": accesses, "payload_members": members, "expected_payload_members": expected_members,
                   "trajectory_case_digests": [{"sample_id": case["sample_id"], "trajectory_digest": case["trajectory_digest"]} for case in cases]}


def _selected_trajectory_rows(cases: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], dict[str, int], dict[str, int]]:
    rows: list[dict[str, object]] = []
    occupancy = Counter({label: 0 for label in DEPTH_BINS})
    selected = Counter({label: 0 for label in DEPTH_BINS})
    for case in cases:
        try:
            family_id = str(case["family_id"])
            trajectory = case["trajectory"]
            batches = trajectory["batches"]  # type: ignore[index]
        except (KeyError, TypeError) as error:
            raise G3Error("trajectory case is malformed") from error
        per_bin: dict[int, list[tuple[int, Mapping[str, object]]]] = {index: [] for index in range(len(DEPTH_BINS))}
        if not isinstance(batches, list):
            raise G3Error("trajectory batches are malformed")
        for batch in batches:
            if not isinstance(batch, Mapping):
                raise G3Error("trajectory batch is malformed")
            candidate = batch.get("candidate")
            if not isinstance(candidate, Mapping) or candidate.get("status") != "success":
                continue
            state_index = batch.get("batch_index")
            fraction = batch.get("proposed_material_reduction")
            if not isinstance(state_index, int):
                raise G3Error("trajectory batch index is invalid")
            bin_index = depth_bin_index(float(fraction))
            if bin_index is None:
                continue
            occupancy[DEPTH_BINS[bin_index]] += 1
            per_bin[bin_index].append((state_index, batch))
        for bin_index, candidates in per_bin.items():
            if not candidates:
                continue
            chosen_index = select_state_index(family_id, bin_index, [state_index for state_index, _ in candidates])
            chosen = next(batch for state_index, batch in candidates if state_index == chosen_index)
            prediction = chosen.get("candidate", {}).get("surrogate_prediction") if isinstance(chosen.get("candidate"), Mapping) else None
            if not isinstance(prediction, Mapping):
                raise G3Error("selected trajectory state lacks prediction")
            rows.append({"sample_id": case["sample_id"], "family_id": family_id, "bin_index": bin_index,
                         "bin_label": DEPTH_BINS[bin_index], "state_index": chosen_index,
                         "fraction_removed": chosen["proposed_material_reduction"], "prediction": prediction,
                         "solver": chosen["candidate"]})
            selected[DEPTH_BINS[bin_index]] += 1
    return rows, dict(occupancy), dict(selected)


def _trajectory_targets(row: Mapping[str, object], normalization: Mapping[str, object]) -> dict[str, float]:
    solver = row.get("solver")
    if not isinstance(solver, Mapping) or solver.get("status") != "success":
        raise G3Error("selected trajectory solver response is unavailable")
    raw = {"compliance": solver.get("compliance_j"), "max_von_mises": solver.get("max_gauss_von_mises_pa", solver.get("max_von_mises_pa")),
           "max_displacement": solver.get("max_displacement_m")}
    try:
        targets = {name: float(value) for name, value in raw.items()}
    except (TypeError, ValueError) as error:
        raise G3Error("selected trajectory target schema is invalid") from error
    return _normalized_targets(targets, normalization)


def _quantiles(rows: Sequence[Mapping[str, object]], *, kappas: Mapping[str, float], normalization: Mapping[str, object], kind: str) -> dict[str, object]:
    scores: dict[str, list[float]] = {name: [] for name in TARGET_NAMES}
    for row in rows:
        y = row["y"] if kind == "baseline" else _trajectory_targets(row, normalization)
        prediction = row.get("prediction")
        if not isinstance(y, Mapping) or not isinstance(prediction, Mapping):
            raise G3Error("calibration row is malformed")
        for name in TARGET_NAMES:
            try:
                score = float(y[name]) - (float(prediction["mu"][name]) + float(kappas[name]) * float(prediction["sigma"][name]))
            except (KeyError, TypeError, ValueError) as error:
                raise G3Error("calibration score is malformed") from error
            if not math.isfinite(score):
                raise G3Error("calibration score is nonfinite")
            scores[name].append(score)
    if not rows:
        raise G3Error("calibration rows are empty")
    return {"schema_version": SCHEMA_VERSION, "kind": kind, "alpha": ALPHA, "J": J, "alpha_j": ALPHA_J,
            "prediction_scale": "normalized_natural_log", "sample_count": len(rows),
            "q": {name: split_conformal_quantile(scores[name], alpha=ALPHA_J) for name in TARGET_NAMES},
            "score_counts": {name: len(scores[name]) for name in TARGET_NAMES}}


def source_bundle(root: Path | None = None) -> tuple[dict[str, str], str]:
    source_root = Path(root) if root is not None else Path(__file__).parents[2]
    try:
        files = {relative: sha256_file(source_root / relative) for relative in SOURCE_BUNDLE_PATHS}
    except ManifestVerificationError as error:
        raise G3Error("cannot hash G3 source bundle") from error
    return files, _digest([{"path": path, "sha256": files[path]} for path in sorted(files)])


def _campaign_manifest(*, preregistration_sha256: str, role: G3Role, predictor: EnsemblePredictor) -> dict[str, object]:
    files, source_sha = source_bundle()
    value: dict[str, object] = {"schema_version": SCHEMA_VERSION, "label": "K6_PRE_COVERAGE_ONLY", "coverage_computed": False,
        "hard_stop": "no_k6_coverage_or_adjudication", "pre_registration_sha256": _lower_digest(preregistration_sha256, "pre-registration"),
        "source_bundle_files": files, "source_bundle_sha256": source_sha, "campaign_seed": CAMPAIGN_SEED,
        "family_seed_namespace": FAMILY_SEED_NAMESPACE, "sampling_namespace": SAMPLING_NAMESPACE, "alpha": ALPHA, "J": J,
        "depth_bins": list(DEPTH_BINS), "trajectory_max_batches": 40, "topology_mode": "conservative_local_6_26",
        "solver": {"fixed_total_force_n": [0.0, 0.0, -100.0], "include_self_weight": False, "relative_tolerance": 2e-8},
        "ensemble_campaign_digest": predictor.campaign["campaign_digest"], "ensemble_checkpoints": predictor.checkpoints,
        "normalization_stats_digest": predictor.normalization["stats_digest"], **role.provenance}
    value["campaign_digest"] = _digest(value)
    return value


def _policy_record(*, preregistration_sha256: str, campaign: Mapping[str, object], predictor: EnsemblePredictor,
                   kappa: Mapping[str, object], q_base: Mapping[str, object], q_traj: Mapping[str, object],
                   trajectory_ledgers: Mapping[str, object]) -> dict[str, object]:
    policy: dict[str, object] = {"schema_version": SCHEMA_VERSION, "label": "K6_FROZEN_PRE_COVERAGE_POLICY", "coverage_computed": False,
        "hard_stop": "no_k6_coverage_or_adjudication", "policy_namespace": POLICY_NAMESPACE,
        "pre_registration_sha256": _lower_digest(preregistration_sha256, "pre-registration"), "campaign_digest": campaign["campaign_digest"],
        "source_bundle_sha256": campaign["source_bundle_sha256"], "source_bundle_files": campaign["source_bundle_files"],
        "split_manifest_sha256": campaign["split_manifest_sha256"], "archive_sha256": campaign["archive_sha256"],
        "cohort_manifest_sha256": campaign["cohort_manifest_sha256"], "cluster_role_manifest_sha256": campaign["cluster_role_manifest_sha256"],
        "ensemble_campaign_digest": predictor.campaign["campaign_digest"], "ensemble_member_checkpoints": predictor.checkpoints,
        "normalization_stats_digest": predictor.normalization["stats_digest"], "kappa": kappa,
        "q_base": q_base["q"], "q_traj": q_traj["q"], "baseline_calibration_sample_count": q_base["sample_count"],
        "trajectory_calibration_sample_count": q_traj["sample_count"], "sampling_rule": {"namespace": SAMPLING_NAMESPACE,
            "identifier_components": ["namespace", "family_id", "bin_index", "state_index"], "response_independent": True,
            "depth_bins": list(DEPTH_BINS), "one_state_per_occupied_family_bin": True},
        "alpha": ALPHA, "J": J, "alpha_j": ALPHA_J, "trajectory_ledgers": trajectory_ledgers}
    policy["policy_sha256"] = _digest(policy)
    return policy


def run_precoverage(*, output_root: Path, split_manifest: Path, expected_split_sha256: str, archive: Path,
                    expected_archive_sha256: str, g1b_root: Path, expected_cohort_manifest_sha256: str,
                    expected_cluster_role_manifest_sha256: str, ensemble_root: Path, normalization_path: Path,
                    preregistration_path: Path, expected_preregistration_sha256: str, device: str = "mps") -> dict[str, object]:
    """Generate all allowed trajectories, calibrate, freeze a policy, then stop."""
    output_root = Path(output_root)
    _ensure_output_root(output_root)
    try:
        observed_preregistration = sha256_file(preregistration_path)
    except ManifestVerificationError as error:
        raise G3Error("frozen pre-registration is unavailable") from error
    if observed_preregistration != _lower_digest(expected_preregistration_sha256, "pre-registration"):
        raise G3Error("frozen pre-registration sha256 mismatch")
    predictor = EnsemblePredictor(ensemble_root=ensemble_root, normalization_path=normalization_path, device=device)
    # Development is opened first. Kappa is completely fixed before calibration is opened.
    development = open_g3_role(role="development", split_manifest=split_manifest, expected_split_sha256=expected_split_sha256,
                               archive=archive, expected_archive_sha256=expected_archive_sha256, g1b_root=g1b_root,
                               expected_cohort_manifest_sha256=expected_cohort_manifest_sha256,
                               expected_cluster_role_manifest_sha256=expected_cluster_role_manifest_sha256)
    campaign = _campaign_manifest(preregistration_sha256=observed_preregistration, role=development, predictor=predictor)
    campaign_path = Path(output_root) / "campaign-manifest.json"
    _write_new_or_match(campaign_path, campaign, "G3 campaign manifest", "campaign_digest")
    development_baselines = _baseline_rows(development, predictor)
    kappa = select_kappas(development_baselines)
    _write_new_or_match(Path(output_root) / "kappa-development-evidence.json", kappa, "G3 kappa evidence", "kappa_evidence_sha256")
    # This is the first calibration-role payload open, after kappa is persisted.
    calibration = open_g3_role(role="calibration", split_manifest=split_manifest, expected_split_sha256=expected_split_sha256,
                               archive=archive, expected_archive_sha256=expected_archive_sha256, g1b_root=g1b_root,
                               expected_cohort_manifest_sha256=expected_cohort_manifest_sha256,
                               expected_cluster_role_manifest_sha256=expected_cluster_role_manifest_sha256)
    calibration_baselines = _baseline_rows(calibration, predictor)
    q_base = _quantiles(calibration_baselines, kappas=kappa["kappa"], normalization=predictor.normalization, kind="baseline")
    _write_new_or_match(Path(output_root) / "baseline-calibration.json", q_base, "G3 baseline calibration", "baseline_calibration_sha256")
    started = time.perf_counter()
    development_cases, development_ledger = _load_or_generate_trajectories(root=Path(output_root), role=development, predictor=predictor)
    calibration_cases, calibration_ledger = _load_or_generate_trajectories(root=Path(output_root), role=calibration, predictor=predictor)
    calibration_rows, calibration_bin_occupancy, calibration_bin_selected = _selected_trajectory_rows(calibration_cases)
    q_traj = _quantiles(calibration_rows, kappas=kappa["kappa"], normalization=predictor.normalization, kind="trajectory")
    q_traj["depth_bin_occupancy_counts"] = calibration_bin_occupancy
    q_traj["selected_state_counts_per_bin"] = calibration_bin_selected
    _write_new_or_match(Path(output_root) / "trajectory-calibration.json", q_traj, "G3 trajectory calibration", "trajectory_calibration_sha256")
    development_rows, development_bin_occupancy, development_bin_selected = _selected_trajectory_rows(development_cases)
    trajectory_ledgers = {"development": development_ledger, "calibration": calibration_ledger,
                          "development_depth_bin_occupancy_counts": development_bin_occupancy,
                          "development_selected_state_counts_per_bin": development_bin_selected,
                          "calibration_depth_bin_occupancy_counts": calibration_bin_occupancy,
                          "calibration_selected_state_counts_per_bin": calibration_bin_selected,
                          "development_selected_state_count": len(development_rows), "calibration_selected_state_count": len(calibration_rows),
                          "wall_seconds": time.perf_counter() - started}
    summary = {"schema_version": SCHEMA_VERSION, "label": "K6_PRE_COVERAGE_ONLY", "coverage_computed": False,
               "trajectory_ledgers": trajectory_ledgers, "q_base": q_base["q"], "q_traj": q_traj["q"], "kappa": kappa["kappa"]}
    _write_new_or_match(Path(output_root) / "precoverage-summary.json", summary, "G3 precoverage summary", "summary_sha256")
    policy = _policy_record(preregistration_sha256=observed_preregistration, campaign=campaign, predictor=predictor,
                            kappa=kappa, q_base=q_base, q_traj=q_traj, trajectory_ledgers=trajectory_ledgers)
    _write_new_or_match(Path(output_root) / "policy.json", policy, "G3 frozen policy", "policy_sha256")
    return {"output_root": str(output_root), "trajectory_count": len(development_cases), "calibration_trajectory_count": len(calibration_cases),
            "development_selected_state_count": len(development_rows), "calibration_selected_state_count": len(calibration_rows),
            "kappa": kappa["kappa"], "q_base": q_base["q"], "q_traj": q_traj["q"], "policy_sha256": policy["policy_sha256"],
            "coverage_computed": False, "trajectory_ledgers": trajectory_ledgers}


def main() -> int:
    parser = argparse.ArgumentParser(description="K6 frozen trajectory generation and pre-coverage calibration only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-manifest-sha256", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--g1b-root", type=Path, required=True)
    parser.add_argument("--expected-cohort-manifest-sha256", required=True)
    parser.add_argument("--expected-cluster-role-manifest-sha256", required=True)
    parser.add_argument("--ensemble-root", type=Path, required=True)
    parser.add_argument("--normalization-path", type=Path, required=True)
    parser.add_argument("--preregistration-path", type=Path, required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    try:
        result = run_precoverage(output_root=args.output, split_manifest=args.split_manifest,
            expected_split_sha256=args.expected_split_manifest_sha256, archive=args.archive,
            expected_archive_sha256=args.expected_archive_sha256, g1b_root=args.g1b_root,
            expected_cohort_manifest_sha256=args.expected_cohort_manifest_sha256,
            expected_cluster_role_manifest_sha256=args.expected_cluster_role_manifest_sha256,
            ensemble_root=args.ensemble_root, normalization_path=args.normalization_path,
            preregistration_path=args.preregistration_path, expected_preregistration_sha256=args.expected_preregistration_sha256,
            device=args.device)
    except (G3Error, OSError) as error:
        print("REJECTED: {}".format(error), file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
