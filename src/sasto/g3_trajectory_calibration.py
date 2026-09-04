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

from .activity_campaign import _activity_config, geometric_trajectory
from .fit_probe import _PayloadAccessLedger, _configuration, _load_occupancy
from .g1b_relabel import load_verified_case, shard_for_id
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
BASELINE_ROWS_FILENAME = "baseline-rows.json"
CHANNEL_CACHE_DIRECTORY = "decoded-channel-cache-v1"
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
    requested = set(sample_ids)
    if len(requested) != len(sample_ids) or not requested <= set(mapping) or len({mapping[sample_id] for sample_id in requested}) != len(requested):
        raise G3Error("eligible role must contain exactly one sample per family")
    if any(not isinstance(item, str) or not item for item in (mapping[sample_id] for sample_id in requested)):
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


_CHANNEL_BYTES = 4 * (64 ** 3 // 8)


class G3ChannelCache:
    """Read-only decoded channels, with development able to reference G2 directly."""

    def __init__(self, *, root: Path, manifest: Mapping[str, object], provenance: Mapping[str, str]) -> None:
        self.root = Path(root)
        self.manifest = dict(manifest)
        self.provenance = dict(provenance)
        roles = manifest.get("roles")
        if not isinstance(roles, Mapping):
            raise G3Error("decoded-channel cache roles are malformed")
        self.roles = roles
        self._maps: dict[str, np.memmap] = {}

    def _data_path(self, role: str, row: Mapping[str, object]) -> Path:
        source = row.get("source")
        filename = row.get("data_file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise G3Error("decoded-channel cache data file is unsafe")
        if source == "g3":
            return self.root / filename
        if source == "g2-reuse":
            return _g2_cache_root(self.provenance) / filename
        raise G3Error("decoded-channel cache source is invalid")

    def channels(self, *, role: str, sample_id: str) -> np.ndarray:
        row = self.roles.get(role)
        if not isinstance(row, Mapping) or not isinstance(row.get("sample_ids"), list):
            raise G3Error("decoded-channel cache role is malformed")
        sample_ids = row["sample_ids"]
        if sample_id not in sample_ids:
            raise G3Error("decoded-channel cache sample is outside its role")
        path = self._data_path(role, row)
        packed = self._maps.get(role)
        if packed is None:
            try:
                packed = np.memmap(path, dtype=np.uint8, mode="r", shape=(len(sample_ids), _CHANNEL_BYTES))
            except (OSError, ValueError) as error:
                raise G3Error("decoded-channel cache payload is unavailable") from error
            self._maps[role] = packed
        payload = packed[sample_ids.index(sample_id)]
        occupancy = np.unpackbits(payload[:32768], bitorder="little", count=64 ** 3).reshape(64, 64, 64)
        parts = np.zeros(64 ** 3, dtype=np.uint8)
        for bit in range(3):
            parts |= np.unpackbits(payload[(bit + 1) * 32768:(bit + 2) * 32768], bitorder="little", count=64 ** 3).astype(np.uint8) << bit
        return np.stack((occupancy.astype(np.float32), parts.reshape(64, 64, 64).astype(np.float32)), axis=0)


def _g2_cache_root(provenance: Mapping[str, str]) -> Path:
    return Path(__file__).parents[2] / "artifacts" / "g2" / "ingest-cache-v1" / "{}-{}".format(provenance["archive_sha256"][:16], provenance["cohort_manifest_sha256"][:16])


def _g2_development_reference(*, role: G3Role) -> dict[str, object] | None:
    """Return a verified G2 development cache row only when all anchors match."""
    root = _g2_cache_root(role.provenance)
    try:
        snapshot = read_regular_path_snapshot(root / "cache-manifest.json", "G2 decoded-channel cache manifest")
        manifest = json.loads(snapshot.bytes.decode("utf-8"))
    except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, Mapping) or manifest.get("cache_digest") != _digest({key: value for key, value in manifest.items() if key != "cache_digest"}):
        return None
    if any(manifest.get(key) != role.provenance.get(key) for key in ("archive_sha256", "cohort_manifest_sha256", "split_manifest_sha256", "cluster_role_manifest_sha256")):
        return None
    roles = manifest.get("roles")
    row = roles.get("development") if isinstance(roles, Mapping) else None
    if not isinstance(row, Mapping) or row.get("sample_ids") != list(role.dataset.sample_ids):
        return None
    filename = row.get("data_file"); digest = row.get("data_sha256")
    if not isinstance(filename, str) or not isinstance(digest, str):
        return None
    try:
        if sha256_file(root / filename) != digest or (root / filename).stat().st_size != len(role.dataset) * _CHANNEL_BYTES:
            return None
    except (ManifestVerificationError, OSError):
        return None
    return {"source": "g2-reuse", "data_file": filename, "data_sha256": digest, "sample_ids": list(role.dataset.sample_ids),
            "source_cache_manifest_sha256": snapshot.sha256}


def _archive_channel_payload(archive: zipfile.ZipFile, sample_id: str) -> bytes:
    arrays: list[np.ndarray] = []
    for leaves in (("occ.npz",), ("part.npz", "parts.npz")):
        data: np.ndarray | None = None
        for leaf in leaves:
            try:
                with archive.open("fea_ml/data/runs_real/{}/{}".format(sample_id, leaf), "r") as opened:
                    with np.load(io.BytesIO(opened.read()), allow_pickle=False) as loaded:
                        if loaded.files != ["data"]:
                            raise G3Error("decoded-channel cache NPZ schema is invalid")
                        data = loaded["data"]
                break
            except KeyError:
                continue
        if data is None:
            raise G3Error("decoded-channel cache payload is missing")
        arrays.append(data)
    occupancy, parts = arrays
    if occupancy.shape != (64, 64, 64) or occupancy.dtype not in (np.dtype(np.uint8), np.dtype(np.bool_)) or not np.all((occupancy == 0) | (occupancy == 1)):
        raise G3Error("decoded-channel cache occupancy schema is invalid")
    if parts.shape != (64, 64, 64) or parts.dtype not in (np.dtype(np.uint8), np.dtype(np.int8)) or not np.all((parts >= 0) & (parts <= 5)):
        raise G3Error("decoded-channel cache part schema is invalid")
    result = bytearray(np.packbits(occupancy.reshape(-1), bitorder="little").tobytes())
    flat_parts = parts.reshape(-1).astype(np.uint8, copy=False)
    for bit in range(3):
        result.extend(np.packbits((flat_parts >> bit) & 1, bitorder="little").tobytes())
    if len(result) != _CHANNEL_BYTES:
        raise G3Error("decoded-channel cache packed size is invalid")
    return bytes(result)


def _open_or_build_channel_cache(*, output_root: Path, development: G3Role, calibration: G3Role) -> G3ChannelCache:
    """Build calibration once; reference certified G2 development bytes when possible."""
    root = Path(output_root) / CHANNEL_CACHE_DIRECTORY
    provenance = development.provenance
    if calibration.provenance != provenance:
        raise G3Error("decoded-channel cache role provenance differs")
    expected_roles = {"development": development, "calibration": calibration}
    if root.exists():
        manifest = _verified_json(root / "cache-manifest.json", "G3 decoded-channel cache", "cache_digest")
        if any(manifest.get(key) != provenance.get(key) for key in provenance):
            raise G3Error("decoded-channel cache provenance mismatch")
        roles = manifest.get("roles")
        if not isinstance(roles, Mapping):
            raise G3Error("decoded-channel cache roles are malformed")
        for name, role in expected_roles.items():
            row = roles.get(name)
            if not isinstance(row, Mapping) or row.get("sample_ids") != list(role.dataset.sample_ids):
                raise G3Error("decoded-channel cache membership mismatch")
            source_root = _g2_cache_root(provenance) if row.get("source") == "g2-reuse" else root
            filename = row.get("data_file")
            if not isinstance(filename, str) or sha256_file(source_root / filename) != row.get("data_sha256"):
                raise G3Error("decoded-channel cache digest mismatch")
        return G3ChannelCache(root=root, manifest=manifest, provenance=provenance)
    _ensure_output_root(root)
    role_rows: dict[str, object] = {}
    development_reference = _g2_development_reference(role=development)
    if development_reference is not None:
        role_rows["development"] = development_reference
    try:
        with zipfile.ZipFile(development.dataset._archive, "r") as archive:  # pylint: disable=protected-access
            for role in (development, calibration):
                if role.role == "development" and development_reference is not None:
                    continue
                filename = "{}-channels-packed.bin".format(role.role)
                with open(root / filename, "xb") as output:
                    for sample_id in role.dataset.sample_ids:
                        output.write(_archive_channel_payload(archive, sample_id))
                role_rows[role.role] = {"source": "g3", "data_file": filename, "data_sha256": sha256_file(root / filename),
                                        "sample_ids": list(role.dataset.sample_ids)}
    except (OSError, zipfile.BadZipFile) as error:
        raise G3Error("cannot build decoded-channel cache") from error
    manifest: dict[str, object] = {"schema_version": SCHEMA_VERSION, "label": "G3_DECODED_CHANNEL_CACHE", "cache_format": "packed-bit-occupancy-and-3bit-parts-v1",
                                   **provenance, "roles": role_rows}
    _write_new_or_match(root / "cache-manifest.json", manifest, "G3 decoded-channel cache", "cache_digest")
    verified = _verified_json(root / "cache-manifest.json", "G3 decoded-channel cache", "cache_digest")
    return G3ChannelCache(root=root, manifest=verified, provenance=provenance)


def _baseline_rows(role: G3Role, predictor: EnsemblePredictor, *, initialization_only: bool = False,
                   channel_cache: G3ChannelCache | None = None) -> list[dict[str, object]]:
    """Use only certified, digest-verified G1b baseline responses for Y."""
    if not initialization_only:
        raise G3Error("baseline rows may only be recomputed by initialize")
    if channel_cache is None:
        raise G3Error("baseline initialization requires the decoded-channel cache")
    rows: list[dict[str, object]] = []
    for sample_id in role.dataset.sample_ids:
        try:
            case = load_verified_case(role.dataset._g1b_root, sample_id)  # pylint: disable=protected-access
            solver = case["solver"]
            raw = {"compliance": solver["compliance_j"], "max_von_mises": solver.get("max_gauss_von_mises_pa", solver.get("max_von_mises_pa")),
                   "max_displacement": solver["max_displacement_m"]}
            targets = {name: float(value) for name, value in raw.items()}
        except (KeyError, TypeError, ValueError) as error:
            raise G3Error("certified G1b baseline target schema is invalid") from error
        if case.get("role") != role.role or case.get("exclusion_reasons") != []:
            raise G3Error("certified G1b baseline is outside the requested eligible role")
        channels = channel_cache.channels(role=role.role, sample_id=sample_id)
        rows.append({"sample_id": sample_id, "family_id": role.family_by_sample[sample_id],
                     "y": _normalized_targets(targets, predictor.normalization), "prediction": predictor.predict(channels),
                     "g1b_case_digest": case["case_digest"]})
    if len(rows) != len(role.dataset):
        raise G3Error("baseline role did not materialize exactly once")
    return rows


def _baseline_rows_artifact(*, development: Sequence[Mapping[str, object]], calibration: Sequence[Mapping[str, object]],
                            campaign: Mapping[str, object]) -> dict[str, object]:
    """Bind the full, shard-independent baseline pass to this frozen campaign."""
    return {
        "schema_version": SCHEMA_VERSION,
        "label": "G3_DIGEST_VERIFIED_BASELINE_ROWS",
        "campaign_digest": campaign["campaign_digest"],
        "roles": {
            "development": {"sample_count": len(development), "rows": list(development)},
            "calibration": {"sample_count": len(calibration), "rows": list(calibration)},
        },
    }


def _load_baseline_rows(*, output_root: Path, campaign: Mapping[str, object], development: G3Role,
                        calibration: G3Role) -> dict[str, list[dict[str, object]]]:
    """Load initialize-only baseline evidence; shard workers cannot recompute it."""
    artifact = _verified_json(output_root / BASELINE_ROWS_FILENAME, "G3 baseline rows", "baseline_rows_sha256")
    if artifact.get("campaign_digest") != campaign.get("campaign_digest"):
        raise G3Error("baseline rows do not bind the frozen campaign")
    roles = artifact.get("roles")
    if not isinstance(roles, Mapping):
        raise G3Error("baseline rows roles are malformed")
    loaded: dict[str, list[dict[str, object]]] = {}
    for role in (development, calibration):
        record = roles.get(role.role)
        if not isinstance(record, Mapping) or not isinstance(record.get("rows"), list):
            raise G3Error("baseline rows role record is malformed")
        rows = record["rows"]
        if record.get("sample_count") != len(rows) or len(rows) != len(role.dataset):
            raise G3Error("baseline rows role cardinality is invalid")
        by_sample: dict[str, dict[str, object]] = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
                raise G3Error("baseline row is malformed")
            sample_id = row["sample_id"]
            if sample_id in by_sample or row.get("family_id") != role.family_by_sample.get(sample_id):
                raise G3Error("baseline row identity is invalid")
            if not isinstance(row.get("y"), Mapping) or not isinstance(row.get("prediction"), Mapping) or not isinstance(row.get("g1b_case_digest"), str):
                raise G3Error("baseline row evidence is incomplete")
            by_sample[sample_id] = row
        if set(by_sample) != set(role.dataset.sample_ids):
            raise G3Error("baseline rows role membership is invalid")
        loaded[role.role] = [by_sample[sample_id] for sample_id in role.dataset.sample_ids]
    return loaded


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
                     predictor: EnsemblePredictor, channel_cache: G3ChannelCache) -> dict[str, object]:
    family_id = role.family_by_sample[sample_id]
    occupancy = _load_occupancy(archive_open, ledger, sample_id)
    base = _configuration(archive_open, ledger, sample_id, (0.0, 0.0, -100.0))
    # The immutable cache supplies decoded part labels; the selected state still uses
    # shard-local archive occupancy/configuration reads for solver provenance.
    parts = channel_cache.channels(role=role.role, sample_id=sample_id)[1].astype(np.uint8, copy=False)
    if occupancy.shape != (64, 64, 64) or parts.shape != occupancy.shape:
        raise G3Error("trajectory source payload has invalid shape")
    config = _activity_config(occupancy, base)

    trajectory, state_volumes = geometric_trajectory(sample_id=sample_id, volume=occupancy, batch_cap=40,
                                                      ranking_seed=family_seed(family_id))
    if trajectory.get("topology", {}).get("topology_mode") != "conservative_local_6_26" or trajectory.get("topology", {}).get("sequential_recheck") is not True:
        raise G3Error("trajectory invariant layer is not certified conservative sequential topology")
    batches = trajectory.get("batches")
    if not isinstance(batches, list) or trajectory.get("solver_call_count") != 0:
        raise G3Error("trajectory geometry unexpectedly consulted the solver")
    per_bin: dict[int, list[int]] = {index: [] for index in range(len(DEPTH_BINS))}
    by_index: dict[int, Mapping[str, object]] = {}
    for batch in batches:
        if not isinstance(batch, Mapping) or not isinstance(batch.get("batch_index"), int):
            raise G3Error("trajectory geometry batch is malformed")
        index = int(batch["batch_index"])
        bin_index = depth_bin_index(float(batch.get("proposed_material_reduction", float("nan"))))
        if bin_index is not None:
            per_bin[bin_index].append(index)
        by_index[index] = batch
    selected_states: list[dict[str, object]] = []
    unsolved_states: list[dict[str, object]] = []
    for bin_index, indices in per_bin.items():
        if not indices:
            continue
        state_index = select_state_index(family_id, bin_index, indices)
        state = state_volumes.get(state_index)
        batch = by_index.get(state_index)
        if state is None or batch is None:
            raise G3Error("selected geometric state is unavailable")
        solver = solve_voxels(state, config)
        if solver.get("status") != "success":
            # A near-singular state is an anticipated outcome, not a defect: as the
            # trajectory removes material a state can stop being solvable.  G1b
            # already records exactly this event as data via cohort_reasons rather
            # than raising, and solve_voxels returns a distinguishing `reason` for
            # precisely this purpose.  Raising here aborted an entire shard over one
            # sample and cost 187 collateral cases on the GB200 run (shard 7 of 16,
            # 2026-08-31).  Record the failure, skip the state, and continue.
            #
            # The exception is a missing preconditioner, which is an environment
            # fault affecting every subsequent solve rather than a property of this
            # state.  That must still fail closed and loudly.
            reason = solver.get("reason")
            if reason == "preconditioner_unavailable":
                raise G3Error("selected trajectory solver preconditioner is unavailable")
            unsolved_states.append({
                "state_index": state_index, "bin_index": bin_index, "bin_label": DEPTH_BINS[bin_index],
                "fraction_removed": batch["proposed_material_reduction"],
                "state_occupancy_sha256": batch["state_occupancy_sha256"],
                "solver_status": solver.get("status"), "solver_reason": reason,
            })
            continue
        selected_states.append({
            "state_index": state_index, "bin_index": bin_index, "bin_label": DEPTH_BINS[bin_index],
            "fraction_removed": batch["proposed_material_reduction"],
            "state_occupancy_sha256": batch["state_occupancy_sha256"],
            "solver": solver, "prediction": predictor.predict(_channels(state, parts)),
        })
    if not selected_states:
        raise G3Error("trajectory has no selected depth-bin state")
    result: dict[str, object] = {"schema_version": SCHEMA_VERSION, "sample_id": sample_id, "family_id": family_id,
        "role": role.role, "family_seed_namespace": FAMILY_SEED_NAMESPACE, "campaign_seed": CAMPAIGN_SEED,
        "family_seed": family_seed(family_id), "trajectory": trajectory, "selected_states": selected_states,
        "selected_solver_call_count": len(selected_states),
        "unsolved_states": unsolved_states,
        "unsolved_state_count": len(unsolved_states),
        "intermediate_solver_call_count": 0}
    result["trajectory_digest"] = _digest(result)
    return result


def _load_or_generate_trajectories(*, root: Path, role: G3Role, predictor: EnsemblePredictor, channel_cache: G3ChannelCache,
                                   shard_index: int = 0, shard_count: int = 1) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not isinstance(shard_index, int) or not isinstance(shard_count, int) or not 0 <= shard_index < shard_count:
        raise G3Error("trajectory shard is invalid")
    cases: list[dict[str, object]] = []
    generated = 0
    resumed = 0
    sample_ids = [sample_id for sample_id in role.dataset.sample_ids if shard_for_id(sample_id, shard_count) == shard_index]
    expected_members = ["fea_ml/data/runs_real/{}/{}".format(sample_id, leaf) for sample_id in sample_ids for leaf in ("occ.npz", "meta.json", "part.npz")]
    ledger = _PayloadAccessLedger(sample_ids)
    with zipfile.ZipFile(role.dataset._archive, "r") as archive_open:  # pylint: disable=protected-access
        for sample_id in sample_ids:
            path = _case_path(root, role.role, sample_id)
            if path.exists():
                case = _verified_json(path, "G3 trajectory case", "trajectory_digest")
                if case.get("sample_id") != sample_id or case.get("family_id") != role.family_by_sample[sample_id] or case.get("role") != role.role:
                    raise G3Error("resumed trajectory identity is invalid")
                cases.append(case); resumed += 1
                continue
            case = _trajectory_case(role=role, sample_id=sample_id, archive_open=archive_open, ledger=ledger, predictor=predictor,
                                    channel_cache=channel_cache)
            _write_new_or_match(path, case, "G3 trajectory case", "trajectory_digest")
            cases.append(case); generated += 1
    members, accesses, nonfit = ledger.evidence()
    if nonfit or accesses != len(members):
        raise G3Error("trajectory payload ledger is inconsistent")
    if len(cases) != len(sample_ids):
        raise G3Error("trajectory cases are incomplete")
    return cases, {"role": role.role, "case_count": len(cases), "generated_count": generated, "resumed_count": resumed,
                   "shard": "{}/{}".format(shard_index + 1, shard_count),
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
            selected_states = case["selected_states"]
        except (KeyError, TypeError) as error:
            raise G3Error("trajectory case is malformed") from error
        # Bins whose frozen-rule state failed to solve are recorded rather than
        # selected.  They are legitimately absent from selected_states and must not
        # be read as a sampling-rule violation.
        unsolved = case.get("unsolved_states", [])
        if not isinstance(unsolved, list):
            raise G3Error("trajectory unsolved state record is malformed")
        unsolved_bins: set[int] = set()
        for entry in unsolved:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("bin_index"), int):
                raise G3Error("trajectory unsolved state entry is malformed")
            unsolved_bins.add(int(entry["bin_index"]))
        per_bin: dict[int, list[tuple[int, Mapping[str, object]]]] = {index: [] for index in range(len(DEPTH_BINS))}
        if not isinstance(batches, list) or not isinstance(selected_states, list):
            raise G3Error("trajectory batches are malformed")
        for batch in batches:
            if not isinstance(batch, Mapping):
                raise G3Error("trajectory batch is malformed")
            state_index = batch.get("batch_index")
            fraction = batch.get("proposed_material_reduction")
            if not isinstance(state_index, int):
                raise G3Error("trajectory batch index is invalid")
            bin_index = depth_bin_index(float(fraction))
            if bin_index is None:
                continue
            occupancy[DEPTH_BINS[bin_index]] += 1
            per_bin[bin_index].append((state_index, batch))
        selected_by_bin: dict[int, Mapping[str, object]] = {}
        for selected_state in selected_states:
            if not isinstance(selected_state, Mapping) or not isinstance(selected_state.get("bin_index"), int):
                raise G3Error("selected trajectory state is malformed")
            bin_index = int(selected_state["bin_index"])
            if bin_index in selected_by_bin:
                raise G3Error("trajectory selects more than one state per depth bin")
            selected_by_bin[bin_index] = selected_state
        for bin_index, candidates in per_bin.items():
            if not candidates:
                if bin_index in selected_by_bin:
                    raise G3Error("trajectory selected an unoccupied depth bin")
                continue
            chosen_index = select_state_index(family_id, bin_index, [state_index for state_index, _ in candidates])
            if bin_index in unsolved_bins and bin_index not in selected_by_bin:
                # The frozen rule chose this state; the solver could not evaluate it.
                # Skipping is correct and the sampling rule is still satisfied,
                # because selection happened before and independently of the solve.
                continue
            chosen = selected_by_bin.get(bin_index)
            if not isinstance(chosen, Mapping) or chosen.get("state_index") != chosen_index:
                raise G3Error("selected trajectory state violates the frozen sampling rule")
            prediction = chosen.get("prediction")
            if not isinstance(prediction, Mapping):
                raise G3Error("selected trajectory state lacks prediction")
            rows.append({"sample_id": case["sample_id"], "family_id": family_id, "bin_index": bin_index,
                         "bin_label": DEPTH_BINS[bin_index], "state_index": chosen_index,
                         "fraction_removed": chosen["fraction_removed"], "prediction": prediction,
                         "solver": chosen["solver"]})
            selected[DEPTH_BINS[bin_index]] += 1
        if set(selected_by_bin) | unsolved_bins != {index for index, candidates in per_bin.items() if candidates}:
            raise G3Error("selected trajectory state set does not exactly cover occupied bins")
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


def _initialize_precoverage(*, output_root: Path, split_manifest: Path, expected_split_sha256: str, archive: Path,
                            expected_archive_sha256: str, g1b_root: Path, expected_cohort_manifest_sha256: str,
                            expected_cluster_role_manifest_sha256: str, ensemble_root: Path, normalization_path: Path,
                            preregistration_path: Path, expected_preregistration_sha256: str, device: str) -> dict[str, object]:
    """Freeze development kappa, then open calibration only for reused G1b baselines."""
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
    campaign["corrected_scope"] = {"selected_states_only": True, "intermediate_solver_calls": 0,
                                   "calibration_baseline_source": "certified_g1b_digest_verified_cases",
                                   "lost_telemetry": "no_per_batch_verified_telemetry; A4_and_risk_cost_frontier_require_a_separate_budget"}
    campaign["campaign_digest"] = _digest({key: value for key, value in campaign.items() if key != "campaign_digest"})
    campaign_path = Path(output_root) / "campaign-manifest.json"
    _write_new_or_match(campaign_path, campaign, "G3 campaign manifest", "campaign_digest")
    development_channel_ref = _g2_development_reference(role=development)
    if development_channel_ref is None:
        raise G3Error("certified G2 decoded-channel cache is unavailable for development initialization")
    development_cache = G3ChannelCache(root=Path(output_root) / CHANNEL_CACHE_DIRECTORY,
                                       manifest={"roles": {"development": development_channel_ref}}, provenance=development.provenance)
    development_baselines = _baseline_rows(development, predictor, initialization_only=True, channel_cache=development_cache)
    kappa = select_kappas(development_baselines)
    _write_new_or_match(Path(output_root) / "kappa-development-evidence.json", kappa, "G3 kappa evidence", "kappa_evidence_sha256")
    # This is the first calibration-role payload open, after kappa is persisted.
    calibration = open_g3_role(role="calibration", split_manifest=split_manifest, expected_split_sha256=expected_split_sha256,
                               archive=archive, expected_archive_sha256=expected_archive_sha256, g1b_root=g1b_root,
                               expected_cohort_manifest_sha256=expected_cohort_manifest_sha256,
                               expected_cluster_role_manifest_sha256=expected_cluster_role_manifest_sha256)
    channel_cache = _open_or_build_channel_cache(output_root=output_root, development=development, calibration=calibration)
    calibration_baselines = _baseline_rows(calibration, predictor, initialization_only=True, channel_cache=channel_cache)
    baseline_rows = _baseline_rows_artifact(development=development_baselines, calibration=calibration_baselines, campaign=campaign)
    _write_new_or_match(Path(output_root) / BASELINE_ROWS_FILENAME, baseline_rows, "G3 baseline rows", "baseline_rows_sha256")
    q_base = _quantiles(calibration_baselines, kappas=kappa["kappa"], normalization=predictor.normalization, kind="baseline")
    q_base["baseline_source"] = "certified_g1b_digest_verified_cases_no_new_solver_calls"
    _write_new_or_match(Path(output_root) / "baseline-calibration.json", q_base, "G3 baseline calibration", "baseline_calibration_sha256")
    return {"output_root": str(output_root), "kappa": kappa["kappa"], "q_base": q_base["q"],
            "development_baseline_count": len(development_baselines), "calibration_baseline_count": len(calibration_baselines),
            "coverage_computed": False}


def prepare_precoverage_cache(*, output_root: Path, split_manifest: Path, expected_split_sha256: str, archive: Path,
                              expected_archive_sha256: str, g1b_root: Path, expected_cohort_manifest_sha256: str,
                              expected_cluster_role_manifest_sha256: str, ensemble_root: Path, normalization_path: Path,
                              device: str = "mps") -> dict[str, object]:
    """Materialize the role-scoped channel cache once, without solver or baseline work."""
    output_root = Path(output_root)
    _verified_json(output_root / "campaign-manifest.json", "G3 campaign manifest", "campaign_digest")
    _verified_json(output_root / "kappa-development-evidence.json", "G3 kappa evidence", "kappa_evidence_sha256")
    _verified_json(output_root / "baseline-calibration.json", "G3 baseline calibration", "baseline_calibration_sha256")
    development = open_g3_role(role="development", split_manifest=split_manifest, expected_split_sha256=expected_split_sha256,
                               archive=archive, expected_archive_sha256=expected_archive_sha256, g1b_root=g1b_root,
                               expected_cohort_manifest_sha256=expected_cohort_manifest_sha256,
                               expected_cluster_role_manifest_sha256=expected_cluster_role_manifest_sha256)
    calibration = open_g3_role(role="calibration", split_manifest=split_manifest, expected_split_sha256=expected_split_sha256,
                               archive=archive, expected_archive_sha256=expected_archive_sha256, g1b_root=g1b_root,
                               expected_cohort_manifest_sha256=expected_cohort_manifest_sha256,
                               expected_cluster_role_manifest_sha256=expected_cluster_role_manifest_sha256)
    cache = _open_or_build_channel_cache(output_root=output_root, development=development, calibration=calibration)
    return {"output_root": str(output_root), "cache_manifest_digest": cache.manifest["cache_digest"],
            "coverage_computed": False, "baseline_predictions_recomputed": False,
            "solver_calls": 0}


def run_precoverage_shard(*, output_root: Path, split_manifest: Path, expected_split_sha256: str, archive: Path,
                          expected_archive_sha256: str, g1b_root: Path, expected_cohort_manifest_sha256: str,
                          expected_cluster_role_manifest_sha256: str, ensemble_root: Path, normalization_path: Path,
                          shard_index: int, shard_count: int, device: str = "mps") -> dict[str, object]:
    """Generate disjoint deterministic role shards after kappa/q_base are frozen."""
    output_root = Path(output_root)
    campaign = _verified_json(output_root / "campaign-manifest.json", "G3 campaign manifest", "campaign_digest")
    _verified_json(output_root / "kappa-development-evidence.json", "G3 kappa evidence", "kappa_evidence_sha256")
    _verified_json(output_root / "baseline-calibration.json", "G3 baseline calibration", "baseline_calibration_sha256")
    predictor = EnsemblePredictor(ensemble_root=ensemble_root, normalization_path=normalization_path, device=device)
    development = open_g3_role(role="development", split_manifest=split_manifest, expected_split_sha256=expected_split_sha256,
                               archive=archive, expected_archive_sha256=expected_archive_sha256, g1b_root=g1b_root,
                               expected_cohort_manifest_sha256=expected_cohort_manifest_sha256,
                               expected_cluster_role_manifest_sha256=expected_cluster_role_manifest_sha256)
    calibration = open_g3_role(role="calibration", split_manifest=split_manifest, expected_split_sha256=expected_split_sha256,
                               archive=archive, expected_archive_sha256=expected_archive_sha256, g1b_root=g1b_root,
                               expected_cohort_manifest_sha256=expected_cohort_manifest_sha256,
                               expected_cluster_role_manifest_sha256=expected_cluster_role_manifest_sha256)
    # Earlier stopped runs froze the certified kappa and q_base records before
    # this optional full baseline-row receipt existed.  Those records are the
    # authoritative reusable baseline evidence; do not regenerate predictions
    # merely to backfill rows while trajectory generation is in progress.
    baseline_rows_path = output_root / BASELINE_ROWS_FILENAME
    if baseline_rows_path.exists():
        _load_baseline_rows(output_root=output_root, campaign=campaign, development=development, calibration=calibration)
    channel_cache = _open_or_build_channel_cache(output_root=output_root, development=development, calibration=calibration)
    started = time.perf_counter()
    development_cases, development_ledger = _load_or_generate_trajectories(root=output_root, role=development, predictor=predictor,
                                                                             channel_cache=channel_cache, shard_index=shard_index, shard_count=shard_count)
    calibration_cases, calibration_ledger = _load_or_generate_trajectories(root=output_root, role=calibration, predictor=predictor,
                                                                             channel_cache=channel_cache, shard_index=shard_index, shard_count=shard_count)
    return {"shard": "{}/{}".format(shard_index + 1, shard_count), "development_case_count": len(development_cases),
            "calibration_case_count": len(calibration_cases), "wall_seconds": time.perf_counter() - started,
            "development": development_ledger, "calibration": calibration_ledger, "coverage_computed": False}


def _all_verified_cases(root: Path, role: G3Role) -> tuple[list[dict[str, object]], dict[str, object]]:
    cases: list[dict[str, object]] = []
    for sample_id in role.dataset.sample_ids:
        case = _verified_json(_case_path(root, role.role, sample_id), "G3 trajectory case", "trajectory_digest")
        if case.get("sample_id") != sample_id or case.get("family_id") != role.family_by_sample[sample_id] or case.get("role") != role.role:
            raise G3Error("merged trajectory case identity is invalid")
        if case.get("intermediate_solver_call_count") != 0:
            raise G3Error("merged trajectory case contains forbidden intermediate solves")
        cases.append(case)
    ledger = {"role": role.role, "case_count": len(cases), "merge_only_after_all_shards_complete": True,
              "trajectory_case_digests": [{"sample_id": case["sample_id"], "trajectory_digest": case["trajectory_digest"]} for case in cases]}
    return cases, ledger


def finalize_precoverage(*, output_root: Path, split_manifest: Path, expected_split_sha256: str, archive: Path,
                         expected_archive_sha256: str, g1b_root: Path, expected_cohort_manifest_sha256: str,
                         expected_cluster_role_manifest_sha256: str, ensemble_root: Path, normalization_path: Path,
                         preregistration_path: Path, expected_preregistration_sha256: str, device: str = "mps") -> dict[str, object]:
    """Merge complete shards, calculate q_traj, freeze the policy, and hard-stop."""
    output_root = Path(output_root)
    observed_preregistration = sha256_file(preregistration_path)
    if observed_preregistration != _lower_digest(expected_preregistration_sha256, "pre-registration"):
        raise G3Error("frozen pre-registration sha256 mismatch")
    campaign = _verified_json(output_root / "campaign-manifest.json", "G3 campaign manifest", "campaign_digest")
    kappa = _verified_json(output_root / "kappa-development-evidence.json", "G3 kappa evidence", "kappa_evidence_sha256")
    q_base = _verified_json(output_root / "baseline-calibration.json", "G3 baseline calibration", "baseline_calibration_sha256")
    predictor = EnsemblePredictor(ensemble_root=ensemble_root, normalization_path=normalization_path, device=device)
    development = open_g3_role(role="development", split_manifest=split_manifest, expected_split_sha256=expected_split_sha256,
                               archive=archive, expected_archive_sha256=expected_archive_sha256, g1b_root=g1b_root,
                               expected_cohort_manifest_sha256=expected_cohort_manifest_sha256,
                               expected_cluster_role_manifest_sha256=expected_cluster_role_manifest_sha256)
    calibration = open_g3_role(role="calibration", split_manifest=split_manifest, expected_split_sha256=expected_split_sha256,
                               archive=archive, expected_archive_sha256=expected_archive_sha256, g1b_root=g1b_root,
                               expected_cohort_manifest_sha256=expected_cohort_manifest_sha256,
                               expected_cluster_role_manifest_sha256=expected_cluster_role_manifest_sha256)
    development_cases, development_ledger = _all_verified_cases(output_root, development)
    calibration_cases, calibration_ledger = _all_verified_cases(output_root, calibration)
    calibration_rows, calibration_bin_occupancy, calibration_bin_selected = _selected_trajectory_rows(calibration_cases)
    q_traj = _quantiles(calibration_rows, kappas=kappa["kappa"], normalization=predictor.normalization, kind="trajectory")
    q_traj["depth_bin_occupancy_counts"] = calibration_bin_occupancy
    q_traj["selected_state_counts_per_bin"] = calibration_bin_selected
    _write_new_or_match(output_root / "trajectory-calibration.json", q_traj, "G3 trajectory calibration", "trajectory_calibration_sha256")
    development_rows, development_bin_occupancy, development_bin_selected = _selected_trajectory_rows(development_cases)
    trajectory_ledgers = {"development": development_ledger, "calibration": calibration_ledger,
                          "development_depth_bin_occupancy_counts": development_bin_occupancy,
                          "development_selected_state_counts_per_bin": development_bin_selected,
                          "calibration_depth_bin_occupancy_counts": calibration_bin_occupancy,
                          "calibration_selected_state_counts_per_bin": calibration_bin_selected,
                          "development_selected_state_count": len(development_rows), "calibration_selected_state_count": len(calibration_rows),
                          "intermediate_solver_calls": 0,
                          "limitation": "per-batch verified telemetry is absent; A4 fixed-period and risk-cost frontier need a separately budgeted run"}
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
    parser.add_argument("--mode", choices=("initialize", "prepare-cache", "run-shard", "finalize", "run"), default="run")
    parser.add_argument("--shard", help="one-indexed deterministic N/K shard for --mode run-shard")
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
        common = {"output_root": args.output, "split_manifest": args.split_manifest,
                  "expected_split_sha256": args.expected_split_manifest_sha256, "archive": args.archive,
                  "expected_archive_sha256": args.expected_archive_sha256, "g1b_root": args.g1b_root,
                  "expected_cohort_manifest_sha256": args.expected_cohort_manifest_sha256,
                  "expected_cluster_role_manifest_sha256": args.expected_cluster_role_manifest_sha256,
                  "ensemble_root": args.ensemble_root, "normalization_path": args.normalization_path, "device": args.device}
        frozen = {"preregistration_path": args.preregistration_path,
                  "expected_preregistration_sha256": args.expected_preregistration_sha256}
        if args.mode == "initialize":
            if args.shard:
                raise G3Error("--shard is valid only for --mode run-shard")
            result = _initialize_precoverage(**common, **frozen)
        elif args.mode == "prepare-cache":
            if args.shard:
                raise G3Error("--shard is valid only for --mode run-shard")
            result = prepare_precoverage_cache(**common)
        elif args.mode == "run-shard":
            if not args.shard:
                raise G3Error("--mode run-shard requires --shard N/K")
            try:
                number, count = (int(value) for value in args.shard.split("/", 1))
            except (AttributeError, ValueError) as error:
                raise G3Error("--shard must be N/K") from error
            if not 1 <= number <= count:
                raise G3Error("--shard must satisfy 1 <= N <= K")
            result = run_precoverage_shard(**common, shard_index=number - 1, shard_count=count)
        elif args.mode == "finalize":
            if args.shard:
                raise G3Error("--shard is valid only for --mode run-shard")
            result = finalize_precoverage(**common, **frozen)
        else:
            if args.shard:
                raise G3Error("--shard is valid only for --mode run-shard")
            _initialize_precoverage(**common, **frozen)
            run_precoverage_shard(**common, shard_index=0, shard_count=1)
            result = finalize_precoverage(**common, **frozen)
    except (G3Error, OSError) as error:
        print("REJECTED: {}".format(error), file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
