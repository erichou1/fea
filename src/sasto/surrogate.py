"""G2 dense-surrogate data and training contracts.

Confirmation is intentionally not representable by a dataset handle.  Every role
is admitted before archive access; inputs are separately SHA-256 anchored and
case targets are accepted only after their G1b case digest verifies.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
import random
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np

from .manifest import ManifestVerificationError, open_new_artifact_root, read_regular_path_snapshot, sha256_file, write_new_regular_path
from .splits import PARTITIONS, validate_family_split_manifest

TARGET_NAMES = ("compliance", "max_von_mises", "max_displacement")
_ARCHIVE_PREFIX = "fea_ml/data/runs_real/"
SEED_NAMESPACE = "sasto-v-g2-dense-ensemble-v1"
SMOKE_LABEL = "SMOKE_ONLY_NONPROMOTABLE"
SOURCE_BUNDLE_CONFIG_PATHS = (".python-version", "pyproject.toml", "uv.lock")


class SurrogateError(ValueError):
    """An anchored G2 data, training, or artifact contract was violated."""


class SurrogateRoleError(SurrogateError):
    """A requested role would violate G2's sealed-partition contract."""


@dataclass(frozen=True)
class SurrogateExample:
    sample_id: str
    channels: np.ndarray
    targets: dict[str, float]
    packed_occupancy_nbytes: int


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _lower_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise SurrogateError("{} must be an exact lowercase SHA-256 digest".format(label))
    return value


def _snapshot_json(path: Path, expected: str, label: str) -> tuple[dict[str, object], str]:
    _lower_digest(expected, label + " anchor")
    try:
        snapshot = read_regular_path_snapshot(path, label)
        value = json.loads(snapshot.bytes.decode("utf-8"))
    except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SurrogateError("{} is unavailable or malformed".format(label)) from error
    if snapshot.sha256 != expected:
        raise SurrogateError("{} sha256 mismatch".format(label))
    if not isinstance(value, dict):
        raise SurrogateError("{} must be an object".format(label))
    return value, snapshot.sha256



def _load_verified_case(root: Path, sample_id: str) -> dict[str, object]:
    """Read one G1b case only after its embedded canonical digest validates."""
    if not isinstance(sample_id, str) or not sample_id or any(character in sample_id for character in "/\\\x00"):
        raise SurrogateError("G1b case sample ID is unsafe")
    try:
        snapshot = read_regular_path_snapshot(root / "cases" / (sample_id + ".json"), "G1b case")
        case = json.loads(snapshot.bytes.decode("utf-8"))
    except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SurrogateError("G1b case is unavailable or malformed") from error
    if not isinstance(case, dict) or case.get("sample_id") != sample_id:
        raise SurrogateError("G1b case identity is invalid")
    if case.get("case_digest") != _digest({key: value for key, value in case.items() if key != "case_digest"}):
        raise SurrogateError("G1b case digest mismatch")
    return case


def _role_ids(split: Mapping[str, object], role: str) -> list[str]:
    try:
        validate_family_split_manifest(split)
        ids = split["partitions"][role]["sample_ids"]  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as error:
        raise SurrogateError("frozen split is invalid") from error
    if not isinstance(ids, list) or not ids or any(not isinstance(item, str) or not item for item in ids):
        raise SurrogateError("{} role membership is invalid".format(role))
    return sorted(ids)


def _cohort_roles(cohort: Mapping[str, object], clusters: Mapping[str, object], split: Mapping[str, object]) -> dict[str, str]:
    records = cohort.get("records")
    eligible = cohort.get("eligible_ids")
    cluster_rows = clusters.get("clusters")
    if not isinstance(records, list) or not isinstance(eligible, list) or not isinstance(cluster_rows, list):
        raise SurrogateError("G1b cohort or cluster role manifest is malformed")
    eligible_ids = set(eligible)
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in eligible_ids):
        raise SurrogateError("G1b eligible IDs are invalid")
    cohort_by_id: dict[str, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("sample_id"), str):
            raise SurrogateError("G1b cohort record is malformed")
        sample_id = record["sample_id"]
        if sample_id in cohort_by_id:
            raise SurrogateError("G1b cohort has duplicate sample IDs")
        cohort_by_id[sample_id] = record
    cluster_role: dict[str, str] = {}
    for row in cluster_rows:
        if not isinstance(row, Mapping) or row.get("role") not in PARTITIONS or not isinstance(row.get("members"), list):
            raise SurrogateError("G1b cluster role manifest is malformed")
        for sample_id in row["members"]:
            if not isinstance(sample_id, str) or sample_id in cluster_role:
                raise SurrogateError("G1b cluster membership is invalid")
            cluster_role[sample_id] = str(row["role"])
    split_roles = {sample_id: role for role in PARTITIONS for sample_id in _role_ids(split, role)}
    if set(split_roles) != set(cohort_by_id) or set(split_roles) != set(cluster_role):
        raise SurrogateError("G1b cohort, cluster roles, and frozen split do not cover the same IDs")
    admitted: dict[str, str] = {}
    for sample_id in sorted(eligible_ids):
        record = cohort_by_id.get(sample_id)
        if record is None or record.get("exclusion_reasons") != []:
            raise SurrogateError("eligible G1b record is inconsistent")
        role = split_roles[sample_id]
        if record.get("role") != role or cluster_role[sample_id] != role:
            raise SurrogateError("G1b role disagreement")
        admitted[sample_id] = role
    return admitted


def _read_npz_member(archive: zipfile.ZipFile, sample_id: str, leaves: tuple[str, ...], label: str) -> np.ndarray:
    last_error: Exception | None = None
    for leaf in leaves:
        try:
            with archive.open(_ARCHIVE_PREFIX + sample_id + "/" + leaf, "r") as opened:
                payload = opened.read()
            with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
                if loaded.files != ["data"]:
                    raise SurrogateError("{} NPZ must contain exactly data".format(label))
                return loaded["data"]
        except KeyError as error:
            last_error = error
    raise SurrogateError("{} payload is missing".format(label)) from last_error


def _targets_from_case(case: Mapping[str, object]) -> dict[str, float]:
    solver = case.get("solver")
    if not isinstance(solver, Mapping) or solver.get("status") != "success":
        raise SurrogateError("G1b case is not a successful canonical baseline")
    raw = {
        "compliance": solver.get("compliance_j"),
        "max_von_mises": solver.get("max_gauss_von_mises_pa", solver.get("max_von_mises_pa")),
        "max_displacement": solver.get("max_displacement_m"),
    }
    try:
        values = {name: float(value) for name, value in raw.items()}
    except (TypeError, ValueError) as error:
        raise SurrogateError("G1b canonical target schema is invalid") from error
    if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
        raise SurrogateError("G1b canonical targets must be finite and positive")
    return values


class RoleDataset:
    """Lazy role-scoped archive reader with packed occupancy cache."""

    def __init__(self, *, role: str, sample_ids: list[str], archive: Path, g1b_root: Path, provenance: Mapping[str, str]) -> None:
        self.role = role
        self.sample_ids = tuple(sample_ids)
        self._archive = archive
        self._g1b_root = g1b_root
        self.provenance = dict(provenance)
        self._packed: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _payload(self, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._packed.get(sample_id)
        if cached is not None:
            return cached
        try:
            with zipfile.ZipFile(self._archive, "r") as archive:
                occupancy = _read_npz_member(archive, sample_id, ("occ.npz",), "occupancy")
                parts = _read_npz_member(archive, sample_id, ("part.npz", "parts.npz"), "part labels")
        except (OSError, zipfile.BadZipFile) as error:
            raise SurrogateError("anchored archive cannot be opened") from error
        if occupancy.shape != (64, 64, 64) or occupancy.dtype not in (np.dtype(np.uint8), np.dtype(np.bool_)):
            raise SurrogateError("occupancy has an invalid 64-cubed binary schema")
        if parts.shape != (64, 64, 64) or parts.dtype not in (np.dtype(np.uint8), np.dtype(np.int8)):
            raise SurrogateError("part labels have an invalid 64-cubed integer schema")
        if not np.all((occupancy == 0) | (occupancy == 1)) or not np.all((parts >= 0) & (parts <= 5)):
            raise SurrogateError("occupancy or part-label value is outside the frozen schema")
        packed = np.packbits(occupancy.reshape(-1), bitorder="little")
        compact_parts = parts.astype(np.uint8, copy=True)
        self._packed[sample_id] = (packed, compact_parts)
        return self._packed[sample_id]

    def __iter__(self) -> Iterator[SurrogateExample]:
        for sample_id in self.sample_ids:
            packed, parts = self._payload(sample_id)
            try:
                case = _load_verified_case(self._g1b_root, sample_id)
            except SurrogateError as error:
                raise SurrogateError("G1b case digest verification failed") from error
            if case.get("role") != self.role or case.get("exclusion_reasons") != []:
                raise SurrogateError("G1b case is outside this eligible role")
            occupancy = np.unpackbits(packed, bitorder="little", count=64 ** 3).reshape(64, 64, 64)
            channels = np.stack((occupancy.astype(np.float32), parts.astype(np.float32)), axis=0)
            yield SurrogateExample(sample_id=sample_id, channels=channels, targets=_targets_from_case(case), packed_occupancy_nbytes=int(packed.nbytes))



class PackedRoleDataset:
    """Role-scoped view of a verified on-disk packed-bit ingest cache."""

    def __init__(self, *, role: str, sample_ids: list[str], cache_root: Path, data_file: str,
                 targets: Mapping[str, Mapping[str, object]], provenance: Mapping[str, str]) -> None:
        self.role = role
        self.sample_ids = tuple(sample_ids)
        self._cache_root = Path(cache_root)
        self._data_file = data_file
        self._targets = {sample_id: dict(values) for sample_id, values in targets.items()}
        self.provenance = dict(provenance)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _example_from_payload(self, *, sample_id: str, payload: np.ndarray) -> SurrogateExample:
        occupancy = np.unpackbits(payload[:32768], bitorder="little", count=64 ** 3).reshape(64, 64, 64)
        parts = np.zeros(64 ** 3, dtype=np.uint8)
        for bit in range(3):
            parts |= np.unpackbits(payload[(bit + 1) * 32768:(bit + 2) * 32768], bitorder="little", count=64 ** 3).astype(np.uint8) << bit
        channels = np.stack((occupancy.astype(np.float32), parts.reshape(64, 64, 64).astype(np.float32)), axis=0)
        target_values = self._targets.get(sample_id)
        if not isinstance(target_values, Mapping):
            raise SurrogateError("packed ingest cache target payload is malformed")
        try:
            targets = {name: float(target_values[name]) for name in TARGET_NAMES}
        except (KeyError, TypeError, ValueError) as error:
            raise SurrogateError("packed ingest cache target payload is malformed") from error
        if not all(math.isfinite(value) and value > 0.0 for value in targets.values()):
            raise SurrogateError("packed ingest cache targets must be finite and positive")
        return SurrogateExample(sample_id=sample_id, channels=channels, targets=targets, packed_occupancy_nbytes=32768)

    def iter_examples(self, indices: list[int] | None = None) -> Iterator[SurrogateExample]:
        item_bytes = 4 * (64 ** 3 // 8)
        ordered = list(range(len(self.sample_ids))) if indices is None else indices
        if any(not isinstance(index, int) or not 0 <= index < len(self.sample_ids) for index in ordered):
            raise SurrogateError("packed ingest cache index is outside the role")
        data_path = self._cache_root / self._data_file
        try:
            packed = np.memmap(data_path, dtype=np.uint8, mode="r", shape=(len(self.sample_ids), item_bytes))
        except (OSError, ValueError) as error:
            raise SurrogateError("packed ingest cache payload is unavailable") from error
        try:
            for index in ordered:
                yield self._example_from_payload(sample_id=self.sample_ids[index], payload=packed[index])
        finally:
            del packed

    def __iter__(self) -> Iterator[SurrogateExample]:
        return self.iter_examples()


class PackedRoleSubset:
    """Deterministic logical subset of a packed role without copying its payload."""

    def __init__(self, dataset: PackedRoleDataset, sample_count: int) -> None:
        if not 1 <= sample_count <= len(dataset):
            raise SurrogateError("packed role subset sample count is outside the admitted role")
        self._dataset = dataset; self.role = dataset.role; self.sample_ids = dataset.sample_ids[:sample_count]; self.provenance = dataset.provenance

    def __len__(self) -> int:
        return len(self.sample_ids)

    def iter_examples(self, indices: list[int] | None = None) -> Iterator[SurrogateExample]:
        selected = list(range(len(self))) if indices is None else indices
        return self._dataset.iter_examples(selected)

    def __iter__(self) -> Iterator[SurrogateExample]:
        return self.iter_examples()


def packed_role_subset(dataset: PackedRoleDataset, *, sample_count: int) -> PackedRoleSubset:
    if not isinstance(dataset, PackedRoleDataset):
        raise SurrogateError("packed role subset requires a packed role dataset")
    return PackedRoleSubset(dataset, sample_count)


def _packed_cache_manifest(path: Path) -> dict[str, object]:
    try:
        snapshot = read_regular_path_snapshot(path / "cache-manifest.json", "packed ingest cache manifest")
        manifest = json.loads(snapshot.bytes.decode("utf-8"))
    except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SurrogateError("packed ingest cache manifest is unavailable or malformed") from error
    if not isinstance(manifest, dict) or manifest.get("cache_digest") != _digest({key: value for key, value in manifest.items() if key != "cache_digest"}):
        raise SurrogateError("packed ingest cache manifest digest mismatch")
    return manifest


def build_packed_ingest_cache(*, cache_root: Path, datasets: list[RoleDataset]) -> dict[str, PackedRoleDataset]:
    """Decode each admitted archive payload once into a digest-bound packed-bit cache.

    Occupancy uses one packed bit-plane and part labels use three packed bit-planes.
    The cache is immutable and its identity binds the exact archive and cohort digests.
    """
    if not datasets or any(not isinstance(dataset, RoleDataset) for dataset in datasets):
        raise SurrogateError("packed ingest cache requires admitted role datasets")
    roles = [dataset.role for dataset in datasets]
    if len(set(roles)) != len(roles) or any(role not in {"fit", "development"} for role in roles):
        raise SurrogateRoleError("packed ingest cache may contain fit and development roles only")
    provenance = datasets[0].provenance
    if any(dataset.provenance != provenance for dataset in datasets[1:]):
        raise SurrogateError("packed ingest cache inputs have inconsistent provenance")
    archive_sha = _lower_digest(provenance.get("archive_sha256"), "packed ingest cache archive")
    cohort_sha = _lower_digest(provenance.get("cohort_manifest_sha256"), "packed ingest cache cohort")
    root = Path(cache_root)

    def instantiate(manifest: Mapping[str, object]) -> dict[str, PackedRoleDataset]:
        if manifest.get("archive_sha256") != archive_sha or manifest.get("cohort_manifest_sha256") != cohort_sha:
            raise SurrogateError("packed ingest cache digest binding does not match admitted inputs")
        rows = manifest.get("roles")
        if not isinstance(rows, Mapping):
            raise SurrogateError("packed ingest cache roles are malformed")
        result: dict[str, PackedRoleDataset] = {}
        for dataset in datasets:
            row = rows.get(dataset.role)
            if not isinstance(row, Mapping) or row.get("sample_ids") != list(dataset.sample_ids):
                raise SurrogateError("packed ingest cache role membership does not match admitted inputs")
            data_file = row.get("data_file"); targets = row.get("targets")
            if not isinstance(data_file, str) or not isinstance(targets, Mapping):
                raise SurrogateError("packed ingest cache role payload is malformed")
            data_path = root / data_file
            if sha256_file(data_path) != row.get("data_sha256") or data_path.stat().st_size != len(dataset) * 4 * (64 ** 3 // 8):
                raise SurrogateError("packed ingest cache role payload digest mismatch")
            result[dataset.role] = PackedRoleDataset(role=dataset.role, sample_ids=list(dataset.sample_ids), cache_root=root,
                data_file=data_file, targets=targets, provenance=provenance)
        return result

    if root.exists():
        return instantiate(_packed_cache_manifest(root))
    try:
        with open_new_artifact_root(root):
            pass
    except (ManifestVerificationError, OSError) as error:
        raise SurrogateError("packed ingest cache root must be new or verified") from error
    role_rows: dict[str, object] = {}
    ingest_started = time.perf_counter()
    try:
        with zipfile.ZipFile(datasets[0]._archive, "r") as archive:
            for dataset in datasets:
                filename = "{}-channels-packed.bin".format(dataset.role)
                targets: dict[str, dict[str, float]] = {}
                with open(root / filename, "xb") as output:
                    for sample_id in dataset.sample_ids:
                        occupancy = _read_npz_member(archive, sample_id, ("occ.npz",), "occupancy")
                        parts = _read_npz_member(archive, sample_id, ("part.npz", "parts.npz"), "part labels")
                        if occupancy.shape != (64, 64, 64) or occupancy.dtype not in (np.dtype(np.uint8), np.dtype(np.bool_)) or not np.all((occupancy == 0) | (occupancy == 1)):
                            raise SurrogateError("occupancy has an invalid 64-cubed binary schema")
                        if parts.shape != (64, 64, 64) or parts.dtype not in (np.dtype(np.uint8), np.dtype(np.int8)) or not np.all((parts >= 0) & (parts <= 5)):
                            raise SurrogateError("part labels have an invalid 64-cubed integer schema")
                        output.write(np.packbits(occupancy.reshape(-1), bitorder="little").tobytes())
                        flat_parts = parts.reshape(-1).astype(np.uint8, copy=False)
                        for bit in range(3):
                            output.write(np.packbits((flat_parts >> bit) & 1, bitorder="little").tobytes())
                        case = _load_verified_case(dataset._g1b_root, sample_id)
                        if case.get("role") != dataset.role or case.get("exclusion_reasons") != []:
                            raise SurrogateError("G1b case is outside this eligible role")
                        targets[sample_id] = _targets_from_case(case)
                role_rows[dataset.role] = {"sample_ids": list(dataset.sample_ids), "data_file": filename,
                    "data_sha256": sha256_file(root / filename), "packed_channel_bytes_per_sample": 4 * (64 ** 3 // 8), "targets": targets}
    except (OSError, zipfile.BadZipFile) as error:
        raise SurrogateError("anchored archive cannot be decoded into packed ingest cache") from error
    ingest_wall_seconds = time.perf_counter() - ingest_started
    ingest_sample_count = sum(len(dataset) for dataset in datasets)
    manifest: dict[str, object] = {"schema_version": "1.0.0", "cache_format": "packed-bit-occupancy-and-3bit-parts-v1",
        "archive_sha256": archive_sha, "cohort_manifest_sha256": cohort_sha,
        "split_manifest_sha256": provenance["split_manifest_sha256"], "cluster_role_manifest_sha256": provenance["cluster_role_manifest_sha256"],
        "input_sample_count": ingest_sample_count, "input_wall_seconds": ingest_wall_seconds,
        "input_samples_per_second": ingest_sample_count / ingest_wall_seconds if ingest_wall_seconds else 0.0, "roles": role_rows}
    manifest["cache_digest"] = _digest(manifest)
    _write_new_json(root / "cache-manifest.json", manifest, "packed ingest cache manifest")
    return instantiate(manifest)


def compute_fit_normalization(dataset: RoleDataset | PackedRoleDataset) -> dict[str, object]:
    """Compute the sole permitted target normalization record from fit examples."""
    if not isinstance(dataset, (RoleDataset, PackedRoleDataset)) or dataset.role != "fit":
        raise SurrogateRoleError("normalization statistics may be computed from fit only")
    if isinstance(dataset, PackedRoleDataset):
        source_ids = list(dataset.sample_ids)
        transformed = {name: np.array([math.log(float(dataset._targets[sample_id][name])) for sample_id in source_ids], dtype=np.float64) for name in TARGET_NAMES}
    else:
        examples = list(dataset)
        if not examples:
            raise SurrogateError("fit role has no examples for normalization")
        source_ids = [example.sample_id for example in examples]
        transformed = {name: np.array([math.log(example.targets[name]) for example in examples], dtype=np.float64) for name in TARGET_NAMES}
    if not source_ids:
        raise SurrogateError("fit role has no examples for normalization")
    record: dict[str, object] = {
        "schema_version": "1.0.0", "role": "fit", "source_sample_ids": source_ids,
        "source_sample_count": len(source_ids), "split_manifest_sha256": dataset.provenance["split_manifest_sha256"],
        "archive_sha256": dataset.provenance["archive_sha256"], "cohort_manifest_sha256": dataset.provenance["cohort_manifest_sha256"],
        "cluster_role_manifest_sha256": dataset.provenance["cluster_role_manifest_sha256"], "target_names": list(TARGET_NAMES),
        "transform": {"name": "natural_log", "domain": "strictly_positive", "clipping": "none"},
        "means": {name: float(values.mean()) for name, values in transformed.items()},
        "scales": {name: max(float(values.std(ddof=0)), 1e-6) for name, values in transformed.items()},
    }
    record["stats_digest"] = _digest(record)
    return record


def role_subset(dataset: RoleDataset, *, sample_count: int) -> RoleDataset:
    """Return the deterministic leading subset of one already-admitted role."""
    if not isinstance(dataset, RoleDataset) or not isinstance(sample_count, int) or isinstance(sample_count, bool):
        raise SurrogateError("role subset requires a dataset and integer sample count")
    if not 1 <= sample_count <= len(dataset):
        raise SurrogateError("role subset sample count is outside the admitted role")
    return RoleDataset(role=dataset.role, sample_ids=list(dataset.sample_ids[:sample_count]), archive=dataset._archive,
                       g1b_root=dataset._g1b_root, provenance=dataset.provenance)


def normalized_tensor_examples(dataset: RoleDataset, normalization: Mapping[str, object]):
    """Materialize a role-scoped subset using fit-only log normalization."""
    torch, _ = _require_torch()
    if not isinstance(dataset, RoleDataset) or not isinstance(normalization, Mapping):
        raise SurrogateError("normalized examples require a role dataset and normalization record")
    if normalization.get("role") != "fit" or normalization.get("target_names") != list(TARGET_NAMES):
        raise SurrogateError("normalization record must be fit-only and target-compatible")
    means = normalization.get("means"); scales = normalization.get("scales")
    if not isinstance(means, Mapping) or not isinstance(scales, Mapping):
        raise SurrogateError("normalization record is malformed")
    try:
        mean_values = {name: float(means[name]) for name in TARGET_NAMES}
        scale_values = {name: float(scales[name]) for name in TARGET_NAMES}
    except (KeyError, TypeError, ValueError) as error:
        raise SurrogateError("normalization values are malformed") from error
    if not all(math.isfinite(value) for value in mean_values.values()) or not all(math.isfinite(value) and value > 0 for value in scale_values.values()):
        raise SurrogateError("normalization values are not finite")
    return [(example.sample_id, torch.from_numpy(example.channels.copy()).to(dtype=torch.float32),
             torch.tensor([(math.log(example.targets[name]) - mean_values[name]) / scale_values[name] for name in TARGET_NAMES], dtype=torch.float32))
            for example in dataset]



def _require_torch():
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as error:
        raise SurrogateError("PyTorch is required; install the locked ml dependency group") from error
    return torch, nn


def DenseSurrogateCNN(*, target_names: tuple[str, ...] = TARGET_NAMES, base_channels: int = 16):
    """Create a compact dense 3-D CNN with heteroscedastic named-target head.

    The factory keeps importing :mod:`sasto.surrogate` possible in the solver-only
    environment; PyTorch is loaded only when model construction is requested.
    """
    torch, nn = _require_torch()
    if not target_names or len(set(target_names)) != len(target_names) or any(not isinstance(name, str) or not name for name in target_names):
        raise SurrogateError("model target names must be unique non-empty strings")
    if not isinstance(base_channels, int) or isinstance(base_channels, bool) or base_channels < 2:
        raise SurrogateError("base_channels must be an integer of at least two")

    class _DenseSurrogateCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c = base_channels
            def block(input_channels: int, output_channels: int, stride: int) -> nn.Sequential:
                return nn.Sequential(nn.Conv3d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                                     nn.GroupNorm(1, output_channels), nn.SiLU())
            self.features = nn.Sequential(block(2, c, 2), block(c, 2 * c, 2), block(2 * c, 4 * c, 2), block(4 * c, 4 * c, 2), nn.AdaptiveAvgPool3d(1))
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(4 * c, 4 * c), nn.SiLU(), nn.Linear(4 * c, 2 * len(target_names)))
            self.target_names = tuple(target_names)

        @property
        def parameter_count(self) -> int:
            return sum(parameter.numel() for parameter in self.parameters())

        def forward(self, channels):
            if channels.ndim != 5 or channels.shape[1] != 2:
                raise SurrogateError("model input must be [batch, 2, 64, 64, 64]")
            raw = self.head(self.features(channels))
            mean, raw_scale = raw.chunk(2, dim=1)
            return {"mean": mean, "dispersion": torch.nn.functional.softplus(raw_scale) + 1e-6}

    return _DenseSurrogateCNN()


def deterministic_seed(namespace: str, campaign_seed: int, member_index: int) -> int:
    """Derive a stable non-Python-hash seed for one independently trained member."""
    if not isinstance(namespace, str) or not namespace or not isinstance(campaign_seed, int) or not isinstance(member_index, int) or member_index < 0:
        raise SurrogateError("seed namespace, campaign seed, and member index are invalid")
    payload = "{}\0{}\0{}".format(namespace, campaign_seed, member_index).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2 ** 31 - 1)



def development_early_stopping_epoch(development_losses, *, patience: int) -> int:
    """Select the 1-indexed epoch using DEVELOPMENT metrics only.

    Callers pass one scalar per completed epoch; calibration and confirmation
    objects are deliberately absent from this API to make them unusable for a
    stopping decision.
    """
    if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
        raise SurrogateError("development early-stopping patience must be positive")
    values = list(development_losses)
    if not values:
        raise SurrogateError("development early stopping requires at least one loss")
    best_loss = float("inf"); best_epoch = 0; stale = 0
    for epoch, loss in enumerate(values, start=1):
        loss = float(loss)
        if not math.isfinite(loss):
            raise SurrogateError("development loss must be finite")
        if loss < best_loss:
            best_loss = loss; best_epoch = epoch; stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_epoch


def _write_new_json(path: Path, value: Mapping[str, object], role: str) -> str:
    try:
        return write_new_regular_path(path, _canonical_bytes(dict(value)) + b"\n", role)
    except ManifestVerificationError as error:
        raise SurrogateError("cannot append {}".format(role)) from error


def _configure_determinism(torch, seed: int) -> None:
    """Set all supported process-local RNGs before each independent member."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_smoke_ensemble(
    *, output_root: Path, examples, target_names: tuple[str, ...], normalization_stats_digest: str, source_bundle_sha256: str,
    split_sha256: str, archive_sha256: str, cohort_manifest_sha256: str, member_count: int = 5, epochs: int = 1,
    base_channels: int = 16, device: str = "cpu", campaign_seed: int = 20260828, data_role: str = "development",
    input_wall_seconds: float = 0.0,
) -> dict[str, object]:
    """Train a bounded role-scoped smoke ensemble; it is explicitly nonpromotable.

    Callers must have passed all sources through :func:`open_role_dataset` before
    materializing ``examples``.  This function remains bounded and cannot become a
    full-fit route while G1b certification is pending.
    """
    torch, _ = _require_torch()
    if not 1 <= len(examples) <= 64 or not 1 <= member_count <= 5 or not 1 <= epochs <= 3:
        raise SurrogateError("SMOKE_ONLY_NONPROMOTABLE requires 1..64 examples, 1..5 members, and 1..3 epochs")
    if data_role != "development":
        raise SurrogateRoleError("SMOKE_ONLY_NONPROMOTABLE may train only on development examples")
    if not math.isfinite(input_wall_seconds) or input_wall_seconds < 0.0:
        raise SurrogateError("smoke input wall clock must be finite and nonnegative")
    for digest, label in ((normalization_stats_digest, "normalization stats"), (source_bundle_sha256, "source bundle"),
                          (split_sha256, "split"), (archive_sha256, "archive"), (cohort_manifest_sha256, "cohort manifest")):
        _lower_digest(digest, label + " digest")
    if device not in {"cpu", "mps"} or (device == "mps" and not torch.backends.mps.is_available()):
        raise SurrogateError("requested smoke device is unavailable")
    converted = []
    for sample_id, channels, targets in examples:
        if not isinstance(sample_id, str) or not sample_id or tuple(channels.shape) != (2, 64, 64, 64) or tuple(targets.shape) != (len(target_names),):
            raise SurrogateError("synthetic smoke example schema is invalid")
        converted.append((sample_id, channels.detach().to(dtype=torch.float32), targets.detach().to(dtype=torch.float32)))
    root = Path(output_root)
    try:
        with open_new_artifact_root(root) as root_fd:
            os.mkdir("members", dir_fd=root_fd)
    except (ManifestVerificationError, OSError) as error:
        raise SurrogateError("smoke artifact root must be new and append-only") from error
    members: list[dict[str, object]] = []
    for member_index in range(member_count):
        seed = deterministic_seed(SEED_NAMESPACE, campaign_seed, member_index)
        _configure_determinism(torch, seed)
        if device == "mps":
            torch.mps.empty_cache()
        model = DenseSurrogateCNN(target_names=target_names, base_channels=base_channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        started = time.perf_counter(); steps = 0; total_loss = 0.0
        model.train()
        for _epoch in range(epochs):
            for _sample_id, channels, targets in converted:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(channels.unsqueeze(0).to(device))
                expected = targets.unsqueeze(0).to(device)
                dispersion = prediction["dispersion"]
                loss = (0.5 * ((expected - prediction["mean"]) / dispersion).square() + torch.log(dispersion)).mean()
                loss.backward(); optimizer.step()
                total_loss += float(loss.detach().cpu()); steps += 1
        training_wall_seconds = time.perf_counter() - started
        wall_seconds = input_wall_seconds + training_wall_seconds
        peak_memory = int(torch.mps.current_allocated_memory()) if device == "mps" else 0
        checkpoint_path = root / "members" / "member-{:02d}.pt".format(member_index)
        if checkpoint_path.exists():
            raise SurrogateError("checkpoint path must be append-only")
        torch.save({"state_dict": model.state_dict(), "target_names": target_names, "base_channels": base_channels, "seed": seed}, checkpoint_path)
        checkpoint_sha = sha256_file(checkpoint_path)
        ledger = {"wall_seconds": wall_seconds, "input_wall_seconds": input_wall_seconds,
                  "training_wall_seconds": training_wall_seconds, "epochs": epochs, "steps": steps, "peak_memory_bytes": peak_memory,
                  "device": device, "samples_per_second": (len(converted) * epochs) / wall_seconds if wall_seconds else 0.0}
        manifest: dict[str, object] = {"schema_version": "1.0.0", "label": SMOKE_LABEL, "member_index": member_index,
            "seed_namespace": SEED_NAMESPACE, "campaign_seed": campaign_seed, "seed": seed, "source_bundle_sha256": source_bundle_sha256,
            "cohort_manifest_sha256": cohort_manifest_sha256, "split_sha256": split_sha256, "archive_sha256": archive_sha256,
            "normalization_stats_digest": normalization_stats_digest, "epoch_count": epochs, "target_names": list(target_names),
            "data_role": data_role, "sample_ids": [sample_id for sample_id, _channels, _targets in converted],
            "parameter_count": model.parameter_count, "checkpoint": {"path": checkpoint_path.name, "sha256": checkpoint_sha},
            "final_metrics": {"train_heteroscedastic_nll": total_loss / steps}, "compute_ledger": ledger}
        manifest["manifest_digest"] = _digest(manifest)
        manifest_sha = _write_new_json(root / "members" / "member-{:02d}.json".format(member_index), manifest, "member manifest")
        members.append({"member_index": member_index, "seed": seed, "manifest_sha256": manifest_sha, "checkpoint_sha256": checkpoint_sha,
                        "parameter_count": model.parameter_count, "samples_per_second": ledger["samples_per_second"]})
    summary: dict[str, object] = {"schema_version": "1.0.0", "label": SMOKE_LABEL, "member_count": member_count,
        "sample_count": len(converted), "members": members, "source_bundle_sha256": source_bundle_sha256,
        "split_sha256": split_sha256, "archive_sha256": archive_sha256, "cohort_manifest_sha256": cohort_manifest_sha256,
        "normalization_stats_digest": normalization_stats_digest, "seed_namespace": SEED_NAMESPACE, "campaign_seed": campaign_seed,
        "data_role": data_role, "sample_ids": [sample_id for sample_id, _channels, _targets in converted]}
    summary["summary_digest"] = _digest(summary)
    summary_sha = _write_new_json(root / "smoke-summary.json", summary, "smoke summary")
    return {"label": SMOKE_LABEL, "members": members, "summary_sha256": summary_sha, "output_root": str(root)}


def _normalized_batches(data, normalization: Mapping[str, object], *, batch_size: int, indices: list[int] | None = None):
    """Yield normalized tensors lazily; packed caches never materialize a role in RAM."""
    torch, _ = _require_torch()
    means = normalization["means"]; scales = normalization["scales"]
    if not isinstance(means, Mapping) or not isinstance(scales, Mapping):
        raise SurrogateError("normalization record is malformed")
    if isinstance(data, (PackedRoleDataset, PackedRoleSubset)):
        iterator = data.iter_examples(indices)
        raw_targets = True
    else:
        if indices is not None:
            iterator = (data[index] for index in indices)
        else:
            iterator = iter(data)
        raw_targets = False
    batch = []
    for item in iterator:
        if raw_targets:
            target = torch.tensor([(math.log(item.targets[name]) - float(means[name])) / float(scales[name]) for name in TARGET_NAMES], dtype=torch.float32)
            row = (item.sample_id, torch.from_numpy(item.channels.copy()).to(dtype=torch.float32), target)
        else:
            row = item
        batch.append(row)
        if len(batch) == batch_size:
            yield batch; batch = []
    if batch:
        yield batch


def _development_metrics(model, data, normalization: Mapping[str, object], *, device: str, batch_size: int = 4) -> dict[str, object]:
    torch, _ = _require_torch()
    means = np.array([float(normalization["means"][name]) for name in TARGET_NAMES])
    scales = np.array([float(normalization["scales"][name]) for name in TARGET_NAMES])
    absolute = np.zeros(len(TARGET_NAMES), dtype=np.float64); normalized = np.zeros(len(TARGET_NAMES), dtype=np.float64); count = 0
    model.eval()
    with torch.no_grad():
        for batch in _normalized_batches(data, normalization, batch_size=batch_size):
            channels = torch.stack([row[1] for row in batch]).to(device)
            expected = torch.stack([row[2] for row in batch]).cpu().numpy()
            predicted = model(channels)["mean"].detach().cpu().numpy()
            absolute += np.abs(np.exp(predicted * scales + means) - np.exp(expected * scales + means)).sum(axis=0)
            normalized += np.abs(predicted - expected).sum(axis=0); count += len(batch)
    if count == 0:
        raise SurrogateError("development metrics require examples")
    raw = {name: float(absolute[index] / count) for index, name in enumerate(TARGET_NAMES)}
    log = {name: float(normalized[index] / count) for index, name in enumerate(TARGET_NAMES)}
    return {"development_mae": raw, "development_normalized_log_mae": log,
            "development_selection_metric": float(np.mean(normalized / count))}


def capacity_study(
    *, fit_examples, development_examples, normalization: Mapping[str, object], widths: tuple[int, ...] = (4, 16, 32),
    epochs: int = 1, device: str = "cpu", campaign_seed: int = 20260828, provenance: Mapping[str, str] | None = None,
    minimum_relative_width_separation: float = 0.02,
) -> dict[str, object]:
    """At-scale capacity comparison with a predeclared no-noise selection rule."""
    torch, _ = _require_torch()
    if len(widths) < 3 or len(set(widths)) != len(widths) or any(not isinstance(width, int) or width < 2 for width in widths):
        raise SurrogateError("capacity study requires at least three distinct valid widths")
    if len(fit_examples) < 1 or len(development_examples) < 1 or not isinstance(epochs, int) or not 1 <= epochs <= 50:
        raise SurrogateError("capacity study requires nonempty examples and 1..50 epochs")
    if not math.isfinite(minimum_relative_width_separation) or not 0.0 < minimum_relative_width_separation < 1.0:
        raise SurrogateError("capacity-study width separation must be between zero and one")
    if device not in {"cpu", "mps"} or (device == "mps" and not torch.backends.mps.is_available()):
        raise SurrogateError("requested capacity-study device is unavailable")
    normalization_digest = _lower_digest(normalization.get("stats_digest"), "capacity study normalization stats")
    rows: list[dict[str, object]] = []
    for width in widths:
        seed = deterministic_seed("sasto-v-g2-capacity-study-v1", campaign_seed, width)
        _configure_determinism(torch, seed)
        if device == "mps": torch.mps.empty_cache()
        model = DenseSurrogateCNN(base_channels=width).to(device)
        epoch_zero = _development_metrics(model, development_examples, normalization, device=device)
        started = time.perf_counter(); steps = 0
        for epoch in range(epochs):
            model.train()
            indices = list(range(len(fit_examples))); random.Random(seed + epoch).shuffle(indices)
            for batch in _normalized_batches(fit_examples, normalization, batch_size=4, indices=indices):
                optimizer = getattr(model, "_g2_optimizer", None)
                if optimizer is None:
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3); model._g2_optimizer = optimizer
                optimizer.zero_grad(set_to_none=True)
                prediction = model(torch.stack([row[1] for row in batch]).to(device)); expected = torch.stack([row[2] for row in batch]).to(device)
                dispersion = prediction["dispersion"]
                loss = (0.5 * ((expected - prediction["mean"]) / dispersion).square() + torch.log(dispersion)).mean()
                loss.backward(); optimizer.step(); steps += len(batch)
        final = _development_metrics(model, development_examples, normalization, device=device)
        rows.append({"base_channels": width, "parameter_count": model.parameter_count, "epochs": epochs,
            "fit_sample_count": len(fit_examples), "development_sample_count": len(development_examples), "training_samples": steps,
            "wall_seconds": time.perf_counter() - started, "development_mae_epoch_0": epoch_zero["development_mae"],
            "development_normalized_log_mae_epoch_0": epoch_zero["development_normalized_log_mae"],
            "development_selection_metric_epoch_0": epoch_zero["development_selection_metric"], "development_mae": final["development_mae"],
            "development_normalized_log_mae": final["development_normalized_log_mae"], "development_selection_metric": final["development_selection_metric"],
            "development_selection_metric_final": final["development_selection_metric"], "device": device})
    cheapest = min(rows, key=lambda row: int(row["base_channels"]))
    best = min(rows, key=lambda row: float(row["development_selection_metric_final"]))
    relative_gain = (float(cheapest["development_selection_metric_final"]) - float(best["development_selection_metric_final"])) / float(cheapest["development_selection_metric_final"])
    separated = int(best["base_channels"]) != int(cheapest["base_channels"]) and relative_gain >= minimum_relative_width_separation
    recommended = best if separated else cheapest
    basis = "best final development normalized log MAE exceeds predeclared {:.1%} separation".format(minimum_relative_width_separation) if separated else "widths statistically indistinguishable under predeclared {:.1%} separation; choose cheapest adequate width".format(minimum_relative_width_separation)
    result: dict[str, object] = {"schema_version": "1.0.0", "label": SMOKE_LABEL, "selection_role": "development", "not_k5_adjudication": True,
        "seed_namespace": "sasto-v-g2-capacity-study-v1", "campaign_seed": campaign_seed, "widths": list(widths), "rows": rows,
        "recommended_base_channels": recommended["base_channels"], "recommendation_basis": basis,
        "predeclared_minimum_relative_width_separation": minimum_relative_width_separation, "observed_best_vs_cheapest_relative_gain": relative_gain,
        "normalization_stats_digest": normalization_digest}
    if provenance is not None:
        for key in ("split_manifest_sha256", "archive_sha256", "cohort_manifest_sha256", "cluster_role_manifest_sha256", "source_bundle_sha256"):
            result[key] = _lower_digest(provenance.get(key), "capacity study " + key)
    result["study_digest"] = _digest(result)
    return result


def _read_verified_member(path: Path) -> dict[str, object]:
    try:
        value = json.loads(read_regular_path_snapshot(path, "G2 member manifest").bytes.decode("utf-8"))
    except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SurrogateError("G2 member manifest is unavailable or malformed") from error
    if not isinstance(value, dict) or value.get("manifest_digest") != _digest({key: item for key, item in value.items() if key != "manifest_digest"}):
        raise SurrogateError("G2 member manifest digest mismatch")
    checkpoint = value.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("path"), str) or sha256_file(path.parent / checkpoint["path"]) != checkpoint.get("sha256"):
        raise SurrogateError("G2 member checkpoint digest mismatch")
    return value


def train_certified_ensemble(
    *, output_root: Path, fit: PackedRoleDataset, development: PackedRoleDataset, normalization: Mapping[str, object],
    source_bundle_sha256: str, cache_manifest_sha256: str, member_count: int = 5, max_epochs: int = 20, patience: int = 4,
    base_channels: int = 4, device: str = "cpu", campaign_seed: int = 20260828, ingest_wall_seconds: float = 0.0,
) -> dict[str, object]:
    """Train the digest-bound M-member G2 ensemble using fit and development only."""
    torch, _ = _require_torch()
    if not isinstance(fit, PackedRoleDataset) or fit.role != "fit" or not isinstance(development, PackedRoleDataset) or development.role != "development":
        raise SurrogateRoleError("certified ensemble requires packed fit training and development evaluation only")
    if not 1 <= member_count <= 5 or not 1 <= max_epochs <= 50 or not 1 <= patience <= max_epochs:
        raise SurrogateError("certified ensemble member, epoch, or patience bounds are invalid")
    if device not in {"cpu", "mps"} or (device == "mps" and not torch.backends.mps.is_available()):
        raise SurrogateError("requested certified ensemble device is unavailable")
    for digest, label in ((source_bundle_sha256, "source bundle"), (cache_manifest_sha256, "cache manifest"),
                          (str(normalization.get("stats_digest")), "normalization stats")):
        _lower_digest(digest, label)
    if fit.provenance != development.provenance:
        raise SurrogateError("certified ensemble role provenance differs")
    root = Path(output_root); members_dir = root / "members"
    campaign = {"schema_version": "1.0.0", "label": "CERTIFIED_G2_ENSEMBLE", "member_count": member_count, "max_epochs": max_epochs,
        "patience": patience, "base_channels": base_channels, "campaign_seed": campaign_seed, "seed_namespace": SEED_NAMESPACE,
        "source_bundle_sha256": source_bundle_sha256, "cache_manifest_sha256": cache_manifest_sha256,
        "normalization_stats_digest": str(normalization["stats_digest"]), **fit.provenance,
        "fit_sample_ids": list(fit.sample_ids), "development_sample_ids": list(development.sample_ids), "data_roles": ["fit", "development"]}
    campaign["campaign_digest"] = _digest(campaign)
    if root.exists():
        prior = _packed_cache_manifest(root) if False else None
        try:
            existing = json.loads(read_regular_path_snapshot(root / "campaign-manifest.json", "G2 campaign manifest").bytes.decode("utf-8"))
        except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SurrogateError("certified ensemble root is incomplete and cannot resume") from error
        if existing != campaign:
            raise SurrogateError("certified ensemble root input digest binding does not match resume request")
    else:
        try:
            with open_new_artifact_root(root) as root_fd:
                os.mkdir("members", dir_fd=root_fd)
        except (ManifestVerificationError, OSError) as error:
            raise SurrogateError("certified ensemble root must be new or resume-verifiable") from error
        _write_new_json(root / "campaign-manifest.json", campaign, "G2 campaign manifest")
    members: list[dict[str, object]] = []
    for member_index in range(member_count):
        manifest_path = members_dir / "member-{:02d}.json".format(member_index)
        if manifest_path.exists():
            manifest = _read_verified_member(manifest_path)
            if manifest.get("campaign_digest") != campaign["campaign_digest"] or manifest.get("member_index") != member_index:
                raise SurrogateError("existing G2 member does not bind this campaign")
            members.append({"member_index": member_index, "manifest_sha256": sha256_file(manifest_path), "checkpoint_sha256": manifest["checkpoint"]["sha256"], "final_metrics": manifest["final_metrics"]})
            continue
        seed = deterministic_seed(SEED_NAMESPACE, campaign_seed, member_index); _configure_determinism(torch, seed)
        if device == "mps": torch.mps.empty_cache()
        model = DenseSurrogateCNN(base_channels=base_channels).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        epoch_zero = _development_metrics(model, development, normalization, device=device)
        losses: list[float] = []; best_metric = float("inf"); best_epoch = 0; stale = 0; best_state = None; steps = 0
        started = time.perf_counter()
        for epoch in range(1, max_epochs + 1):
            model.train(); indices = list(range(len(fit))); random.Random(seed + epoch).shuffle(indices)
            for batch in _normalized_batches(fit, normalization, batch_size=4, indices=indices):
                optimizer.zero_grad(set_to_none=True); prediction = model(torch.stack([row[1] for row in batch]).to(device)); expected = torch.stack([row[2] for row in batch]).to(device)
                dispersion = prediction["dispersion"]; loss = (0.5 * ((expected - prediction["mean"]) / dispersion).square() + torch.log(dispersion)).mean()
                loss.backward(); optimizer.step(); steps += len(batch)
            metrics = _development_metrics(model, development, normalization, device=device); current = float(metrics["development_selection_metric"]); losses.append(current)
            if current < best_metric:
                best_metric = current; best_epoch = epoch; stale = 0; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience: break
        training_wall = time.perf_counter() - started
        if best_state is None: raise SurrogateError("certified ensemble produced no development metric")
        model.load_state_dict(best_state); final = _development_metrics(model, development, normalization, device=device)
        checkpoint_path = members_dir / "member-{:02d}.pt".format(member_index)
        torch.save({"state_dict": model.state_dict(), "target_names": TARGET_NAMES, "base_channels": base_channels, "seed": seed}, checkpoint_path)
        checkpoint_sha = sha256_file(checkpoint_path)
        ledger = {"input_wall_seconds": ingest_wall_seconds, "input_sample_count": len(fit) + len(development),
            "input_samples_per_second": (len(fit) + len(development)) / ingest_wall_seconds if ingest_wall_seconds else 0.0,
            "training_wall_seconds": training_wall, "training_samples": steps,
            "training_samples_per_second": steps / training_wall if training_wall else 0.0, "epochs": len(losses), "device": device}
        manifest: dict[str, object] = {"schema_version": "1.0.0", "label": "CERTIFIED_G2_ENSEMBLE", "campaign_digest": campaign["campaign_digest"],
            "member_index": member_index, "campaign_seed": campaign_seed, "seed_namespace": SEED_NAMESPACE, "seed": seed,
            "source_bundle_sha256": source_bundle_sha256, "cache_manifest_sha256": cache_manifest_sha256, "normalization_stats_digest": normalization["stats_digest"],
            **fit.provenance, "data_role": "fit", "development_role": "development", "fit_sample_count": len(fit), "development_sample_count": len(development),
            "epoch_count": len(losses), "selected_epoch": best_epoch, "base_channels": base_channels, "parameter_count": model.parameter_count,
            "checkpoint": {"path": checkpoint_path.name, "sha256": checkpoint_sha}, "development_mae_epoch_0": epoch_zero["development_mae"],
            "development_normalized_log_mae_epoch_0": epoch_zero["development_normalized_log_mae"], "development_mae_final": final["development_mae"],
            "development_normalized_log_mae_final": final["development_normalized_log_mae"], "final_metrics": final, "compute_ledger": ledger}
        manifest["manifest_digest"] = _digest(manifest); manifest_sha = _write_new_json(manifest_path, manifest, "G2 member manifest")
        members.append({"member_index": member_index, "manifest_sha256": manifest_sha, "checkpoint_sha256": checkpoint_sha, "final_metrics": final})
    summary: dict[str, object] = {"schema_version": "1.0.0", "label": "CERTIFIED_G2_ENSEMBLE", "campaign_digest": campaign["campaign_digest"], "member_count": member_count,
        "members": members, "k5_not_adjudicated": True, "ensemble_mean_final_development_mae": {name: float(np.mean([member["final_metrics"]["development_mae"][name] for member in members])) for name in TARGET_NAMES},
        "ensemble_member_mae_std": {name: float(np.std([member["final_metrics"]["development_mae"][name] for member in members])) for name in TARGET_NAMES}}
    summary["summary_digest"] = _digest(summary)
    summary_path = root / "ensemble-summary.json"
    if summary_path.exists():
        existing = json.loads(read_regular_path_snapshot(summary_path, "G2 ensemble summary").bytes.decode("utf-8"))
        if existing != summary: raise SurrogateError("existing G2 ensemble summary does not match verified members")
        summary_sha = sha256_file(summary_path)
    else:
        summary_sha = _write_new_json(summary_path, summary, "G2 ensemble summary")
    return {"member_count": member_count, "members": members, "summary_sha256": summary_sha, "output_root": str(root)}


def open_role_dataset(
    *,
    role: str,
    split_manifest: Path,
    expected_split_sha256: str,
    archive: Path,
    expected_archive_sha256: str,
    g1b_root: Path,
    expected_cohort_manifest_sha256: str,
    expected_cluster_role_manifest_sha256: str,
    calibration_pass: bool = False,
) -> RoleDataset:
    """Open one explicitly allowed G2 role with all source anchors verified."""
    if role == "confirmation":
        raise SurrogateRoleError("confirmation is sealed and cannot be opened by G2")
    if role not in {"fit", "development", "calibration"}:
        raise SurrogateRoleError("G2 role must be fit, development, or calibration")
    if role == "calibration" and not calibration_pass:
        raise SurrogateRoleError("calibration may be opened only for the calibration pass")
    if role != "calibration" and calibration_pass:
        raise SurrogateRoleError("calibration pass flag is valid only for calibration")
    split, split_sha = _snapshot_json(split_manifest, expected_split_sha256, "split manifest")
    cohort, cohort_sha = _snapshot_json(Path(g1b_root) / "cohort-manifest.json", expected_cohort_manifest_sha256, "cohort manifest")
    clusters, cluster_sha = _snapshot_json(Path(g1b_root) / "cluster-role-manifest.json", expected_cluster_role_manifest_sha256, "cluster role manifest")
    expected_archive_sha256 = _lower_digest(expected_archive_sha256, "archive anchor")
    try:
        archive_sha = sha256_file(archive)
    except ManifestVerificationError as error:
        raise SurrogateError("archive is unavailable") from error
    if archive_sha != expected_archive_sha256:
        raise SurrogateError("archive sha256 mismatch")
    admitted = _cohort_roles(cohort, clusters, split)
    sample_ids = [sample_id for sample_id in _role_ids(split, role) if sample_id in admitted]
    if not sample_ids:
        raise SurrogateError("role has no G1b-eligible samples")
    return RoleDataset(role=role, sample_ids=sample_ids, archive=Path(archive), g1b_root=Path(g1b_root), provenance={
        "split_manifest_sha256": split_sha, "archive_sha256": archive_sha,
        "cohort_manifest_sha256": cohort_sha, "cluster_role_manifest_sha256": cluster_sha,
    })


def _local_import_closure(source_root: Path, entry: str) -> tuple[str, ...]:
    """Return the AST-resolved relative-import closure for a local module."""
    pending = [entry]
    seen: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in seen:
            continue
        try:
            source = read_regular_path_snapshot(source_root / relative, "G2 local source").bytes
            tree = ast.parse(source, filename=relative)
        except (ManifestVerificationError, SyntaxError) as error:
            raise SurrogateError("cannot parse G2 local import closure") from error
        seen.add(relative)
        module_parts = Path(relative).with_suffix("").parts
        if module_parts[:2] != ("src", "sasto"):
            raise SurrogateError("G2 local source is outside the sasto package")
        package = module_parts[1:-1]
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level < 1:
                continue
            if node.level > len(package):
                raise SurrogateError("G2 relative import escapes the sasto package")
            base = package[:len(package) - node.level + 1]
            module = tuple(node.module.split(".")) if node.module else ()
            candidates = [base + module]
            if not module:
                candidates.extend(base + (alias.name,) for alias in node.names)
            for candidate in candidates:
                local = Path("src", *candidate).with_suffix(".py").as_posix()
                if (source_root / local).is_file():
                    pending.append(local)
    return tuple(sorted(seen))


def surrogate_source_bundle(root: Path | None = None) -> tuple[dict[str, str], str]:
    """Hash configuration plus the AST-proven transitive local G2 closure."""
    source_root = Path(root) if root is not None else Path(__file__).parents[2]
    paths = SOURCE_BUNDLE_CONFIG_PATHS + _local_import_closure(source_root, "src/sasto/surrogate.py")
    files: dict[str, str] = {}
    try:
        for relative in paths:
            files[relative] = sha256_file(source_root / relative)
    except ManifestVerificationError as error:
        raise SurrogateError("cannot hash G2 source bundle") from error
    return files, _digest([{"path": path, "sha256": files[path]} for path in sorted(files)])


def run_anchored_smoke(
    *, output_root: Path, split_manifest: Path, expected_split_sha256: str, archive: Path, expected_archive_sha256: str,
    g1b_root: Path, expected_cohort_manifest_sha256: str, expected_cluster_role_manifest_sha256: str,
    sample_count: int = 64, member_count: int = 5, epochs: int = 1, base_channels: int = 16, device: str = "cpu",
    campaign_seed: int = 20260828,
) -> dict[str, object]:
    """Run the only allowed G2 training path: bounded, anchored, development smoke."""
    if Path(output_root).exists():
        raise SurrogateError("anchored smoke artifact root must be new")
    development = role_subset(open_role_dataset(
        role="development", split_manifest=split_manifest, expected_split_sha256=expected_split_sha256, archive=archive,
        expected_archive_sha256=expected_archive_sha256, g1b_root=g1b_root,
        expected_cohort_manifest_sha256=expected_cohort_manifest_sha256,
        expected_cluster_role_manifest_sha256=expected_cluster_role_manifest_sha256), sample_count=sample_count)
    fit = role_subset(open_role_dataset(
        role="fit", split_manifest=split_manifest, expected_split_sha256=expected_split_sha256, archive=archive,
        expected_archive_sha256=expected_archive_sha256, g1b_root=g1b_root,
        expected_cohort_manifest_sha256=expected_cohort_manifest_sha256,
        expected_cluster_role_manifest_sha256=expected_cluster_role_manifest_sha256), sample_count=sample_count)
    normalization = compute_fit_normalization(fit)
    input_started = time.perf_counter()
    development_examples = normalized_tensor_examples(development, normalization)
    development_input_wall_seconds = time.perf_counter() - input_started
    fit_examples = normalized_tensor_examples(fit, normalization)
    files, bundle_sha = surrogate_source_bundle()
    provenance = dict(development.provenance)
    provenance["source_bundle_sha256"] = bundle_sha
    study = capacity_study(fit_examples=fit_examples, development_examples=development_examples, normalization=normalization,
                           device=device, campaign_seed=campaign_seed, provenance=provenance)
    result = train_smoke_ensemble(
        output_root=output_root, examples=development_examples, target_names=TARGET_NAMES,
        normalization_stats_digest=str(normalization["stats_digest"]), source_bundle_sha256=bundle_sha,
        split_sha256=development.provenance["split_manifest_sha256"], archive_sha256=development.provenance["archive_sha256"],
        cohort_manifest_sha256=development.provenance["cohort_manifest_sha256"], member_count=member_count, epochs=epochs,
        base_channels=base_channels, device=device, campaign_seed=campaign_seed, data_role="development",
        input_wall_seconds=development_input_wall_seconds,
    )
    root = Path(output_root)
    normalization_sha = _write_new_json(root / "normalization-stats.json", normalization, "smoke normalization statistics")
    study_sha = _write_new_json(root / "capacity-study.json", study, "capacity study")
    result.update({"normalization_stats_sha256": normalization_sha, "capacity_study_sha256": study_sha,
                   "source_bundle_files": files, "capacity_study": study})
    return result


def _read_smoke_summary(path: Path) -> dict[str, object]:
    try:
        snapshot = read_regular_path_snapshot(path, "smoke summary")
        summary = json.loads(snapshot.bytes.decode("utf-8"))
    except (ManifestVerificationError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SurrogateError("smoke summary is unavailable or malformed") from error
    if not isinstance(summary, dict) or summary.get("label") != "SMOKE_ONLY_NONPROMOTABLE":
        raise SurrogateError("smoke summary is not explicitly nonpromotable")
    if summary.get("summary_digest") != _digest({key: value for key, value in summary.items() if key != "summary_digest"}):
        raise SurrogateError("smoke summary digest mismatch")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="G2 dense surrogate ensemble (confirmation sealed)")
    parser.add_argument("--mode", choices=("prepare-stats", "train", "predict", "report"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-only-nonpromotable", action="store_true")
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--campaign-seed", type=int, default=20260828)
    parser.add_argument("--smoke-sample-count", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--expected-split-manifest-sha256")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--g1b-root", type=Path)
    parser.add_argument("--expected-cohort-manifest-sha256")
    parser.add_argument("--expected-cluster-role-manifest-sha256")
    args = parser.parse_args()
    try:
        if args.mode == "prepare-stats":
            if not all((args.split_manifest, args.expected_split_manifest_sha256, args.archive, args.expected_archive_sha256, args.g1b_root,
                        args.expected_cohort_manifest_sha256, args.expected_cluster_role_manifest_sha256)):
                raise SurrogateError("prepare-stats requires every anchored split, archive, and G1b manifest input")
            fit = open_role_dataset(role="fit", split_manifest=args.split_manifest, expected_split_sha256=args.expected_split_manifest_sha256,
                archive=args.archive, expected_archive_sha256=args.expected_archive_sha256, g1b_root=args.g1b_root,
                expected_cohort_manifest_sha256=args.expected_cohort_manifest_sha256,
                expected_cluster_role_manifest_sha256=args.expected_cluster_role_manifest_sha256)
            result = compute_fit_normalization(fit)
            _write_new_json(args.output, result, "normalization statistics")
        elif args.mode == "train":
            if not args.smoke_only_nonpromotable:
                raise SurrogateError("G1b independent certification is pending: train requires --smoke-only-nonpromotable (SMOKE_ONLY_NONPROMOTABLE); full fitting is blocked")
            if not all((args.split_manifest, args.expected_split_manifest_sha256, args.archive, args.expected_archive_sha256, args.g1b_root,
                        args.expected_cohort_manifest_sha256, args.expected_cluster_role_manifest_sha256)):
                raise SurrogateError("anchored smoke train requires every split, archive, and G1b manifest input")
            result = run_anchored_smoke(
                output_root=args.output, split_manifest=args.split_manifest, expected_split_sha256=args.expected_split_manifest_sha256,
                archive=args.archive, expected_archive_sha256=args.expected_archive_sha256, g1b_root=args.g1b_root,
                expected_cohort_manifest_sha256=args.expected_cohort_manifest_sha256,
                expected_cluster_role_manifest_sha256=args.expected_cluster_role_manifest_sha256,
                sample_count=args.smoke_sample_count, member_count=args.members, epochs=args.epochs,
                base_channels=args.base_channels, device=args.device, campaign_seed=args.campaign_seed,
            )
        else:
            summary = _read_smoke_summary(args.output / "smoke-summary.json")
            result = {"mode": args.mode, "label": summary["label"], "member_count": summary["member_count"],
                      "sample_count": summary["sample_count"], "summary_digest": summary["summary_digest"]}
    except (SurrogateError, OSError) as error:
        print("REJECTED: {}".format(error), file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
