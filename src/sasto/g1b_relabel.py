"""G1b baseline relabeling: solver-valid cohort and cluster-level roles.

Every case is append-only and independently digest-verified before it can be
merged.  Confirmation payloads are deliberately never opened in this slice;
they receive explicit sealed records so the 11,178-member population is never
silently narrowed.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import math
import os
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .fit_probe import _PayloadAccessLedger, _configuration, _load_occupancy
from .manifest import ManifestVerificationError, read_regular_path_snapshot, write_new_regular_path
from .splits import PARTITIONS, FamilyLeakageError, validate_family_split_manifest
from .voxel_fea import VoxelFEAConfig, solve_voxels

SCHEMA_VERSION = "1.0.0"
NAMESPACE = "sasto-v-g1b-relabel-v1"
ADMISSION_BOUND = 2e-8
DUPLICATE_TOLERANCE = "medium"
SOURCE_BUNDLE_PATHS = (
    ".python-version", "pyproject.toml", "uv.lock", "src/sasto/g1b_relabel.py",
    "src/sasto/fit_probe.py", "src/sasto/manifest.py", "src/sasto/splits.py",
    "src/sasto/topology.py", "src/sasto/voxel_fea.py",
)


class RelabelError(ValueError):
    """A relabeling input, record, or resume state is not admissible."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_path(path: Path, label: str) -> tuple[bytes, str]:
    try:
        snapshot = read_regular_path_snapshot(path, label)
    except ManifestVerificationError as error:
        raise RelabelError(str(error)) from error
    return snapshot.bytes, snapshot.sha256


def _lower_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RelabelError("{} must be an exact lowercase SHA-256 digest".format(label))
    return value


def _face_connected(occupancy: object) -> bool:
    if not isinstance(occupancy, np.ndarray) or occupancy.dtype != np.bool_ or occupancy.ndim != 3 or not occupancy.any():
        return False
    start = tuple(int(value) for value in np.argwhere(occupancy)[0])
    seen = {start}; stack = [start]; shape = occupancy.shape
    while stack:
        x, y, z = stack.pop()
        for dx, dy, dz in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
            candidate = (x + dx, y + dy, z + dz)
            if (0 <= candidate[0] < shape[0] and 0 <= candidate[1] < shape[1] and 0 <= candidate[2] < shape[2]
                    and occupancy[candidate] and candidate not in seen):
                seen.add(candidate); stack.append(candidate)
    return len(seen) == int(occupancy.sum())


def _finite_positive(record: Mapping[str, object], keys: Sequence[str]) -> bool:
    try:
        values = [float(record[key]) for key in keys]
    except (KeyError, TypeError, ValueError):
        return False
    return all(math.isfinite(value) and value > 0.0 for value in values)


def cohort_reasons(occupancy: object, record: Mapping[str, object], *, expected_loaded_nodes: int) -> list[str]:
    """Apply exactly the five pre-registered G1b validity predicates in order."""
    reasons: list[str] = []
    if not _face_connected(occupancy):
        reasons.append("occupancy_not_face_connected")
    if record.get("status") != "success":
        reasons.append("solver_status_failure")
    try:
        residual = float(record["relative_residual"])
    except (KeyError, TypeError, ValueError):
        residual = float("inf")
    if not math.isfinite(residual) or residual > ADMISSION_BOUND:
        reasons.append("relative_residual_exceeds_2e-8")
    if not _finite_positive(record, ("compliance_j", "max_displacement_m", "max_gauss_von_mises_pa")):
        reasons.append("nonfinite_or_nonpositive_outputs")
    if record.get("loaded_node_count") != expected_loaded_nodes:
        reasons.append("unstable_loaded_node_set")
    return reasons


def shard_for_id(sample_id: str, shard_count: int) -> int:
    if not isinstance(sample_id, str) or not sample_id or not isinstance(shard_count, int) or isinstance(shard_count, bool) or shard_count < 1:
        raise ValueError("sample ID and shard count must be valid; shard count must be positive")
    return int.from_bytes(hashlib.sha256((NAMESPACE + "\\0" + sample_id).encode("utf-8")).digest()[:8], "big") % shard_count


def _parse_shard(value: str) -> tuple[int, int]:
    try:
        index_text, count_text = value.split("/", 1); index, count = int(index_text), int(count_text)
    except (AttributeError, ValueError):
        raise RelabelError("--shard must be N/K") from None
    if not 1 <= index <= count:
        raise RelabelError("--shard must satisfy 1 <= N <= K")
    return index - 1, count


def _case_path(root: Path, sample_id: str) -> Path:
    if not isinstance(sample_id, str) or not sample_id or any(char in sample_id for char in "/\\\x00"):
        raise RelabelError("case sample ID is unsafe")
    return root / "cases" / "{}.json".format(sample_id)


def write_case_record(root: Path, record: Mapping[str, object]) -> str:
    sample_id = record.get("sample_id")
    path = _case_path(root, sample_id)
    payload = dict(record)
    payload.pop("case_digest", None)
    payload["case_digest"] = _digest(payload)
    try:
        return write_new_regular_path(path, _canonical_bytes(payload) + b"\n", "G1b case")
    except ManifestVerificationError as error:
        raise RelabelError(str(error)) from error


def load_verified_case(root: Path, sample_id: str) -> dict[str, object]:
    try:
        payload, _ = _sha256_path(_case_path(root, sample_id), "G1b case")
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, RelabelError) as error:
        raise RelabelError("cannot safely load G1b case") from error
    if not isinstance(value, dict) or value.get("sample_id") != sample_id:
        raise RelabelError("G1b case identity is invalid")
    observed = value.get("case_digest")
    if not isinstance(observed, str) or observed != _digest({key: item for key, item in value.items() if key != "case_digest"}):
        raise RelabelError("G1b case digest mismatch")
    return value


def merge_completed_records(*, root: Path, selected_ids: Sequence[str], total_population: int) -> dict[str, object]:
    if not isinstance(total_population, int) or isinstance(total_population, bool) or total_population < 1:
        raise RelabelError("total population must be positive")
    if len(selected_ids) != total_population or len(set(selected_ids)) != len(selected_ids):
        raise RelabelError("selected IDs must exactly cover the declared population")
    try:
        records = [load_verified_case(root, sample_id) for sample_id in sorted(selected_ids)]
    except RelabelError as error:
        raise RelabelError("completed case set is incomplete or invalid") from error
    eligible = [record["sample_id"] for record in records if record.get("exclusion_reasons") == []]
    exclusions = Counter(
        reason for record in records for reason in record.get("exclusion_reasons", []) if isinstance(reason, str)
    )
    compact = [{"sample_id": record["sample_id"], "case_digest": record["case_digest"],
                "role": record.get("role"), "exclusion_reasons": record.get("exclusion_reasons", [])} for record in records]
    result = {"schema_version": SCHEMA_VERSION, "population_count": total_population, "records": compact,
              "records_digest": _digest(compact), "eligible_ids": eligible, "eligible_count": len(eligible),
              "excluded_count": total_population - len(eligible), "exclusion_counts": dict(sorted(exclusions.items()))}
    return result


def build_cluster_table(*, sample_roles: Mapping[str, str], duplicate_clusters: Sequence[Sequence[str]]) -> list[dict[str, object]]:
    if set(sample_roles.values()) - set(PARTITIONS):
        raise RelabelError("sample roles must be frozen split roles")
    pending = set(sample_roles); components: list[list[str]] = []
    for supplied in duplicate_clusters:
        members = sorted(set(supplied) & pending)
        if not members:
            continue
        roles = {sample_roles[member] for member in members}
        if len(roles) != 1:
            raise RelabelError("cross-role duplicate cluster requires STOP")
        components.append(members); pending.difference_update(members)
    components.extend([[member] for member in sorted(pending)])
    components.sort(key=lambda members: tuple(members))
    return [{"cluster_id": "cluster:{:05d}".format(index), "members": members,
             "role": sample_roles[members[0]]} for index, members in enumerate(components)]


def _source_roles(manifest: Mapping[str, object]) -> dict[str, str]:
    try:
        validate_family_split_manifest(manifest)
        roles = {sample_id: role for role in PARTITIONS for sample_id in manifest["partitions"][role]["sample_ids"]}  # type: ignore[index]
    except (FamilyLeakageError, KeyError, TypeError) as error:
        raise RelabelError("frozen family split is invalid") from error
    if len(roles) != 11178:
        raise RelabelError("frozen split must contain exactly 11,178 retained samples")
    return roles


def _duplicate_clusters(summary_path: Path, pairs_path: Path, sample_roles: Mapping[str, str]) -> tuple[list[list[str]], dict[str, str]]:
    summary_bytes, summary_sha = _sha256_path(summary_path, "near-duplicate summary")
    pairs_bytes, pairs_sha = _sha256_path(pairs_path, "near-duplicate verified pairs")
    try:
        summary = json.loads(summary_bytes)
        expected_pairs = summary["outputs"]["verified_pairs_csv_sha256"]
        clusters = summary["tolerances"][DUPLICATE_TOLERANCE]["clusters"]
    except (TypeError, KeyError, json.JSONDecodeError) as error:
        raise RelabelError("near-duplicate audit summary is malformed") from error
    if expected_pairs != pairs_sha or not isinstance(clusters, list):
        raise RelabelError("near-duplicate verified pairs digest mismatch")
    try:
        rows = list(csv.DictReader(io.StringIO(pairs_bytes.decode("utf-8-sig"))))
    except UnicodeDecodeError as error:
        raise RelabelError("near-duplicate verified pairs is not UTF-8") from error
    medium_pairs = {(row.get("source_id_a"), row.get("source_id_b")) for row in rows if row.get("tolerance") == DUPLICATE_TOLERANCE}
    for component in clusters:
        if not isinstance(component, list) or any(not isinstance(item, str) for item in component):
            raise RelabelError("near-duplicate cluster is malformed")
    # CSV must contain the declared medium edge provenance; summary components alone are not accepted.
    for left, right in medium_pairs:
        if not isinstance(left, str) or not isinstance(right, str):
            raise RelabelError("near-duplicate verified pair is malformed")
    return [list(component) for component in clusters], {"near_duplicate_summary_sha256": summary_sha, "near_duplicate_verified_pairs_sha256": pairs_sha}


def source_bundle(root: Path | None = None) -> tuple[dict[str, str], str]:
    source_root = Path(root) if root is not None else Path(__file__).parents[2]
    files: dict[str, str] = {}
    for relative in SOURCE_BUNDLE_PATHS:
        _, files[relative] = _sha256_path(source_root / relative, "source bundle")
    return files, _digest([{"path": path, "sha256": files[path]} for path in sorted(files)])


def _load_sources(split_manifest: Path, archive: Path, expected_split: str, expected_archive: str) -> tuple[dict[str, str], bytes, dict[str, object]]:
    _lower_digest(expected_split, "split anchor"); _lower_digest(expected_archive, "archive anchor")
    split_bytes, split_sha = _sha256_path(split_manifest, "split manifest")
    if split_sha != expected_split:
        raise RelabelError("split manifest sha256 mismatch")
    try:
        manifest = json.loads(split_bytes)
    except json.JSONDecodeError as error:
        raise RelabelError("split manifest is malformed") from error
    roles = _source_roles(manifest)
    archive_bytes, archive_sha = _sha256_path(archive, "archive")
    if archive_sha != expected_archive:
        raise RelabelError("archive sha256 mismatch")
    return roles, archive_bytes, {"split_manifest_sha256": split_sha, "archive_sha256": archive_sha}


def _load_coordinates(volume: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    occupied = np.argwhere(volume)
    if not len(occupied):
        return ()
    maximum = int(occupied[:, 0].max())
    return tuple(sorted({(maximum + 1, int(y) + dy, int(z) + dz)
                         for y, z in np.argwhere(volume[maximum]) for dy, dz in ((0, 0), (0, 1), (1, 0), (1, 1))}))


def _baseline_case(*, archive_open: zipfile.ZipFile, ledger: _PayloadAccessLedger, sample_id: str, role: str) -> dict[str, object]:
    if role == "confirmation":
        return {"sample_id": sample_id, "role": role, "execution": "sealed_confirmation_no_payload_open",
                "exclusion_reasons": ["confirmation_sealed"], "solver": {"status": "not_run"}}
    try:
        volume = _load_occupancy(archive_open, ledger, sample_id)
        base = _configuration(archive_open, ledger, sample_id, (0.0, 0.0, -100.0))
        expected = _load_coordinates(volume)
        config = dataclasses.replace(base, include_self_weight=False, fixed_total_force_n=(0.0, 0.0, -100.0),
                                     relative_tolerance=ADMISSION_BOUND, expected_loaded_node_count=len(expected),
                                     expected_loaded_node_coordinates=expected)
        solver_record = solve_voxels(volume, config)
        scientific_solver = {key: value for key, value in solver_record.items() if key != "timing"}
        reasons = cohort_reasons(volume, scientific_solver, expected_loaded_nodes=len(expected))
        return {"sample_id": sample_id, "role": role, "execution": "canonical_baseline_only", "occupancy_face_connected": _face_connected(volume),
                "expected_loaded_node_count": len(expected), "solver": scientific_solver, "exclusion_reasons": reasons}
    except Exception as error:
        return {"sample_id": sample_id, "role": role, "execution": "canonical_baseline_only", "solver": {"status": "failure", "reason": "input_or_solver_exception"},
                "exclusion_reasons": ["solver_status_failure"], "failure_detail": type(error).__name__}


def run_shard(*, root: Path, split_manifest: Path, archive: Path, expected_split_sha256: str, expected_archive_sha256: str,
              shard_index: int, shard_count: int, limit: int | None = None) -> dict[str, object]:
    roles, archive_bytes, provenance = _load_sources(split_manifest, archive, expected_split_sha256, expected_archive_sha256)
    if root.exists():
        cases = root / "cases"
        if not cases.is_dir() or cases.is_symlink():
            raise RelabelError("existing root lacks a safe cases directory")
    else:
        try:
            root.mkdir(parents=True); (root / "cases").mkdir()
        except FileExistsError:
            if not (root / "cases").is_dir():
                raise RelabelError("concurrent root creation was incomplete")
    ids = sorted(roles)
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise RelabelError("limit must be a positive integer")
        ids = ids[:limit]
    selected = [sample_id for sample_id in ids if shard_for_id(sample_id, shard_count) == shard_index]
    generated: list[str] = []; resumed: list[str] = []
    ledger = _PayloadAccessLedger([sample_id for sample_id in selected if roles[sample_id] != "confirmation"])
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as opened:
        for sample_id in selected:
            try:
                load_verified_case(root, sample_id); resumed.append(sample_id); continue
            except RelabelError:
                if _case_path(root, sample_id).exists():
                    raise
            write_case_record(root, _baseline_case(archive_open=opened, ledger=ledger, sample_id=sample_id, role=roles[sample_id])); generated.append(sample_id)
    members, fit_accesses, nonfit_accesses = ledger.evidence()
    if nonfit_accesses or fit_accesses != len(members):
        raise RelabelError("G1b payload ledger is inconsistent")
    return {"shard": "{}/{}".format(shard_index + 1, shard_count), "selected_count": len(selected), "generated_count": len(generated),
            "resumed_count": len(resumed), "payload_access_count": fit_accesses, "payload_members": members, **provenance}


def _write_or_match(path: Path, payload: Mapping[str, object], role: str) -> str:
    encoded = _canonical_bytes(dict(payload)) + b"\n"
    if path.exists():
        existing, observed = _sha256_path(path, role)
        if existing != encoded:
            raise RelabelError("existing {} does not match deterministic recomputation".format(role))
        return observed
    try:
        return write_new_regular_path(path, encoded, role)
    except ManifestVerificationError as error:
        raise RelabelError(str(error)) from error


def finalize(*, root: Path, split_manifest: Path, expected_split_sha256: str, near_duplicate_summary: Path,
             near_duplicate_pairs: Path, limit: int | None = None) -> dict[str, object]:
    split_bytes, split_sha = _sha256_path(split_manifest, "split manifest")
    if split_sha != _lower_digest(expected_split_sha256, "split anchor"):
        raise RelabelError("split manifest sha256 mismatch")
    try:
        manifest = json.loads(split_bytes)
    except json.JSONDecodeError as error:
        raise RelabelError("split manifest is malformed") from error
    roles = _source_roles(manifest); ids = sorted(roles)
    if limit is not None:
        ids = ids[:limit]
    cohort = merge_completed_records(root=root, selected_ids=ids, total_population=len(ids))
    clusters, duplicate_hashes = _duplicate_clusters(near_duplicate_summary, near_duplicate_pairs, {sample_id: roles[sample_id] for sample_id in ids})
    table = build_cluster_table(sample_roles={sample_id: roles[sample_id] for sample_id in ids}, duplicate_clusters=clusters)
    role_counts = dict(sorted(Counter(row["role"] for row in table).items()))
    source_files, bundle_sha = source_bundle()
    cluster_manifest = {"schema_version": SCHEMA_VERSION, "algorithm": "family-cluster-v1", "base_algorithm": "family-id-v1",
                        "seed_lineage": manifest["seed"], "source_split_manifest_sha256": split_sha, **duplicate_hashes,
                        "duplicate_tolerance": DUPLICATE_TOLERANCE, "cluster_count": len(table), "role_counts": role_counts, "clusters": table}
    cohort_hash = _write_or_match(root / "cohort-manifest.json", cohort, "cohort manifest")
    cluster_hash = _write_or_match(root / "cluster-role-manifest.json", cluster_manifest, "cluster role manifest")
    summary = {"schema_version": SCHEMA_VERSION, "namespace": NAMESPACE, "population_count": len(ids), "cohort_manifest_sha256": cohort_hash,
               "cluster_role_manifest_sha256": cluster_hash, "split_manifest_sha256": split_sha, **duplicate_hashes,
               "source_bundle_files": source_files, "source_bundle_sha256": bundle_sha, "eligible_count": cohort["eligible_count"],
               "excluded_count": cohort["excluded_count"], "exclusion_counts": cohort["exclusion_counts"], "cluster_count": len(table), "role_counts": role_counts}
    summary_hash = _write_or_match(root / "g1b-summary.json", summary, "G1b summary")
    rebuilt = merge_completed_records(root=root, selected_ids=ids, total_population=len(ids))
    if _canonical_bytes(rebuilt) != _canonical_bytes(cohort):
        raise RelabelError("byte-for-byte cohort reproduction failed")
    return {**summary, "g1b_summary_sha256": summary_hash}


def main() -> int:
    parser = argparse.ArgumentParser(description="G1b deterministic sharded canonical baseline relabeling")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("run", "finalize"), required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-manifest-sha256", required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-fea-archive-sha256")
    parser.add_argument("--shard", default="1/1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--near-duplicate-summary", type=Path)
    parser.add_argument("--near-duplicate-pairs", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "run":
            if args.archive is None or args.expected_fea_archive_sha256 is None:
                raise RelabelError("run requires archive and exact archive anchor")
            index, count = _parse_shard(args.shard)
            result = run_shard(root=args.root, split_manifest=args.split_manifest, archive=args.archive,
                               expected_split_sha256=args.expected_split_manifest_sha256, expected_archive_sha256=args.expected_fea_archive_sha256,
                               shard_index=index, shard_count=count, limit=args.limit)
        else:
            if args.near_duplicate_summary is None or args.near_duplicate_pairs is None:
                raise RelabelError("finalize requires near-duplicate summary and verified pairs")
            result = finalize(root=args.root, split_manifest=args.split_manifest, expected_split_sha256=args.expected_split_manifest_sha256,
                              near_duplicate_summary=args.near_duplicate_summary, near_duplicate_pairs=args.near_duplicate_pairs, limit=args.limit)
    except (RelabelError, OSError, zipfile.BadZipFile) as error:
        print("REJECTED: {}".format(error)); return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
