"""Versioned, fail-closed run-manifest construction and SHA-256 verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .targets import TargetRegistry


SCHEMA_VERSION = "1.0.0"


class ManifestVerificationError(ValueError):
    """The artifact cannot be admitted as a complete SASTO-V evidence record."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_records(files: Mapping[str, Path]) -> list:
    records = []
    for logical_id, path_like in sorted(files.items()):
        path = Path(path_like).resolve()
        if not path.is_file():
            raise ManifestVerificationError("declared file is missing: {}".format(path))
        records.append({"logical_id": logical_id, "path": str(path), "sha256": sha256_file(path)})
    return records


def build_run_manifest(
    *,
    run_id: str,
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
    targets: TargetRegistry,
    split_sha256: str,
) -> dict:
    """Build a complete record after all declared files already exist."""
    if not run_id:
        raise ManifestVerificationError("run_id must be non-empty")
    if len(split_sha256) != 64 or any(char not in "0123456789abcdef" for char in split_sha256):
        raise ManifestVerificationError("split_sha256 must be a lowercase SHA-256 digest")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "complete",
        "targets": targets.as_list(),
        "split": {"algorithm": "family-id-v1", "sha256": split_sha256},
        "inputs": _file_records(inputs),
        "outputs": _file_records(outputs),
    }


def _verify_records(records: object, role: str) -> None:
    if not isinstance(records, list) or not records:
        raise ManifestVerificationError("{} must be a non-empty list".format(role))
    logical_ids = set()
    paths = set()
    for record in records:
        if not isinstance(record, dict):
            raise ManifestVerificationError("{} record must be an object".format(role))
        logical_id = record.get("logical_id")
        path_value = record.get("path")
        expected = record.get("sha256")
        if not isinstance(logical_id, str) or not logical_id or logical_id in logical_ids:
            raise ManifestVerificationError("{} logical_id must be unique and non-empty".format(role))
        if not isinstance(path_value, str) or not path_value or path_value in paths:
            raise ManifestVerificationError("{} path must be unique and non-empty".format(role))
        if not isinstance(expected, str) or len(expected) != 64:
            raise ManifestVerificationError("{} sha256 must be a SHA-256 digest".format(role))
        path = Path(path_value)
        if not path.is_file():
            raise ManifestVerificationError("{} file is missing: {}".format(role, path))
        observed = sha256_file(path)
        if observed != expected:
            raise ManifestVerificationError(
                "{} sha256 mismatch for {}: expected {}, observed {}".format(role, path, expected, observed)
            )
        logical_ids.add(logical_id)
        paths.add(path_value)


def verify_run_manifest(manifest_path: Path) -> None:
    """Reject incomplete, unsupported, malformed, missing, or hash-mutated evidence."""
    manifest_path = Path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestVerificationError("cannot read manifest: {}".format(error)) from error
    if not isinstance(manifest, dict):
        raise ManifestVerificationError("manifest root must be an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestVerificationError("unsupported schema_version")
    if manifest.get("status") != "complete":
        raise ManifestVerificationError("manifest status must be complete (fail closed)")
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]:
        raise ManifestVerificationError("run_id must be non-empty")
    targets = manifest.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ManifestVerificationError("targets must be a non-empty named registry")
    target_names = set()
    for target in targets:
        if not isinstance(target, dict):
            raise ManifestVerificationError("target must be an object")
        name, unit, direction, threshold = (
            target.get("name"),
            target.get("unit"),
            target.get("direction"),
            target.get("threshold"),
        )
        if not isinstance(name, str) or not name or name in target_names:
            raise ManifestVerificationError("targets require unique names")
        if not isinstance(unit, str) or not unit or direction not in {"upper", "lower"}:
            raise ManifestVerificationError("target unit/direction is invalid")
        if not isinstance(threshold, (int, float)):
            raise ManifestVerificationError("target threshold is invalid")
        target_names.add(name)
    split = manifest.get("split")
    if not isinstance(split, dict) or split.get("algorithm") != "family-id-v1":
        raise ManifestVerificationError("split must use family-id-v1")
    split_hash = split.get("sha256")
    if not isinstance(split_hash, str) or len(split_hash) != 64:
        raise ManifestVerificationError("split sha256 is invalid")
    _verify_records(manifest.get("inputs"), "input")
    _verify_records(manifest.get("outputs"), "output")
