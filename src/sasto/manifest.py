"""Versioned, fail-closed run-manifest construction and SHA-256 verification."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path, PurePosixPath
from typing import Mapping

from .splits import FamilyLeakageError, split_sha256, validate_family_split_manifest
from .targets import TargetRegistry, TargetSpec


SCHEMA_VERSION = "1.0.0"


class ManifestVerificationError(ValueError):
    """The artifact cannot be admitted as a complete SASTO-V evidence record."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_beneath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _has_symlink_component(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            return True
    return False


def has_lexical_symlink_component(path: Path) -> bool:
    """Inspect each caller-supplied path component before any resolution."""
    if path.is_absolute():
        current = Path(path.anchor)
        parts = path.parts[1:]
    else:
        current = Path(".")
        parts = path.parts
    for component in parts:
        if component in ("", "."):
            continue
        current = current / component
        if current.is_symlink():
            return True
    return False


def _record_path(path_value: object, root: Path, role: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ManifestVerificationError("{} path must be unique and non-empty".format(role))
    portable = PurePosixPath(path_value)
    if portable.is_absolute() or ".." in portable.parts or portable == PurePosixPath("."):
        raise ManifestVerificationError("{} path must be a relative non-traversal path".format(role))
    return root.joinpath(*portable.parts)


def _check_regular_artifact(path: Path, root: Path, role: str) -> None:
    if not _is_beneath(path, root):
        raise ManifestVerificationError("{} file must remain beneath artifact root".format(role))
    if _has_symlink_component(path, root):
        raise ManifestVerificationError("{} file must not be a symlink".format(role))
    resolved = path.resolve()
    if not _is_beneath(resolved, root):
        raise ManifestVerificationError("{} file must remain beneath artifact root".format(role))
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise ManifestVerificationError("{} file is missing: {}".format(role, path)) from error
    if not stat.S_ISREG(mode):
        raise ManifestVerificationError("{} file must be a regular file: {}".format(role, path))


def _file_records(files: Mapping[str, Path], artifact_root: Path) -> list:
    records = []
    for logical_id, path_like in sorted(files.items()):
        if not isinstance(logical_id, str) or not logical_id:
            raise ManifestVerificationError("declared logical_id must be non-empty")
        path = Path(path_like)
        if not path.is_absolute():
            path = artifact_root / path
        _check_regular_artifact(path, artifact_root, "declared")
        relative = path.relative_to(artifact_root).as_posix()
        records.append({"logical_id": logical_id, "path": relative, "sha256": sha256_file(path)})
    return records


def build_run_manifest(
    *,
    run_id: str,
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
    targets: TargetRegistry,
    split_sha256: str,
    split_artifact: str,
    artifact_root: Path,
) -> dict:
    """Build a complete, self-contained record after declared files exist."""
    if not run_id:
        raise ManifestVerificationError("run_id must be non-empty")
    if len(split_sha256) != 64 or any(char not in "0123456789abcdef" for char in split_sha256):
        raise ManifestVerificationError("split_sha256 must be a lowercase SHA-256 digest")
    if not isinstance(split_artifact, str) or not split_artifact:
        raise ManifestVerificationError("split_artifact must name a declared output")
    root = Path(artifact_root)
    if root.is_symlink() or not root.is_dir():
        raise ManifestVerificationError("artifact_root must be a real directory")
    root = root.resolve()
    output_records = _file_records(outputs, root)
    if split_artifact not in {record["logical_id"] for record in output_records}:
        raise ManifestVerificationError("split_artifact must name a declared output")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "complete",
        "targets": targets.as_list(),
        "split": {"algorithm": "family-id-v1", "artifact": split_artifact, "sha256": split_sha256},
        "inputs": _file_records(inputs, root),
        "outputs": output_records,
    }


def _verify_records(records: object, role: str, artifact_root: Path) -> dict[str, Path]:
    if not isinstance(records, list) or not records:
        raise ManifestVerificationError("{} must be a non-empty list".format(role))
    logical_ids = set()
    paths = set()
    verified = {}
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
        path = _record_path(path_value, artifact_root, role)
        _check_regular_artifact(path, artifact_root, role)
        observed = sha256_file(path)
        if observed != expected:
            raise ManifestVerificationError(
                "{} sha256 mismatch for {}: expected {}, observed {}".format(role, path, expected, observed)
            )
        logical_ids.add(logical_id)
        paths.add(path_value)
        verified[logical_id] = path
    return verified


def _verify_targets(targets: object) -> None:
    if not isinstance(targets, list) or not targets:
        raise ManifestVerificationError("targets must be a non-empty named registry")
    parsed = []
    required_fields = {"name", "unit", "direction", "threshold", "normalization", "base_target"}
    for target in targets:
        if not isinstance(target, dict):
            raise ManifestVerificationError("target must be an object")
        if set(target) != required_fields:
            raise ManifestVerificationError("target contract is invalid: malformed target fields")
        try:
            parsed.append(
                TargetSpec(
                    name=target["name"],
                    unit=target["unit"],
                    direction=target["direction"],
                    threshold=target["threshold"],
                    normalization=target["normalization"],
                    base_target=target.get("base_target"),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ManifestVerificationError("target contract is invalid: {}".format(error)) from error
    try:
        TargetRegistry(parsed)
    except ValueError as error:
        raise ManifestVerificationError("targets require unique names") from error


def verify_run_manifest(manifest_path: Path) -> None:
    """Reject incomplete, nonportable, malformed, or hash-mutated evidence."""
    manifest_path = Path(manifest_path)
    if has_lexical_symlink_component(manifest_path):
        raise ManifestVerificationError("manifest must reside in a real artifact root")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestVerificationError("cannot read manifest: {}".format(error)) from error
    if not isinstance(manifest, dict):
        raise ManifestVerificationError("manifest root must be an object")
    artifact_root = manifest_path.parent.resolve()
    if manifest_path.is_symlink() or not artifact_root.is_dir():
        raise ManifestVerificationError("manifest must reside in a real artifact root")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestVerificationError("unsupported schema_version")
    if manifest.get("status") != "complete":
        raise ManifestVerificationError("manifest status must be complete (fail closed)")
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]:
        raise ManifestVerificationError("run_id must be non-empty")
    _verify_targets(manifest.get("targets"))
    split = manifest.get("split")
    if not isinstance(split, dict) or split.get("algorithm") != "family-id-v1":
        raise ManifestVerificationError("split must use family-id-v1")
    split_hash = split.get("sha256")
    if not isinstance(split_hash, str) or len(split_hash) != 64:
        raise ManifestVerificationError("split sha256 is invalid")
    inputs = _verify_records(manifest.get("inputs"), "input", artifact_root)
    outputs = _verify_records(manifest.get("outputs"), "output", artifact_root)
    split_artifact = split.get("artifact")
    if not isinstance(split_artifact, str) or not split_artifact:
        raise ManifestVerificationError("split artifact must name a declared output")
    split_path = outputs.get(split_artifact)
    if split_path is None:
        raise ManifestVerificationError("split artifact must name a declared output")
    try:
        declared_split = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestVerificationError("cannot parse declared split artifact: {}".format(error)) from error
    if split_sha256(declared_split) != split_hash:
        raise ManifestVerificationError("split sha256 does not match declared split artifact")
    try:
        validate_family_split_manifest(declared_split)
    except (FamilyLeakageError, TypeError, ValueError) as error:
        raise ManifestVerificationError("declared split artifact is invalid: {}".format(error)) from error
