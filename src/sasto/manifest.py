"""Versioned, fail-closed run-manifest construction and descriptor snapshots.

Verification takes a bounded snapshot rooted at an opened artifact-directory file
descriptor.  Every manifest and declared record is opened with no-follow and
nonblocking flags, checked as a regular file on that same descriptor, and read
and hashed from that descriptor.  The verifier reopens every member through the
held root descriptor before returning and rejects observed inode, metadata, or
byte drift.  Mutation after this function returns is outside the snapshot; this
does not claim filesystem immutability or ongoing soundness.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator, Mapping

from .splits import FamilyLeakageError, split_sha256, validate_family_split_manifest
from .targets import TargetRegistry, TargetSpec


SCHEMA_VERSION = "1.0.0"
_READ_CHUNK_SIZE = 1024 * 1024


class ManifestVerificationError(ValueError):
    """The artifact cannot be admitted as a complete SASTO-V evidence record."""


@dataclass(frozen=True)
class _FileSnapshot:
    """Exact regular-file bytes and the descriptor metadata observed with them."""

    path: Path
    relative_parts: tuple[str, ...]
    role: str
    bytes: bytes
    sha256: str
    identity: tuple[int, int, int, int, int, int]


def _is_lowercase_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """Stable fields used to notice replacement or metadata mutation while verifying."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_open_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ManifestVerificationError("descriptor no-follow directory opening is unavailable")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _file_open_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise ManifestVerificationError("descriptor no-follow file opening is unavailable")
    return os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)


def _path_parts(path: Path) -> tuple[str, ...]:
    parts = path.parts[1:] if path.is_absolute() else path.parts
    if not parts or any(component in ("", ".", "..") for component in parts):
        raise ManifestVerificationError("artifact path must not contain empty or traversal components")
    return tuple(parts)


@contextlib.contextmanager
def _open_directory_path(path: Path) -> Iterator[int]:
    """Open each lexical directory component without following a symlink."""
    path = Path(path)
    parts = _path_parts(path)
    current_fd = os.open(path.anchor if path.is_absolute() else ".", _directory_open_flags())
    try:
        for component in parts:
            try:
                next_fd = os.open(component, _directory_open_flags(), dir_fd=current_fd)
            except OSError as error:
                if error.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise ManifestVerificationError("artifact_root must not contain symlink components") from error
                raise ManifestVerificationError("cannot safely open artifact root: {}".format(error)) from error
            os.close(current_fd)
            current_fd = next_fd
        yield current_fd
    finally:
        os.close(current_fd)


def _relative_artifact_parts(path: Path, root: Path, role: str) -> tuple[str, ...]:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ManifestVerificationError("{} file must remain beneath artifact root".format(role)) from error
    portable = PurePosixPath(relative.as_posix())
    if not portable.parts or any(component in ("", ".", "..") for component in portable.parts):
        raise ManifestVerificationError("{} path must be a relative non-traversal path".format(role))
    return tuple(portable.parts)


def _record_path(path_value: object, root: Path, role: str) -> tuple[Path, tuple[str, ...]]:
    if not isinstance(path_value, str) or not path_value:
        raise ManifestVerificationError("{} path must be unique and non-empty".format(role))
    portable = PurePosixPath(path_value)
    if portable.is_absolute() or ".." in portable.parts or portable == PurePosixPath("."):
        raise ManifestVerificationError("{} path must be a relative non-traversal path".format(role))
    return root.joinpath(*portable.parts), tuple(portable.parts)


def _open_error(role: str, path: Path, error: OSError) -> ManifestVerificationError:
    if error.errno == errno.ELOOP:
        return ManifestVerificationError("{} file must not be a symlink: {}".format(role, path))
    if error.errno == errno.ENOENT:
        return ManifestVerificationError("{} file is missing: {}".format(role, path))
    return ManifestVerificationError("cannot safely open {} file {}: {}".format(role, path, error))


def _read_regular_snapshot(root_fd: int, relative_parts: tuple[str, ...], role: str, path: Path) -> _FileSnapshot:
    """Read one nonblocking, no-follow regular file through a held root descriptor."""
    if not relative_parts or any(component in ("", ".", "..") for component in relative_parts):
        raise ManifestVerificationError("{} path must be a relative non-traversal path".format(role))
    parent_fd = os.dup(root_fd)
    file_fd: int | None = None
    try:
        for component in relative_parts[:-1]:
            try:
                next_fd = os.open(component, _directory_open_flags(), dir_fd=parent_fd)
            except OSError as error:
                raise _open_error(role, path, error) from error
            os.close(parent_fd)
            parent_fd = next_fd
        try:
            file_fd = os.open(relative_parts[-1], _file_open_flags(), dir_fd=parent_fd)
        except OSError as error:
            raise _open_error(role, path, error) from error
    finally:
        os.close(parent_fd)
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ManifestVerificationError("{} file must be a regular file: {}".format(role, path))
        try:
            chunks = []
            while True:
                chunk = os.read(file_fd, _READ_CHUNK_SIZE)
                if not chunk:
                    break
                chunks.append(chunk)
        except OSError as error:
            raise ManifestVerificationError("cannot read {} file {}: {}".format(role, path, error)) from error
        after = os.fstat(file_fd)
        if _identity(before) != _identity(after):
            raise ManifestVerificationError("{} file changed while being read: {}".format(role, path))
        payload = b"".join(chunks)
        return _FileSnapshot(
            path=path,
            relative_parts=relative_parts,
            role=role,
            bytes=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            identity=_identity(after),
        )
    finally:
        os.close(file_fd)


def _revalidate_snapshot(root_fd: int, snapshot: _FileSnapshot) -> None:
    """Bound acceptance to the exact descriptor snapshot first observed."""
    try:
        current = _read_regular_snapshot(root_fd, snapshot.relative_parts, snapshot.role, snapshot.path)
    except ManifestVerificationError as error:
        raise ManifestVerificationError("{} file changed during verification: {} ({})".format(snapshot.role, snapshot.path, error)) from error
    if current.identity != snapshot.identity or current.bytes != snapshot.bytes or current.sha256 != snapshot.sha256:
        raise ManifestVerificationError("{} file changed during verification: {}".format(snapshot.role, snapshot.path))


def sha256_file(path: Path) -> str:
    """Hash one descriptor-anchored regular-file snapshot without reopening its leaf."""
    path = Path(path)
    with _open_directory_path(path.parent) as root_fd:
        snapshot = _read_regular_snapshot(root_fd, (path.name,), "declared", path)
    return snapshot.sha256


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
    """Best-effort diagnostic for callers; security decisions use descriptor opens."""
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


def _file_records(files: Mapping[str, Path], artifact_root: Path, artifact_root_fd: int) -> list[dict[str, str]]:
    records = []
    for logical_id, path_like in sorted(files.items()):
        if not isinstance(logical_id, str) or not logical_id:
            raise ManifestVerificationError("declared logical_id must be non-empty")
        path = Path(path_like)
        if not path.is_absolute():
            path = artifact_root / path
        relative_parts = _relative_artifact_parts(path, artifact_root, "declared")
        snapshot = _read_regular_snapshot(artifact_root_fd, relative_parts, "declared", path)
        records.append({"logical_id": logical_id, "path": "/".join(relative_parts), "sha256": snapshot.sha256})
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
    """Build a record from descriptor snapshots of files that already exist.

    The resulting manifest records bytes captured during this call.  It makes no
    claim that callers will keep those paths immutable after this function returns.
    """
    if not run_id:
        raise ManifestVerificationError("run_id must be non-empty")
    if not _is_lowercase_sha256(split_sha256):
        raise ManifestVerificationError("split_sha256 must be a lowercase SHA-256 digest")
    if not isinstance(split_artifact, str) or not split_artifact:
        raise ManifestVerificationError("split_artifact must name a declared output")
    root = Path(artifact_root)
    if not root.is_absolute():
        root = Path.cwd() / root
    with _open_directory_path(root) as root_fd:
        output_records = _file_records(outputs, root, root_fd)
        if split_artifact not in {record["logical_id"] for record in output_records}:
            raise ManifestVerificationError("split_artifact must name a declared output")
        input_records = _file_records(inputs, root, root_fd)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "complete",
        "targets": targets.as_list(),
        "split": {"algorithm": "family-id-v1", "artifact": split_artifact, "sha256": split_sha256},
        "inputs": input_records,
        "outputs": output_records,
    }


def _verify_records(
    records: object, role: str, artifact_root: Path, artifact_root_fd: int
) -> dict[str, _FileSnapshot]:
    """Return the exact verified bytes, not paths that would need a later reread."""
    if not isinstance(records, list) or not records:
        raise ManifestVerificationError("{} must be a non-empty list".format(role))
    logical_ids = set()
    paths = set()
    verified: dict[str, _FileSnapshot] = {}
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
        if not _is_lowercase_sha256(expected):
            raise ManifestVerificationError("{} sha256 must be a SHA-256 digest".format(role))
        path, relative_parts = _record_path(path_value, artifact_root, role)
        snapshot = _read_regular_snapshot(artifact_root_fd, relative_parts, role, path)
        if snapshot.sha256 != expected:
            raise ManifestVerificationError(
                "{} sha256 mismatch for {}: expected {}, observed {}".format(role, path, expected, snapshot.sha256)
            )
        logical_ids.add(logical_id)
        paths.add(path_value)
        verified[logical_id] = snapshot
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


@contextlib.contextmanager
def _open_manifest_root(path: Path) -> Iterator[int]:
    """Translate root-opening failures into the public manifest-root contract."""
    manager = _open_directory_path(path)
    try:
        descriptor = manager.__enter__()
    except ManifestVerificationError as error:
        raise ManifestVerificationError("manifest must reside in a real artifact root") from error
    try:
        yield descriptor
    finally:
        manager.__exit__(None, None, None)


def verify_run_manifest(manifest_path: Path, expected_manifest_sha256: str) -> dict:
    """Verify a bounded descriptor snapshot of a complete evidence artifact.

    The returned object was parsed from the captured manifest bytes.  All declared
    records are revalidated before return; external mutation after return remains
    outside this snapshot and is not a filesystem-immutability claim.
    """
    manifest_path = Path(manifest_path)
    if not _is_lowercase_sha256(expected_manifest_sha256):
        raise ManifestVerificationError("expected manifest sha256 must be a lowercase SHA-256 digest")
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    try:
        with _open_manifest_root(manifest_path.parent) as artifact_root_fd:
            try:
                manifest_snapshot = _read_regular_snapshot(
                    artifact_root_fd, (manifest_path.name,), "manifest", manifest_path
                )
            except ManifestVerificationError as error:
                if "must not be a symlink" in str(error):
                    raise ManifestVerificationError("manifest must reside in a real artifact root") from error
                raise
            if manifest_snapshot.sha256 != expected_manifest_sha256:
                raise ManifestVerificationError(
                    "expected manifest sha256 mismatch: expected {}, observed {}".format(
                        expected_manifest_sha256, manifest_snapshot.sha256
                    )
                )
            try:
                manifest = json.loads(manifest_snapshot.bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ManifestVerificationError("cannot parse manifest: {}".format(error)) from error
            if not isinstance(manifest, dict):
                raise ManifestVerificationError("manifest root must be an object")
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
            if not _is_lowercase_sha256(split_hash):
                raise ManifestVerificationError("split sha256 is invalid")
            inputs = _verify_records(manifest.get("inputs"), "input", manifest_path.parent, artifact_root_fd)
            outputs = _verify_records(manifest.get("outputs"), "output", manifest_path.parent, artifact_root_fd)
            split_artifact = split.get("artifact")
            if not isinstance(split_artifact, str) or not split_artifact:
                raise ManifestVerificationError("split artifact must name a declared output")
            split_snapshot = outputs.get(split_artifact)
            if split_snapshot is None:
                raise ManifestVerificationError("split artifact must name a declared output")
            try:
                declared_split = json.loads(split_snapshot.bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ManifestVerificationError("cannot parse declared split artifact: {}".format(error)) from error
            if split_sha256(declared_split) != split_hash:
                raise ManifestVerificationError("split sha256 does not match declared split artifact")
            try:
                validate_family_split_manifest(declared_split)
            except (FamilyLeakageError, TypeError, ValueError) as error:
                raise ManifestVerificationError("declared split artifact is invalid: {}".format(error)) from error
            _revalidate_snapshot(artifact_root_fd, manifest_snapshot)
            for snapshot in (*inputs.values(), *outputs.values()):
                _revalidate_snapshot(artifact_root_fd, snapshot)
            return manifest
    except ManifestVerificationError:
        raise
    except OSError as error:
        raise ManifestVerificationError("cannot safely inspect manifest: {}".format(error)) from error
