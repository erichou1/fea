from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sasto.manifest import ManifestVerificationError, build_run_manifest, verify_run_manifest
from sasto.smoke import run_smoke
from sasto.targets import TargetRegistry, TargetSpec
from sasto.topology import is_simple_point_6_26


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path: Path) -> dict:
    return verify_run_manifest(path, digest(path))


def smoke_manifest(tmp_path: Path) -> Path:
    fixture = Path(__file__).parents[1] / "fixtures" / "smoke" / "families.json"
    return run_smoke(fixture, tmp_path / "artifact")


def test_manifest_verification_requires_a_well_formed_external_digest(tmp_path: Path) -> None:
    manifest_path = smoke_manifest(tmp_path)
    with pytest.raises(TypeError):
        verify_run_manifest(manifest_path)
    for invalid in ("", "0" * 63, "A" * 64, 7):
        with pytest.raises(ManifestVerificationError, match="expected manifest sha256"):
            verify_run_manifest(manifest_path, invalid)


@pytest.mark.parametrize("mutation", ("run_id", "target", "path_hash_pair"))
def test_external_manifest_digest_rejects_self_consistent_metadata_tampering(tmp_path: Path, mutation: str) -> None:
    manifest_path = smoke_manifest(tmp_path)
    trusted_digest = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "run_id":
        manifest["run_id"] = "forged-run-id"
    elif mutation == "target":
        manifest["targets"][0]["threshold"] = 999.0
    else:
        record = manifest["outputs"][0]
        original = manifest_path.parent / record["path"]
        replacement = manifest_path.parent / "renamed-output.json"
        original.rename(replacement)
        record["path"] = replacement.name
        record["sha256"] = hashlib.sha256(replacement.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="expected manifest sha256 mismatch"):
        verify_run_manifest(manifest_path, trusted_digest)


def test_identical_manifest_bytes_verify_after_relocation_with_same_external_digest(tmp_path: Path) -> None:
    manifest_path = smoke_manifest(tmp_path)
    trusted_digest = digest(manifest_path)
    relocated = tmp_path / "relocated"
    manifest_path.parent.rename(relocated)
    result = verify_run_manifest(relocated / "run-manifest.json", trusted_digest)
    assert result["run_id"] == "sasto-v-smoke-v1"


def test_manifest_fifo_is_rejected_before_opening_within_bound(tmp_path: Path) -> None:
    fifo = tmp_path / "run-manifest.json"
    os.mkfifo(fifo)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sasto.verify_artifact",
            "--expected-manifest-sha256",
            "0" * 64,
            str(fifo),
        ],
        cwd=Path(__file__).parents[1],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=0.75,
        check=False,
    )
    assert result.returncode == 2
    assert "manifest file must be a regular file" in result.stdout


@pytest.mark.parametrize("kind", ("directory", "fifo"))
def test_manifest_nonregular_leaves_are_rejected_before_read(tmp_path: Path, kind: str) -> None:
    manifest = tmp_path / "run-manifest.json"
    if kind == "directory":
        manifest.mkdir()
    else:
        os.mkfifo(manifest)
    with pytest.raises(ManifestVerificationError, match="manifest file must be a regular file"):
        verify_run_manifest(manifest, "0" * 64)


def test_builder_rejects_lexical_symlink_ancestor_before_resolution(tmp_path: Path) -> None:
    real_root = tmp_path / "real" / "artifact"
    real_root.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "real", target_is_directory=True)
    registry = TargetRegistry((TargetSpec("compliance", "J", "upper", 1.0),))
    with pytest.raises(ManifestVerificationError, match="symlink components"):
        build_run_manifest(
            run_id="alias",
            inputs={},
            outputs={},
            targets=registry,
            split_sha256="0" * 64,
            split_artifact="split",
            artifact_root=alias / "artifact",
        )


def test_target_thresholds_normalize_accepted_numpy_reals_to_json_native_float() -> None:
    for value in (np.int64(1), np.float32(1.25), np.float64(1.5)):
        target = TargetSpec("compliance", "J", "upper", value)
        assert isinstance(target.threshold, float)
        encoded = json.dumps(TargetRegistry((target,)).as_list())
        assert isinstance(json.loads(encoded)[0]["threshold"], float)


def test_numpy_and_nested_boolean_topology_match_controller_64_cube_regression() -> None:
    volume = np.zeros((64, 64, 64), dtype=bool)
    volume[16, 20, 33] = True
    volume[16, 20, 32] = True
    point = (16, 20, 33)
    nested = volume.tolist()
    assert is_simple_point_6_26(nested, point) is True
    assert is_simple_point_6_26(volume, point) is True


@pytest.mark.parametrize(
    "volume",
    (
        np.zeros((2, 2, 2), dtype=int),
        np.array([[[True, False], [False, True]]], dtype=object),
        [[[True], [1]]],
        [[[True], []]],
        np.zeros((2, 2), dtype=bool),
    ),
)
def test_topology_fails_closed_for_non_boolean_or_malformed_grids(volume: object) -> None:
    assert is_simple_point_6_26(volume, (0, 0, 0)) is False
