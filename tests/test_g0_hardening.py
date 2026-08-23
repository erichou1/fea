from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

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


def test_manifest_fifo_substitution_after_snapshot_is_bounded_and_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sasto.manifest as manifest_module

    manifest_path = smoke_manifest(tmp_path)
    trusted_digest = digest(manifest_path)
    original_snapshot = manifest_module._read_regular_snapshot
    substituted = False

    def replace_manifest_after_snapshot(*args: object, **kwargs: object) -> object:
        nonlocal substituted
        snapshot = original_snapshot(*args, **kwargs)
        if not substituted and snapshot.role == "manifest":
            substituted = True
            snapshot.path.unlink()
            os.mkfifo(snapshot.path)
        return snapshot

    monkeypatch.setattr(manifest_module, "_read_regular_snapshot", replace_manifest_after_snapshot)
    with pytest.raises(ManifestVerificationError, match="changed during verification"):
        verify_run_manifest(manifest_path, trusted_digest)
    assert substituted


def test_declared_fifo_substitution_after_snapshot_is_bounded_and_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sasto.manifest as manifest_module

    manifest_path = smoke_manifest(tmp_path)
    trusted_digest = digest(manifest_path)
    original_snapshot = manifest_module._read_regular_snapshot
    substituted = False

    def replace_record_after_snapshot(*args: object, **kwargs: object) -> object:
        nonlocal substituted
        snapshot = original_snapshot(*args, **kwargs)
        if not substituted and snapshot.path.name == "summary.json":
            substituted = True
            snapshot.path.unlink()
            os.mkfifo(snapshot.path)
        return snapshot

    monkeypatch.setattr(manifest_module, "_read_regular_snapshot", replace_record_after_snapshot)
    with pytest.raises(ManifestVerificationError, match="changed during verification"):
        verify_run_manifest(manifest_path, trusted_digest)
    assert substituted


@pytest.mark.parametrize("substitution", ("leaf", "parent"))
def test_declared_leaf_and_parent_symlink_substitutions_are_rejected(tmp_path: Path, substitution: str) -> None:
    manifest_path = smoke_manifest(tmp_path)
    trusted_digest = digest(manifest_path)
    artifact_root = manifest_path.parent
    external = tmp_path / "external"
    external.mkdir()
    if substitution == "leaf":
        target = artifact_root / "summary.json"
        target.unlink()
        target.symlink_to(external / "summary.json")
    else:
        original = artifact_root / "inputs"
        original.rename(artifact_root / "inputs-original")
        (external / "families.json").write_text("[]\n", encoding="utf-8")
        original.symlink_to(external, target_is_directory=True)

    with pytest.raises(ManifestVerificationError, match="must not be a symlink|cannot safely open"):
        verify_run_manifest(manifest_path, trusted_digest)


def test_make_smoke_treats_malicious_artifact_dir_as_literal_or_rejects_without_markers(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    marker = tmp_path / "unexpected-marker"
    artifact = tmp_path / "literal-artifact"
    malicious = "{}; touch {}; : 'quoted' $$(printf injected) spaced *".format(artifact, marker)

    result = subprocess.run(
        ["make", "smoke", "ARTIFACT_DIR={}".format(malicious)],
        cwd=repository,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert not marker.exists(), result.stdout + result.stderr
    assert result.returncode in (0, 2), result.stdout + result.stderr


@pytest.mark.parametrize(
    "template",
    (
        '/definitely/missing/python"; touch {marker}; #',
        "/definitely/missing/python$(touch {marker})",
        "/definitely/missing/python\n touch {marker}",
        "/definitely missing/python *",
        "-definitely-missing-python",
    ),
    ids=("semicolon-quote", "command-substitution", "newline", "spaces-glob", "leading-dash"),
)
@pytest.mark.parametrize("target", ("smoke", "verify-artifact", "reproduce-paper", "test"))
def test_make_python_is_environment_data_not_recipe_source(
    tmp_path: Path, template: str, target: str
) -> None:
    repository = Path(__file__).parents[1]
    marker = tmp_path / "unexpected-python-marker"
    malicious = template.format(marker=marker)
    environment = {
        **os.environ,
        "PYTHONPATH": "src",
        "EXPECTED_MANIFEST_SHA256": "0" * 64,
        "ARTIFACT_DIR": str(tmp_path / "missing-artifact"),
    }
    result = subprocess.run(
        ["make", target, "PYTHON={}".format(malicious)],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert not marker.exists(), result.stdout + result.stderr
    assert result.returncode != 0, result.stdout + result.stderr


def test_split_rewrite_after_its_verified_snapshot_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A semantic-equivalent split rewrite cannot evade record-byte verification."""
    import sasto.manifest as manifest_module

    manifest_path = smoke_manifest(tmp_path)
    trusted_digest = digest(manifest_path)
    original_snapshot = manifest_module._read_regular_snapshot
    rewritten = False

    def rewrite_after_snapshot(*args: object, **kwargs: object) -> object:
        nonlocal rewritten
        snapshot = original_snapshot(*args, **kwargs)
        if not rewritten and snapshot.path.name == "split-manifest.json":
            rewritten = True
            payload = json.loads(snapshot.bytes.decode("utf-8"))
            snapshot.path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        return snapshot

    monkeypatch.setattr(manifest_module, "_read_regular_snapshot", rewrite_after_snapshot)
    with pytest.raises(ManifestVerificationError, match="changed during verification"):
        verify_run_manifest(manifest_path, trusted_digest)
    assert rewritten


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


def test_smoke_ancestor_swap_cannot_redirect_descriptor_anchored_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Swap a checked ancestor after its FD is opened; external stays untouched."""
    import sasto.manifest as manifest_module

    fixture = Path(__file__).parents[1] / "fixtures" / "smoke" / "families.json"
    parent = tmp_path / "ancestor"
    parent.mkdir()
    original_parent = tmp_path / "ancestor-original"
    external = tmp_path / "external"
    external.mkdir()
    real_open = manifest_module.os.open
    opened_parent = threading.Event()
    swap_finished = threading.Event()
    failures: list[BaseException] = []

    def swap_ancestor() -> None:
        try:
            assert opened_parent.wait(timeout=2)
            parent.rename(original_parent)
            parent.symlink_to(external, target_is_directory=True)
            swap_finished.set()
        except BaseException as error:  # surfaced in the foreground assertion below
            failures.append(error)
            swap_finished.set()

    worker = threading.Thread(target=swap_ancestor)
    worker.start()

    def swap_after_parent_opened(path: object, flags: int, *args: object, **kwargs: object) -> int:
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == "ancestor":
            opened_parent.set()
            assert swap_finished.wait(timeout=2)
        return descriptor

    monkeypatch.setattr(manifest_module.os, "open", swap_after_parent_opened)
    try:
        with pytest.raises(ManifestVerificationError, match="artifact root changed"):
            run_smoke(fixture, parent / "artifact")
    finally:
        worker.join(timeout=2)
    assert not worker.is_alive()
    assert not failures
    assert (original_parent / "artifact" / "run-manifest.json").exists()
    assert not (external / "artifact").exists()


def test_smoke_rejects_final_root_replacement_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sasto.manifest as manifest_module
    import sasto.smoke as smoke_module

    fixture = Path(__file__).parents[1] / "fixtures" / "smoke" / "families.json"
    output = tmp_path / "artifact"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    original_write = manifest_module._write_new_regular_file
    replaced = False

    def replace_after_manifest_write(*args: object, **kwargs: object) -> object:
        nonlocal replaced
        result = original_write(*args, **kwargs)
        if not replaced and args[1] == "run-manifest.json":
            replaced = True
            output.rename(tmp_path / "artifact-original")
            output.symlink_to(replacement, target_is_directory=True)
        return result

    monkeypatch.setattr(smoke_module, "_write_new_regular_file", replace_after_manifest_write)
    with pytest.raises(ManifestVerificationError, match="artifact root changed"):
        run_smoke(fixture, output)
    assert replaced
    assert not (replacement / "run-manifest.json").exists()


@pytest.mark.parametrize("bad_path", ("bad\x00path", "inputs/\x00families.json"))
def test_digest_consistent_nul_record_path_is_rejected_cleanly(tmp_path: Path, bad_path: str) -> None:
    manifest_path = smoke_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"][0]["path"] = bad_path
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    expected = digest(manifest_path)

    with pytest.raises(ManifestVerificationError, match="NUL|relative non-traversal"):
        verify_run_manifest(manifest_path, expected)

    result = subprocess.run(
        [sys.executable, "-m", "sasto.verify_artifact", "--expected-manifest-sha256", expected, str(manifest_path)],
        cwd=Path(__file__).parents[1],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout.startswith("REJECTED:")
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("bad", ("bad\x00fixture", "bad\x00output"))
def test_smoke_rejects_nul_paths_as_normal_public_errors(tmp_path: Path, bad: str) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "smoke" / "families.json"
    fixture_path = Path(bad) if bad.endswith("fixture") else fixture
    output = Path(bad) if bad.endswith("output") else tmp_path / "artifact"
    with pytest.raises(ManifestVerificationError, match="NUL|invalid"):
        run_smoke(fixture_path, output)


def test_verifier_cli_rejects_a_nul_manifest_path_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sasto.verify_artifact as verifier_cli

    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_artifact", "--expected-manifest-sha256", "0" * 64, "manifest\x00path"],
    )
    assert verifier_cli.main() == 2
    captured = capsys.readouterr()
    assert captured.out.startswith("REJECTED:")
    assert "Traceback" not in captured.err


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
