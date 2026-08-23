from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from sasto.manifest import ManifestVerificationError, verify_run_manifest
from sasto.smoke import run_smoke


def _manifest_digest(manifest_path: Path) -> str:
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _verify(manifest_path: Path) -> dict:
    return verify_run_manifest(manifest_path, _manifest_digest(manifest_path))


def test_smoke_fixture_is_deterministic_and_verifiable(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "smoke" / "families.json"
    output = tmp_path / "artifact"
    manifest_path = run_smoke(fixture, output)
    _verify(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["split"]["algorithm"] == "family-id-v1"
    assert manifest["split"]["artifact"] == "smoke/split-manifest"
    assert {record["logical_id"] for record in manifest["outputs"]} == {
        "smoke/split-manifest",
        "smoke/summary",
    }
    assert {record["logical_id"] for record in manifest["inputs"]} == {"smoke/families"}
    assert all(not Path(record["path"]).is_absolute() for record in manifest["inputs"] + manifest["outputs"])
    assert (output / manifest["inputs"][0]["path"]).read_bytes() == fixture.read_bytes()

    relocated = tmp_path / "relocated"
    output.rename(relocated)
    _verify(relocated / "run-manifest.json")
    with pytest.raises(FileExistsError):
        run_smoke(fixture, relocated)


def test_smoke_rejects_a_lexically_symlinked_output_ancestor_before_creation(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "smoke" / "families.json"
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(ManifestVerificationError, match="artifact root must not contain symlink components"):
        run_smoke(fixture, alias / "artifact")
    assert not (real / "artifact").exists()


def test_smoke_manifest_encodes_baseline_compliance_ratio(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "smoke" / "families.json"
    manifest = json.loads(run_smoke(fixture, tmp_path / "artifact").read_text(encoding="utf-8"))

    assert manifest["targets"] == [
        {
            "name": "compliance_ratio",
            "unit": "1",
            "direction": "upper",
            "threshold": 1.15,
            "normalization": "baseline_ratio",
            "base_target": "compliance",
        },
        {
            "name": "max_von_mises",
            "unit": "Pa",
            "direction": "upper",
            "threshold": 5_000_000.0,
            "normalization": "absolute",
            "base_target": None,
        },
        {
            "name": "max_displacement",
            "unit": "m",
            "direction": "upper",
            "threshold": 0.028,
            "normalization": "absolute",
            "base_target": None,
        },
    ]


def test_verifier_binds_split_digest_to_declared_split_artifact(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "smoke" / "families.json"
    manifest_path = run_smoke(fixture, tmp_path / "artifact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_record = next(record for record in manifest["outputs"] if record["logical_id"] == manifest["split"]["artifact"])
    split_path = manifest_path.parent / split_record["path"]
    split_path.write_text('{"tampered": true}\n', encoding="utf-8")
    for record in manifest["outputs"]:
        if record["logical_id"] == manifest["split"]["artifact"]:
            record["sha256"] = hashlib.sha256(split_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ManifestVerificationError, match="split sha256"):
        _verify(manifest_path)


def test_verifier_rejects_digest_bound_semantically_malformed_split_artifact(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "smoke" / "families.json"
    manifest_path = run_smoke(fixture, tmp_path / "artifact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_record = next(record for record in manifest["outputs"] if record["logical_id"] == manifest["split"]["artifact"])
    split_path = manifest_path.parent / split_record["path"]
    split_path.write_text('{"not_a_split":true,"partitions":{}}\n', encoding="utf-8")
    split_record["sha256"] = hashlib.sha256(split_path.read_bytes()).hexdigest()
    malformed = json.loads(split_path.read_text(encoding="utf-8"))
    manifest["split"]["sha256"] = hashlib.sha256(
        json.dumps(malformed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ManifestVerificationError, match="declared split artifact is invalid"):
        _verify(manifest_path)


def test_verifier_rejects_digest_bound_split_family_leakage(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "smoke" / "families.json"
    manifest_path = run_smoke(fixture, tmp_path / "artifact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_record = next(record for record in manifest["outputs"] if record["logical_id"] == manifest["split"]["artifact"])
    split_path = manifest_path.parent / split_record["path"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["partitions"]["development"]["sample_ids"].append(split["partitions"]["fit"]["sample_ids"][0])
    split_path.write_text(json.dumps(split, sort_keys=True), encoding="utf-8")
    split_record["sha256"] = hashlib.sha256(split_path.read_bytes()).hexdigest()
    manifest["split"]["sha256"] = hashlib.sha256(
        json.dumps(split, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ManifestVerificationError, match="declared split artifact is invalid"):
        _verify(manifest_path)


def test_verifier_rejects_exact_string_threshold_target_reproduction(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "smoke" / "families.json"
    manifest_path = run_smoke(fixture, tmp_path / "artifact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["targets"][0]["threshold"] = "not-a-number"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ManifestVerificationError, match="target contract is invalid"):
        _verify(manifest_path)


def test_verifier_rejects_malformed_target_fields(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "smoke" / "families.json"
    manifest_path = run_smoke(fixture, tmp_path / "artifact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["targets"][0]["unit"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ManifestVerificationError, match="target contract is invalid"):
        _verify(manifest_path)


def test_verifier_fails_closed_when_manifest_is_not_complete(tmp_path: Path) -> None:
    manifest = tmp_path / "run-manifest.json"
    manifest.write_text('{"schema_version":"1.0.0","status":"running"}', encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="fail closed"):
        _verify(manifest)


def test_verifier_rejects_symlinked_manifest_before_reading_contents(tmp_path: Path) -> None:
    manifest = tmp_path / "run-manifest.json"
    manifest.symlink_to(tmp_path / "missing-manifest.json")

    with pytest.raises(ManifestVerificationError, match="manifest must reside in a real artifact root"):
        verify_run_manifest(manifest, "0" * 64)


def test_verifier_rejects_exact_symlinked_manifest_parent_reproduction(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "smoke" / "families.json"
    real_artifact = tmp_path / "real-artifact"
    run_smoke(fixture, real_artifact)
    symlinked_parent = tmp_path / "symlinked-artifact"
    symlinked_parent.symlink_to(real_artifact, target_is_directory=True)

    with pytest.raises(ManifestVerificationError, match="manifest must reside in a real artifact root"):
        verify_run_manifest(symlinked_parent / "run-manifest.json", "0" * 64)


def test_verifier_rejects_symlinked_declared_output_control(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "smoke" / "families.json"
    manifest_path = run_smoke(fixture, tmp_path / "artifact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_record = next(record for record in manifest["outputs"] if record["logical_id"] == "smoke/summary")
    output_path = manifest_path.parent / output_record["path"]
    replacement = tmp_path / "replacement-summary.json"
    replacement.write_text(output_path.read_text(encoding="utf-8"), encoding="utf-8")
    output_path.unlink()
    output_path.symlink_to(replacement)

    with pytest.raises(ManifestVerificationError, match="output file must not be a symlink"):
        _verify(manifest_path)


def test_reproduce_paper_fails_closed_until_g1_assets_and_runner_exist() -> None:
    repository = Path(__file__).parents[1]
    result = subprocess.run(
        ["make", "reproduce-paper"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "G1 UNAVAILABLE" in result.stdout
    assert "does not reproduce paper results" in result.stdout
