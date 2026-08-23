from __future__ import annotations

import json
from pathlib import Path

import pytest

from sasto.manifest import ManifestVerificationError, verify_run_manifest
from sasto.smoke import run_smoke


def test_smoke_fixture_is_deterministic_and_verifiable(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "smoke" / "families.json"
    output = tmp_path / "artifact"
    manifest_path = run_smoke(fixture, output)
    verify_run_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["split"]["algorithm"] == "family-id-v1"
    assert {record["logical_id"] for record in manifest["outputs"]} == {
        "smoke/split-manifest",
        "smoke/summary",
    }
    with pytest.raises(FileExistsError):
        run_smoke(fixture, output)


def test_verifier_fails_closed_when_manifest_is_not_complete(tmp_path: Path) -> None:
    manifest = tmp_path / "run-manifest.json"
    manifest.write_text('{"schema_version":"1.0.0","status":"running"}', encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="fail closed"):
        verify_run_manifest(manifest)
