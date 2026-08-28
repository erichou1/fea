"""Focused byte-preserving v1 -> v2 G3 migration contract."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: dict[str, object], field: str) -> dict[str, object]:
    value[field] = hashlib.sha256(_canonical(value)).hexdigest()
    return value


def _load_module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "migrate_g3_v2.py"
    spec = importlib.util.spec_from_file_location("migrate_g3_v2", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_copies_frozen_values_and_rebinds_current_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    v1, v2 = tmp_path / "v1", tmp_path / "v2"
    v1.mkdir()
    campaign = {
        "source_bundle_sha256": module.V1_BUNDLE_SHA256,
        "archive_sha256": "a" * 64,
        "cohort_manifest_sha256": "b" * 64,
        "cluster_role_manifest_sha256": "c" * 64,
        "split_manifest_sha256": "d" * 64,
    }
    campaign = _digest(campaign, "campaign_digest")
    (v1 / "campaign-manifest.json").write_bytes(_canonical(campaign))
    kappa = _digest({"kappa": module.EXPECTED_KAPPA}, "kappa_evidence_sha256")
    q_base = _digest({"q": module.EXPECTED_Q_BASE}, "baseline_calibration_sha256")
    (v1 / "kappa-development-evidence.json").write_bytes(_canonical(kappa))
    (v1 / "baseline-calibration.json").write_bytes(_canonical(q_base))
    cache = v1 / "decoded-channel-cache-v1"
    cache.mkdir()
    rows = {}
    for role in ("development", "calibration"):
        filename = f"{role}.bin"
        payload = role.encode("ascii")
        (cache / filename).write_bytes(payload)
        rows[role] = {"source": "g3", "data_file": filename, "data_sha256": hashlib.sha256(payload).hexdigest(), "sample_ids": [role]}
    cache_manifest = _digest({**{key: campaign[key] for key in ("archive_sha256", "cohort_manifest_sha256", "cluster_role_manifest_sha256", "split_manifest_sha256")}, "roles": rows}, "cache_digest")
    (cache / "cache-manifest.json").write_bytes(_canonical(cache_manifest))
    for index in range(193):
        case = _digest({"sample_id": f"{index:05d}", "role": "development", "intermediate_solver_call_count": 0}, "trajectory_digest")
        (v1 / f"trajectory-development-{index:05d}.json").write_bytes(_canonical(case))
    adjudication = tmp_path / "adjudication.json"
    adjudication.write_text("controller decision", encoding="utf-8")
    monkeypatch.setattr(module, "ADJUDICATION_SHA256", hashlib.sha256(adjudication.read_bytes()).hexdigest())
    monkeypatch.setattr(module, "source_bundle", lambda: ({"current.py": "e" * 64}, "f" * 64))

    result = module.migrate(v1_root=v1, v2_root=v2, adjudication_path=adjudication)

    assert result["imported_development_trajectory_count"] == 193
    assert json.loads((v2 / "campaign-manifest.json").read_text())["source_bundle_sha256"] == "f" * 64
    assert (v2 / "kappa-development-evidence.json").read_bytes() == (v1 / "kappa-development-evidence.json").read_bytes()
    assert (v2 / "baseline-calibration.json").read_bytes() == (v1 / "baseline-calibration.json").read_bytes()
    assert (v2 / "trajectory-development-00000.json").read_bytes() == (v1 / "trajectory-development-00000.json").read_bytes()
    provenance = json.loads((v2 / "migration-provenance.json").read_text())
    assert provenance["intermediate_solver_calls_verified_zero_for_all_imports"] is True
    assert provenance["frozen_constants_imported_by_value"]["q_base"] == module.EXPECTED_Q_BASE
