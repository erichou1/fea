#!/usr/bin/env python3
"""Create a provenance-preserving G3 v2 campaign from a valid v1 campaign.

The only permissible migration is the controller-adjudicated v1 -> v2 source
bundle correction.  Existing trajectory files and frozen baseline constants are
copied by byte value after integrity checks; no solver, baseline, or trajectory
recomputation occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from sasto.g3_trajectory_calibration import _digest, source_bundle

V1_BUNDLE_SHA256 = "ab323527b620703bd16ca15d68412c4a8135f073939a9d9fd4055566f0939a57"
ADJUDICATION_SHA256 = "9a7fe787faa9326f770272a28ea6f88afd3d970ea1ed9cd5b2626b88b4109c28"
EXPECTED_KAPPA = {"compliance": 1.5, "max_displacement": 1.75, "max_von_mises": 2.5}
EXPECTED_Q_BASE = {
    "compliance": 0.11861603856441827,
    "max_displacement": 0.06926176456487498,
    "max_von_mises": 0.014884569734212372,
}
SCIENCE_IDENTICAL = (
    "select_state_index", "family_seed", "_sha_text", "split_conformal_quantile",
    "SAMPLING_NAMESPACE", "FAMILY_SEED_NAMESPACE", "CAMPAIGN_SEED", "DEPTH_BINS", "TARGET_NAMES",
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unavailable or malformed") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_digest(value: Mapping[str, Any], field: str, label: str) -> None:
    observed = value.get(field)
    expected = _digest({key: item for key, item in value.items() if key != field})
    if observed != expected:
        raise ValueError(f"{label} {field} verification failed")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_case(path: Path) -> dict[str, Any]:
    case = _load(path, "v1 trajectory case")
    _require_digest(case, "trajectory_digest", path.name)
    if case.get("role") != "development":
        raise ValueError(f"{path.name} is not a development trajectory")
    if case.get("intermediate_solver_call_count") != 0:
        raise ValueError(f"{path.name} has forbidden intermediate solver calls")
    return case


def _validate_cache(root: Path, campaign: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = root / "decoded-channel-cache-v1" / "cache-manifest.json"
    manifest = _load(manifest_path, "v1 decoded-channel cache manifest")
    _require_digest(manifest, "cache_digest", "v1 decoded-channel cache")
    for key in ("archive_sha256", "cohort_manifest_sha256", "cluster_role_manifest_sha256", "split_manifest_sha256"):
        if manifest.get(key) != campaign.get(key):
            raise ValueError(f"v1 decoded-channel cache {key} does not match v1 campaign")
    roles = manifest.get("roles")
    if not isinstance(roles, dict) or set(roles) != {"development", "calibration"}:
        raise ValueError("v1 decoded-channel cache roles are incomplete")
    for role, row in roles.items():
        if not isinstance(row, dict) or not isinstance(row.get("sample_ids"), list):
            raise ValueError(f"v1 decoded-channel cache {role} row is malformed")
        if row.get("source") == "g3":
            data_file, data_sha = row.get("data_file"), row.get("data_sha256")
            if not isinstance(data_file, str) or not isinstance(data_sha, str):
                raise ValueError(f"v1 decoded-channel cache {role} data reference is malformed")
            if _sha256_file(manifest_path.parent / data_file) != data_sha:
                raise ValueError(f"v1 decoded-channel cache {role} payload digest mismatch")
        elif row.get("source") != "g2-reuse":
            raise ValueError(f"v1 decoded-channel cache {role} source is unsupported")
    return manifest


def migrate(*, v1_root: Path, v2_root: Path, adjudication_path: Path) -> dict[str, Any]:
    """Validate v1, copy its allowed bytes, and bind v2 to current source."""
    if v2_root.exists():
        raise ValueError("v2 root already exists; refuse to mingle migration histories")
    if _sha256_file(adjudication_path) != ADJUDICATION_SHA256:
        raise ValueError("controller migration adjudication SHA-256 mismatch")

    campaign = _load(v1_root / "campaign-manifest.json", "v1 campaign manifest")
    _require_digest(campaign, "campaign_digest", "v1 campaign manifest")
    if campaign.get("source_bundle_sha256") != V1_BUNDLE_SHA256:
        raise ValueError("v1 campaign is not bound to the adjudicated source bundle")
    kappa = _load(v1_root / "kappa-development-evidence.json", "v1 kappa evidence")
    _require_digest(kappa, "kappa_evidence_sha256", "v1 kappa evidence")
    if kappa.get("kappa") != EXPECTED_KAPPA:
        raise ValueError("v1 kappa does not equal the adjudicated frozen value")
    q_base = _load(v1_root / "baseline-calibration.json", "v1 baseline calibration")
    _require_digest(q_base, "baseline_calibration_sha256", "v1 baseline calibration")
    if q_base.get("q") != EXPECTED_Q_BASE:
        raise ValueError("v1 q_base does not equal the adjudicated frozen value")
    cache = _validate_cache(v1_root, campaign)

    cases = sorted(v1_root.glob("trajectory-development-*.json"))
    if len(cases) != 193:
        raise ValueError(f"expected exactly 193 v1 development trajectories, found {len(cases)}")
    case_digests: list[dict[str, str]] = []
    for path in cases:
        case = _validate_case(path)
        sample_id, digest = case.get("sample_id"), case.get("trajectory_digest")
        if not isinstance(sample_id, str) or not isinstance(digest, str):
            raise ValueError(f"{path.name} trajectory identity is malformed")
        case_digests.append({"sample_id": sample_id, "trajectory_digest": digest})
    if len({row["sample_id"] for row in case_digests}) != len(case_digests):
        raise ValueError("v1 trajectory sample identities are not unique")

    current_files, current_bundle_sha = source_bundle()
    provenance: dict[str, Any] = {
        "schema_version": "1.0.0",
        "label": "G3_V1_TO_V2_MIGRATION_PROVENANCE",
        "migration_status": "CONTROLLER_ADJUDICATED_IMPORT_ONLY",
        "v1_root": str(v1_root),
        "v1_source_bundle_sha256": V1_BUNDLE_SHA256,
        "v1_campaign_digest": campaign["campaign_digest"],
        "current_source_bundle_sha256": current_bundle_sha,
        "current_source_bundle_files": current_files,
        "controller_adjudication_path": str(adjudication_path),
        "controller_adjudication_sha256": ADJUDICATION_SHA256,
        "imported_development_trajectory_count": len(case_digests),
        "imported_trajectory_digests": case_digests,
        "frozen_constants_imported_by_value": {"kappa": EXPECTED_KAPPA, "q_base": EXPECTED_Q_BASE},
        "reason": "v1 was bound to an older execution bundle while all 193 imported trajectories use the corrected geometric semantics; v2 fixes the provenance label without regenerating valid science artifacts.",
        "science_determining_code_verified_identical": {
            "statement": "Controller direct AST/artifact comparison verified the listed functions and constants byte-identical across the v1-bound and current source bundles.",
            "identical_functions_and_constants": list(SCIENCE_IDENTICAL),
            "verification_evidence": "controller adjudication SHA-256 above; compare its science_determining_code_unchanged object",
        },
        "intermediate_solver_calls_verified_zero_for_all_imports": True,
        "decoded_channel_cache": {
            "action": "reused_by_byte_copy_after_manifest_and_payload_digest_verification",
            "v1_cache_digest": cache["cache_digest"],
        },
        "coverage_computed": False,
        "hard_stop": "no_k6_coverage_or_adjudication",
    }
    provenance["migration_provenance_sha256"] = _digest(provenance)

    v2_root.mkdir(parents=True)
    try:
        (v2_root / "migration-provenance.json").write_bytes(_canonical(provenance) + b"\n")
        for source in (v1_root / "kappa-development-evidence.json", v1_root / "baseline-calibration.json"):
            shutil.copyfile(source, v2_root / source.name)
        for source in cases:
            shutil.copyfile(source, v2_root / source.name)
        shutil.copytree(v1_root / "decoded-channel-cache-v1", v2_root / "decoded-channel-cache-v1", copy_function=shutil.copyfile)

        v2_campaign = dict(campaign)
        v2_campaign["source_bundle_files"] = current_files
        v2_campaign["source_bundle_sha256"] = current_bundle_sha
        v2_campaign["migration"] = {
            "from_campaign_root": str(v1_root),
            "from_source_bundle_sha256": V1_BUNDLE_SHA256,
            "migration_provenance_path": "migration-provenance.json",
            "migration_provenance_sha256": provenance["migration_provenance_sha256"],
            "imported_development_trajectory_count": len(case_digests),
            "science_determining_code_verified_identical": True,
        }
        v2_campaign["campaign_digest"] = _digest({key: value for key, value in v2_campaign.items() if key != "campaign_digest"})
        (v2_root / "campaign-manifest.json").write_bytes(_canonical(v2_campaign) + b"\n")

        # Verify copied bytes and every new root anchor before returning success.
        if _sha256_file(v2_root / "kappa-development-evidence.json") != _sha256_file(v1_root / "kappa-development-evidence.json"):
            raise ValueError("kappa byte-value import verification failed")
        if _sha256_file(v2_root / "baseline-calibration.json") != _sha256_file(v1_root / "baseline-calibration.json"):
            raise ValueError("q_base byte-value import verification failed")
        for source in cases:
            target = v2_root / source.name
            if _sha256_file(target) != _sha256_file(source):
                raise ValueError(f"trajectory byte import verification failed: {source.name}")
            _validate_case(target)
        _require_digest(_load(v2_root / "campaign-manifest.json", "v2 campaign manifest"), "campaign_digest", "v2 campaign manifest")
        _require_digest(_load(v2_root / "migration-provenance.json", "v2 migration provenance"), "migration_provenance_sha256", "v2 migration provenance")
    except Exception:
        shutil.rmtree(v2_root, ignore_errors=True)
        raise
    return {
        "v2_root": str(v2_root), "current_source_bundle_sha256": current_bundle_sha,
        "imported_development_trajectory_count": len(case_digests),
        "migration_provenance_sha256": provenance["migration_provenance_sha256"],
        "campaign_digest": v2_campaign["campaign_digest"], "coverage_computed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Controller-adjudicated, byte-preserving G3 v1 -> v2 migration")
    parser.add_argument("--v1-root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = migrate(v1_root=args.v1_root, v2_root=args.v2_root, adjudication_path=args.adjudication)
    except (OSError, ValueError) as error:
        print(f"REJECTED: {error}")
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
