"""Deterministic public SASTO-V smoke artifact generator."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from .manifest import (
    ManifestVerificationError,
    _mkdir_open_new_directory,
    _revalidate_snapshot,
    _write_new_regular_file,
    assert_lexical_root_identity,
    build_run_manifest_from_snapshots,
    open_new_artifact_root,
    read_regular_path_snapshot,
    sha256_file,
)
from .splits import build_family_split_manifest, split_sha256
from .targets import TargetRegistry, TargetSpec
from .topology import is_simple_point_6_26


SMOKE_TARGETS = TargetRegistry(
    (
        TargetSpec(
            "compliance_ratio",
            "1",
            "upper",
            1.15,
            normalization="baseline_ratio",
            base_target="compliance",
        ),
        TargetSpec("max_von_mises", "Pa", "upper", 5_000_000.0),
        TargetSpec("max_displacement", "m", "upper", 0.028),
    )
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _public_path(path: Path, role: str) -> Path:
    path = Path(path)
    if "\x00" in str(path):
        raise ManifestVerificationError("{} path contains an embedded NUL".format(role))
    return path


def run_smoke(fixture_path: Path, output_dir: Path) -> Path:
    """Write one immutable smoke artifact through held no-follow descriptors."""
    fixture_path = _public_path(fixture_path, "fixture")
    output_dir = _public_path(output_dir, "artifact root")
    try:
        fixture_snapshot = read_regular_path_snapshot(fixture_path, "fixture")
    except ManifestVerificationError:
        raise
    except FileNotFoundError as error:
        raise FileNotFoundError("smoke fixture is missing: {}".format(fixture_path)) from error
    try:
        samples = json.loads(fixture_snapshot.bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestVerificationError("smoke fixture must be valid JSON") from error
    if not isinstance(samples, list):
        raise ManifestVerificationError("smoke fixture must be a JSON list")

    with open_new_artifact_root(output_dir) as root_fd:
        inputs_fd = _mkdir_open_new_directory(root_fd, "inputs", output_dir / "inputs")
        try:
            input_snapshot = _write_new_regular_file(
                inputs_fd,
                "families.json",
                fixture_snapshot.bytes,
                output_dir / "inputs" / "families.json",
                "input",
            )
        finally:
            os.close(inputs_fd)
        input_snapshot = replace(input_snapshot, relative_parts=("inputs", "families.json"))
        split = build_family_split_manifest(samples, seed=42)
        split_snapshot = _write_new_regular_file(
            root_fd,
            "split-manifest.json",
            _json_bytes(split),
            output_dir / "split-manifest.json",
            "output",
        )

        bridge = [[[False for _ in range(3)] for _ in range(3)] for _ in range(3)]
        bridge[1][1][0] = bridge[1][1][1] = bridge[1][1][2] = True
        response = {
            "compliance_ratio": 1.10,
            "max_von_mises": 4_000_000.0,
            "max_displacement": 0.020,
        }
        summary = {
            "fixture": "deterministic-small-family-split-v1",
            "sample_count": len(samples),
            "family_count": len({sample["family_id"] for sample in samples}),
            "target_evaluation": {
                name: result.passed for name, result in SMOKE_TARGETS.evaluate(response).items()
            },
            "topology": {
                "foreground_connectivity": 6,
                "background_connectivity": 26,
                "bridge_center_is_simple": is_simple_point_6_26(bridge, (1, 1, 1)),
                "bridge_endpoint_is_simple": is_simple_point_6_26(bridge, (1, 1, 0)),
            },
        }
        summary_snapshot = _write_new_regular_file(
            root_fd,
            "summary.json",
            _json_bytes(summary),
            output_dir / "summary.json",
            "output",
        )
        manifest = build_run_manifest_from_snapshots(
            run_id="sasto-v-smoke-v1",
            inputs={"smoke/families": input_snapshot},
            outputs={
                "smoke/split-manifest": split_snapshot,
                "smoke/summary": summary_snapshot,
            },
            targets=SMOKE_TARGETS,
            split_sha256=split_sha256(split),
            split_artifact="smoke/split-manifest",
        )
        manifest_snapshot = _write_new_regular_file(
            root_fd,
            "run-manifest.json",
            _json_bytes(manifest),
            output_dir / "run-manifest.json",
            "manifest",
        )
        for snapshot in (input_snapshot, split_snapshot, summary_snapshot, manifest_snapshot):
            _revalidate_snapshot(root_fd, snapshot)
        assert_lexical_root_identity(output_dir, root_fd)
    return output_dir / "run-manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a deterministic SASTO-V smoke artifact")
    parser.add_argument("--fixture", default="fixtures/smoke/families.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = run_smoke(Path(args.fixture), Path(args.output))
    print("created {}".format(manifest_path))
    print("manifest_sha256={}".format(sha256_file(manifest_path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
