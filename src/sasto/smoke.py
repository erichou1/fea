"""Deterministic public SASTO-V smoke artifact generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .manifest import ManifestVerificationError, build_run_manifest, has_lexical_symlink_component
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


def run_smoke(fixture_path: Path, output_dir: Path) -> Path:
    """Write one immutable smoke artifact; an existing run root is rejected."""
    fixture_path = Path(fixture_path).resolve()
    output_dir = Path(output_dir)
    if has_lexical_symlink_component(output_dir):
        raise ManifestVerificationError("artifact root must not contain symlink components")
    output_dir = output_dir.resolve()
    if not fixture_path.is_file():
        raise FileNotFoundError("smoke fixture is missing: {}".format(fixture_path))
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite existing artifact root: {}".format(output_dir))
    samples = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("smoke fixture must be a JSON list")
    output_dir.mkdir(parents=True)
    bundled_fixture = output_dir / "inputs" / "families.json"
    bundled_fixture.parent.mkdir()
    shutil.copyfile(fixture_path, bundled_fixture)
    split = build_family_split_manifest(samples, seed=42)
    split_path = output_dir / "split-manifest.json"
    split_path.write_text(json.dumps(split, sort_keys=True, indent=2) + "\n", encoding="utf-8")

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
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    manifest = build_run_manifest(
        run_id="sasto-v-smoke-v1",
        inputs={"smoke/families": bundled_fixture},
        outputs={"smoke/split-manifest": split_path, "smoke/summary": summary_path},
        targets=SMOKE_TARGETS,
        split_sha256=split_sha256(split),
        split_artifact="smoke/split-manifest",
        artifact_root=output_dir,
    )
    manifest_path = output_dir / "run-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a deterministic SASTO-V smoke artifact")
    parser.add_argument("--fixture", default="fixtures/smoke/families.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = run_smoke(Path(args.fixture), Path(args.output))
    print("created {}".format(manifest_path))
    print("manifest_sha256={}".format(hashlib.sha256(manifest_path.read_bytes()).hexdigest()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
