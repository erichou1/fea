"""Strict command-line verification of a SASTO-V run manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import ManifestVerificationError, verify_run_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify all declared SASTO-V artifact hashes")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        verify_run_manifest(args.manifest)
    except ManifestVerificationError as error:
        print("REJECTED: {}".format(error))
        return 2
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    print(
        "VERIFIED: run_id={} schema_version={} inputs={} outputs={}".format(
            manifest["run_id"],
            manifest["schema_version"],
            len(manifest["inputs"]),
            len(manifest["outputs"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
