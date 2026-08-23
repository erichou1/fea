"""Fit-role-only real-data probe for the canonical T0 Hex8 verifier.

The role guard completes before a Zip member is opened.  It deliberately never
lists or reports non-fit members, and it does not provide a confirmation override.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .voxel_fea import VoxelFEAConfig, solve_voxels


class FitOnlyAccessError(ValueError):
    """Requested input would breach the frozen fit-only role boundary."""


def _role_ids(manifest: Mapping[str, object], role: str) -> set[str]:
    try:
        values = manifest["partitions"][role]["sample_ids"]  # type: ignore[index]
    except (KeyError, TypeError):
        raise FitOnlyAccessError("split manifest lacks {} membership".format(role)) from None
    if (not isinstance(values, list) or not values or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)):
        raise FitOnlyAccessError("split manifest has invalid {} membership".format(role))
    return set(values)


def select_fit_sample_ids(manifest: Mapping[str, object], requested: Iterable[str] | None, *, limit: int) -> list[str]:
    """Select an ordered fit-only batch after proving every other role is excluded."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise FitOnlyAccessError("limit must be a positive integer")
    fit = _role_ids(manifest, "fit")
    development = _role_ids(manifest, "development")
    calibration = _role_ids(manifest, "calibration")
    confirmation = _role_ids(manifest, "confirmation")
    nonfit = development | calibration | confirmation
    if fit & nonfit or development & calibration or development & confirmation or calibration & confirmation:
        raise FitOnlyAccessError("split manifest roles overlap")
    if requested is None:
        selected = sorted(fit)[:limit]
    else:
        if isinstance(requested, (str, bytes)):
            raise FitOnlyAccessError("requested sample IDs must be a non-string iterable")
        try:
            complete_request = list(requested)
        except TypeError as error:
            raise FitOnlyAccessError("requested sample IDs must be iterable") from error
        if not complete_request or any(not isinstance(sample_id, str) or not sample_id for sample_id in complete_request):
            raise FitOnlyAccessError("requested sample IDs must be non-empty strings")
        if len(set(complete_request)) != len(complete_request):
            raise FitOnlyAccessError("requested sample IDs must be unique")
        confirmation_requested = set(complete_request) & confirmation
        if confirmation_requested:
            raise FitOnlyAccessError("confirmation role requested; fit-only probe denies confirmation access")
        rejected = set(complete_request) - fit
        if rejected:
            raise FitOnlyAccessError("non-fit role requested; fit-only probe denies data access")
        selected = complete_request[:limit]
    return selected


def _member_name(sample_id: str, leaf: str) -> str:
    return "fea_ml/data/runs_real/{}/{}".format(sample_id, leaf)


def _read_member(archive: zipfile.ZipFile, sample_id: str, leaf: str) -> bytes:
    try:
        with archive.open(_member_name(sample_id, leaf), "r") as member:
            return member.read()
    except KeyError as error:
        raise FitOnlyAccessError("fit sample payload is missing required {}".format(leaf)) from error


def _load_occupancy(archive: zipfile.ZipFile, sample_id: str) -> np.ndarray:
    payload = _read_member(archive, sample_id, "occ.npz")
    with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
        if loaded.files != ["data"]:
            raise FitOnlyAccessError("fit occupancy archive must contain exactly data")
        data = loaded["data"]
    if data.shape != (64, 64, 64) or data.dtype not in (np.dtype(np.uint8), np.dtype(np.bool_)):
        raise FitOnlyAccessError("fit occupancy has an invalid schema")
    if not np.all((data == 0) | (data == 1)):
        raise FitOnlyAccessError("fit occupancy must be binary")
    return data.astype(bool, copy=True)


def _configuration(archive: zipfile.ZipFile, sample_id: str, fixed_force: tuple[float, float, float]) -> VoxelFEAConfig:
    raw = _read_member(archive, sample_id, "meta.json")
    try:
        metadata = json.loads(raw)
        voxel_size = float(metadata["voxel_size"])
        youngs = float(metadata.get("E", 25.0e9))
        poisson = float(metadata.get("nu", 0.20))
        density = float(metadata.get("density", 2400.0))
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise FitOnlyAccessError("fit metadata is malformed") from error
    return VoxelFEAConfig(
        voxel_size=(voxel_size, voxel_size, voxel_size), youngs_modulus_pa=youngs, poisson_ratio=poisson,
        density_kg_m3=density, include_self_weight=True, fixed_total_force_n=fixed_force,
    )


def run_fit_probe(
    *, split_manifest: Path, archive_path: Path, sample_ids: Iterable[str] | None, limit: int,
    fixed_force: tuple[float, float, float],
) -> dict[str, object]:
    """Run the declared batch; role validation occurs before opening the archive."""
    manifest = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    selected = select_fit_sample_ids(manifest, sample_ids, limit=limit)
    records = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        for sample_id in selected:
            occupancy = _load_occupancy(archive, sample_id)
            result = solve_voxels(occupancy, _configuration(archive, sample_id, fixed_force))
            records.append({"sample_id": sample_id, "record": result})
    return {
        "schema_version": "1.0.0", "role": "fit", "confirmation_accessed": False,
        "development_accessed": False, "calibration_accessed": False, "sample_count": len(records), "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the canonical Hex8 verifier only on frozen fit-role samples.")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--fixed-force-z", type=float, default=-100.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_fit_probe(
            split_manifest=args.split_manifest, archive_path=args.archive, sample_ids=args.sample_ids, limit=args.limit,
            fixed_force=(0.0, 0.0, args.fixed_force_z),
        )
        if args.output.exists():
            raise FitOnlyAccessError("output must be a new append-only path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print("REJECTED: {}".format(error))
        return 2
    print("FIT_ONLY_PROBE: samples={} output={}".format(result["sample_count"], args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
