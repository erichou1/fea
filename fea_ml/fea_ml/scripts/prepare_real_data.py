"""
Prepare real FEA simulation data for voxel-based surrogate training.

Reads per-sample STL parts + FEA CSV results and produces the run-directory
layout expected by ``VoxelFEADataset`` / ``build_index.py``:

    <output>/<sample_id>/
        occ.npz          – binary occupancy grid  (D,H,W) uint8
        part.npz         – part labels            (D,H,W) uint8
        edit_mask.npz    – editable regions       (D,H,W) uint8
        protected_mask.npz – protected regions    (D,H,W) uint8
        meta.json        – material / load-case metadata
        targets.json     – envelope FEA targets (keys match voxel_config.yaml)

Usage (full):
    python -m fea_ml.scripts.prepare_real_data \\
        --parts-dir optimization/data/3dwire_parts_combined \\
        --fea-dir  optimization/fea_gmsh_run/fea_results \\
        --output-dir fea_ml/data/runs_real \\
        --resolution 64 --workers 4

Quick sanity check:
    python -m fea_ml.scripts.prepare_real_data \\
        --parts-dir ... --fea-dir ... --output-dir ... \\
        --dry-run --limit 50
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
from tqdm import tqdm

from fea_ml.geometry.voxelize import (
    VoxelizationConfig,
    generate_masks,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOXEL_RESOLUTION = 64

# Part-label mapping (consistent with fea_ml.geometry.voxelize)
PART_LABEL_MAPPING = {
    "exterior_walls": 1,
    "interior_rooms": 2,
    "roof": 3,
    "floor": 4,
    "attic_floor": 4,  # Treat attic floor same as floor
}

# Default material properties used by the FEA solver when not specified
# (see solve_asce7_22_asd_sfepy_ai_labels.py defaults: young=25e9, poisson=0.20, rho=2400)
DEFAULT_YOUNGS_MODULUS = 25e9       # Pa  (concrete-class)
DEFAULT_POISSON_RATIO = 0.20
DEFAULT_DENSITY = 2400.0            # kg/m^3
DEFAULT_YIELD_STRESS = 30e6         # Pa  (conservative concrete tensile-like)
DEFAULT_MATERIAL_TYPE = "concrete"
DEFAULT_LOAD_CASE = "combined"      # envelope over ASD combos
DEFAULT_LENGTH_UNIT = "meters"

# ---------------------------------------------------------------------------
# Helpers – mesh loading & voxelization
# ---------------------------------------------------------------------------
def load_mesh(path: Path) -> Optional[trimesh.Trimesh]:
    """Load a mesh file using trimesh."""
    try:
        mesh = trimesh.load(path, force="mesh")
        return mesh
    except Exception as e:
        logger.error(f"Failed to load mesh {path}: {e}")
        return None


def process_sample_parts(
    sample_id: str, parts_dir: Path, resolution: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    Voxelize all STL parts for one sample into composite occupancy + label grids.

    Uses a single global coordinate system across all parts of the same house.
    Filters out ``_complete`` files – only individual part STLs are used.
    """
    sample_files = sorted(parts_dir.glob(f"{sample_id}_*.stl"))
    # Only individual parts, not combined complete meshes
    sample_files = [f for f in sample_files if "_complete" not in f.stem]

    if not sample_files:
        return None, None, {}

    # ---- collect meshes & compute global bounds ----
    all_meshes: List[Dict[str, Any]] = []
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])

    for f in sample_files:
        part_suffix = f.stem.replace(f"{sample_id}_", "")
        label_id = 0
        for key, val in PART_LABEL_MAPPING.items():
            if key in part_suffix:
                label_id = val
                break

        mesh = load_mesh(f)
        if mesh is None or mesh.vertices.shape[0] == 0:
            continue

        all_meshes.append({"mesh": mesh, "label": label_id, "file": f.name})
        global_min = np.minimum(global_min, mesh.bounds[0])
        global_max = np.maximum(global_max, mesh.bounds[1])

    if not all_meshes:
        return None, None, {}

    # ---- voxel grid parameters ----
    extents = global_max - global_min
    max_extent = float(np.max(extents))
    padding = max_extent * 0.1
    padded = max_extent + 2 * padding
    voxel_size = padded / resolution
    center = (global_min + global_max) / 2
    grid_origin = center - (padded / 2)

    full_occ = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    label_grid = np.zeros((resolution, resolution, resolution), dtype=np.uint8)

    for p in all_meshes:
        mesh: trimesh.Trimesh = p["mesh"]
        label: int = p["label"]
        try:
            vgrid = mesh.voxelized(pitch=voxel_size)
            points = vgrid.points
            indices = np.floor((points - grid_origin) / voxel_size).astype(int)
            mask = (
                (indices[:, 0] >= 0) & (indices[:, 0] < resolution)
                & (indices[:, 1] >= 0) & (indices[:, 1] < resolution)
                & (indices[:, 2] >= 0) & (indices[:, 2] < resolution)
            )
            vi = indices[mask]
            full_occ[vi[:, 0], vi[:, 1], vi[:, 2]] = 1
            label_grid[vi[:, 0], vi[:, 1], vi[:, 2]] = label
        except Exception as e:
            logger.error(f"Voxelizing part {p['file']}: {e}")

    meta = {
        "voxel_size": float(voxel_size),
        "origin": grid_origin.tolist(),
        "bounds_min": global_min.tolist(),
        "bounds_max": global_max.tolist(),
    }
    return full_occ, label_grid, meta


# ---------------------------------------------------------------------------
# Helpers – FEA target extraction
# ---------------------------------------------------------------------------
def _find_fea_csv(sample_id: str, fea_dir: Path) -> Optional[Path]:
    """
    Locate ``fea_labels_combos.csv`` for *sample_id* inside *fea_dir*.

    Convention:  ``<fea_dir>/<id>/fea_labels_combos.csv``
    Tries exact match, then zero-padded (width 5, 3), then un-padded.
    """
    candidates = [sample_id]
    try:
        numeric = int(sample_id)
        for width in (5, 3):
            padded = str(numeric).zfill(width)
            if padded != sample_id:
                candidates.append(padded)
        unpadded = str(numeric)
        if unpadded != sample_id:
            candidates.append(unpadded)
    except ValueError:
        pass
    for cand in candidates:
        csv_path = fea_dir / cand / "fea_labels_combos.csv"
        if csv_path.exists():
            return csv_path
    return None


def extract_fea_targets(
    sample_id: str,
    fea_dir: Path,
    yield_stress: float = DEFAULT_YIELD_STRESS,
) -> Optional[Dict[str, float]]:
    """
    Read ``fea_labels_combos.csv`` and compute **envelope** (worst-case)
    targets keyed exactly as ``voxel_config.yaml`` expects::

        max_von_mises, max_displacement, min_safety_factor, compliance
    """
    import pandas as pd

    csv_path = _find_fea_csv(sample_id, fea_dir)
    if csv_path is None:
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning(f"[{sample_id}] Cannot parse CSV {csv_path}: {e}")
        return None

    if df.empty:
        logger.warning(f"[{sample_id}] CSV is empty: {csv_path}")
        return None

    # Prefer combo rows (the envelope load-combinations)
    if "type" in df.columns:
        combo_df = df[df["type"] == "combo"]
        if combo_df.empty:
            combo_df = df
    else:
        combo_df = df

    required_cols = {"max_u", "max_von_mises", "compliance_like"}
    missing = required_cols - set(combo_df.columns)
    if missing:
        logger.warning(f"[{sample_id}] CSV missing columns {missing}")
        return None

    max_displacement = float(combo_df["max_u"].max())
    max_von_mises = float(combo_df["max_von_mises"].max())
    compliance = float(combo_df["compliance_like"].max())

    # Safety factor: yield / von_mises per combo → take worst (min)
    per_combo_sf = yield_stress / combo_df["max_von_mises"].replace(0, np.nan)
    min_safety_factor = float(per_combo_sf.min())

    return {
        "max_von_mises": max_von_mises,
        "max_displacement": max_displacement,
        "min_safety_factor": min_safety_factor,
        "compliance": compliance,
    }


# ---------------------------------------------------------------------------
# Per-sample pipeline
# ---------------------------------------------------------------------------
def process_single_sample(
    sample_id: str,
    parts_dir: Path,
    fea_dir: Path,
    output_dir: Path,
    resolution: int,
    yield_stress: float,
    dry_run: bool = False,
) -> Tuple[bool, str]:
    """Full preparation for one sample.  Returns ``(success, reason)``."""
    # 1. Geometry → voxels
    occ, labels, geo_meta = process_sample_parts(sample_id, parts_dir, resolution)
    if occ is None:
        return (False, "no STL parts found")
    if int(occ.sum()) == 0:
        return (False, "empty occupancy grid after voxelization")

    # 2. Masks via existing voxelize.py logic
    config = VoxelizationConfig(resolution=resolution)
    edit_mask, protected_mask = generate_masks(occ, labels, config)

    # 3. FEA targets
    targets = extract_fea_targets(sample_id, fea_dir, yield_stress=yield_stress)
    if targets is None:
        return (False, "no FEA CSV or un-parseable")

    # 4. Meta (material + load-case features for VoxelFEADataset)
    meta = {
        # Geometry info
        **geo_meta,
        # Material properties matching FEA solver defaults
        "E": DEFAULT_YOUNGS_MODULUS,
        "youngs_modulus": DEFAULT_YOUNGS_MODULUS,
        "nu": DEFAULT_POISSON_RATIO,
        "poisson_ratio": DEFAULT_POISSON_RATIO,
        "density": DEFAULT_DENSITY,
        "yield_stress": yield_stress,
        # Classification features (for one-hot encoding in VoxelFEADataset)
        "material_type": DEFAULT_MATERIAL_TYPE,
        "material_label": DEFAULT_MATERIAL_TYPE,
        "load_case_id": DEFAULT_LOAD_CASE,
        "load_case": DEFAULT_LOAD_CASE,
        "length_unit": DEFAULT_LENGTH_UNIT,
    }

    if dry_run:
        return (True, "dry-run OK")

    # 5. Save
    sample_out = output_dir / sample_id
    sample_out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(sample_out / "occ.npz", data=occ)
    np.savez_compressed(sample_out / "part.npz", data=labels)
    np.savez_compressed(sample_out / "edit_mask.npz", data=edit_mask)
    np.savez_compressed(sample_out / "protected_mask.npz", data=protected_mask)

    with open(sample_out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(sample_out / "targets.json", "w") as f:
        json.dump(targets, f, indent=2)

    return (True, "saved")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare real-world FEA data for ML surrogate training"
    )
    parser.add_argument(
        "--parts-dir", type=str, required=True,
        help="Directory with STL parts  (e.g. optimization/data/3dwire_parts_combined)",
    )
    parser.add_argument(
        "--fea-dir", type=str, required=True,
        help="Parent dir of per-sample FEA results  "
             "(e.g. optimization/fea_gmsh_run/fea_results – "
             "each subfolder <id>/ contains fea_labels_combos.csv)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/runs_real",
        help="Output dir for processed voxel run directories",
    )
    parser.add_argument("--resolution", type=int, default=VOXEL_RESOLUTION)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--yield-stress", type=float, default=DEFAULT_YIELD_STRESS,
        help=f"Yield stress (Pa) for safety-factor calc.  Default: {DEFAULT_YIELD_STRESS:.0e}",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N samples (debugging)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan & report counts without writing output files",
    )
    args = parser.parse_args()

    parts_dir = Path(args.parts_dir)
    fea_dir = Path(args.fea_dir)
    output_dir = Path(args.output_dir)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---- discover samples: ONLY those with FEA results ----
    # This avoids scanning 78k STL samples when only ~14k have FEA
    fea_sample_ids = set()
    for fea_subdir in fea_dir.iterdir():
        if fea_subdir.is_dir():
            csv_path = fea_subdir / "fea_labels_combos.csv"
            if csv_path.exists():
                fea_sample_ids.add(fea_subdir.name)
    
    # Cross-check: only keep samples that have both FEA results AND STL parts
    stls = sorted(parts_dir.glob("*_exterior_walls.stl"))
    stl_sample_ids = {f.stem.replace("_exterior_walls", "") for f in stls}
    
    # Intersection: samples with both
    sample_ids = sorted(fea_sample_ids & stl_sample_ids)
    
    logger.info(f"Found {len(fea_sample_ids)} FEA results, {len(stl_sample_ids)} STL samples")
    logger.info(f"Processing {len(sample_ids)} samples with both FEA + STL")

    if args.limit:
        sample_ids = sample_ids[: args.limit]
        logger.info(f"Limited to first {args.limit} samples")

    # ---- dry-run report ----
    if args.dry_run:
        # Spot-check a few CSVs for parseability
        n_csv_parseable = 0
        check_csv_n = min(10, len(sample_ids))
        for sid in sample_ids[:check_csv_n]:
            targets = extract_fea_targets(sid, fea_dir, args.yield_stress)
            if targets is not None:
                n_csv_parseable += 1

        logger.info("=== DRY-RUN REPORT ===")
        logger.info(f"  FEA results available       : {len(fea_sample_ids)}")
        logger.info(f"  STL parts available         : {len(stl_sample_ids)}")
        logger.info(f"  Samples with BOTH           : {len(sample_ids)}")
        logger.info(f"  Will process                : {len(sample_ids)}")
        logger.info(f"  FEA CSVs parseable (spot)   : {n_csv_parseable}/{check_csv_n}")

        # Quick voxelization spot-check on first 3
        check_n = min(3, len(sample_ids))
        vox_ok = 0
        for sid in sample_ids[:check_n]:
            occ, _, _ = process_sample_parts(sid, parts_dir, args.resolution)
            if occ is not None and int(occ.sum()) > 0:
                vox_ok += 1
        logger.info(f"  Voxelization spot-check     : {vox_ok}/{check_n} OK")
        return

    # ---- full processing ----
    success_count = 0
    skip_reasons: Dict[str, int] = {}

    def _record(ok: bool, reason: str):
        nonlocal success_count
        if ok:
            success_count += 1
        else:
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    if args.workers <= 1:
        for sid in tqdm(sample_ids, desc="Preparing"):
            try:
                ok, reason = process_single_sample(
                    sid, parts_dir, fea_dir, output_dir,
                    args.resolution, args.yield_stress,
                )
                _record(ok, reason)
            except Exception:
                logger.error(f"[{sid}] exception:\n{traceback.format_exc()}")
                _record(False, "exception")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_single_sample,
                    sid, parts_dir, fea_dir, output_dir,
                    args.resolution, args.yield_stress,
                ): sid
                for sid in sample_ids
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Preparing"):
                sid = futures[fut]
                try:
                    ok, reason = fut.result()
                    _record(ok, reason)
                except Exception:
                    logger.error(f"[{sid}] exception:\n{traceback.format_exc()}")
                    _record(False, "exception")

    logger.info("=== PROCESSING COMPLETE ===")
    logger.info(f"  Success : {success_count}/{len(sample_ids)}")
    if skip_reasons:
        logger.info("  Skipped :")
        for reason, cnt in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            logger.info(f"    {reason:40s} : {cnt}")
    logger.info(f"  Output  : {output_dir}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
