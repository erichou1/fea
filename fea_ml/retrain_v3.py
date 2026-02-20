"""
V3 Retraining Pipeline – end-to-end script.

Steps:
  1. Filter diverged FEA data  → runs/v3/clean_manifest.json
  2. Generate 128³ voxels for all clean samples (skip existing)
  3. Train v3 ensemble with clean data, weighted loss, R² monitoring

Usage (local — will generate 128³ data, then train):
    python retrain_v3.py --generate-128

Usage (remote GPU server — assume 128³ data already exists):
    python retrain_v3.py

Usage (just generate 128³ data, no training):
    python retrain_v3.py --generate-128 --skip-train
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths relative to fea_ml/
PROJECT_ROOT = Path(__file__).resolve().parent
PARTS_DIR = PROJECT_ROOT.parent / "optimization" / "data" / "3dwire_parts_combined"
FEA_DIR = PROJECT_ROOT.parent / "optimization" / "fea_gmsh_run" / "fea_results"
RUNS_64 = PROJECT_ROOT / "data" / "runs_real"
RUNS_128 = PROJECT_ROOT / "data" / "runs_real_128"
V3_DIR = PROJECT_ROOT / "runs" / "v3"
CONFIG_PATH = PROJECT_ROOT / "configs" / "voxel_config_v3.yaml"


# ── Step 1: Filter bad data ──────────────────────────────────────────────
def step_filter(runs_dir: Path = RUNS_64, output_dir: Path = V3_DIR) -> list[str]:
    """Filter diverged FEA samples. Returns list of clean sample IDs."""
    from fea_ml.scripts.filter_bad_data import filter_dataset

    manifest_path = output_dir / "clean_manifest.json"

    # Skip if already done
    if manifest_path.exists():
        with open(manifest_path) as f:
            data = json.load(f)
        clean_ids = data["clean_sample_ids"]
        logger.info(f"Step 1 SKIP: Clean manifest already exists ({len(clean_ids)} samples)")
        return clean_ids

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_dirs, rejected, summary = filter_dataset(runs_dir)

    clean_ids = [d.name for d in clean_dirs]
    with open(manifest_path, "w") as f:
        json.dump({"clean_sample_ids": clean_ids, "count": len(clean_ids)}, f, indent=2)

    with open(output_dir / "rejected_manifest.json", "w") as f:
        json.dump(rejected, f, indent=2)
    with open(output_dir / "filter_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Step 1 DONE: {summary['clean_samples']} clean / "
                f"{summary['rejected_samples']} rejected ({summary['rejection_rate']})")
    return clean_ids


# ── Step 2: Generate 128³ voxels ─────────────────────────────────────────
def _generate_one(args_tuple):
    """Worker function for parallel 128³ generation."""
    sample_id, parts_dir, fea_dir, output_dir, resolution, yield_stress = args_tuple
    from fea_ml.scripts.prepare_real_data import process_single_sample
    try:
        ok, reason = process_single_sample(
            sample_id, parts_dir, fea_dir, output_dir, resolution, yield_stress
        )
        return sample_id, ok, reason
    except Exception as e:
        return sample_id, False, str(e)


def step_generate_128(
    clean_ids: list[str],
    workers: int = 12,
    resolution: int = 128,
) -> int:
    """Generate 128³ voxel data for all clean samples. Skip existing."""
    RUNS_128.mkdir(parents=True, exist_ok=True)

    # Find which ones already exist
    existing = {d.name for d in RUNS_128.iterdir() if d.is_dir() and (d / "occ.npz").exists()}
    todo = [sid for sid in clean_ids if sid not in existing]

    if not todo:
        logger.info(f"Step 2 SKIP: All {len(clean_ids)} clean samples already have 128³ data")
        return len(existing)

    logger.info(f"Step 2: Generating 128³ for {len(todo)} samples "
                f"({len(existing)} already exist, {len(clean_ids)} total)")

    from fea_ml.scripts.prepare_real_data import DEFAULT_YIELD_STRESS

    args_list = [
        (sid, PARTS_DIR, FEA_DIR, RUNS_128, resolution, DEFAULT_YIELD_STRESS)
        for sid in todo
    ]

    t0 = time.time()
    success = 0
    fail = 0
    from tqdm import tqdm

    if workers <= 1:
        for args in tqdm(args_list, desc="Generating 128³"):
            sid, ok, reason = _generate_one(args)
            if ok:
                success += 1
            else:
                fail += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_generate_one, a): a[0] for a in args_list}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating 128³"):
                sid, ok, reason = fut.result()
                if ok:
                    success += 1
                else:
                    fail += 1

    elapsed = time.time() - t0
    total = len(existing) + success
    logger.info(f"Step 2 DONE: {success} generated, {fail} failed, "
                f"{total} total 128³ samples ({elapsed/60:.1f} min)")
    return total


# ── Step 3: Train ────────────────────────────────────────────────────────
def step_train(device: str = None):
    """Train v3 ensemble using the training script."""
    import subprocess

    cmd = [
        sys.executable, "-m", "fea_ml.scripts.train",
        "--config", str(CONFIG_PATH),
        "--output", str(V3_DIR),
        "--manifest", str(V3_DIR / "clean_manifest.json"),
    ]
    if device:
        cmd += ["--device", device]

    logger.info(f"Step 3: Launching training...")
    logger.info(f"  Command: {' '.join(cmd)}")

    # Run training as subprocess so it gets full stdout/stderr
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        logger.error(f"Training failed with return code {result.returncode}")
        sys.exit(1)
    else:
        logger.info("Step 3 DONE: Training complete")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="V3 Retraining Pipeline")
    parser.add_argument("--generate-128", action="store_true",
                        help="Generate 128³ voxels for all clean samples (slow, ~45 min)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (useful with --generate-128 to just prep data)")
    parser.add_argument("--workers", type=int, default=12,
                        help="Number of parallel workers for 128³ generation (default: 12)")
    parser.add_argument("--device", type=str, default=None,
                        help="Training device (cuda/cpu). Default: auto-detect")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("V3 Retraining Pipeline")
    logger.info("=" * 60)

    # Step 1: Filter
    clean_ids = step_filter()

    # Step 2: Generate 128³ (optional)
    if args.generate_128:
        step_generate_128(clean_ids, workers=args.workers)
    else:
        # Check how many 128³ samples exist
        existing_128 = sum(1 for d in RUNS_128.iterdir()
                           if d.is_dir() and (d / "occ.npz").exists()) if RUNS_128.exists() else 0
        clean_with_128 = sum(1 for sid in clean_ids
                             if (RUNS_128 / sid / "occ.npz").exists()) if RUNS_128.exists() else 0
        logger.info(f"128³ data: {clean_with_128}/{len(clean_ids)} clean samples have 128³ voxels")
        if clean_with_128 < len(clean_ids) * 0.5:
            logger.warning(f"WARNING: Only {clean_with_128}/{len(clean_ids)} 128^3 samples exist! "
                           f"Run with --generate-128 to generate the rest.")

    # Step 3: Train
    if not args.skip_train:
        step_train(device=args.device)
    else:
        logger.info("Step 3 SKIP: --skip-train specified")

    logger.info("=" * 60)
    logger.info("Pipeline complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
