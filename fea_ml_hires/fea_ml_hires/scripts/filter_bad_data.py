"""
Filter out corrupted / diverged FEA samples from the dataset.

~21.5% of the 14,293 FEA simulations diverged (displacement >> 1 m, compliance ≈ 0).
These corrupt samples make the target distribution bimodal and unlearnable.

Criteria for REMOVING a sample:
  1. max_displacement > DISP_THRESHOLD  (default 1.0 m — physically impossible)
  2. compliance < COMPLIANCE_FLOOR      (default 1e-6 — solver didn't converge)
  3. max_von_mises <= 0 or NaN/Inf in any target

Outputs:
  - <output_dir>/clean_manifest.json  — list of clean sample directory paths
  - <output_dir>/rejected_manifest.json — list of rejected samples + reasons
  - <output_dir>/filter_report.json — summary statistics

Usage:
    python -m fea_ml.scripts.filter_bad_data \
        --runs-dir data/runs_real \
        --output-dir runs/v3
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- Default thresholds ----
DISP_THRESHOLD = 1.0        # meters — any house displacing > 1 m is diverged
COMPLIANCE_FLOOR = 1e-6     # compliance near zero ⇒ solver failure
VON_MISES_FLOOR = 0.0       # stress must be strictly positive

TARGET_NAMES = ["max_von_mises", "max_displacement", "min_safety_factor", "compliance"]


def load_targets(run_dir: Path) -> Optional[Dict[str, float]]:
    """Load targets.json from a run directory."""
    targets_path = run_dir / "targets.json"
    if not targets_path.exists():
        return None
    try:
        with open(targets_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def check_sample(
    targets: Dict[str, float],
    disp_threshold: float = DISP_THRESHOLD,
    compliance_floor: float = COMPLIANCE_FLOOR,
    von_mises_floor: float = VON_MISES_FLOOR,
) -> Tuple[bool, str]:
    """
    Check if a sample has valid (non-diverged) FEA results.

    Returns:
        (is_valid, reason) — reason is "ok" if valid, else describes the problem.
    """
    # Check for NaN / Inf
    for name in TARGET_NAMES:
        val = targets.get(name, None)
        if val is None:
            return False, f"missing target: {name}"
        if not np.isfinite(val):
            return False, f"{name} is NaN/Inf ({val})"

    disp = targets["max_displacement"]
    comp = targets["compliance"]
    vm = targets["max_von_mises"]
    sf = targets["min_safety_factor"]

    # Diverged simulation: absurdly large displacement
    if abs(disp) > disp_threshold:
        return False, f"displacement too large: {disp:.4g} > {disp_threshold}"

    # Solver failure: compliance near zero
    if comp < compliance_floor:
        return False, f"compliance too small: {comp:.4g} < {compliance_floor}"

    # Non-physical: von Mises must be positive
    if vm <= von_mises_floor:
        return False, f"von_mises non-positive: {vm:.4g}"

    # Safety factor should be positive
    if sf <= 0:
        return False, f"safety_factor non-positive: {sf:.4g}"

    return True, "ok"


def filter_dataset(
    runs_dir: Path,
    disp_threshold: float = DISP_THRESHOLD,
    compliance_floor: float = COMPLIANCE_FLOOR,
    von_mises_floor: float = VON_MISES_FLOOR,
) -> Tuple[List[Path], List[Dict], Dict]:
    """
    Scan all samples in runs_dir and classify them as clean or rejected.

    Returns:
        (clean_dirs, rejected_info_list, summary_stats)
    """
    all_dirs = sorted([
        d for d in runs_dir.iterdir()
        if d.is_dir() and (d / "targets.json").exists()
    ])

    clean_dirs: List[Path] = []
    rejected: List[Dict] = []

    # For statistics
    all_targets = {name: [] for name in TARGET_NAMES}
    clean_targets = {name: [] for name in TARGET_NAMES}
    reject_reasons: Dict[str, int] = {}

    for d in tqdm(all_dirs, desc="Filtering samples"):
        targets = load_targets(d)
        if targets is None:
            rejected.append({"path": str(d), "reason": "cannot load targets.json"})
            reject_reasons["cannot load targets.json"] = reject_reasons.get("cannot load targets.json", 0) + 1
            continue

        # Collect all targets for stats
        for name in TARGET_NAMES:
            val = targets.get(name, float("nan"))
            all_targets[name].append(val)

        is_valid, reason = check_sample(targets, disp_threshold, compliance_floor, von_mises_floor)

        if is_valid:
            clean_dirs.append(d)
            for name in TARGET_NAMES:
                clean_targets[name].append(targets[name])
        else:
            rejected.append({
                "path": str(d),
                "reason": reason,
                "targets": {k: targets.get(k) for k in TARGET_NAMES},
            })
            reject_reasons[reason[:60]] = reject_reasons.get(reason[:60], 0) + 1

    # Summary stats
    summary = {
        "total_samples": len(all_dirs),
        "clean_samples": len(clean_dirs),
        "rejected_samples": len(rejected),
        "rejection_rate": f"{len(rejected) / max(len(all_dirs), 1) * 100:.1f}%",
        "reject_reasons": dict(sorted(reject_reasons.items(), key=lambda x: -x[1])),
        "thresholds": {
            "disp_threshold": disp_threshold,
            "compliance_floor": compliance_floor,
            "von_mises_floor": von_mises_floor,
        },
    }

    # Per-target stats (before and after filtering)
    for label, tgt_dict in [("all", all_targets), ("clean", clean_targets)]:
        for name, values in tgt_dict.items():
            arr = np.array(values)
            finite = arr[np.isfinite(arr)]
            if len(finite) > 0:
                summary[f"{label}_{name}_mean"] = float(np.mean(finite))
                summary[f"{label}_{name}_median"] = float(np.median(finite))
                summary[f"{label}_{name}_std"] = float(np.std(finite))
                summary[f"{label}_{name}_min"] = float(np.min(finite))
                summary[f"{label}_{name}_max"] = float(np.max(finite))
                summary[f"{label}_{name}_p01"] = float(np.percentile(finite, 1))
                summary[f"{label}_{name}_p99"] = float(np.percentile(finite, 99))

    return clean_dirs, rejected, summary


def main():
    parser = argparse.ArgumentParser(description="Filter bad FEA samples")
    parser.add_argument("--runs-dir", type=str, required=True,
                        help="Directory with processed run subdirectories (64³ or 128³)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for manifests and report")
    parser.add_argument("--disp-threshold", type=float, default=DISP_THRESHOLD,
                        help=f"Max displacement threshold (default: {DISP_THRESHOLD})")
    parser.add_argument("--compliance-floor", type=float, default=COMPLIANCE_FLOOR,
                        help=f"Min compliance threshold (default: {COMPLIANCE_FLOOR})")
    parser.add_argument("--von-mises-floor", type=float, default=VON_MISES_FLOOR,
                        help=f"Min von Mises threshold (default: {VON_MISES_FLOOR})")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning {runs_dir} ...")

    clean_dirs, rejected, summary = filter_dataset(
        runs_dir,
        disp_threshold=args.disp_threshold,
        compliance_floor=args.compliance_floor,
        von_mises_floor=args.von_mises_floor,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info(f"Total samples  : {summary['total_samples']}")
    logger.info(f"Clean samples  : {summary['clean_samples']}")
    logger.info(f"Rejected       : {summary['rejected_samples']} ({summary['rejection_rate']})")
    logger.info("Rejection reasons:")
    for reason, count in summary["reject_reasons"].items():
        logger.info(f"  {reason:50s} : {count}")
    logger.info("=" * 60)

    # Save manifests
    # Clean manifest: list of sample IDs (not full paths — more portable)
    clean_ids = [d.name for d in clean_dirs]
    with open(output_dir / "clean_manifest.json", "w") as f:
        json.dump({"clean_sample_ids": clean_ids, "count": len(clean_ids)}, f, indent=2)
    logger.info(f"Saved clean manifest: {output_dir / 'clean_manifest.json'} ({len(clean_ids)} samples)")

    # Rejected manifest
    with open(output_dir / "rejected_manifest.json", "w") as f:
        json.dump(rejected, f, indent=2)

    # Filter report
    with open(output_dir / "filter_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved filter report: {output_dir / 'filter_report.json'}")


if __name__ == "__main__":
    main()
