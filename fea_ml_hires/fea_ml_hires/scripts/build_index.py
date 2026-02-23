"""
Dataset index builder for voxel FEA surrogate training.

Scans the runs directory and creates train/val/test split manifests.

Usage:
    python -m fea_ml.scripts.build_index --runs-dir data/runs --output data/manifests
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def validate_run(run_dir: Path) -> Tuple[bool, str]:
    """Check if a run directory has all required files."""
    required_files = [
        "occ.npz",
        "part.npz",
        "edit_mask.npz",
        "protected_mask.npz",
        "meta.json",
        "targets.json",
    ]
    
    missing = []
    for f in required_files:
        if not (run_dir / f).exists():
            missing.append(f)
    
    if missing:
        return False, f"Missing: {missing}"
    
    return True, "OK"


def extract_design_family(run_name: str) -> str:
    """
    Extract design family from run name for splitting.
    
    Convention: run names like "house_001_v1", "house_001_v2" share family "house_001"
    """
    parts = run_name.split("_")
    if len(parts) >= 2:
        # Assume last part is version, rest is family
        return "_".join(parts[:-1])
    return run_name


def build_splits(
    runs: List[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    split_by_family: bool = True,
) -> Dict[str, List[Path]]:
    """Split runs into train/val/test sets."""
    rng = np.random.default_rng(seed)
    
    if split_by_family:
        # Group by design family
        families: Dict[str, List[Path]] = {}
        for run in runs:
            family = extract_design_family(run.name)
            if family not in families:
                families[family] = []
            families[family].append(run)
        
        # Shuffle families
        family_names = list(families.keys())
        rng.shuffle(family_names)
        
        n_train = int(len(family_names) * train_ratio)
        n_val = int(len(family_names) * val_ratio)
        
        train_families = family_names[:n_train]
        val_families = family_names[n_train:n_train + n_val]
        test_families = family_names[n_train + n_val:]
        
        train = [r for f in train_families for r in families[f]]
        val = [r for f in val_families for r in families[f]]
        test = [r for f in test_families for r in families[f]]
    else:
        # Simple random split
        runs_shuffled = list(runs)
        rng.shuffle(runs_shuffled)
        
        n_train = int(len(runs_shuffled) * train_ratio)
        n_val = int(len(runs_shuffled) * val_ratio)
        
        train = runs_shuffled[:n_train]
        val = runs_shuffled[n_train:n_train + n_val]
        test = runs_shuffled[n_train + n_val:]
    
    return {
        "train": train,
        "val": val,
        "test": test,
    }


def compute_statistics(runs: List[Path]) -> Dict:
    """Compute dataset statistics for documentation."""
    n_runs = len(runs)
    
    # Sample first run for shape info
    if runs:
        sample_run = runs[0]
        occ = np.load(sample_run / "occ.npz")["data"]
        resolution = occ.shape[0]
        
        with open(sample_run / "targets.json", "r") as f:
            targets = json.load(f)
        target_names = list(targets.keys())
    else:
        resolution = None
        target_names = []
    
    # Collect target statistics
    target_values: Dict[str, List[float]] = {n: [] for n in target_names}
    
    for run in runs:
        try:
            with open(run / "targets.json", "r") as f:
                targets = json.load(f)
            for name in target_names:
                if name in targets:
                    target_values[name].append(targets[name])
        except Exception:
            pass
    
    target_stats = {}
    for name, values in target_values.items():
        if values:
            arr = np.array(values)
            target_stats[name] = {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            }
    
    return {
        "n_runs": n_runs,
        "resolution": resolution,
        "target_names": target_names,
        "target_stats": target_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Build dataset index")
    parser.add_argument("--runs-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-by-family", action="store_true", default=True)
    parser.add_argument("--no-split-by-family", action="store_false", dest="split_by_family")
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all run directories
    print(f"Scanning {runs_dir}...")
    all_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    
    # Validate each run
    valid_runs = []
    invalid_runs = []
    
    for d in all_dirs:
        is_valid, msg = validate_run(d)
        if is_valid:
            valid_runs.append(d)
        else:
            invalid_runs.append((d, msg))
    
    print(f"Found {len(valid_runs)} valid runs, {len(invalid_runs)} invalid")
    
    if invalid_runs:
        print("\nInvalid runs:")
        for d, msg in invalid_runs[:10]:
            print(f"  {d.name}: {msg}")
        if len(invalid_runs) > 10:
            print(f"  ... and {len(invalid_runs) - 10} more")
    
    if not valid_runs:
        print("ERROR: No valid runs found")
        return
    
    # Create splits
    splits = build_splits(
        valid_runs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        split_by_family=args.split_by_family,
    )
    
    print(f"\nSplits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    
    # Compute statistics
    stats = compute_statistics(valid_runs)
    
    # Save manifests
    for split_name, runs in splits.items():
        manifest_path = output_dir / f"{split_name}.json"
        with open(manifest_path, "w") as f:
            json.dump([str(r) for r in runs], f, indent=2)
        print(f"Wrote {manifest_path}")
    
    # Save combined info
    index_info = {
        "runs_dir": str(runs_dir),
        "n_valid": len(valid_runs),
        "n_invalid": len(invalid_runs),
        "splits": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "statistics": stats,
        "config": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "split_by_family": args.split_by_family,
        },
    }
    
    with open(output_dir / "index_info.json", "w") as f:
        json.dump(index_info, f, indent=2)
    
    print(f"\nDataset index saved to {output_dir}")


if __name__ == "__main__":
    main()
