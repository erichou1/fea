"""
Batch convert STEP files to mesh (.msh) with physical groups.
Wraps gmsh_fragment_mesh.py for batch processing.
"""
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_single_step(step_path: Path, output_dir: Path, h: float, overwrite: bool, labels_dir: Path = None) -> bool:
    """Process a single STEP file to mesh with fallback strategies."""
    msh_path = output_dir / (step_path.stem + ".msh")
    
    # Skip if exists and not overwriting
    if msh_path.exists() and not overwrite:
        return True
    
    # Check for part labels JSON
    label_json = None
    if labels_dir:
        label_json = labels_dir / f"{step_path.stem}_labels.json"
        if not label_json.exists():
            label_json = None
    
    # Fallback strategy: Try multiple algorithms and mesh sizes
    # Based on testing: Frontal (algo=4) with h=0.10 produces best quality
    strategies = [
        {"algo": 4, "h": h, "name": "Frontal"},        # Best quality (primary)
        {"algo": 1, "h": h, "name": "Delaunay"},       # Robust alternative
        {"algo": 10, "h": h, "name": "HXT"},           # Fast parallel (if others fail)
        {"algo": 4, "h": h * 1.5, "name": "Frontal-coarse"},   # Coarser if needed
        {"algo": 1, "h": h * 2.0, "name": "Delaunay-coarse"},  # Very coarse fallback
    ]
    
    for i, strat in enumerate(strategies):
        try:
            cmd = [
                "python", "gmsh_fragment_mesh.py",
                str(step_path),
                str(msh_path),
                "--h", str(strat["h"]),
                "--msh-version", "2.2",
                "--algo3d", str(strat["algo"]),
            ]
            
            # Add part labels if available
            if label_json:
                cmd.extend(["--part-labels-json", str(label_json)])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                if i > 0:  # Only log if fallback was needed
                    print(f"SUCCESS (fallback {strat['name']}): {step_path.name}")
                return True
            
            # If last strategy, report failure
            if i == len(strategies) - 1:
                print(f"FAILED (all strategies): {step_path.name}")
                print(f"  Last error: {result.stderr[:200]}")
                return False
                
        except subprocess.TimeoutExpired:
            if i == len(strategies) - 1:
                print(f"TIMEOUT: {step_path.name}")
                return False
            continue
        except Exception as e:
            if i == len(strategies) - 1:
                print(f"ERROR: {step_path.name} - {e}")
                return False
            continue
    
    return False


def main():
    parser = argparse.ArgumentParser(description="Batch convert STEP to mesh")
    parser.add_argument("--step-dir", type=str, required=True, help="Directory containing STEP files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for mesh files")
    parser.add_argument("--labels-dir", type=str, default=None, help="Directory containing part label JSONs")
    parser.add_argument("--h", type=float, default=0.10, help="Mesh element size (smaller = more detail, default 0.10)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mesh files")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N files (for testing)")
    
    args = parser.parse_args()
    
    step_dir = Path(args.step_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    labels_dir = Path(args.labels_dir) if args.labels_dir else None
    
    # Find all STEP files
    step_files = sorted(step_dir.glob("*.step"))
    
    if not step_files:
        print(f"No .step files found in {step_dir}")
        return
    
    if args.limit:
        step_files = step_files[:args.limit]
    
    print(f"Found {len(step_files)} STEP files to process")
    print(f"Mesh element size: {args.h}")
    print(f"Workers: {args.workers}")
    if labels_dir:
        print(f"Using part labels from: {labels_dir}")
    else:
        print(f"WARNING: No labels directory - meshes will have generic 'House' label only")
    
    success_count = 0
    failed_files = []
    
    # Process files
    if args.workers == 1:
        # Serial processing for debugging
        for step_file in tqdm(step_files, desc="Processing"):
            if process_single_step(step_file, output_dir, args.h, args.overwrite, labels_dir):
                success_count += 1
            else:
                failed_files.append(step_file.name)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_single_step, step_file, output_dir, args.h, args.overwrite, labels_dir): step_file
                for step_file in step_files
            }
            
            for future in tqdm(as_completed(futures), total=len(step_files), desc="Processing"):
                step_file = futures[future]
                try:
                    if future.result():
                        success_count += 1
                    else:
                        failed_files.append(step_file.name)
                except Exception as e:
                    print(f"Exception processing {step_file.name}: {e}")
                    failed_files.append(step_file.name)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Success: {success_count}/{len(step_files)}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print(f"\nFailed files (first 10):")
        for f in failed_files[:10]:
            print(f"  - {f}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
