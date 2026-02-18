"""
Batch Runner for ASCE 7-22 FEA Simulations (SfePy)
Orchestrates parallel execution of solve_asce7_22_asd_sfepy_ai_labels.py
"""
import argparse
import subprocess
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os

def check_solver_exists(solver_name="solve_asce7_22_asd_sfepy_ai_labels.py"):
    """Verify the solver script exists."""
    solver = Path(solver_name).resolve()
    if not solver.exists():
        raise FileNotFoundError(f"Solver script not found at {solver}")
    return solver

def run_single_simulation(args_tuple):
    """
    Worker function to run a single FEA simulation.
    args_tuple: (msh_path, output_root, solver_script, overwrite, timeout)
    """
    msh_path, output_root, solver_script, overwrite, timeout = args_tuple
    
    # IMPORTANT: Convert to absolute paths to avoid path resolution issues
    msh_path = Path(msh_path).resolve()
    output_root = Path(output_root).resolve()
    
    sample_id = msh_path.stem.replace('_parts', '').replace('_house_coupled', '')
    
    # Create isolated output directory for this sample
    # structure: output_root/00000/
    sample_out_dir = output_root / sample_id
    
    # Check if result already exists
    result_csv = sample_out_dir / "fea_labels_combos.csv"
    if result_csv.exists() and not overwrite:
        return (sample_id, "SKIPPED", 0.0)

    # Clean partial runs
    if sample_out_dir.exists():
        shutil.rmtree(sample_out_dir)
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    
    # Build command - use ABSOLUTE paths
    cmd = [
        "python", str(solver_script),
        str(msh_path),  # Now absolute
        "--out-dir", str(sample_out_dir),  # Now absolute
        "--young", "25e9",   # Default Concrete E
        "--poisson", "0.20", # Default Concrete nu
    ]

    try:
        # Run process
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=solver_script.parent 
        )

        duration = time.time() - start_time

        if result.returncode == 0 and result_csv.exists():
            return (sample_id, "SUCCESS", duration)
        else:
            # Save error log
            with open(sample_out_dir / "error.log", "w") as f:
                f.write(f"Return Code: {result.returncode}\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
            return (sample_id, "FAILED", duration)

    except subprocess.TimeoutExpired:
        with open(sample_out_dir / "error.log", "w") as f:
            f.write(f"Process timed out after {timeout} seconds")
        return (sample_id, "TIMEOUT", timeout)
    except Exception as e:
        return (sample_id, f"ERROR: {str(e)}", 0.0)

def aggregate_results(output_root):
    """Combine all individual CSVs into one master dataset."""
    print("Aggregating results...")
    all_dfs = []
    
    csv_files = list(output_root.glob("*/fea_labels_combos.csv"))
    for f in tqdm(csv_files, desc="Reading CSVs"):
        try:
            df = pd.read_csv(f)
            # Add sample_id column if not present or infer from path
            sample_id = f.parent.name
            df['sample_id'] = sample_id
            all_dfs.append(df)
        except Exception:
            pass
            
    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)
        master_csv = output_root / "master_fea_results.csv"
        master_df.to_csv(master_csv, index=False)
        print(f"Saved master results to: {master_csv}")
        print(f"Total samples: {len(master_df['sample_id'].unique())}")
    else:
        print("No results found to aggregate.")

def main():
    parser = argparse.ArgumentParser(description="Batch FEA Runner")
    parser.add_argument("--mesh-dir", type=str, required=True, help="Directory containing .msh files")
    parser.add_argument("--output-dir", type=str, required=True, help="Root directory for results")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N meshes")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per simulation (seconds)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--fast-mode", action="store_true", help="Use fast solver (4 load cases instead of 8, ~3x faster)")
    args = parser.parse_args()

    mesh_dir = Path(args.mesh_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select solver based on mode
    solver_name = "solve_fast_fea.py" if args.fast_mode else "solve_asce7_22_asd_sfepy_ai_labels.py"
    solver_script = check_solver_exists(solver_name)

    # Find meshes
    # Prioritize meshes that look like '00000_parts.msh'
    meshes = sorted(list(mesh_dir.glob("*_parts.msh")))
    if not meshes:
        # Fallback to any msh
        meshes = sorted(list(mesh_dir.glob("*.msh")))
    
    if args.limit:
        meshes = meshes[:args.limit]

    print(f"Found {len(meshes)} meshes to process.")
    print(f"Output Directory: {output_dir}")
    print(f"Workers: {args.workers}")

    # Prepare Tasks
    tasks = [
        (m, output_dir, solver_script, args.overwrite, args.timeout) 
        for m in meshes
    ]

    # Run
    stats = {"SUCCESS": 0, "FAILED": 0, "TIMEOUT": 0, "SKIPPED": 0, "ERROR": 0}
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Use simple map or as_completed
        futures = {executor.submit(run_single_simulation, t): t[0] for t in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Simulating"):
            sid, status, duration = future.result()
            
            # Categorize status generically if error
            cat = status if status in stats else "ERROR"
            stats[cat] += 1
            
            if status == "FAILED":
                print(f"Failed: {sid} (See logs in {output_dir}/{sid}/)")

    print("\n" + "="*40)
    print("Processing Complete.")
    print(f"Stats: {stats}")
    
    # Aggregate results at the end
    aggregate_results(output_dir)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
