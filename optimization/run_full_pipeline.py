import subprocess
import sys
from pathlib import Path
import os

def run_step(cmd, desc):
    print(f"\n=== {desc} ===")
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)

def main():
    # optimization/run_full_pipeline.py
    # Assumes it is located in optimization/
    ROOT = Path(__file__).parent.resolve()
    
    # Scripts
    wire_script = ROOT / "wireframe_to_volume.py"
    step_script = ROOT / "fea_gmsh_run" / "freecad_00000_parts_to_step.py"
    gmsh_script = ROOT / "fea_gmsh_run" / "gmsh_fragment_mesh.py"
    solver_script = ROOT / "fea_gmsh_run" / "solve_asce7_22_asd_sfepy_ai_labels.py"
    
    # Directories
    data_raw = ROOT / "data" / "3dwire_raw"
    fea_out_dir = ROOT / "fea_gmsh_run" / "fea_out"
    
    # 1. Wireframe -> STL
    print(f"Checking for wireframes in {data_raw}...")
    if not data_raw.exists() or not list(data_raw.glob("*.npz")):
        print("No .npz wireframes found. Please ensure data/3dwire_raw has data.")
        # Proceeding anyway as script might handle generation or user might have paths relative
    
    cmd_wire = [sys.executable, str(wire_script)]
    # You can add arguments to process_batch if needed, e.g. limit count
    # cmd_wire.extend(["--max-files", "5"]) 
    run_step(cmd_wire, "1. Generating STLs from Wireframe")
    
    # 2. STL -> STEP
    print("Converting STLs to STEP...")
    # This invokes FreeCAD. If it fails, user might need to use `FreeCADCmd <script>`
    cmd_step = [sys.executable, str(step_script)]
    try:
        run_step(cmd_step, "2. Converting STLs to STEP")
    except SystemExit:
        print("\n[One-time Warning] If FreeCAD import failed, try running this part with 'FreeCADCmd' or set up PYTHONPATH.")
        print("Continuing check for existing STEP files...")

    # 3 & 4. Mesh and FEA Loop
    # Find STEP files generated in fea_out
    step_files = sorted(fea_out_dir.glob("*_parts.step"))
    if not step_files:
        print(f"No STEP files found in {fea_out_dir}. Cannot proceed to Meshing/FEA.")
        sys.exit(1)
        
    print(f"Found {len(step_files)} STEP files to process.")
    
    for step_file in step_files:
        base_name = step_file.stem.replace("_parts", "")
        msh_file = fea_out_dir / f"{base_name}.msh"
        
        # 3. Mesh
        cmd_mesh = [sys.executable, str(gmsh_script), str(step_file), str(msh_file)]
        # We don't have part-labels-json from FreeCAD, so we run without it.
        # This means Physical Groups will default to "House". 
        # prepare_real_data.py handles labeling via STL voxelization.
        run_step(cmd_mesh, f"3. Meshing {base_name}")
        
        # 4. FEA
        # --write-combo-vtk is CRITICAL for prepare_real_data.py to read fields
        # --write-case-vtk is optional
        cmd_fea = [
            sys.executable, str(solver_script), 
            str(msh_file), 
            "--out-dir", str(fea_out_dir),
            "--write-combo-vtk" 
        ]
        run_step(cmd_fea, f"4. Running FEA for {base_name}")

    print("\n=== Pipeline Execution Complete ===")
    print("Next Step: Run 'fea_ml/fea_ml/scripts/prepare_real_data.py' to generate the ML dataset.")

if __name__ == "__main__":
    main()
