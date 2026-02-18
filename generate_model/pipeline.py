import subprocess
import sys
from pathlib import Path

def run_pipeline():
    print("\n" + "="*70)
    print("COMPLETE FLOOR PLAN TO 3D PIPELINE")
    print("CubiCasa5k + Plan2Scene Integration")
    print("="*70 + "\n")
    
    # Step 1: CubiCasa vectorization
    print("STEP 1: Vectorizing floor plans with CubiCasa5k")
    print("-"*70)
    cubicasa_script = Path("./CubiCasa5k/vectorize.py")
    
    if not cubicasa_script.exists():
        print(f"✗ CubiCasa script not found: {cubicasa_script}")
        return
    
    result = subprocess.run(
        [sys.executable, str(cubicasa_script)],
        cwd=Path("generate_model/CubiCasa5k")
    )
    
    if result.returncode != 0:
        print("\n✗ Vectorization failed!")
        return
    
    # Step 2: 3D generation
    print("\n" + "="*70)
    print("STEP 2: Generating 3D models with Plan2Scene")
    print("-"*70)
    
    result = subprocess.run(
        [sys.executable, "plan2scene_3d_generator.py"]
    )
    
    if result.returncode != 0:
        print("\n✗ 3D generation failed!")
        return
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)
    print("\nYour 3D models are ready:")
    print("  Location: final_3d_models/")
    print("  Formats: .obj, .stl, .ply")
    print("\nView with:")
    print("  - Blender (free)")
    print("  - Windows 3D Viewer")
    print("  - MeshLab (free)")
    print("  - Online: https://3dviewer.net/")

if __name__ == "__main__":
    run_pipeline()
