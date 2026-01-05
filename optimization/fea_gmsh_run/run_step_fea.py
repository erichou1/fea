import subprocess
from pathlib import Path


def main():
    # optimization/fea_gmsh_run/run_batch_fea.py  -> ROOT = optimization
    ROOT = Path(__file__).resolve().parent.parent

    STL_DIR = ROOT / "data" / "3dwire_parts_combined"
    OUT_DIR = Path(__file__).resolve().parent / "fea_out"   # optimization/fea_gmsh_run/fea_out

    H = 0.25
    ANGLE = 40.0

    YOUNG = 2.0e11
    POISSON = 0.30
    DISP_X = 1e-3

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stls = sorted(STL_DIR.glob("*_complete.stl"))
    if not stls:
        raise FileNotFoundError(f"No *_complete.stl found in {STL_DIR}")

    for stl in stls:
        msh = OUT_DIR / (stl.stem + ".msh")
        vtk = OUT_DIR / (stl.stem + "_u.vtk")

        subprocess.check_call([
            "python", "mesh_stl_to_tet_msh.py",
            str(stl), str(msh),
            "--h", str(H),
            "--angle", str(ANGLE),
        ])

        subprocess.check_call([
            "python", "solve_linear_elasticity_sfepy.py",
            str(msh),
            "--young", str(YOUNG),
            "--poisson", str(POISSON),
            "--disp-x", str(DISP_X),
            "--out", str(vtk),
        ])

        print(f"OK: {stl.name} -> {vtk.name}")

    print(f"Done. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
