import argparse
from pathlib import Path
import gmsh

def safe_remove_all_duplicates():
    try:
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
    except Exception as e:
        print(f"[warn] removeAllDuplicates skipped: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", type=str)
    ap.add_argument("msh", type=str)
    ap.add_argument("--h", type=float, default=0.10)
    ap.add_argument("--msh-version", type=float, default=2.2)
    ap.add_argument("--algo3d", type=int, default=10, help="Try 10 (HXT), 1 (Delaunay), 4 (Frontal).")
    ap.add_argument("--debug-brep", type=int, default=1)
    args = ap.parse_args()

    step_path = Path(args.step)
    msh_path = Path(args.msh)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(step_path.stem)

    # IMPORTANT: do NOT enable OCCAutoFix/MakeSolids here, since it can erase solids
    # when it fails to "make solid". [web:297]
    gmsh.model.occ.importShapes(str(step_path))
    gmsh.model.occ.synchronize()

    safe_remove_all_duplicates()

    vols = gmsh.model.getEntities(3)
    print(f"Imported volumes (dim=3): {len(vols)}")
    if not vols:
        if args.debug_brep:
            dbg = msh_path.with_suffix(".import_only.brep")
            gmsh.write(str(dbg))
            print("Wrote debug geometry:", dbg)
        gmsh.finalize()
        raise RuntimeError("No volumes in STEP (dim=3) after import (before fragment).")

    if len(vols) > 1:
        # Fragment ALL volumes against each other so none gets treated as a removable "tool".
        # Signature can vary, so try the explicit form first.
        try:
            gmsh.model.occ.fragment(vols, [], removeObject=True, removeTool=True)
        except TypeError:
            gmsh.model.occ.fragment(vols, [])
        gmsh.model.occ.synchronize()
        safe_remove_all_duplicates()
        vols = gmsh.model.getEntities(3)
        print(f"Volumes after fragment: {len(vols)}")

    if args.debug_brep:
        dbg = msh_path.with_suffix(".brep")
        gmsh.write(str(dbg))
        print("Wrote debug geometry:", dbg)

    tags = [t for (d, t) in vols]
    gmsh.model.addPhysicalGroup(3, tags, 1)
    gmsh.model.setPhysicalName(3, 1, "House")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(args.h))
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(args.h))
    gmsh.option.setNumber("Mesh.MshFileVersion", float(args.msh_version))
    gmsh.option.setNumber("Mesh.Algorithm3D", int(args.algo3d))  # algorithm choice documented by Gmsh. [web:19]

    gmsh.model.mesh.generate(3)

    msh_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write(str(msh_path))
    gmsh.finalize()
    print("Wrote mesh:", msh_path)

if __name__ == "__main__":
    main()
