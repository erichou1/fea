from pathlib import Path
import os
import sys
import traceback

import FreeCAD as App
import Mesh
import Part


ROOT = Path(__file__).resolve().parent.parent
PARTS_DIR = ROOT / "data" / "3dwire_parts_combined"
OUT_STEP = Path(__file__).resolve().parent / "fea_out" / "00000_parts.step"

PREFIX = "00000"

# ---- Include attic floor explicitly (your previous script did not) ----
PARTS = [
    f"{PREFIX}_exterior_walls.stl",
    f"{PREFIX}_interior_rooms.stl",
    f"{PREFIX}_floor.stl",
    f"{PREFIX}_attic_floor.stl",
    f"{PREFIX}_roof.stl",
]

# removeSplitter can help sometimes, but may break some meshes.
REFINE_BY_FILE = {
    f"{PREFIX}_exterior_walls.stl": True,
    f"{PREFIX}_interior_rooms.stl": False,
    f"{PREFIX}_floor.stl": True,
    f"{PREFIX}_attic_floor.stl": True,
    f"{PREFIX}_roof.stl": True,
}

# Optional per-file sewing tolerance override (else: auto from bbox)
SEW_TOL_BY_FILE = {
    # f"{PREFIX}_roof.stl": 0.002,
    # f"{PREFIX}_attic_floor.stl": 0.002,
}

# Auto tolerance: small fraction of bbox diagonal, clamped.
AUTO_TOL_FRAC_OF_DIAG = 1.0e-4
AUTO_TOL_MIN = 1.0e-6
AUTO_TOL_MAX = 5.0e-3  # critical for thin roof/attic (avoid "over-sew") [web:884]


def log(msg: str):
    print(msg, flush=True)
    try:
        App.Console.PrintMessage(msg + "\n")
    except Exception:
        pass


def auto_sew_tol(mesh: Mesh.Mesh) -> float:
    bb = mesh.BoundBox
    diag = float(bb.DiagonalLength)
    tol = diag * float(AUTO_TOL_FRAC_OF_DIAG)
    tol = max(float(AUTO_TOL_MIN), tol)
    tol = min(float(AUTO_TOL_MAX), tol)
    return float(tol)


def try_mesh_cleanup(mesh: Mesh.Mesh, name: str):
    """
    Best-effort mesh cleanup. Not all methods exist in all FreeCAD versions,
    so everything is guarded.
    """
    # Some FreeCAD builds expose these as methods; some don’t. Keep it safe.
    for meth in [
        "removeDuplicatedPoints",
        "removeDuplicatedFacets",
        "removeDegeneratedFacets",
        "removeNonManifolds",
        "harmonizeNormals",
        "flipNormals",
    ]:
        fn = getattr(mesh, meth, None)
        if callable(fn):
            try:
                fn()
                log(f"[mesh_cleanup] {name}: {meth}() OK")
            except Exception as e:
                log(f"[mesh_cleanup] {name}: {meth}() failed: {e}")


def ensure_valid_solid(sol, tag: str):
    if getattr(sol, "ShapeType", None) != "Solid":
        raise RuntimeError(f"{tag}: not a Solid (ShapeType={getattr(sol,'ShapeType',None)})")
    try:
        if not sol.isValid():
            raise RuntimeError(f"{tag}: Solid isValid() == False")
    except Exception:
        pass
    vol = float(getattr(sol, "Volume", 0.0))
    if vol <= 0.0:
        raise RuntimeError(f"{tag}: Solid volume <= 0 ({vol})")


def shape_to_solids(shape, tag: str):
    """
    Return a list of TopoDS solids to export as separate STEP bodies.
    This avoids exporting Compounds/Shells (which many tools treat as surfaces). [web:948]
    """
    st = getattr(shape, "ShapeType", None)
    log(f"[shape] {tag}: ShapeType={st} Faces={len(getattr(shape,'Faces',[]))}")

    # If FreeCAD already recognizes solids, use them directly
    try:
        solids = list(shape.Solids)
    except Exception:
        solids = []

    if solids:
        out = []
        for i, s in enumerate(solids):
            ensure_valid_solid(s, f"{tag}:solid[{i}]")
            out.append(s)
        return out

    # Otherwise, try to build solids from closed shells
    shells = []
    try:
        shells = list(shape.Shells)
    except Exception:
        shells = []

    if not shells:
        # fallback: construct a shell from all faces
        try:
            shells = [Part.Shell(shape.Faces)]
        except Exception as e:
            raise RuntimeError(f"{tag}: cannot build Shell from faces: {e}")

    out = []
    for i, sh in enumerate(shells):
        try:
            closed = bool(sh.isClosed())
            log(f"[shell] {tag}: shell[{i}] closed={closed} type={getattr(sh,'ShapeType',None)}")
            if not closed:
                continue
            sol = Part.makeSolid(sh)  # standard mesh->shape->solid pipeline [web:948]
            ensure_valid_solid(sol, f"{tag}:fromShell[{i}]")
            out.append(sol)
        except Exception as e:
            log(f"[warn] {tag}: shell[{i}] -> makeSolid failed: {e}")

    if not out:
        raise RuntimeError(f"{tag}: no solids could be created (shape is not a closed shell)")

    return out


def stl_to_solids(stl_path: Path, do_refine: bool):
    log(f"[stl] Loading: {stl_path}")
    mesh = Mesh.Mesh(str(stl_path))

    try_mesh_cleanup(mesh, stl_path.name)

    tol = float(SEW_TOL_BY_FILE.get(stl_path.name, auto_sew_tol(mesh)))
    log(f"[stl] {stl_path.name}: sew_tol={tol:g}")

    shape = Part.Shape()
    # Create B-Rep faces from triangles + sew within tolerance. [web:881]
    shape.makeShapeFromMesh(mesh.Topology, tol)

    if do_refine:
        try:
            shape = shape.removeSplitter()
            log(f"[stl] {stl_path.name}: removeSplitter OK")
        except Exception as e:
            log(f"[warn] {stl_path.name}: removeSplitter failed: {e}")

    # Optional: enforce a smaller tolerance on the resulting shape (can help export stability)
    try:
        shape.fixTolerance(1e-7)
    except Exception:
        pass

    solids = shape_to_solids(shape, stl_path.name)
    log(f"[stl] {stl_path.name}: produced {len(solids)} solid(s)")
    return solids


def main():
    log(f"START FreeCAD STL->STEP (pid={os.getpid()})")
    log(f"PARTS_DIR={PARTS_DIR}")
    log(f"OUT_STEP={OUT_STEP}")

    paths = [PARTS_DIR / n for n in PARTS]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing input STL: {p}")

    doc = App.newDocument(f"{PREFIX}_parts")
    export_objs = []

    for p in paths:
        do_refine = REFINE_BY_FILE.get(p.name, True)
        log(f"\nConverting: {p.name} (refine={do_refine})")

        solids = stl_to_solids(p, do_refine=do_refine)

        # Export each solid as its own object (avoid exporting Compounds) [web:948]
        if len(solids) == 1:
            obj = doc.addObject("Part::Feature", p.stem)
            obj.Shape = solids[0]
            export_objs.append(obj)
        else:
            for i, s in enumerate(solids):
                obj = doc.addObject("Part::Feature", f"{p.stem}_s{i:02d}")
                obj.Shape = s
                export_objs.append(obj)

    doc.recompute()

    OUT_STEP.parent.mkdir(parents=True, exist_ok=True)
    Part.export(export_objs, str(OUT_STEP))  # STEP export behavior depends on prefs [web:960]
    log(f"Wrote: {OUT_STEP}")


# IMPORTANT for FreeCADCmd: run unconditionally
try:
    main()
except Exception:
    log("FATAL ERROR (traceback below):")
    log(traceback.format_exc())
    sys.exit(1)
