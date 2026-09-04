"""Geometry-stage yield on never-used wireframes. No solver. Prints the reject
reasons so the GB200 handoff can size the tranche honestly."""
import sys, json, zipfile, tempfile, io, contextlib, time
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import fresh_tranche as ft
from pathlib import Path
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
ids = ft.list_unused_ids()[:N]
ok, fails = 0, {}
t = time.perf_counter()
for sid in ids:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            occ, prt, prov = ft.wireframe_to_voxels(sid, Path(tmp))
            ok += 1
        except Exception as e:
            fails.setdefault(f"{type(e).__name__}: {str(e)[:80]}", []).append(sid)
print(f"{ok}/{N} passed geometry in {time.perf_counter()-t:.0f}s")
for k, v in fails.items():
    print(f"  {len(v):3d}  {k}  e.g. {v[:4]}")
