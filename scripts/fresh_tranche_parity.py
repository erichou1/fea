"""Parity: run three already-archived wireframes through the fresh chain and compare
their voxelization to the archived one. The legacy thickness seed was Python's
salted hash and cannot be reproduced, so exact equality is not expected; IoU and
label composition tell us whether the fresh houses live in the same distribution."""
import sys, json, io, zipfile, tempfile
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import numpy as np, fresh_tranche as ft
from pathlib import Path
a = zipfile.ZipFile("/Users/eric/workspace/sasto-modernization-control/archives/fea_ml.zip")
rows = []
for sid in ("00000", "00739", "15490"):
    occA = np.load(io.BytesIO(a.read(f"fea_ml/data/runs_real/{sid}/occ.npz")))["data"].astype(bool)
    prtA = np.load(io.BytesIO(a.read(f"fea_ml/data/runs_real/{sid}/part.npz")))["data"]
    metaA = json.loads(a.read(f"fea_ml/data/runs_real/{sid}/meta.json"))
    with tempfile.TemporaryDirectory() as tmp:
        occF, prtF, prov = ft.wireframe_to_voxels(sid, Path(tmp))
    inter = (occA & occF).sum(); union = (occA | occF).sum()
    labA = np.bincount(prtA[occA], minlength=5)[1:]; labF = np.bincount(prtF[occF], minlength=5)[1:]
    rows.append((sid, int(occA.sum()), int(occF.sum()), round(inter / union, 3), round(metaA["voxel_size"], 5),
                 round(prov["voxel_meta"]["voxel_size"], 5), labA.tolist(), labF.tolist()))
print("sid | archived occ | fresh-chain occ | IoU | voxel archived | voxel fresh | labels archived | labels fresh")
for r in rows:
    print(r)
