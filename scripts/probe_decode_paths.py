"""Do the two decode paths produce bit-identical channels?

Path A (old, used when v1/v2 q_base was computed): archive -> RoleDataset._payload
Path B (new, used by seqtest):                     certified G2 ingest cache

If they differ for any calibration sample, that is the source of the q_base drift
and it is an INPUT inconsistency, not floating-point nondeterminism.
"""
import json, sys, zipfile, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from sasto.g3_trajectory_calibration import open_g3_role, _channels

SPLIT = "/Users/eric/workspace/sasto-modernization-control/v2/g1/split-manifest.json"
ARCH  = "/Users/eric/workspace/sasto-modernization-control/archives/fea_ml.zip"
CACHE = Path("artifacts/g2/ingest-cache-v1/79640406e1e0921c-b7066e14c6713eb6")

role = open_g3_role(role="development", split_manifest=Path(SPLIT),
    expected_split_sha256="ca526a068137308ca4bb05325d62bab5a7ad45c81d54566d5fa8e3ef62a91650",
    archive=Path(ARCH), expected_archive_sha256="79640406e1e0921c0ccfdc1df7ce51e05a8ecfece2ceccb7dec42c981736beda",
    g1b_root=Path("artifacts/g1b/relabel-v3"),
    expected_cohort_manifest_sha256="b7066e14c6713eb69e1555f7ccacae4d82bb1fd092eef61ab113bf3ee540b8d8",
    expected_cluster_role_manifest_sha256="9c3691f523b681b0bffaa26f9559b7a4008c096cea52c32d2b74ab1c20394227")

manifest = json.loads((CACHE / "cache-manifest.json").read_text())
row = manifest["roles"]["development"]
ids = row["sample_ids"]
PER = 131_072

def cache_channels(i: int) -> np.ndarray:
    with open(CACHE / row["data_file"], "rb") as fh:
        fh.seek(i * PER); raw = fh.read(PER)
    occ = np.unpackbits(np.frombuffer(raw[:32_768], dtype=np.uint8), bitorder="little")[:64**3].reshape(64,64,64)
    pb = np.unpackbits(np.frombuffer(raw[32_768:], dtype=np.uint8), bitorder="little")[:64**3*3].reshape(64**3,3)
    parts = (pb[:,0] | (pb[:,1]<<1) | (pb[:,2]<<2)).reshape(64,64,64)
    return np.stack([occ, parts]).astype(np.float32)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
mism = 0
for i in range(N):
    sid = ids[i]
    packed, parts = role.dataset._payload(sid)
    occ = np.unpackbits(packed, bitorder="little")[:64**3].reshape(64,64,64).astype(np.uint8)
    a = _channels(occ, parts)
    b = cache_channels(i)
    if a.shape != b.shape or not np.array_equal(a, b):
        mism += 1
        d = int(np.sum(a != b))
        print(f"MISMATCH {sid}: {d} differing voxels; occ diff={int(np.sum(a[0]!=b[0]))} parts diff={int(np.sum(a[1]!=b[1]))}")
print(f"\ncompared {N} development samples: {mism} mismatched")
print("VERDICT:", "decode paths DIFFER -> q drift is an input inconsistency" if mism else "decode paths identical -> drift source is elsewhere")
