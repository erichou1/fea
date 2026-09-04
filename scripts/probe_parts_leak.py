"""Are part labels nonzero anywhere occupancy is zero?

G2 training used   channels = [occ, parts]            (raw parts)
G3 trajectories use channels = [occ, parts * occ]      (masked parts)

If parts is ever nonzero where occ == 0, those two are DIFFERENT inputs to the
same surrogate, and G3 is feeding the ensemble a distribution it was not
trained on. That is a train/inference mismatch, not a numerics issue.
"""
import json, sys
from pathlib import Path
import numpy as np

CACHE = Path("artifacts/g2/ingest-cache-v1/79640406e1e0921c-b7066e14c6713eb6")
manifest = json.loads((CACHE / "cache-manifest.json").read_text())
row = manifest["roles"]["development"]
PER = 131_072
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200

leak_samples = 0
leak_voxels_total = 0
with open(CACHE / row["data_file"], "rb") as fh:
    for i in range(N):
        fh.seek(i * PER); raw = fh.read(PER)
        occ = np.unpackbits(np.frombuffer(raw[:32_768], dtype=np.uint8), bitorder="little")[:64**3]
        pb = np.unpackbits(np.frombuffer(raw[32_768:], dtype=np.uint8), bitorder="little")[:64**3*3].reshape(64**3, 3)
        parts = pb[:,0] | (pb[:,1]<<1) | (pb[:,2]<<2)
        leak = int(np.sum((parts != 0) & (occ == 0)))
        if leak:
            leak_samples += 1; leak_voxels_total += leak
            if leak_samples <= 5:
                print(f"  {row['sample_ids'][i]}: {leak:,} voxels have part!=0 but occ==0  (of {int(occ.sum()):,} occupied)")

print(f"\nchecked {N} development samples")
print(f"samples where parts leak outside occupancy: {leak_samples}/{N}")
print(f"total leaking voxels: {leak_voxels_total:,}")
print()
if leak_samples:
    print("VERDICT: G2 (raw parts) and G3 (masked parts) feed the surrogate DIFFERENT inputs.")
    print("         This is a train/inference mismatch. q_base drift is a symptom of it.")
else:
    print("VERDICT: parts is always zero outside occupancy; masking is a no-op; inputs identical.")
