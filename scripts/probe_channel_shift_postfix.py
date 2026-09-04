"""Post-fix confirmation: G3's _channels on a BASELINE state must equal the G2
training representation bit-for-bit. Before the fix, |dmu| median was ~0.58."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from sasto.g3_trajectory_calibration import EnsemblePredictor, _channels

ROOT  = Path("artifacts/g2/ensemble-v1")
CACHE = Path("artifacts/g2/ingest-cache-v1/79640406e1e0921c-b7066e14c6713eb6")
row = json.loads((CACHE / "cache-manifest.json").read_text())["roles"]["development"]
PER = 131_072
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
p = EnsemblePredictor(ensemble_root=ROOT, normalization_path=ROOT/"normalization-stats.json", device="cpu")
T = ("compliance", "max_displacement", "max_von_mises")
maxd = {t: 0.0 for t in T}; chan_mismatch = 0
with open(CACHE / row["data_file"], "rb") as fh:
    for i in range(N):
        fh.seek(i * PER); raw = fh.read(PER)
        occ = np.unpackbits(np.frombuffer(raw[:32_768], dtype=np.uint8), bitorder="little")[:64**3].reshape(64,64,64)
        pb = np.unpackbits(np.frombuffer(raw[32_768:], dtype=np.uint8), bitorder="little")[:64**3*3].reshape(64**3,3)
        parts = (pb[:,0] | (pb[:,1]<<1) | (pb[:,2]<<2)).reshape(64,64,64).astype(np.uint8)
        g2_repr = np.stack([occ, parts]).astype(np.float32)          # what G2 trained on
        g3_repr = _channels(occ.astype(np.bool_), parts)              # what G3 now produces
        if not np.array_equal(g2_repr, g3_repr): chan_mismatch += 1
        a = p.predict(g2_repr); b = p.predict(g3_repr)
        for t in T: maxd[t] = max(maxd[t], abs(b["mu"][t] - a["mu"][t]))
print(f"n={N}")
print(f"channel representation mismatches: {chan_mismatch}")
for t in T: print(f"  max |dmu| {t:<18} {maxd[t]:.2e}")
ok = chan_mismatch == 0 and all(v == 0.0 for v in maxd.values())
print("\nVERDICT:", "FIXED -- G3 channels identical to G2 training representation, zero prediction shift" if ok else "STILL BROKEN")
