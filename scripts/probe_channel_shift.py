"""How much does the G2/G3 channel mismatch move the surrogate's output?

Feed the same development samples through the frozen ensemble twice:
  raw    = [occ, parts]         (what G2 trained on)
  masked = [occ, parts * occ]   (what G3 uses at every trajectory state)
and measure the shift in mu and sigma on the normalized-log scale.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from sasto.g3_trajectory_calibration import EnsemblePredictor

ROOT  = Path("artifacts/g2/ensemble-v1")
CACHE = Path("artifacts/g2/ingest-cache-v1/79640406e1e0921c-b7066e14c6713eb6")
row = json.loads((CACHE / "cache-manifest.json").read_text())["roles"]["development"]
PER = 131_072
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100

p = EnsemblePredictor(ensemble_root=ROOT, normalization_path=ROOT/"normalization-stats.json", device="cpu")
T = ("compliance", "max_displacement", "max_von_mises")
dmu = {t: [] for t in T}; dsig = {t: [] for t in T}; sig_raw = {t: [] for t in T}

with open(CACHE / row["data_file"], "rb") as fh:
    for i in range(N):
        fh.seek(i * PER); raw = fh.read(PER)
        occ = np.unpackbits(np.frombuffer(raw[:32_768], dtype=np.uint8), bitorder="little")[:64**3].reshape(64,64,64)
        pb = np.unpackbits(np.frombuffer(raw[32_768:], dtype=np.uint8), bitorder="little")[:64**3*3].reshape(64**3,3)
        parts = (pb[:,0] | (pb[:,1]<<1) | (pb[:,2]<<2)).reshape(64,64,64)
        ch_raw    = np.stack([occ, parts]).astype(np.float32)
        ch_masked = np.stack([occ, parts * occ]).astype(np.float32)
        a = p.predict(ch_raw); b = p.predict(ch_masked)
        for t in T:
            dmu[t].append(b["mu"][t] - a["mu"][t])
            dsig[t].append(b["sigma"][t] - a["sigma"][t])
            sig_raw[t].append(a["sigma"][t])

print(f"n={N} development samples, normalized natural-log scale\n")
print(f"{'target':<18}{'mean dmu':>10}{'|dmu| p50':>11}{'|dmu| p95':>11}{'|dmu| max':>11}{'  sigma_raw p50':>16}")
for t in T:
    d = np.array(dmu[t]); s = np.array(sig_raw[t])
    print(f"{t:<18}{d.mean():>+10.4f}{np.median(np.abs(d)):>11.4f}{np.percentile(np.abs(d),95):>11.4f}{np.abs(d).max():>11.4f}{np.median(s):>16.4f}")
print()
print("Interpretation: |dmu| is the prediction shift caused purely by the channel")
print("representation. Compare to sigma_raw, the model's own claimed uncertainty.")
print("If |dmu| is a sizeable fraction of sigma, the mismatch is material to K6.")
