"""Cross-PROCESS determinism probe. Prints one digest and exits.
Run it several times from a shell and compare."""
import json, sys, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from sasto.g3_trajectory_calibration import EnsemblePredictor

ROOT = Path("artifacts/g2/ensemble-v1")
cache = Path("artifacts/g2/ingest-cache-v1/79640406e1e0921c-b7066e14c6713eb6")
row = json.loads((cache / "cache-manifest.json").read_text())["roles"]["development"]
with open(cache / row["data_file"], "rb") as fh:
    raw = fh.read(131_072)
occ = np.unpackbits(np.frombuffer(raw[:32_768], dtype=np.uint8), bitorder="little")[:64**3].reshape(64,64,64).astype(np.float32)
pb = np.unpackbits(np.frombuffer(raw[32_768:], dtype=np.uint8), bitorder="little")[:64**3*3].reshape(64**3,3)
parts = (pb[:,0] | (pb[:,1]<<1) | (pb[:,2]<<2)).reshape(64,64,64).astype(np.float32)
ch = np.stack([occ, parts])

mode = sys.argv[1] if len(sys.argv) > 1 else "default"
if mode == "det":
    import torch
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
p = EnsemblePredictor(ensemble_root=ROOT, normalization_path=ROOT/"normalization-stats.json", device="cpu")
pred = p.predict(ch)
print(mode, hashlib.sha256(json.dumps(pred, sort_keys=True, separators=(",",":")).encode()).hexdigest()[:20],
      f"mu.compliance={pred['mu']['compliance']:.17g}")
