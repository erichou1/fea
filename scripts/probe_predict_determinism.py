"""Probe: is EnsemblePredictor.predict bitwise deterministic across runs?

Runs the same input through the frozen ensemble N times under the current
(uncontrolled) torch config, then again with deterministic controls, and
reports whether the outputs are byte-identical.
"""
import json
import sys
import hashlib
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from sasto.g3_trajectory_calibration import EnsemblePredictor  # noqa: E402

ROOT = Path("artifacts/g2/ensemble-v1")


def load_sample_channels() -> np.ndarray:
    # Real decoded channels from the certified cache, first development sample.
    cache = Path("artifacts/g2/ingest-cache-v1/79640406e1e0921c-b7066e14c6713eb6")
    manifest = json.loads((cache / "cache-manifest.json").read_text())
    row = manifest["roles"]["development"]
    per = 131_072
    with open(cache / row["data_file"], "rb") as fh:
        raw = fh.read(per)
    occ_bits = np.unpackbits(np.frombuffer(raw[:32_768], dtype=np.uint8), bitorder="little")[: 64 ** 3]
    occ = occ_bits.reshape(64, 64, 64).astype(np.float32)
    parts_packed = np.frombuffer(raw[32_768:], dtype=np.uint8)
    parts_bits = np.unpackbits(parts_packed, bitorder="little")[: 64 ** 3 * 3].reshape(64 ** 3, 3)
    parts = (parts_bits[:, 0] | (parts_bits[:, 1] << 1) | (parts_bits[:, 2] << 2)).reshape(64, 64, 64).astype(np.float32)
    return np.stack([occ, parts])


def digest(pred: dict) -> str:
    payload = json.dumps(pred, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run(label: str, n: int = 6) -> set[str]:
    import torch
    # A fresh predictor per call, so intra-process cached state does not mask
    # cross-process variation.
    seen = set()
    for _ in range(n):
        p = EnsemblePredictor(ensemble_root=ROOT, normalization_path=ROOT / "normalization-stats.json", device="cpu")
        seen.add(digest(p.predict(load_sample_channels())))
    print(f"{label:<22} distinct outputs over {n} runs: {len(seen)}  threads={torch.get_num_threads()}")
    return seen


if __name__ == "__main__":
    import torch
    baseline = run("uncontrolled")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    controlled = run("deterministic+1thread")
    print()
    print("VERDICT:", "non-deterministic under default config" if len(baseline) > 1 else "already deterministic")
    print("        ", "deterministic under controls" if len(controlled) == 1 else "STILL NON-DETERMINISTIC under controls")
