"""Train architecture B on the frozen fit role. Amendment 07.

Governed by K6_AMENDMENT_07_SECOND_ARCHITECTURE.md sha256
b6d27643c6a01fa19f37502d17ee5adf7a96aa19d8f8cbc9472d7baf8f030ce4.

The training protocol MIRRORS architecture A's certified run in every respect
the comparison depends on -- read directly from
``surrogate.train_certified_ensemble``:

- AdamW, lr 1e-3, no scheduler, no gradient clipping, batch size 4
- Gaussian NLL ``(0.5*((y-mu)/d)^2 + log d)`` with softplus dispersion
- per-epoch development selection metric, patience 4, best epoch restored
- ``deterministic_seed`` formula and ``use_deterministic_algorithms(True)``
- the same fit role (6,643) and development role (2,214), same normalization

Only the network differs. That is the entire point.

One addition A lacks: every member's development-set predictions are stored so
member-subsampling analyses are possible later; the manuscript currently lists
their absence for A as a limitation.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from sasto.surrogate_b import (  # noqa: E402
    SEED_NAMESPACE_B,
    TARGET_NAMES,
    ResidualSurrogateCNN,
)

AMENDMENT_SHA = "91b14bdaaa4bead2861a4afb34247851ef610cdc89ff2fc94b5bfe3b3648cb1c"

# Arm registry, per K6_AMENDMENT_09. Each arm isolates one factor.
ARMS = {
    "a4-conv":  {"arch": "A", "width": 4,  "ns": "a4-converged-v1",      "cap": 100},
    "a16-conv": {"arch": "A", "width": 16, "ns": "a16-converged-v1",     "cap": 100},
    "b-conv":   {"arch": "B", "width": 12, "ns": "b-converged-v1",       "cap": 100},
    "a4-rep":   {"arch": "A", "width": 4,  "ns": "a4-converged-rep1",    "cap": 100},
    "b-rep":    {"arch": "B", "width": 12, "ns": "b-converged-rep1",     "cap": 100},
}
CACHE = REPO / "artifacts/g2/ingest-cache-v1/79640406e1e0921c-b7066e14c6713eb6"
OUT = REPO / "artifacts/g2b/ensemble-v1"
MEMBERS = 5
PATIENCE = 4
BATCH = 4          # matches A
LR = 1e-3          # matches A
CAMPAIGN_SEED = 20260828


def deterministic_seed(namespace: str, campaign_seed: int, member_index: int) -> int:
    """Byte-identical to surrogate.deterministic_seed, new namespace."""
    payload = "{}\0{}\0{}".format(namespace, campaign_seed, member_index).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2 ** 31 - 1)


class Packed:
    """Reads the same packed ingest cache architecture A trained from."""

    def __init__(self, role: str, norm: dict) -> None:
        manifest = json.loads((CACHE / "cache-manifest.json").read_text())
        row = manifest["roles"][role]
        self.ids = row["sample_ids"]
        self.targets = row["targets"]
        self.path = CACHE / row["data_file"]
        self.item_bytes = 4 * (64 ** 3 // 8)
        self.mean = {n: float(norm["means"][n]) for n in TARGET_NAMES}
        self.scale = {n: float(norm["scales"][n]) for n in TARGET_NAMES}
        self._mm = None

    def __len__(self) -> int:
        return len(self.ids)

    def example(self, i: int):
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r",
                                 shape=(len(self.ids), self.item_bytes))
        payload = self._mm[i]
        occ = np.unpackbits(payload[:32768], bitorder="little", count=64 ** 3)
        parts = np.zeros(64 ** 3, dtype=np.uint8)
        for bit in range(3):
            plane = payload[(bit + 1) * 32768:(bit + 2) * 32768]
            parts |= np.unpackbits(plane, bitorder="little", count=64 ** 3).astype(np.uint8) << bit
        channels = np.stack((occ.reshape(64, 64, 64).astype(np.float32),
                             parts.reshape(64, 64, 64).astype(np.float32)), axis=0)
        t = self.targets[self.ids[i]]
        y = np.array([(np.log(float(t[n])) - self.mean[n]) / self.scale[n]
                      for n in TARGET_NAMES], dtype=np.float32)
        return torch.from_numpy(channels), torch.from_numpy(y)

    def batches(self, batch_size: int, order: list[int] | None = None):
        indices = order if order is not None else list(range(len(self)))
        for start in range(0, len(indices), batch_size):
            chunk = [self.example(i) for i in indices[start:start + batch_size]]
            yield (torch.stack([c for c, _ in chunk]), torch.stack([y for _, y in chunk]))


def development_metric(model, data: Packed, device) -> float:
    """Mean normalized-log MAE over targets; A's selection metric exactly."""
    model.eval()
    total = np.zeros(len(TARGET_NAMES), dtype=np.float64)
    count = 0
    with torch.no_grad():
        for xb, yb in data.batches(4):
            mu, _ = model(xb.to(device))
            total += (mu.cpu() - yb).abs().sum(dim=0).numpy()
            count += xb.shape[0]
    return float(np.mean(total / count))


def build_model(arch: str, width: int):
    """Architecture A (dense, from the frozen bundle) or B (residual+SE)."""
    if arch == "A":
        from sasto.surrogate import DenseSurrogateCNN
        model = DenseSurrogateCNN(base_channels=width)

        class _AWrap(torch.nn.Module):
            """Adapt A's dict output to the (mu, dispersion) tuple used here."""

            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            @property
            def parameter_count(self):
                return self.inner.parameter_count

            def forward(self, x):
                out = self.inner(x)
                return out["mean"], out["dispersion"]

        return _AWrap(model)
    from sasto.surrogate_b import ResidualSurrogateCNN
    return ResidualSurrogateCNN(width=width)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    arm = ARMS[args.arm]
    global OUT, SEED_NAMESPACE_B, MAX_EPOCHS
    OUT = REPO / "artifacts/g2b" / args.arm
    SEED_NAMESPACE_B = arm["ns"]
    MAX_EPOCHS = arm["cap"]

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"arm {args.arm}: arch {arm['arch']} width {arm['width']} "
          f"ns {arm['ns']} cap {arm['cap']}", flush=True)

    norm = json.loads((REPO / "artifacts/g2/ensemble-v1/normalization-stats.json").read_text())
    if norm["role"] != "fit":
        raise SystemExit("normalization record must be fit-only")

    fit = Packed("fit", norm)
    dev_set = Packed("development", norm)
    print(f"fit {len(fit)} | development {len(dev_set)} | device {device}", flush=True)

    members = []
    for index in range(MEMBERS):
        done = OUT / f"member-{index:02d}.json"
        if done.exists():
            record = json.loads(done.read_text())
            ckpt = OUT / record["checkpoint"]
            if hashlib.sha256(ckpt.read_bytes()).hexdigest() == record["checkpoint_sha256"]:
                print(f"member {index} already trained, resuming past it", flush=True)
                members.append(record)
                continue
        seed = deterministic_seed(SEED_NAMESPACE_B, CAMPAIGN_SEED, index)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if device.type == "mps":
            torch.mps.empty_cache()

        model = build_model(arm["arch"], arm["width"]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        best_metric = float("inf")
        best_epoch = 0
        best_state = None
        stale = 0
        losses = []
        started = time.perf_counter()
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            order = list(range(len(fit)))
            random.Random(seed + epoch).shuffle(order)      # A's shuffle rule
            for xb, yb in fit.batches(BATCH, order):
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                mu, dispersion = model(xb)
                loss = (0.5 * ((yb - mu) / dispersion).square() + torch.log(dispersion)).mean()
                loss.backward()
                opt.step()
            current = development_metric(model, dev_set, device)
            losses.append(current)
            print(f"  member {index} epoch {epoch:2d} dev metric {current:.4f}", flush=True)
            if current < best_metric:
                best_metric = current
                best_epoch = epoch
                stale = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        wall = time.perf_counter() - started
        if best_state is None:
            raise SystemExit("no development metric was produced")
        model.load_state_dict(best_state)

        # store per-member development predictions: the capability A lacks
        model.eval()
        dev_mu = []
        with torch.no_grad():
            for xb, _ in dev_set.batches(8):
                mu, _d = model(xb.to(device))
                dev_mu.append(mu.cpu().numpy())
        dev_mu = np.concatenate(dev_mu)

        ckpt = OUT / f"member-{index:02d}.pt"
        torch.save({"state_dict": model.state_dict(), "target_names": TARGET_NAMES,
                    "seed": seed}, ckpt)
        np.save(OUT / f"member-{index:02d}-dev-mu.npy", dev_mu)
        record = {
            "member_index": index, "seed": seed, "seed_namespace": SEED_NAMESPACE_B,
            "amendment_sha256": AMENDMENT_SHA,
            "parameter_count": int(model.parameter_count),
            "epoch_count": len(losses), "selected_epoch": best_epoch,
            "development_selection_metric_final": best_metric,
            "training_wall_seconds": wall, "checkpoint": ckpt.name,
            "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
            "dev_mu_file": f"member-{index:02d}-dev-mu.npy",
        }
        done.write_text(json.dumps(record, indent=1, sort_keys=True))
        members.append(record)
        print(f"member {index} done: metric {best_metric:.4f} @ epoch {best_epoch}, "
              f"{wall/60:.1f} min", flush=True)

    summary = {
        "schema_version": "1.0.0",
        "label": args.arm,
        "amendment_sha256": AMENDMENT_SHA,
        "architecture": arm["arch"], "width": arm["width"], "epoch_cap": arm["cap"],
        "member_count": MEMBERS,
        "protocol": {
            "optimizer": "AdamW", "lr": LR, "batch_size": BATCH,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "selection": "best development normalized-log MAE, best epoch restored",
            "loss": "gaussian nll, softplus dispersion, matches arch A",
            "seed_formula": "sha256(ns\\0seed\\0idx)[:8] % (2^31-1), matches arch A",
        },
        "normalization_stats_digest": norm["stats_digest"],
        "campaign_seed": CAMPAIGN_SEED,
        "members": members,
    }
    (OUT / "ensemble-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(f"\nwrote {OUT}/ensemble-summary.json", flush=True)


if __name__ == "__main__":
    main()
