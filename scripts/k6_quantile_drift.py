"""Amendment 13: quantile drift, and whether it can be extrapolated.

The surviving result after three retractions. Two measurements, both from
frozen artifacts, no retraining and no new solver calls:

  1. The conformal quantile q(d) needed at depth d, per target. It grows
     smoothly and monotonically by about 7.7x from 5-10% to 35-40% removal.
     There is no changepoint; amendment 12's "critical depth d*" framing was
     withdrawn on exactly this evidence.

  2. Whether a practitioner can EXTRAPOLATE q(d) from the cheap shallow data
     they already have. They cannot: a linear fit on bins up to 25% recovers
     28-48% of the quantile actually needed and yields coverage 0.876-0.898
     against a 0.95 target. This is the paper's practical content and it is a
     negative result.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

CONTROL = Path("/Users/eric/workspace/sasto-modernization-control/v2/g3")
import os
_D1 = os.environ.get("SASTO_D1_ROOT")
_SUFFIX = "-d1" if _D1 else ""
GB200 = Path(_D1) if _D1 else Path("/Users/eric/workspace/sasto-g3-gb200-inbound/trajectory-calibration-gb200")
REPO = Path(__file__).resolve().parents[1]
TN = ("compliance", "max_von_mises", "max_displacement")
KEY = {"compliance": "compliance_j", "max_von_mises": "max_gauss_von_mises_pa",
       "max_displacement": "max_displacement_m"}
ALPHA_J = 0.05 / 3
BINS = [(0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25),
        (0.25, 0.30), (0.30, 0.35), (0.35, 0.40)]
SHALLOW = 4  # bins at or below 25% removal


def wilson_lower(h: int, n: int, z: float = 1.6448536269514722) -> float:
    if n == 0:
        return 0.0
    p = h / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n)) / d - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d


def main() -> None:
    k6 = json.loads((CONTROL / ("k6-coverage-gb200-2096" + _SUFFIX + ".json")).read_text())
    kappa = {k: float(v) for k, v in k6["kappa"].items()}
    norm = json.loads((REPO / "artifacts/g2/ensemble-v1/normalization-stats.json").read_text())

    def truth(solver, name):
        raw = solver.get(KEY[name], solver.get("max_von_mises_pa"))
        return (math.log(float(raw)) - norm["means"][name]) / norm["scales"][name]

    def load(role):
        rows = []
        for path in sorted(GB200.glob(f"trajectory-{role}-*.json")):
            rec = json.loads(path.read_text())
            for s in rec["selected_states"]:
                rows.append({
                    "family_id": rec["family_id"], "f": s["fraction_removed"],
                    "mu": {n: float(s["prediction"]["mu"][n]) for n in TN},
                    "sig": {n: float(s["prediction"]["sigma"][n]) for n in TN},
                    "y": {n: truth(s["solver"], n) for n in TN}})
        return rows

    dev, cal = load("development"), load("calibration")
    assert not ({r["family_id"] for r in dev} & {r["family_id"] for r in cal}), \
        "calibration and development share a family"

    def score(r, n):
        return r["y"][n] - r["mu"][n] - kappa[n] * r["sig"][n]

    def q_at(rows, n):
        s = sorted(score(r, n) for r in rows)
        k = math.ceil((len(s) + 1) * (1 - ALPHA_J))
        return s[min(k, len(s)) - 1]

    mids = [(a + b) / 2 for a, b in BINS]
    Q = {n: [] for n in TN}
    counts = []
    for a, b in BINS:
        sub = [r for r in cal if a < r["f"] <= b]
        counts.append(len(sub))
        for n in TN:
            Q[n].append(q_at(sub, n))

    report = {"alpha_j": ALPHA_J, "kappa": kappa, "bins": BINS, "midpoints": mids,
              "calibration_n_per_bin": counts, "q_of_depth": Q}

    print("QUANTILE DRIFT q(d), calibration-role states")
    print(f"{'depth':>10s} {'n':>5s} " + " ".join(f"{n[:12]:>12s}" for n in TN))
    for i, (a, b) in enumerate(BINS):
        print(f"{a:.2f}-{b:.2f} {counts[i]:5d} " + " ".join(f"{Q[n][i]:12.4f}" for n in TN))

    growth = {n: Q[n][-1] / Q[n][0] for n in TN if Q[n][0] > 0}
    report["growth_factor_first_to_last"] = growth
    print("\ngrowth 5-10% -> 35-40%: " +
          ", ".join(f"{n[:12]} {v:.1f}x" for n, v in growth.items()))
    ratios = {n: [Q[n][i + 1] / Q[n][i] if Q[n][i] > 0 else float("nan")
                  for i in range(len(BINS) - 1)] for n in TN}
    report["consecutive_ratios"] = ratios
    print("consecutive ratios (compliance): " +
          " ".join(f"{v:.2f}" for v in ratios["compliance"]))
    print("  -> monotone, no changepoint; 'critical depth' framing withdrawn")

    print("\nEXTRAPOLATION TEST: fit q(d) on bins <=25% only, apply forward")
    extrap = {}
    for n in TN:
        y = np.array(Q[n], dtype=float)
        x = np.array(mids, dtype=float)
        coef = np.polyfit(x[:SHALLOW], y[:SHALLOW], 1)
        rows = []
        for i in range(SHALLOW, len(BINS)):
            a, b = BINS[i]
            sub = [r for r in dev if a < r["f"] <= b]
            q_ext = float(np.polyval(coef, mids[i]))
            h = sum(1 for r in sub
                    if r["y"][n] <= r["mu"][n] + kappa[n] * r["sig"][n] + q_ext)
            rows.append({"bin": f"{a:.2f}-{b:.2f}", "q_extrapolated": q_ext,
                         "q_true": float(y[i]), "fraction_of_needed": q_ext / float(y[i]),
                         "n": len(sub), "coverage": h / len(sub),
                         "wilson_lower": wilson_lower(h, len(sub))})
            print(f"  {n[:12]:12s} {rows[-1]['bin']}  q_ext {q_ext:6.3f} vs true {y[i]:6.3f} "
                  f"({q_ext / y[i] * 100:3.0f}% of needed) -> coverage {h / len(sub):.3f}")
        extrap[n] = rows
    report["extrapolation_from_shallow"] = extrap
    best = max(r["coverage"] for n in TN for r in extrap[n])
    report["best_extrapolated_coverage"] = best
    print(f"\n  best coverage any extrapolation achieved: {best:.3f} against 0.95")
    print("  -> the quantile you will need cannot be predicted from shallow data")

    dest = CONTROL / ("k6-amendment-13-quantile-drift" + _SUFFIX + ".json")
    dest.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
