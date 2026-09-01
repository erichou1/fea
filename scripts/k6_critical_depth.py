"""Amendment 12/13: quantile drift along the trajectory.

NOTE: amendment 13 WITHDREW the "critical depth d*" framing. The quantile
drifts smoothly (ratios 1.04, 1.26, 1.83, 2.49, 1.17, 1.09); there is no
changepoint. The tail is now split to 40-45 and 45+ because amendment 12
repeated the open-ended-bin mistake that amendment 11 retracted.

Re-bins the open-ended >25% tail into 25-30, 30-35, 35+ and recomputes the
full calibration transfer matrix. Uses only frozen artifacts: no retraining,
no new solver calls, no new geometry.

Amendment 11 showed the published shelf-life staircase was forced by the bin
grid, because the coarse grid had a single open-ended tail bin and every
calibration failed there and nowhere else. Splitting that tail asks whether
the cliff moves with calibration depth (it does not) and whether anything
else is there (it is).

Outputs everything needed to adjudicate amendment 12's predictions:
  - fine transfer matrix, coverage + Wilson lower bound per cell
  - conformal quantile per calibration bin, and its multiple of q_base
  - multiplicative interval width on the raw scale at deep states
  - vacuity control: surrogate bound vs model-free constant, both calibrated
    on the same deep states
  - small-sample control: bootstrap the deep calibration bins down to n=288
"""
from __future__ import annotations

import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

CONTROL = Path("/Users/eric/workspace/sasto-modernization-control/v2/g3")
GB200 = Path("/Users/eric/workspace/sasto-g3-gb200-inbound/trajectory-calibration-gb200")
REPO = Path(__file__).resolve().parents[1]

TN = ("compliance", "max_von_mises", "max_displacement")
KEY = {"compliance": "compliance_j", "max_von_mises": "max_gauss_von_mises_pa",
       "max_displacement": "max_displacement_m"}
ALPHA = 0.05
J = 3
ALPHA_J = ALPHA / J
EDGES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 1.01]
LBL = ["5-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-45", "45+"]
MIN_CAL = 30


def wilson_lower(h: int, n: int, z: float = 1.6448536269514722) -> float:
    if n == 0:
        return 0.0
    p = h / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n)) / d - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d


def bin_of(f: float):
    for i in range(len(EDGES) - 1):
        if EDGES[i] < f <= EDGES[i + 1]:
            return LBL[i]
    return None


def main() -> None:
    k6 = json.loads((CONTROL / "k6-coverage-gb200-2096.json").read_text())
    kappa = {k: float(v) for k, v in k6["kappa"].items()}
    q_base = {k: float(v) for k, v in k6["q_base"].items()}
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
                    "b": bin_of(s["fraction_removed"]),
                    "mu": {n: float(s["prediction"]["mu"][n]) for n in TN},
                    "sig": {n: float(s["prediction"]["sigma"][n]) for n in TN},
                    "y": {n: truth(s["solver"], n) for n in TN},
                })
        return rows

    dev, cal = load("development"), load("calibration")
    assert not ({r["family_id"] for r in dev} & {r["family_id"] for r in cal}), \
        "calibration and development share a family; transfer matrix would be circular"

    def fit_q(rows):
        q = {}
        for n in TN:
            s = sorted(r["y"][n] - r["mu"][n] - kappa[n] * r["sig"][n] for r in rows)
            k = math.ceil((len(s) + 1) * (1 - ALPHA_J))
            q[n] = s[min(k, len(s)) - 1]
        return q

    def covered(r, q):
        return all(r["y"][n] <= r["mu"][n] + kappa[n] * r["sig"][n] + q[n] for n in TN)

    cal_by = defaultdict(list)
    dev_by = defaultdict(list)
    for r in cal:
        cal_by[r["b"]].append(r)
    for r in dev:
        dev_by[r["b"]].append(r)

    report = {"alpha": ALPHA, "J": J, "alpha_j": ALPHA_J, "kappa": kappa, "q_base": q_base,
              "edges": EDGES, "labels": LBL,
              "n_cal": {l: len(cal_by[l]) for l in LBL},
              "n_dev": {l: len(dev_by[l]) for l in LBL}}

    # ---- fine transfer matrix
    matrix, quantiles, fails = {}, {}, {}
    for bi in LBL:
        if len(cal_by[bi]) < MIN_CAL:
            continue
        q = fit_q(cal_by[bi])
        quantiles[bi] = {"q": q, "n": len(cal_by[bi]),
                         "multiple_of_q_base": {n: q[n] / q_base[n] for n in TN}}
        row, failed = {}, []
        for bj in LBL:
            ev = dev_by[bj]
            if not ev:
                continue
            h = sum(1 for r in ev if covered(r, q))
            lo = wilson_lower(h, len(ev))
            row[bj] = {"covered": h, "n": len(ev), "coverage": h / len(ev), "wilson_lower": lo}
            if lo < 0.95:
                failed.append(bj)
        matrix[bi] = row
        fails[bi] = failed
    report["matrix"] = matrix
    report["quantiles"] = quantiles
    report["failing_eval_bins"] = fails

    print("FINE TRANSFER MATRIX (coverage, * = Wilson LB >= 0.95)")
    print(f"{'fit/eval':>9s} " + " ".join(f"{l:>9s}" for l in LBL))
    for bi in LBL:
        if bi not in matrix:
            continue
        cells = []
        for bj in LBL:
            c = matrix[bi].get(bj)
            cells.append("   --    " if not c else
                         f"  {c['coverage']:.3f}{'*' if c['wilson_lower'] >= 0.95 else ' '} ")
        print(f"{bi:>9s} " + " ".join(cells))

    # ---- d*: smallest calibration bin whose row fails nowhere
    d_star = next((bi for bi in LBL if bi in fails and not fails[bi]), None)
    report["d_star_bin"] = d_star
    report["d_star_lower_edge"] = EDGES[LBL.index(d_star)] if d_star else None
    print(f"\nd* = shallowest calibration bin valid EVERYWHERE: {d_star}")

    # ---- width: multiplicative interval on the raw scale at deep states
    deep_dev = [r for r in dev if r["b"] in ("25-30", "30-35", "35+")]
    widths = {}
    for bi, rec in quantiles.items():
        q = rec["q"]
        widths[bi] = {n: float(np.median([
            math.exp(norm["scales"][n] * (kappa[n] * r["sig"][n] + q[n])) for r in deep_dev]))
            for n in TN}
    report["median_multiplicative_width_at_deep_states"] = widths
    print("\nmedian multiplicative width U/exp(mu) at deep states:")
    for bi in LBL:
        if bi in widths:
            print(f"  calibrated {bi:>6s}: " +
                  "  ".join(f"{n[:10]} {widths[bi][n]:6.2f}x" for n in TN))

    # ---- vacuity control: surrogate vs model-free constant, same deep calibration
    deep_cal = [r for r in cal if r["b"] in ("25-30", "30-35", "35+")]
    vac = {}
    for n in TN:
        sc = norm["scales"][n]
        s = sorted(r["y"][n] - r["mu"][n] - kappa[n] * r["sig"][n] for r in deep_cal)
        q = s[min(math.ceil((len(s) + 1) * (1 - ALPHA_J)), len(s)) - 1]
        s2 = sorted(r["y"][n] for r in deep_cal)
        c = s2[min(math.ceil((len(s2) + 1) * (1 - ALPHA_J)), len(s2)) - 1]
        vac[n] = {
            "surrogate_coverage": float(np.mean(
                [r["y"][n] <= r["mu"][n] + kappa[n] * r["sig"][n] + q for r in deep_dev])),
            "constant_coverage": float(np.mean([r["y"][n] <= c for r in deep_dev])),
            "surrogate_median_U": float(np.median(
                [math.exp(sc * (r["mu"][n] + kappa[n] * r["sig"][n] + q)) for r in deep_dev])),
            "constant_U": float(math.exp(sc * c)),
        }
        vac[n]["ratio_surrogate_over_constant"] = vac[n]["surrogate_median_U"] / vac[n]["constant_U"]
    report["vacuity_control"] = vac
    print("\nvacuity control (both calibrated on the same deep states):")
    for n in TN:
        v = vac[n]
        print(f"  {n:18s} surrogate cov {v['surrogate_coverage']:.3f} vs constant "
              f"{v['constant_coverage']:.3f} | width ratio {v['ratio_surrogate_over_constant']:.3f}")

    # ---- small-sample control: does a SHALLOW bin subsampled to deep-bin n still fail?
    rng = np.random.default_rng(20260828)
    n_deep = min(len(cal_by[b]) for b in ("25-30", "30-35", "35+"))
    sub = {}
    for bi in ("5-10", "20-25"):
        covs = []
        for _ in range(200):
            pick = [cal_by[bi][i] for i in rng.choice(len(cal_by[bi]), n_deep, replace=False)]
            q = fit_q(pick)
            ev = dev_by["35+"]
            covs.append(sum(1 for r in ev if covered(r, q)) / len(ev))
        sub[bi] = {"n_subsampled_to": n_deep, "mean_coverage_on_35plus": float(np.mean(covs)),
                   "p95": float(np.percentile(covs, 95)), "max": float(np.max(covs))}
        print(f"\nshallow bin {bi} subsampled to n={n_deep}, evaluated on 35+: "
              f"mean coverage {np.mean(covs):.3f}, max over 200 draws {np.max(covs):.3f}")
    report["small_sample_control"] = sub
    print("  => if these stay far below 0.95, small n does NOT explain the deep rows")

    dest = CONTROL / "k6-amendment-12-critical-depth.json"
    dest.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
