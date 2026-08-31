"""Amendment 05: calibration shelf life, and verifier-free failure prediction.

Frozen at K6_AMENDMENT_05_SHELF_LIFE.md, SHA-256
e3ca83ec8f0c10d7d23b10cfafe321a8dee4dcea7c13a69cc72a648c907eb7ba, before either
experiment was computed.

A  5x5 calibration transfer matrix.  Calibrate in bin i, evaluate in bin j.
   Shelf life = largest forward displacement d with Wilson lower bound >= 0.95.

B  Within-bin AUC for predicting "this state is uncovered" from verifier-free
   statistics.  Within-bin so a predictor cannot win by proxying for depth.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sasto.g3_trajectory_calibration import (  # noqa: E402
    ALPHA_J, TARGET_NAMES, _normalized_targets, _selected_trajectory_rows,
    _verified_json, split_conformal_quantile,
)
from sasto.k6_coverage import wilson_lower_bound  # noqa: E402

GB200 = Path("/Users/eric/workspace/sasto-g3-gb200-inbound/trajectory-calibration-gb200")
FROZEN = Path("/Users/eric/workspace/fea-sasto-v/artifacts/g3/trajectory-calibration-v2")
NORM = Path("/Users/eric/workspace/fea-sasto-v/artifacts/g2/ensemble-v1/normalization-stats.json")
CONTROL = Path("/Users/eric/workspace/sasto-modernization-control/v2/g3")
AMENDMENT_05 = "e3ca83ec8f0c10d7d23b10cfafe321a8dee4dcea7c13a69cc72a648c907eb7ba"
ORDER = ["(5,10%]", "(10,15%]", "(15,20%]", "(20,25%]", ">25%"]
SHORT = ["5-10", "10-15", "15-20", "20-25", ">25"]
AUC_MATERIAL = 0.60


def load(role, normalization):
    cases = [_verified_json(p, "G3 trajectory case", "trajectory_digest")
             for p in sorted(GB200.glob(f"trajectory-{role}-*.json"))]
    rows, _, _ = _selected_trajectory_rows(cases)
    out = []
    for r in rows:
        s = r["solver"]
        y = _normalized_targets({
            "compliance": s["compliance_j"],
            "max_von_mises": s.get("max_gauss_von_mises_pa", s.get("max_von_mises_pa")),
            "max_displacement": s["max_displacement_m"],
        }, normalization)
        out.append({"bin": r["bin_label"], "y": y,
                    "mu": {n: float(r["prediction"]["mu"][n]) for n in TARGET_NAMES},
                    "sigma": {n: float(r["prediction"]["sigma"][n]) for n in TARGET_NAMES}})
    return out


def fit_q(rows, kappa):
    return {n: split_conformal_quantile(
        [r["y"][n] - (r["mu"][n] + kappa[n] * r["sigma"][n]) for r in rows],
        alpha=ALPHA_J) for n in TARGET_NAMES}


def covered(r, kappa, q):
    return all(r["y"][n] <= r["mu"][n] + kappa[n] * r["sigma"][n] + q[n] for n in TARGET_NAMES)


def auc(scores, labels):
    """AUC via rank sum; labels 1 = positive (uncovered)."""
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(ranks[i] for i in range(len(labels)) if labels[i])
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def main() -> int:
    normalization = json.loads(NORM.read_text())
    kappa = {k: float(v) for k, v in _verified_json(
        FROZEN / "kappa-development-evidence.json", "G3 kappa evidence",
        "kappa_evidence_sha256")["kappa"].items()}

    dev = load("development", normalization)
    cal = load("calibration", normalization)
    dev_by = defaultdict(list)
    cal_by = defaultdict(list)
    for r in dev:
        dev_by[r["bin"]].append(r)
    for r in cal:
        cal_by[r["bin"]].append(r)

    report = {"amendment_05_sha256": AMENDMENT_05, "kappa": kappa}

    # ------------------------------------------------------- EXPERIMENT A
    print("=" * 78)
    print("EXPERIMENT A  calibration transfer matrix: fit in bin i, evaluate in bin j")
    print("=" * 78)
    qs = {b: fit_q(cal_by[b], kappa) for b in ORDER if cal_by[b]}
    matrix = {}
    print(f"\n  coverage (Wilson lower bound in parentheses)")
    header = "fit\\eval"
    print(f"  {header:>10s} " + " ".join(f"{s:>15s}" for s in SHORT))
    for i, bi in enumerate(ORDER):
        cells = []
        row = {}
        for bj in ORDER:
            ev = dev_by[bj]
            h = sum(1 for r in ev if covered(r, kappa, qs[bi]))
            cov = h / len(ev)
            lo = wilson_lower_bound(h, len(ev))
            row[bj] = {"covered": h, "n": len(ev), "coverage": cov, "wilson_lower": lo}
            mark = "*" if lo >= 0.95 else " "
            cells.append(f"{cov:.3f}({lo:.3f}){mark}")
        matrix[bi] = row
        print(f"  {SHORT[i]:>10s} " + " ".join(f"{c:>15s}" for c in cells))
    print("\n  * = Wilson lower bound >= 0.95 (valid)")

    print("\n  SHELF LIFE (forward displacement, per amendment 05 section 2)")
    shelf = {}
    for i, bi in enumerate(ORDER):
        d = 0
        for step in range(1, len(ORDER) - i):
            if matrix[bi][ORDER[i + step]]["wilson_lower"] >= 0.95:
                d = step
            else:
                break
        shelf[bi] = d
        pts = "n/a (last bin)" if i == len(ORDER) - 1 else f"{d * 5} percentage points"
        print(f"    fit at {SHORT[i]:>6s}: shelf life {d} bin(s)   = {pts}")
    report["experiment_a"] = {"matrix": matrix, "shelf_life_bins": shelf}

    fwd = [shelf[b] for b in ORDER[:-1]]
    if all(v == 0 for v in fwd):
        a_verdict = "shelf life is 0 bins from every origin: calibration must be refreshed every bin"
    elif len(set(fwd)) == 1:
        a_verdict = f"constant shelf life of {fwd[0]} bin(s): a fixed recalibration interval suffices"
    else:
        a_verdict = f"shelf life varies with depth: {fwd}"
    print(f"\n  -> {a_verdict}")
    report["experiment_a"]["verdict"] = a_verdict

    print("\n  BACKWARD CONTROL (fit deep, apply shallow; should over-cover)")
    back = []
    for i, bi in enumerate(ORDER):
        for j in range(0, i):
            back.append(matrix[bi][ORDER[j]]["coverage"])
    print(f"    {len(back)} backward cells, min coverage {min(back):.4f}, all >= 0.95: {all(c >= 0.95 for c in back)}")
    report["experiment_a"]["backward_control"] = {
        "n_cells": len(back), "min_coverage": min(back),
        "all_at_least_0.95": bool(all(c >= 0.95 for c in back))}

    # ------------------------------------------------------- EXPERIMENT B
    print()
    print("=" * 78)
    print("EXPERIMENT B  verifier-free prediction of coverage failure (within-bin AUC)")
    print("=" * 78)
    q_base = {k: float(v) for k, v in _verified_json(
        FROZEN / "baseline-calibration.json", "G3 baseline calibration",
        "baseline_calibration_sha256")["q"].items()}
    mean_base_sigma = {n: sum(r["sigma"][n] for r in cal) / len(cal) for n in TARGET_NAMES}

    predictors = {
        "sigma_compliance": lambda r: r["sigma"]["compliance"],
        "sigma_mean_all": lambda r: sum(r["sigma"][n] for n in TARGET_NAMES) / 3.0,
        "mu_compliance": lambda r: r["mu"]["compliance"],
        "sigma_rel_baseline": lambda r: sum(r["sigma"][n] / mean_base_sigma[n] for n in TARGET_NAMES) / 3.0,
    }
    print(f"\n  {'bin':>8s} {'n':>6s} {'uncov':>6s} " + " ".join(f"{k:>19s}" for k in predictors))
    b_res = {}
    for i, b in enumerate(ORDER):
        ev = dev_by[b]
        labels = [0 if covered(r, kappa, q_base) else 1 for r in ev]
        cells, row = [], {"n": len(ev), "n_uncovered": sum(labels)}
        for name, fn in predictors.items():
            a = auc([fn(r) for r in ev], labels)
            row[name] = a
            cells.append("     n/a          " if a != a else f"{a:19.3f}")
        b_res[b] = row
        print(f"  {SHORT[i]:>8s} {len(ev):6d} {sum(labels):6d} " + " ".join(cells))

    deep = b_res[">25%"]
    sig_auc = deep["sigma_mean_all"]
    print(f"\n  DEEPEST BIN, sigma AUC = {sig_auc:.3f} against the 0.60 threshold fixed in advance")
    if sig_auc != sig_auc:
        b_verdict = "undefined (no variation in labels)"
    elif sig_auc >= AUC_MATERIAL:
        b_verdict = f"sigma HAS predictive power at depth (AUC {sig_auc:.3f} >= 0.60); a cheap triage signal exists"
    else:
        b_verdict = (f"sigma has NO useful predictive power at depth (AUC {sig_auc:.3f} < 0.60): "
                     "the ensemble cannot tell which of its own deep predictions are wrong")
    print(f"  -> {b_verdict}")
    print(f"  PREDICTION recorded in amendment 05 section 3 was: AUC near 0.5 in the deepest bin")
    print(f"  PREDICTION {'CONFIRMED' if sig_auc < AUC_MATERIAL else 'WRONG'}")
    report["experiment_b"] = {"per_bin": b_res, "auc_threshold": AUC_MATERIAL,
                              "deepest_sigma_auc": sig_auc, "verdict": b_verdict,
                              "prediction_confirmed": bool(sig_auc < AUC_MATERIAL)}

    out = CONTROL / "k6-amendment-05-shelf-life.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
