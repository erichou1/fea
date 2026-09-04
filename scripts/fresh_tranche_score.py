"""Amendment 15: score the fresh tranche against constants frozen on the old data.

This is the out-of-sample test. Every constant it uses (kappa, q_base, the
per-depth quantiles q(d), the shallow linear extrapolation) is imported by value
from artifacts frozen BEFORE any fresh wireframe was solved. The fresh records
are evaluated only; nothing is fitted on them.

Hashed before the first fresh solve. Prints and writes:

  A. coverage of the BASELINE-calibrated bound (kappa, q_base) per fresh depth
     bin, with Wilson lower bounds                  [reproduces the K6 collapse?]
  B. coverage of each frozen depth-calibrated q(d_i) on each fresh bin d_j
     (the transfer matrix, rows frozen, columns fresh) [validity at/below own depth?]
  C. coverage of the frozen shallow-fit extrapolation on fresh deep bins
                                                    [extrapolation still fails?]
  D. the fresh quantile q_fresh(d) alongside q_old(d), and their ratio
                                                    [drift magnitude reproduces?]
  E. sigma vs |residual| Pearson per bin            [sigma still positive?]

Usage:
  python scripts/fresh_tranche_score.py --fresh-root ROOT --out REPORT.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CONTROL = Path("/Users/eric/workspace/sasto-modernization-control/v2/g3")
TN = ("compliance", "max_von_mises", "max_displacement")
KEY = {"compliance": "compliance_j", "max_von_mises": "max_gauss_von_mises_pa", "max_displacement": "max_displacement_m"}
ALPHA_J = 0.05 / 3
# Amendment-13 bins on the old data; the fresh tranche uses the same edges
# from 5% up, plus 40-45 and 45-50 which the old data could not populate.
OLD_BINS = [(0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.35), (0.35, 0.40)]
FRESH_BINS = OLD_BINS + [(0.40, 0.45), (0.45, 0.50)]
SHALLOW = 4
MIN_N = 30
import os
if os.environ.get("FRESH_SCORE_SMOKE"):
    MIN_N = 1


def wilson_lower(h: int, n: int, z: float = 1.6448536269514722) -> float:
    if n == 0:
        return float("nan")
    p = h / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n)) / d - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d


def label(b):
    return f"{b[0]:.2f}-{b[1]:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("report exists; refusing to overwrite an adjudicated record")

    # ---- frozen constants, by value
    k6 = json.loads((CONTROL / "k6-coverage-gb200-2096.json").read_text())
    kappa = {k: float(v) for k, v in k6["kappa"].items()}
    q_base = {k: float(v) for k, v in k6["q_base"].items()}
    a13 = json.loads((CONTROL / "k6-amendment-13-quantile-drift.json").read_text())
    q_old = {n: list(map(float, a13["q_of_depth"][n])) for n in TN}
    assert [tuple(b) for b in a13["bins"]] == OLD_BINS
    old_mids = [(a + b) / 2 for a, b in OLD_BINS]
    shallow_fit = {n: np.polyfit(old_mids[:SHALLOW], q_old[n][:SHALLOW], 1).tolist() for n in TN}
    norm = json.loads((REPO / "artifacts/g2/ensemble-v1/normalization-stats.json").read_text())

    def truth(solver, n):
        raw = solver.get(KEY[n], solver.get("max_von_mises_pa"))
        return (math.log(float(raw)) - norm["means"][n]) / norm["scales"][n]

    # ---- fresh rows
    rows, families, statuses = [], set(), defaultdict(int)
    for path in sorted(args.fresh_root.glob("trajectory-fresh-*.json")):
        rec = json.loads(path.read_text())
        statuses[rec.get("status")] += 1
        if rec.get("status") != "complete":
            continue
        if rec.get("role") != "fresh" or rec.get("intermediate_solver_call_count") != 0:
            raise SystemExit(f"{path.name}: not a clean fresh record")
        families.add(rec["family_id"])
        for s in rec["selected_states"]:
            rows.append({"family": rec["family_id"], "f": float(s["fraction_removed"]),
                         "mu": {n: float(s["prediction"]["mu"][n]) for n in TN},
                         "sig": {n: float(s["prediction"]["sigma"][n]) for n in TN},
                         "y": {n: truth(s["solver"], n) for n in TN}})
    if not rows:
        raise SystemExit("no complete fresh records")

    def in_bin(r, b):
        return b[0] < r["f"] <= b[1]

    def score(r, n):
        return r["y"][n] - r["mu"][n] - kappa[n] * r["sig"][n]

    def covered_joint(r, q):
        return all(score(r, n) <= q[n] for n in TN)

    def fit_q(sub):
        out = {}
        for n in TN:
            s = sorted(score(r, n) for r in sub)
            k = math.ceil((len(s) + 1) * (1 - ALPHA_J))
            out[n] = s[min(k, len(s)) - 1]
        return out

    by_bin = {label(b): [r for r in rows if in_bin(r, b)] for b in FRESH_BINS}
    report = {"fresh_root": str(args.fresh_root), "families": len(families), "states": len(rows),
              "record_statuses": dict(statuses), "kappa": kappa, "q_base": q_base,
              "n_per_bin": {l: len(v) for l, v in by_bin.items()}}

    print(f"fresh tranche: {len(families)} families, {len(rows)} states, statuses {dict(statuses)}")

    # A. baseline-calibrated bound
    print("\nA. baseline-calibrated (kappa, q_base) coverage on fresh bins")
    A = {}
    for l, sub in by_bin.items():
        if len(sub) < MIN_N:
            continue
        h = sum(covered_joint(r, q_base) for r in sub)
        A[l] = {"n": len(sub), "covered": h, "coverage": h / len(sub), "wilson_lower": wilson_lower(h, len(sub))}
        print(f"  {l}  n={len(sub):4d}  cov={h/len(sub):.3f}  wl={A[l]['wilson_lower']:.3f}")
    report["A_baseline_coverage"] = A

    # B. transfer matrix, frozen rows x fresh columns
    print("\nB. frozen q(d_i) rows evaluated on fresh bins (columns)")
    B = {}
    for i, bi in enumerate(OLD_BINS):
        qi = {n: q_old[n][i] for n in TN}
        row = {}
        for l, sub in by_bin.items():
            if len(sub) < MIN_N:
                continue
            h = sum(covered_joint(r, qi) for r in sub)
            row[l] = {"n": len(sub), "coverage": h / len(sub), "wilson_lower": wilson_lower(h, len(sub))}
        B[label(bi)] = row
        print(f"  q@{label(bi)}: " + " ".join(f"{v['coverage']:.3f}{'*' if v['wilson_lower']>=0.95 else ' '}" for v in row.values()))
    report["B_transfer_frozen_rows"] = B

    # C. shallow extrapolation, frozen fit
    print("\nC. frozen shallow linear fit (bins <=25%) extrapolated to fresh deep bins")
    C = {}
    for b in FRESH_BINS[SHALLOW:]:
        l = label(b)
        sub = by_bin[l]
        if len(sub) < MIN_N:
            continue
        q_ext = {n: float(np.polyval(shallow_fit[n], (b[0] + b[1]) / 2)) for n in TN}
        h = sum(covered_joint(r, q_ext) for r in sub)
        C[l] = {"n": len(sub), "q_extrapolated": q_ext, "coverage": h / len(sub), "wilson_lower": wilson_lower(h, len(sub))}
        print(f"  {l}  cov={h/len(sub):.3f}")
    report["C_extrapolation_coverage"] = C
    report["C_best_extrapolated_coverage"] = max((v["coverage"] for v in C.values()), default=float("nan"))

    # D. fresh quantile vs old
    print("\nD. q_fresh(d) vs q_old(d)")
    D = {}
    for i, b in enumerate(FRESH_BINS):
        l = label(b)
        sub = by_bin[l]
        if len(sub) < MIN_N:
            continue
        qf = fit_q(sub)
        entry = {"n": len(sub), "q_fresh": qf}
        if i < len(OLD_BINS):
            entry["q_old"] = {n: q_old[n][i] for n in TN}
            entry["ratio_fresh_over_old"] = {n: (qf[n] / q_old[n][i] if q_old[n][i] > 0 else None) for n in TN}
        D[l] = entry
        print(f"  {l}  " + "  ".join(f"{n[:4]} {qf[n]:+.3f}" + (f"/{q_old[n][i]:+.3f}" if i < len(OLD_BINS) else "") for n in TN))
    report["D_quantiles"] = D
    qc = [D[label(b)]["q_fresh"]["compliance"] for b in FRESH_BINS if label(b) in D]
    if len(qc) >= 2 and qc[0] > 0:
        report["D_fresh_growth_first_to_last_compliance"] = qc[-1] / qc[0]
        report["D_fresh_monotone_compliance"] = all(qc[i] <= qc[i + 1] + 1e-9 for i in range(len(qc) - 1))

    # E. sigma vs |residual|
    print("\nE. Pearson(sigma, |y-mu|) per bin, per target")
    E = {}
    for l, sub in by_bin.items():
        if len(sub) < MIN_N:
            continue
        E[l] = {}
        for n in TN:
            s = np.array([r["sig"][n] for r in sub]); e = np.abs(np.array([r["y"][n] - r["mu"][n] for r in sub]))
            E[l][n] = float(np.corrcoef(s, e)[0, 1]) if s.std() > 0 and e.std() > 0 else float("nan")
        print(f"  {l}  " + "  ".join(f"{n[:4]} {E[l][n]:+.2f}" for n in TN))
    report["E_sigma_residual_pearson"] = E

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
