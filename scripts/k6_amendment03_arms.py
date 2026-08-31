"""K6 amendment 03: three diagnostic arms.

Frozen at K6_AMENDMENT_03_DIAGNOSTIC_ARMS.md, SHA-256
abb290be4ac1cf7ca2fbf3b600839201d81908ac9a07895b47afbb6c2ca97f8f, before any of
these was computed.

Arm A  trajectory calibration      q_traj from calibration-role TRAJECTORY states
Arm B  depth-conditional Mondrian  per-bin q from calibration-role states in-bin
Arm C  sigma ablation              kappa = 0, does the normalizer contribute?

All three are recomputations over already-verified records.  No solver call, no
confirmation access, no kappa tuning.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sasto.g3_trajectory_calibration import (  # noqa: E402
    ALPHA_J, DEPTH_BINS, TARGET_NAMES, _normalized_targets,
    _selected_trajectory_rows, _verified_json, split_conformal_quantile,
)
from sasto.k6_coverage import wilson_lower_bound, binomial_cdf  # noqa: E402

GB200 = Path("/Users/eric/workspace/sasto-g3-gb200-inbound/trajectory-calibration-gb200")
FROZEN = Path("/Users/eric/workspace/fea-sasto-v/artifacts/g3/trajectory-calibration-v2")
NORM = Path("/Users/eric/workspace/fea-sasto-v/artifacts/g2/ensemble-v1/normalization-stats.json")
CONTROL = Path("/Users/eric/workspace/sasto-modernization-control/v2/g3")
AMENDMENT_03 = "abb290be4ac1cf7ca2fbf3b600839201d81908ac9a07895b47afbb6c2ca97f8f"
ORDER = ["(5,10%]", "(10,15%]", "(15,20%]", "(20,25%]", ">25%"]


def load(role: str):
    return [_verified_json(p, "G3 trajectory case", "trajectory_digest")
            for p in sorted(GB200.glob(f"trajectory-{role}-*.json"))]


def rows_with_y(cases, normalization):
    """Selected states with normalized truth attached."""
    rows, _, _ = _selected_trajectory_rows(cases)
    out = []
    for row in rows:
        solver = row["solver"]
        y = _normalized_targets({
            "compliance": solver["compliance_j"],
            "max_von_mises": solver.get("max_gauss_von_mises_pa", solver.get("max_von_mises_pa")),
            "max_displacement": solver["max_displacement_m"],
        }, normalization)
        out.append({"bin_label": row["bin_label"], "y": y,
                    "mu": {n: float(row["prediction"]["mu"][n]) for n in TARGET_NAMES},
                    "sigma": {n: float(row["prediction"]["sigma"][n]) for n in TARGET_NAMES}})
    return out


def quantiles(rows, kappa):
    """Split-conformal quantile per target from scores y - (mu + kappa*sigma)."""
    q = {}
    for name in TARGET_NAMES:
        scores = [r["y"][name] - (r["mu"][name] + kappa[name] * r["sigma"][name]) for r in rows]
        q[name] = split_conformal_quantile(scores, alpha=ALPHA_J)
    return q


def coverage(rows, kappa, q, per_bin_q=None):
    """Joint coverage over all targets, by depth bin."""
    hit = defaultdict(int)
    tot = defaultdict(int)
    for r in rows:
        qq = per_bin_q[r["bin_label"]] if per_bin_q else q
        ok = all(r["y"][n] <= r["mu"][n] + kappa[n] * r["sigma"][n] + qq[n] for n in TARGET_NAMES)
        tot[r["bin_label"]] += 1
        hit[r["bin_label"]] += 1 if ok else 0
    return {b: (hit[b], tot[b]) for b in ORDER if tot[b]}


def show(title, cov, extra=None):
    print(f"\n{title}")
    print(f"  {'bin':10s} {'n':>5s} {'cov':>8s} {'WilsonL':>8s}" + ("  " + extra["header"] if extra else ""))
    for b in ORDER:
        if b not in cov:
            continue
        h, n = cov[b]
        line = f"  {b:10s} {n:5d} {h/n:8.4f} {wilson_lower_bound(h, n):8.4f}"
        if extra:
            line += "  " + extra["row"](b)
        print(line)


def main() -> int:
    normalization = json.loads(NORM.read_text())
    kappa_rec = _verified_json(FROZEN / "kappa-development-evidence.json",
                               "G3 kappa evidence", "kappa_evidence_sha256")
    q_rec = _verified_json(FROZEN / "baseline-calibration.json",
                           "G3 baseline calibration", "baseline_calibration_sha256")
    kappa = {k: float(v) for k, v in kappa_rec["kappa"].items()}
    q_base = {k: float(v) for k, v in q_rec["q"].items()}
    zero = {n: 0.0 for n in TARGET_NAMES}

    dev = rows_with_y(load("development"), normalization)
    cal = rows_with_y(load("calibration"), normalization)
    print(f"development selected states: {len(dev)}   calibration selected states: {len(cal)}")
    print(f"amendment 03: {AMENDMENT_03[:16]}...")

    report = {"amendment_03_sha256": AMENDMENT_03,
              "development_selected_states": len(dev),
              "calibration_selected_states": len(cal),
              "kappa": kappa, "q_base": q_base}

    # ---------------------------------------------------------------- baseline
    base_cov = coverage(dev, kappa, q_base)
    show("REFERENCE  baseline-calibrated q_base (the adjudicated K6 result)", base_cov)
    report["reference"] = {b: {"covered": h, "n": n, "coverage": h / n} for b, (h, n) in base_cov.items()}

    # --------------------------------------------------------------- ARM A
    q_traj = quantiles(cal, kappa)
    a_cov = coverage(dev, kappa, q_traj)
    show("ARM A  trajectory-calibrated q_traj (H2)", a_cov,
         {"header": "q_traj/q_base", "row": lambda b: ""})
    print(f"    q_base = {{{', '.join(f'{n}: {q_base[n]:.4f}' for n in TARGET_NAMES)}}}")
    print(f"    q_traj = {{{', '.join(f'{n}: {q_traj[n]:.4f}' for n in TARGET_NAMES)}}}")
    print(f"    ratio  = {{{', '.join(f'{n}: {q_traj[n]/q_base[n]:.2f}x' for n in TARGET_NAMES)}}}")
    report["arm_a"] = {"q_traj": q_traj,
                       "ratio_to_base": {n: q_traj[n] / q_base[n] for n in TARGET_NAMES},
                       "coverage": {b: {"covered": h, "n": n, "coverage": h / n,
                                        "wilson_lower": wilson_lower_bound(h, n)}
                                    for b, (h, n) in a_cov.items()}}

    # --------------------------------------------------------------- ARM B
    by_bin = defaultdict(list)
    for r in cal:
        by_bin[r["bin_label"]].append(r)
    per_bin_q = {b: quantiles(by_bin[b], kappa) for b in ORDER if by_bin[b]}
    b_cov = coverage(dev, kappa, q_base, per_bin_q=per_bin_q)
    show("ARM B  depth-conditional (Mondrian) per-bin q", b_cov,
         {"header": "q_compliance  (xq_base)",
          "row": lambda b: f"{per_bin_q[b]['compliance']:11.4f}  ({per_bin_q[b]['compliance']/q_base['compliance']:.1f}x)"})
    print(f"    calibration states per bin: {{{', '.join(f'{b}: {len(by_bin[b])}' for b in ORDER if by_bin[b])}}}")
    report["arm_b"] = {"per_bin_q": per_bin_q,
                       "calibration_n_per_bin": {b: len(by_bin[b]) for b in ORDER if by_bin[b]},
                       "coverage": {b: {"covered": h, "n": n, "coverage": h / n,
                                        "wilson_lower": wilson_lower_bound(h, n)}
                                    for b, (h, n) in b_cov.items()}}

    # --------------------------------------------------------------- ARM C
    q_nosigma = quantiles(cal if False else [r for r in cal], zero)
    # calibrate the no-sigma bound on the SAME baseline-role states the primary
    # result used, so the comparison isolates sigma rather than the calibration set
    q_nosigma_base = None
    baseline_rows_path = GB200 / "baseline-rows.json"
    if baseline_rows_path.exists():
        br = json.loads(baseline_rows_path.read_text())
        rows_b = br.get("roles", {}).get("calibration", {}).get("rows", [])
        if rows_b:
            conv = [{"y": r["y"], "mu": {n: float(r["prediction"]["mu"][n]) for n in TARGET_NAMES},
                     "sigma": {n: float(r["prediction"]["sigma"][n]) for n in TARGET_NAMES}}
                    for r in rows_b]
            q_nosigma_base = quantiles(conv, zero)
    q_c = q_nosigma_base if q_nosigma_base else q_nosigma
    c_cov = coverage(dev, zero, q_c)
    show("ARM C  sigma ablation, kappa = 0" +
         ("  [q from baseline states]" if q_nosigma_base else "  [q from trajectory states]"), c_cov)
    print(f"    q(kappa=0) = {{{', '.join(f'{n}: {q_c[n]:.4f}' for n in TARGET_NAMES)}}}")
    report["arm_c"] = {"q_zero_kappa": q_c,
                       "calibrated_on": "baseline" if q_nosigma_base else "trajectory",
                       "coverage": {b: {"covered": h, "n": n, "coverage": h / n,
                                        "wilson_lower": wilson_lower_bound(h, n)}
                                    for b, (h, n) in c_cov.items()}}

    # -------------------------------------------------------- ARM C adjudication
    print("\nARM C adjudication, deepest bin, 2pp materiality margin")
    hb, nb = base_cov[">25%"]
    hc, nc = c_cov[">25%"]
    delta = (hb / nb) - (hc / nc)
    print(f"    with sigma:    {hb/nb:.4f}   without sigma: {hc/nc:.4f}   delta: {delta:+.4f}")
    if abs(delta) < 0.02:
        verdict = "SIGMA CONTRIBUTES NOTHING at depth (|delta| < 2pp)"
    elif delta > 0:
        verdict = "sigma helps at depth; mechanism claim must soften"
    else:
        verdict = "sigma is ACTIVELY HARMFUL at depth"
    print(f"    -> {verdict}")
    report["arm_c_adjudication"] = {"with_sigma": hb / nb, "without_sigma": hc / nc,
                                    "delta": delta, "verdict": verdict}

    out = CONTROL / "k6-amendment-03-arms.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
