"""Amendment 04 arms: score construction, and whether the wide bound is vacuous.

Two audits independently converged on the same two gaps.

GAP 1, score construction.  The theory audit points out that the campaign's score
    A_add = Y - mu - kappa*sigma
is SHIFTED but not NORMALIZED: sigma enters additively, scaled by a constant.
The genuinely normalized (studentized, locally-weighted) score is
    A_norm = (Y - mu)/sigma,      giving the endpoint  mu + q*sigma.
Calling the campaign's sigma a "normalizer" is therefore imprecise, and the
mechanism claim rests on a construction that was never compared against the
standard alternative.  Arm C tested kappa=0 (drop sigma entirely) but never
tested using sigma correctly.  Three constructions, one of them untested:

    D1  raw          Y - mu                 (= arm C)
    D2  shifted      Y - mu - kappa*sigma   (= the campaign's, current result)
    D3  normalized   (Y - mu)/sigma         (UNTESTED)

If D3 transports across depth materially better than D2, the finding is about
score construction and the mechanism claim must be restated.  If D3 also fails,
the mechanism claim survives a real attempt to rescue it.

GAP 2, vacuity.  The framing audit: "q_j-relative multiples are not evidence of
practical vacuity unless the inflated bounds are compared with physically
meaningful tolerances and trivial baselines."  Correct.  A bound 8.7x a small
q_base may still be tight.  Compare the valid deep bound against trivial
baselines that need no model at all.
"""

from __future__ import annotations

import json
import math
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
ORDER = ["(5,10%]", "(10,15%]", "(15,20%]", "(20,25%]", ">25%"]


def traj_rows(role, normalization):
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


def baseline_rows(role):
    br = json.loads((GB200 / "baseline-rows.json").read_text())
    return [{"y": r["y"],
             "mu": {n: float(r["prediction"]["mu"][n]) for n in TARGET_NAMES},
             "sigma": {n: float(r["prediction"]["sigma"][n]) for n in TARGET_NAMES}}
            for r in br["roles"][role]["rows"]]


def cover_by_bin(rows, hit_fn):
    h, t = defaultdict(int), defaultdict(int)
    for r in rows:
        t[r["bin"]] += 1
        h[r["bin"]] += 1 if hit_fn(r) else 0
    return {b: (h[b], t[b]) for b in ORDER if t[b]}


def show(title, cov):
    print(f"\n{title}")
    print(f"  {'bin':10s} {'n':>5s} {'cov':>8s} {'WilsonL':>8s}")
    for b in ORDER:
        if b in cov:
            hh, nn = cov[b]
            print(f"  {b:10s} {nn:5d} {hh/nn:8.4f} {wilson_lower_bound(hh, nn):8.4f}")


def main() -> int:
    normalization = json.loads(NORM.read_text())
    kappa = {k: float(v) for k, v in _verified_json(
        FROZEN / "kappa-development-evidence.json", "G3 kappa evidence",
        "kappa_evidence_sha256")["kappa"].items()}

    dev = traj_rows("development", normalization)
    cal_base = baseline_rows("calibration")
    report = {"kappa": kappa}

    print("=" * 74)
    print("GAP 1  SCORE CONSTRUCTION: shifted vs genuinely normalized")
    print("=" * 74)
    print("all three calibrated on the SAME 1,108 baseline calibration states")

    # D2 shifted (the campaign's construction)
    q_add = {n: split_conformal_quantile(
        [r["y"][n] - (r["mu"][n] + kappa[n] * r["sigma"][n]) for r in cal_base],
        alpha=ALPHA_J) for n in TARGET_NAMES}
    cov_add = cover_by_bin(dev, lambda r: all(
        r["y"][n] <= r["mu"][n] + kappa[n] * r["sigma"][n] + q_add[n] for n in TARGET_NAMES))
    show("D2  shifted   U = mu + kappa*sigma + q      (the campaign's score)", cov_add)

    # D1 raw
    q_raw = {n: split_conformal_quantile([r["y"][n] - r["mu"][n] for r in cal_base],
                                         alpha=ALPHA_J) for n in TARGET_NAMES}
    cov_raw = cover_by_bin(dev, lambda r: all(
        r["y"][n] <= r["mu"][n] + q_raw[n] for n in TARGET_NAMES))
    show("D1  raw       U = mu + q                    (sigma dropped)", cov_raw)

    # D3 normalized -- the untested one
    q_norm = {n: split_conformal_quantile(
        [(r["y"][n] - r["mu"][n]) / r["sigma"][n] for r in cal_base],
        alpha=ALPHA_J) for n in TARGET_NAMES}
    cov_norm = cover_by_bin(dev, lambda r: all(
        r["y"][n] <= r["mu"][n] + q_norm[n] * r["sigma"][n] for n in TARGET_NAMES))
    show("D3  normalized U = mu + q*sigma             (studentized, UNTESTED)", cov_norm)

    print(f"\n  q_add  = {{{', '.join(f'{n}: {q_add[n]:.4f}' for n in TARGET_NAMES)}}}")
    print(f"  q_raw  = {{{', '.join(f'{n}: {q_raw[n]:.4f}' for n in TARGET_NAMES)}}}")
    print(f"  q_norm = {{{', '.join(f'{n}: {q_norm[n]:.4f}' for n in TARGET_NAMES)}}}")

    d2, d3 = cov_add[">25%"], cov_norm[">25%"]
    delta = d3[0] / d3[1] - d2[0] / d2[1]
    print(f"\n  DEEPEST BIN: shifted {d2[0]/d2[1]:.4f}  normalized {d3[0]/d3[1]:.4f}  delta {delta:+.4f}")
    verdict = ("normalized score MATERIALLY BETTER; finding is about score construction"
               if delta > 0.02 else
               "normalized score MATERIALLY WORSE" if delta < -0.02 else
               "NO MATERIAL DIFFERENCE; using sigma correctly does not rescue the bound")
    print(f"  -> {verdict}")
    report["gap1"] = {"q_add": q_add, "q_raw": q_raw, "q_norm": q_norm,
                      "coverage": {"shifted": {b: {"covered": h, "n": n, "coverage": h / n}
                                               for b, (h, n) in cov_add.items()},
                                   "raw": {b: {"covered": h, "n": n, "coverage": h / n}
                                           for b, (h, n) in cov_raw.items()},
                                   "normalized": {b: {"covered": h, "n": n, "coverage": h / n}
                                                  for b, (h, n) in cov_norm.items()}},
                      "deepest_delta_norm_minus_shifted": delta, "verdict": verdict}

    print()
    print("=" * 74)
    print("GAP 2  IS THE VALID DEEP BOUND ACTUALLY VACUOUS?")
    print("=" * 74)
    arms = json.loads((CONTROL / "k6-amendment-03-arms.json").read_text())
    q_bin = arms["arm_b"]["per_bin_q"][">25%"]
    deep = [r for r in dev if r["bin"] == ">25%"]
    scales = normalization["scales"]

    print("\nAll quantities in normalized log units unless marked physical.\n")
    print(f"  {'target':18s} {'boundP95':>9s} {'y_p95':>8s} {'y_range':>8s} {'trivial':>8s} {'ratio':>7s}")
    g2 = {}
    for n in TARGET_NAMES:
        bounds = sorted(r["mu"][n] + kappa[n] * r["sigma"][n] + q_bin[n] for r in deep)
        ys = sorted(r["y"][n] for r in deep)
        base_y = sorted(r["y"][n] for r in cal_base)
        p95_bound = bounds[int(0.95 * len(bounds))]
        p95_y = ys[int(0.95 * len(ys))]
        rng = ys[-1] - ys[0]
        # trivial model-free bound: the 95th percentile of BASELINE observed y,
        # available with no surrogate at all
        trivial = base_y[int(0.95 * len(base_y))]
        # how much of the bound's excess over truth is the conformal correction?
        ratio = (p95_bound - p95_y) / rng if rng else float("nan")
        print(f"  {n:18s} {p95_bound:9.3f} {p95_y:8.3f} {rng:8.3f} {trivial:8.3f} {ratio:7.3f}")
        g2[n] = {"bound_p95": p95_bound, "y_p95": p95_y, "y_range": rng,
                 "trivial_baseline_p95": trivial,
                 "excess_over_truth_as_fraction_of_range": ratio,
                 "bound_exceeds_trivial": bool(p95_bound > trivial),
                 "physical_bound_over_prediction": math.exp(q_bin[n] * scales[n])}

    print("\n  KEY TEST: does the valid deep bound exceed a trivial model-free bound")
    print("  (the 95th percentile of observed BASELINE values, needing no surrogate)?")
    for n in TARGET_NAMES:
        v = g2[n]
        tag = "VACUOUS (worse than trivial)" if v["bound_exceeds_trivial"] else "still tighter than trivial"
        print(f"    {n:18s} {tag}")
    n_vac = sum(1 for n in TARGET_NAMES if g2[n]["bound_exceeds_trivial"])
    print(f"\n  -> {n_vac}/3 targets vacuous by this test")
    report["gap2"] = {"per_target": g2, "n_targets_vacuous": n_vac,
                      "test": "valid deep Mondrian bound vs 95th pct of baseline observed y"}

    out = CONTROL / "k6-amendment-04-score-and-vacuity.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
