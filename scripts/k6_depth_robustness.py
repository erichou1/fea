"""EXPLORATORY robustness checks on the K6 depth-coverage finding.

Not pre-registered.  Purpose is adversarial: try to make the depth effect go
away.  If it survives these, it is probably real; if any one of them explains
it, the headline finding is an artifact and must not be written up.

Checks:

A.  Paired within-family.  Does coverage drop with depth *within* the same
    family?  This removes family heterogeneity entirely.  If the effect is
    driven by a subpopulation of hard families that only appear at depth, the
    paired test kills it.
B.  Concentration.  Is failure spread across families, or does a small set of
    pathological families account for it?
C.  Is the deepest bin selecting different families than the shallow bins?
    Bin (10,15%] has n=343 not 355, so 12 families lack that bin; the deepest
    has 343 too.  Check whether the missing families differ systematically.
D.  Sign of the effect per family: does error increase monotonically with depth
    within a family, or is it noise that happens to average upward?
E.  Does the surrogate's sigma grow proportionally with |error| within family?
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sasto.g3_trajectory_calibration import _normalized_targets, _selected_trajectory_rows, _verified_json  # noqa: E402
from sasto.k6_coverage import binomial_cdf, wilson_lower_bound  # noqa: E402

TARGETS = ("compliance", "max_displacement", "max_von_mises")
ORDER = ["(5,10%]", "(10,15%]", "(15,20%]", "(20,25%]", ">25%"]


def build(root: Path):
    normalization = json.loads(Path("/Users/eric/workspace/fea-sasto-v/artifacts/g2/ensemble-v1/normalization-stats.json").read_text())
    kappa_record = _verified_json(Path(__import__('os').environ.get('K6_CONST', str(root))) / "kappa-development-evidence.json", "G3 kappa evidence", "kappa_evidence_sha256")
    q_record = _verified_json(Path(__import__('os').environ.get('K6_CONST', str(root))) / "baseline-calibration.json", "G3 baseline calibration", "baseline_calibration_sha256")
    kappa = {k: float(v) for k, v in kappa_record["kappa"].items()}
    q_base = {k: float(v) for k, v in q_record["q"].items()}
    cases = [_verified_json(p, "G3 trajectory case", "trajectory_digest")
             for p in sorted(root.glob("trajectory-development-*.json"))]
    rows, _, _ = _selected_trajectory_rows(cases)
    records = []
    for row in rows:
        solver = row["solver"]
        y = _normalized_targets({
            "compliance": solver["compliance_j"],
            "max_von_mises": solver.get("max_gauss_von_mises_pa", solver.get("max_von_mises_pa")),
            "max_displacement": solver["max_displacement_m"],
        }, normalization)
        prediction = row["prediction"]
        covered = True
        errors, sigmas = {}, {}
        for name in TARGETS:
            mu = float(prediction["mu"][name])
            sigma = float(prediction["sigma"][name])
            covered = covered and (y[name] <= mu + kappa[name] * sigma + q_base[name])
            errors[name] = y[name] - mu
            sigmas[name] = sigma
        records.append({"family_id": row["family_id"], "bin_label": row["bin_label"],
                        "fraction_removed": float(row["fraction_removed"]),
                        "covered": covered, "errors": errors, "sigmas": sigmas})
    return records


def sign_test(pairs):
    """Exact two-sided sign test; pairs of (shallow, deep) booleans or floats."""
    wins = sum(1 for a, b in pairs if b > a)
    losses = sum(1 for a, b in pairs if b < a)
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    lower = binomial_cdf(min(wins, losses), n, 0.5)
    return wins, losses, min(1.0, 2.0 * lower)


def main() -> int:
    root = Path(__import__('os').environ.get('K6_ROOT', '/Users/eric/workspace/fea-sasto-v/artifacts/g3/trajectory-calibration-v2'))
    records = build(root)
    by_family = defaultdict(dict)
    for record in records:
        by_family[record["family_id"]][record["bin_label"]] = record

    print("EXPLORATORY robustness — adversarial checks on the depth effect\n")

    print("A. Paired within-family, shallowest (5,10%] vs deepest >25%")
    paired = [(f["(5,10%]"], f[">25%"]) for f in by_family.values()
              if "(5,10%]" in f and ">25%" in f]
    print(f"   families with both bins: {len(paired)}")
    both = sum(1 for a, b in paired if a["covered"] and b["covered"])
    only_shallow = sum(1 for a, b in paired if a["covered"] and not b["covered"])
    only_deep = sum(1 for a, b in paired if not a["covered"] and b["covered"])
    neither = sum(1 for a, b in paired if not a["covered"] and not b["covered"])
    print(f"   covered both={both}  shallow only={only_shallow}  deep only={only_deep}  neither={neither}")
    n = only_shallow + only_deep
    p = min(1.0, 2.0 * binomial_cdf(min(only_shallow, only_deep), n, 0.5)) if n else 1.0
    print(f"   McNemar exact discordant {only_shallow}/{only_deep}, p={p:.3e}")
    print("   -> a family that is covered shallow and uncovered deep is the dominant pattern\n"
          if only_shallow > only_deep else "   -> no directional pattern\n")

    print("B. Concentration of failure across families")
    fail_counts = defaultdict(int)
    for record in records:
        if not record["covered"]:
            fail_counts[record["family_id"]] += 1
    total_failures = sum(fail_counts.values())
    families_failing = len(fail_counts)
    print(f"   total uncovered states: {total_failures} across {families_failing} distinct families "
          f"of {len(by_family)}")
    distribution = defaultdict(int)
    for count in fail_counts.values():
        distribution[count] += 1
    for count in sorted(distribution):
        print(f"     families with exactly {count} uncovered bin(s): {distribution[count]}")
    top = sorted(fail_counts.values(), reverse=True)[:20]
    print(f"   top-20 families account for {sum(top)}/{total_failures} = {sum(top)/total_failures:.2%}")
    print("   -> failure is broad, not a handful of pathological families\n"
          if families_failing > 0.3 * len(by_family) else "   -> failure is concentrated\n")

    print("C. Do families missing a bin differ?")
    for label in ORDER:
        present = [f for f in by_family.values() if label in f]
        print(f"   {label:10s} present in {len(present):4d} families")
    missing_deep = [fid for fid, f in by_family.items() if ">25%" not in f]
    print(f"   families lacking >25%: {len(missing_deep)}")
    if missing_deep:
        depths = [max(r["fraction_removed"] for r in by_family[fid].values()) for fid in missing_deep]
        print(f"   their max depth reached: min={min(depths):.4f} max={max(depths):.4f}")
        print("   -> these trajectories simply stopped shallower; not a coverage-related exclusion\n")

    print("D. Within-family error monotonicity with depth")
    for name in TARGETS:
        pairs = [(f["(5,10%]"]["errors"][name], f[">25%"]["errors"][name])
                 for f in by_family.values() if "(5,10%]" in f and ">25%" in f]
        wins, losses, p = sign_test(pairs)
        print(f"   {name:18s} error increases in {wins}/{wins+losses} families, p={p:.3e}")

    print("\nE. Does sigma keep pace with |error| within family?")
    for name in TARGETS:
        pairs = [f for f in by_family.values() if "(5,10%]" in f and ">25%" in f]
        sigma_growth = [f[">25%"]["sigmas"][name] / f["(5,10%]"]["sigmas"][name] for f in pairs]
        error_growth = [abs(f[">25%"]["errors"][name]) / max(abs(f["(5,10%]"]["errors"][name]), 1e-9) for f in pairs]
        sigma_growth.sort(); error_growth.sort()
        median_sigma = sigma_growth[len(sigma_growth) // 2]
        median_error = error_growth[len(error_growth) // 2]
        print(f"   {name:18s} median sigma growth {median_sigma:6.2f}x   "
              f"median |error| growth {median_error:8.2f}x")
    print("\n   -> if |error| growth greatly exceeds sigma growth, the predictor is")
    print("      confidently wrong at depth and recalibration alone cannot fix it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
