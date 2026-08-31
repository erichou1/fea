"""EXPLORATORY post-hoc characterisation of the K6 depth-coverage breakdown.

Nothing here is pre-registered.  This script exists to describe the shape of a
finding that the frozen per-bin analysis already established, and every output
it produces is labelled EXPLORATORY.  It may not be used to support a claim; the
frozen adjudication in k6-coverage-interim-355.json is the only decision record.

Specifically it asks three questions the frozen bins cannot answer:

1.  Is the >25% collapse a genuine depth effect, or an artifact of that bin
    being unbounded above while the others span five points each?
2.  Does the collapse track distance beyond the deepest state the surrogate saw
    in fit-role training, i.e. is it extrapolation rather than depth per se?
3.  Is sigma growing with depth (the predictor knows it is unsure) or flat (the
    predictor is confidently wrong)?

Question 3 matters most.  A predictor whose sigma grows with depth is
recoverable by recalibration; one whose sigma stays flat while error grows is
not, and that distinction changes what the paper can propose.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sasto.g3_trajectory_calibration import _normalized_targets, _selected_trajectory_rows, _verified_json  # noqa: E402
from sasto.k6_coverage import wilson_lower_bound  # noqa: E402

TARGETS = ("compliance", "max_displacement", "max_von_mises")


def load(root: Path):
    cases = []
    for path in sorted(root.glob("trajectory-development-*.json")):
        cases.append(_verified_json(path, "G3 trajectory case", "trajectory_digest"))
    rows, _, _ = _selected_trajectory_rows(cases)
    return cases, rows


def residuals(rows, kappa, q_base, normalization):
    out = []
    for row in rows:
        solver = row["solver"]
        y = _normalized_targets({
            "compliance": solver["compliance_j"],
            "max_von_mises": solver.get("max_gauss_von_mises_pa", solver.get("max_von_mises_pa")),
            "max_displacement": solver["max_displacement_m"],
        }, normalization)
        prediction = row["prediction"]
        record = {"fraction_removed": float(row["fraction_removed"]), "bin_label": row["bin_label"],
                  "sample_id": row["sample_id"]}
        covered_all = True
        for name in TARGETS:
            mu = float(prediction["mu"][name])
            sigma = float(prediction["sigma"][name])
            upper = mu + kappa[name] * sigma + q_base[name]
            record[f"{name}_error"] = y[name] - mu
            record[f"{name}_sigma"] = sigma
            record[f"{name}_slack"] = upper - y[name]
            record[f"{name}_covered"] = y[name] <= upper
            covered_all = covered_all and (y[name] <= upper)
        record["covered"] = covered_all
        out.append(record)
    return out


def fine_bins(records, edges):
    buckets = defaultdict(list)
    for record in records:
        fraction = record["fraction_removed"]
        for low, high in zip(edges, edges[1:]):
            if low < fraction <= high:
                buckets[(low, high)].append(record)
                break
    return buckets


def main() -> int:
    root = Path("/Users/eric/workspace/fea-sasto-v/artifacts/g3/trajectory-calibration-v2")
    normalization = json.loads((Path("/Users/eric/workspace/fea-sasto-v/artifacts/g2/ensemble-v1/normalization-stats.json")).read_text())
    kappa_record = _verified_json(root / "kappa-development-evidence.json", "G3 kappa evidence", "kappa_evidence_sha256")
    q_record = _verified_json(root / "baseline-calibration.json", "G3 baseline calibration", "baseline_calibration_sha256")
    kappa = {k: float(v) for k, v in kappa_record["kappa"].items()}
    q_base = {k: float(v) for k, v in q_record["q"].items()}

    cases, rows = load(root)
    records = residuals(rows, kappa, q_base, normalization)

    print("EXPLORATORY — not pre-registered, not usable as evidence for a claim\n")

    deep = [r for r in records if r["bin_label"] == ">25%"]
    fractions = sorted(r["fraction_removed"] for r in deep)
    print(f"Q1. Shape of the >25% bin (n={len(deep)})")
    print(f"    range {fractions[0]:.4f} to {fractions[-1]:.4f}, "
          f"median {fractions[len(fractions)//2]:.4f}")
    print("    the bin is NOT unbounded in practice; trajectories stop at candidate exhaustion\n")

    edges = [0.25, 0.27, 0.29, 0.30, 0.31, 0.32, 0.35]
    print("    fine slices inside >25%:")
    for (low, high), bucket in sorted(fine_bins(deep, edges).items()):
        if not bucket:
            continue
        n = len(bucket)
        covered = sum(1 for r in bucket if r["covered"])
        print(f"      ({low:.2f},{high:.2f}]  n={n:4d}  cov={covered/n:.4f}  L={wilson_lower_bound(covered, n):.4f}")

    print("\nQ2. Coverage across the whole depth range, 2-point slices")
    edges = [0.05 * i for i in range(1, 8)]
    edges = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    for (low, high), bucket in sorted(fine_bins(records, edges).items()):
        if not bucket:
            continue
        n = len(bucket)
        covered = sum(1 for r in bucket if r["covered"])
        print(f"      ({low:.2f},{high:.2f}]  n={n:4d}  cov={covered/n:.4f}  L={wilson_lower_bound(covered, n):.4f}")

    print("\nQ3. Does sigma grow with depth, or is the predictor confidently wrong?")
    print(f"      {'bin':12s} {'n':>4s} " + " ".join(f"{t[:9]:>10s}" for t in TARGETS))
    by_bin = defaultdict(list)
    for record in records:
        by_bin[record["bin_label"]].append(record)
    order = ["(5,10%]", "(10,15%]", "(15,20%]", "(20,25%]", ">25%"]
    for label in order:
        bucket = by_bin.get(label)
        if not bucket:
            continue
        means = []
        for name in TARGETS:
            means.append(sum(r[f"{name}_sigma"] for r in bucket) / len(bucket))
        print(f"      sigma {label:6s} {len(bucket):4d} " + " ".join(f"{value:10.4f}" for value in means))
    for label in order:
        bucket = by_bin.get(label)
        if not bucket:
            continue
        means = []
        for name in TARGETS:
            means.append(sum(r[f"{name}_error"] for r in bucket) / len(bucket))
        print(f"      error {label:6s} {len(bucket):4d} " + " ".join(f"{value:10.4f}" for value in means))
    for label in order:
        bucket = by_bin.get(label)
        if not bucket:
            continue
        ratios = []
        for name in TARGETS:
            mean_error = sum(r[f"{name}_error"] for r in bucket) / len(bucket)
            mean_sigma = sum(r[f"{name}_sigma"] for r in bucket) / len(bucket)
            ratios.append(mean_error / mean_sigma if mean_sigma else float("nan"))
        print(f"      ratio {label:6s} {len(bucket):4d} " + " ".join(f"{value:10.4f}" for value in ratios))

    print("\n      per-target coverage by bin")
    for label in order:
        bucket = by_bin.get(label)
        if not bucket:
            continue
        rates = [sum(1 for r in bucket if r[f"{name}_covered"]) / len(bucket) for name in TARGETS]
        print(f"      {label:12s} {len(bucket):4d} " + " ".join(f"{value:10.4f}" for value in rates))

    print("\nQ4. How much would q have to grow to restore 0.95 at the deepest bin?")
    deep_records = by_bin[">25%"]
    for name in TARGETS:
        slacks = sorted(r[f"{name}_slack"] for r in deep_records)
        index = math.floor(0.05 * len(slacks))
        deficit = -slacks[index]
        print(f"      {name:18s} q_base={q_base[name]:.4f}  needs +{deficit:.4f}  "
              f"({(q_base[name] + deficit) / q_base[name]:.1f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
