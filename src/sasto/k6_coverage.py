"""K6 coverage evaluation against the frozen pre-registration and amendment 01.

This module computes the one and only quantity the K6 kill test needs: does a
one-sided upper predictor calibrated on unoptimized baseline states cover
optimization-trajectory states at the target rate?

It is deliberately separate from ``g3_trajectory_calibration``.  That module is
frozen under ``SOURCE_BUNDLE_PATHS`` and its records were produced with
``coverage_computed: False`` and a ``no_k6_coverage_or_adjudication`` hard stop.
Adding coverage there would change the source bundle digest of already-written
evidence.  This module reads those records and never writes into their root.

Role discipline is enforced, not assumed: the evaluator refuses to run if any
confirmation-role artifact is reachable, and it imports ``kappa`` and ``q_base``
by value from the frozen records rather than recomputing them.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .g3_trajectory_calibration import (
    ALPHA,
    ALPHA_J,
    DEPTH_BINS,
    J,
    TARGET_NAMES,
    G3Error,
    _normalized_targets,
    _selected_trajectory_rows,
    _verified_json,
    depth_bin_index,
    select_state_index,
)
from .manifest import sha256_file

SCHEMA_VERSION = "1.0.0"

#: Branch thresholds.  Values are transcribed from the frozen pre-registration
#: and amendment 01; the digests below bind them to those documents.
TARGET_COVERAGE = 0.95
PREMISE_FALSE_WILSON_LOWER = 0.95
PREMISE_TRUE_WILSON_LOWER = 0.93
#: Added by amendment 01 §5.  n-invariant, so a widened interval alone cannot
#: manufacture a positive finding at reduced n.
PREMISE_TRUE_POINT = 0.93

PREREGISTRATION_SHA256 = "79bd228a1a3b778a424594b33134fe99fdf72c77837d96209e7a41e402becdca"
AMENDMENT_01_SHA256 = "47256af82f4053cd8bfb44b9d400e8ed45139ef9edbe76deaee70de0e2f23312"
AMENDMENT_02_SHA256 = "d9ec781610666b50888a4d32dd353c5e264535d67c861b62be1ee01370c22132"

#: Accepted amendment digests.  A run must cite one of these.
ACCEPTED_AMENDMENTS = {
    AMENDMENT_01_SHA256: "01_REDUCED_N",
    AMENDMENT_02_SHA256: "02_GB200_POPULATION",
}

#: Full pre-registered development population.  A run below this is INTERIM.
PREREGISTERED_FAMILY_COUNT = 2235

#: G3-D1 (amendment 14).  Records whose source bundle pins
#: ``g3_trajectory_calibration.py`` at one of these hashes were produced with
#: part labels masked by occupancy, a representation the G2 ensemble never saw.
#: Their prediction fields are invalid unless a ``d1_correction`` block shows
#: they were recomputed on the raw representation.
D1_DEFECTIVE_G3_HASHES = frozenset({
    "7ada646364bc0427",  # 92432e6
    "2587d40b127d0959",  # 182599b (GB200 run)
    "9b2977e8c538d7c5",  # 4f9ad42
})
D1_FIX_COMMIT = "00856c4b1d5fb555ef586686de3c0e7dade007cb"


def _assert_d1_corrected(output_root: Path, cases: Sequence[Mapping[str, object]]) -> str:
    """Refuse to evaluate predictions produced on masked part labels."""
    manifest_path = output_root / "campaign-manifest.json"
    pinned = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            pinned = str(manifest.get("source_bundle_files", {}).get("src/sasto/g3_trajectory_calibration.py", ""))[:16]
        except (OSError, json.JSONDecodeError, AttributeError):
            pinned = None
    if pinned is not None and pinned not in D1_DEFECTIVE_G3_HASHES and pinned != "":
        return "produced_after_fix"
    missing = [str(case.get("sample_id")) for case in cases
               if not isinstance(case.get("d1_correction"), Mapping)
               or case["d1_correction"].get("fix_commit") != D1_FIX_COMMIT]  # type: ignore[index]
    if missing:
        raise K6Error(
            "G3-D1: {} record(s) carry predictions computed on masked part labels and no "
            "d1_correction at commit {}; refusing to evaluate (amendment 14)".format(len(missing), D1_FIX_COMMIT[:7]))
    return "d1_corrected"


class K6Error(G3Error):
    """A K6 coverage invariant was violated."""


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def wilson_lower_bound(successes: int, n: int, *, alpha: float = 0.05) -> float:
    """One-sided Wilson lower bound.  Pure stdlib; no scipy dependency."""
    if n <= 0 or successes < 0 or successes > n:
        raise K6Error("wilson bound inputs are invalid")
    z = _normal_quantile(1.0 - alpha)
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (centre - radius) / denominator


def _normal_quantile(p: float) -> float:
    """Acklam inverse normal CDF.  Accurate to ~1.15e-9, ample here."""
    if not 0.0 < p < 1.0:
        raise K6Error("normal quantile argument is out of range")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return _refine_normal_quantile(x, p)


def _refine_normal_quantile(x: float, p: float) -> float:
    """One Halley step against the exact CDF, taking Acklam to machine precision.

    The raw approximation is accurate to ~1.15e-9.  That is comfortably inside
    any K6 threshold, but the refined value costs nothing and removes the
    question of approximation error from a quantity that sits underneath a
    decision boundary.
    """
    error = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    density = math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)
    if density == 0.0:
        return x
    u = error / density
    return x - u / (1.0 + x * u / 2.0)


def _log_binom_coefficient(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Bin(n, p), summed in log space for numerical safety."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if not 0.0 <= p <= 1.0:
        raise K6Error("binomial probability is out of range")
    if p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    total = 0.0
    for i in range(0, k + 1):
        total += math.exp(_log_binom_coefficient(n, i) + i * math.log(p) + (n - i) * math.log1p(-p))
    return min(1.0, total)


def one_sided_binomial_p_value(successes: int, n: int, *, p0: float = TARGET_COVERAGE) -> float:
    """P(X <= successes | n, p0).  Small values argue coverage is below p0."""
    return binomial_cdf(successes, n, p0)


def binomial_power(n: int, true_p: float, *, p0: float = TARGET_COVERAGE, alpha: float = 0.05) -> dict[str, object]:
    """Power of the one-sided test, and the critical value it rejects at."""
    critical = -1
    for k in range(0, n + 1):
        if binomial_cdf(k, n, p0) > alpha:
            break
        critical = k
    power = binomial_cdf(critical, n, true_p) if critical >= 0 else 0.0
    return {"n": n, "true_coverage": true_p, "critical_value": critical, "power": power}


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, order preserved."""
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, (index, value) in enumerate(indexed):
        candidate = min(1.0, (m - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BinResult:
    bin_label: str
    n: int
    covered: int
    coverage: float
    wilson_lower: float
    p_value: float
    per_target_covered: dict[str, int]


def _joint_covered(row: Mapping[str, object], *, kappa: Mapping[str, float],
                   q: Mapping[str, float], normalization: Mapping[str, object]) -> tuple[bool, dict[str, bool]]:
    """Does U_j = mu_j + kappa_j sigma_j + q_j cover Y_j for every target?"""
    y = _normalized_targets(
        {
            "compliance": row["solver"]["compliance_j"],  # type: ignore[index]
            "max_von_mises": row["solver"].get("max_gauss_von_mises_pa", row["solver"].get("max_von_mises_pa")),  # type: ignore[union-attr]
            "max_displacement": row["solver"]["max_displacement_m"],  # type: ignore[index]
        },
        normalization,
    )
    prediction = row["prediction"]
    per_target: dict[str, bool] = {}
    for name in TARGET_NAMES:
        mu = float(prediction["mu"][name])  # type: ignore[index]
        sigma = float(prediction["sigma"][name])  # type: ignore[index]
        upper = mu + float(kappa[name]) * sigma + float(q[name])
        if not math.isfinite(upper):
            raise K6Error("upper predictor is nonfinite")
        per_target[name] = y[name] <= upper
    return all(per_target.values()), per_target


def evaluate_bins(rows: Sequence[Mapping[str, object]], *, kappa: Mapping[str, float],
                  q: Mapping[str, float], normalization: Mapping[str, object]) -> tuple[list[BinResult], BinResult]:
    """Per-bin results, plus the pooled marginal (context only, never adjudicated)."""
    by_bin: dict[str, list[bool]] = {label: [] for label in DEPTH_BINS}
    per_target_by_bin: dict[str, Counter] = {label: Counter() for label in DEPTH_BINS}
    pooled: list[bool] = []
    pooled_targets: Counter = Counter()
    for row in rows:
        covered, per_target = _joint_covered(row, kappa=kappa, q=q, normalization=normalization)
        label = str(row["bin_label"])
        by_bin[label].append(covered)
        pooled.append(covered)
        for name, hit in per_target.items():
            if hit:
                per_target_by_bin[label][name] += 1
                pooled_targets[name] += 1

    results: list[BinResult] = []
    for label in DEPTH_BINS:
        flags = by_bin[label]
        if not flags:
            continue
        results.append(_bin_result(label, flags, per_target_by_bin[label]))
    marginal = _bin_result("POOLED_NOT_ADJUDICATED", pooled, pooled_targets)
    return results, marginal


def _bin_result(label: str, flags: Sequence[bool], per_target: Mapping[str, int]) -> BinResult:
    n = len(flags)
    covered = sum(1 for flag in flags if flag)
    return BinResult(
        bin_label=label,
        n=n,
        covered=covered,
        coverage=covered / n,
        wilson_lower=wilson_lower_bound(covered, n),
        p_value=one_sided_binomial_p_value(covered, n),
        per_target_covered=dict(per_target),
    )


def adjudicate(results: Sequence[BinResult], *, family_count: int,
               interim_power_floor: float = 0.95) -> dict[str, object]:
    """Apply the amended decision rule.  Per bin, Holm-corrected across bins.

    INTERIM status is decided by realized power against the pre-registration's own
    effect size (true coverage 0.93), not by raw family count.  Amendment 02 §3
    fixes this: a run at 2,096 of 2,235 families retains power 0.986 against the
    pre-registered 0.989 and is not meaningfully underpowered, whereas amendment
    01's run at 355 families had power 0.435 and was.
    """
    if not results:
        raise K6Error("no occupied depth bin produced a result")
    adjusted = holm_adjust([result.p_value for result in results])

    per_bin: list[dict[str, object]] = []
    premise_false_bins: list[str] = []
    premise_true_bins: list[str] = []
    for result, adjusted_p in zip(results, adjusted):
        false_branch = result.wilson_lower >= PREMISE_FALSE_WILSON_LOWER
        true_branch = (
            result.coverage < TARGET_COVERAGE
            and result.wilson_lower < PREMISE_TRUE_WILSON_LOWER
            and result.coverage < PREMISE_TRUE_POINT
            and adjusted_p < 0.05
        )
        if false_branch:
            premise_false_bins.append(result.bin_label)
        if true_branch:
            premise_true_bins.append(result.bin_label)
        per_bin.append({
            "bin_label": result.bin_label,
            "n": result.n,
            "covered": result.covered,
            "coverage": result.coverage,
            "wilson_lower_95": result.wilson_lower,
            "p_value_one_sided": result.p_value,
            "p_value_holm_adjusted": adjusted_p,
            "per_target_covered": result.per_target_covered,
            "branch": "PREMISE_FALSE" if false_branch else ("PREMISE_TRUE" if true_branch else "AMBIGUOUS"),
        })

    if premise_true_bins and premise_false_bins:
        verdict = "MIXED"
        rationale = "bins disagree; no single verdict is licensed"
    elif premise_false_bins and len(premise_false_bins) == len(results):
        verdict = "PREMISE_FALSE"
        rationale = "every occupied bin reaches the kill threshold L >= 0.95"
    elif premise_false_bins:
        verdict = "PREMISE_FALSE_PARTIAL"
        rationale = "some but not all occupied bins reach the kill threshold"
    elif premise_true_bins:
        verdict = "PREMISE_TRUE"
        rationale = "at least one bin meets point, interval, and Holm-corrected conditions"
    else:
        verdict = "AMBIGUOUS"
        rationale = "no bin satisfies either branch under the amended rule"

    # Power is evaluated at the smallest occupied bin, the weakest link.
    smallest_n = min(result.n for result in results)
    realized_power = float(binomial_power(smallest_n, 0.93)["power"])  # type: ignore[arg-type]
    interim = realized_power < interim_power_floor
    return {
        "verdict": verdict,
        "rationale": rationale,
        "premise_false_bins": premise_false_bins,
        "premise_true_bins": premise_true_bins,
        "per_bin": per_bin,
        "status": "INTERIM" if interim else "FULL_POPULATION",
        "family_count": family_count,
        "preregistered_family_count": PREREGISTERED_FAMILY_COUNT,
        "smallest_bin_n": smallest_n,
        "realized_power_vs_0.93": realized_power,
        "interim_power_floor": interim_power_floor,
        "interim_note": (
            "Amendment 01 §6: an INTERIM PREMISE_FALSE is a sufficient kill; an INTERIM "
            "PREMISE_TRUE or AMBIGUOUS is not sufficient to advance to controller arms."
            if interim else
            "Amendment 02 §3: realized power at the smallest occupied bin is at or above "
            "the floor, so this run may be used to advance as well as to stop."
        ),
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def _assert_no_confirmation_reachable(root: Path) -> int:
    """Invariant 1.  Refuse to run if any confirmation artifact exists here."""
    breaches = sorted(str(path) for path in Path(root).rglob("trajectory-confirmation-*"))
    if breaches:
        raise K6Error("confirmation-role artifact is present in the evaluated root")
    return len(breaches)


def run_coverage(*, output_root: Path, ensemble_normalization: Path, report_path: Path,
                 preregistration_path: Path, amendment_path: Path,
                 frozen_constants_root: Path | None = None,
                 interim_threshold: int = PREREGISTERED_FAMILY_COUNT) -> dict[str, object]:
    """Evaluate K6 coverage.

    ``frozen_constants_root`` selects where the authoritative kappa and q_base are
    read from, which may differ from the root holding the trajectory records.  Per
    amendment 02 §4 the Mac's frozen record is authoritative even when the
    trajectories were produced elsewhere, because invariant 2 requires those
    constants to be imported by value rather than recomputed per host.
    """
    output_root = Path(output_root)
    _assert_no_confirmation_reachable(output_root)

    observed_prereg = sha256_file(preregistration_path)
    if observed_prereg != PREREGISTRATION_SHA256:
        raise K6Error("frozen pre-registration sha256 mismatch")
    observed_amendment = sha256_file(amendment_path)
    if observed_amendment not in ACCEPTED_AMENDMENTS:
        raise K6Error("amendment sha256 is not an accepted frozen amendment")

    constants_root = Path(frozen_constants_root) if frozen_constants_root is not None else output_root
    _assert_no_confirmation_reachable(constants_root)
    kappa_record = _verified_json(constants_root / "kappa-development-evidence.json",
                                  "G3 kappa evidence", "kappa_evidence_sha256")
    q_base_record = _verified_json(constants_root / "baseline-calibration.json",
                                   "G3 baseline calibration", "baseline_calibration_sha256")
    kappa = {name: float(value) for name, value in kappa_record["kappa"].items()}  # type: ignore[union-attr]
    q_base = {name: float(value) for name, value in q_base_record["q"].items()}  # type: ignore[union-attr]

    # Pre-declared sensitivity check (amendment 02 §4): when the trajectories were
    # produced on a host that recomputed its own constants, evaluate against those
    # too and report whether any branch assignment moves.
    local_q: dict[str, float] | None = None
    local_kappa: dict[str, float] | None = None
    if constants_root != output_root:
        try:
            local_kappa_record = _verified_json(output_root / "kappa-development-evidence.json",
                                               "G3 kappa evidence", "kappa_evidence_sha256")
            local_q_record = _verified_json(output_root / "baseline-calibration.json",
                                            "G3 baseline calibration", "baseline_calibration_sha256")
            local_kappa = {n: float(v) for n, v in local_kappa_record["kappa"].items()}  # type: ignore[union-attr]
            local_q = {n: float(v) for n, v in local_q_record["q"].items()}  # type: ignore[union-attr]
        except (K6Error, G3Error, OSError):
            local_q = local_kappa = None

    normalization = json.loads(Path(ensemble_normalization).read_text(encoding="utf-8"))
    if normalization.get("role") != "fit":
        raise K6Error("normalization stats must be fit-role")

    cases = []
    for path in sorted(output_root.glob("trajectory-development-*.json")):
        case = _verified_json(path, "G3 trajectory case", "trajectory_digest")
        if case.get("role") != "development":
            raise K6Error("non-development case in development coverage set")
        if case.get("intermediate_solver_call_count") != 0:
            raise K6Error("case contains forbidden intermediate solves")
        cases.append(case)
    if not cases:
        raise K6Error("no development trajectory cases found")
    d1_status = _assert_d1_corrected(output_root, cases)

    rows, occupancy, selected = _selected_trajectory_rows(cases)
    results, marginal = evaluate_bins(rows, kappa=kappa, q=q_base, normalization=normalization)
    adjudication = adjudicate(results, family_count=len(cases))

    # Sensitivity: re-adjudicate under the producing host's own recomputed
    # constants.  Any branch disagreement is pre-declared to be the headline.
    sensitivity: dict[str, object] | None = None
    if local_q is not None and local_kappa is not None:
        alt_results, _ = evaluate_bins(rows, kappa=local_kappa, q=local_q, normalization=normalization)
        alt_adjudication = adjudicate(alt_results, family_count=len(cases))
        primary_branches = {row["bin_label"]: row["branch"] for row in adjudication["per_bin"]}  # type: ignore[index]
        alt_branches = {row["bin_label"]: row["branch"] for row in alt_adjudication["per_bin"]}  # type: ignore[index]
        disagreements = sorted(label for label in primary_branches
                               if primary_branches[label] != alt_branches.get(label))
        sensitivity = {
            "alternate_kappa": local_kappa,
            "alternate_q_base": local_q,
            "alternate_verdict": alt_adjudication["verdict"],
            "alternate_per_bin_coverage": {result.bin_label: result.coverage for result in alt_results},
            "branch_disagreements": disagreements,
            "agrees_with_primary": not disagreements and alt_adjudication["verdict"] == adjudication["verdict"],
        }

    power_table = [
        binomial_power(result.n, true_p)
        for result in results
        for true_p in (0.94, 0.93, 0.90)
    ]

    report = {
        "schema_version": SCHEMA_VERSION,
        "label": "K6_COVERAGE",
        "coverage_computed": True,
        "evaluated_root": str(output_root),
        "pre_registration_sha256": observed_prereg,
        "amendment_sha256": observed_amendment,
        "amendment_label": ACCEPTED_AMENDMENTS[observed_amendment],
        "frozen_constants_root": str(constants_root),
        "q_base_authority": ("external frozen record per amendment 02 §4"
                             if constants_root != output_root else "same root"),
        "sensitivity_alternate_constants": sensitivity,
        "alpha": ALPHA,
        "alpha_j": ALPHA_J,
        "J": J,
        "target_coverage": TARGET_COVERAGE,
        "calibration_source": "q_base imported by value from frozen baseline-calibration.json",
        "d1_channel_status": d1_status,
        "kappa": kappa,
        "q_base": q_base,
        "normalization_stats_digest": normalization.get("stats_digest"),
        "development_family_count": len(cases),
        "preregistered_family_count": PREREGISTERED_FAMILY_COUNT,
        "selected_state_count": len(rows),
        "depth_bin_occupancy_counts": occupancy,
        "selected_state_counts_per_bin": selected,
        "pooled_marginal_context_only": {
            "n": marginal.n,
            "covered": marginal.covered,
            "coverage": marginal.coverage,
            "wilson_lower_95": marginal.wilson_lower,
            "note": "clustered within family; reported for context and never adjudicated (amendment 01 §4)",
        },
        "power": power_table,
        "adjudication": adjudication,
        "confirmation_artifacts_present": 0,
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        raise K6Error("coverage report already exists; refusing to overwrite adjudicated evidence")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="K6 coverage evaluation and adjudication")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--frozen-constants-root", type=Path, default=None,
                        help="root holding the authoritative kappa/q_base (default: --output-root)")
    args = parser.parse_args()
    report = run_coverage(output_root=args.output_root, ensemble_normalization=args.normalization,
                          report_path=args.report, preregistration_path=args.preregistration,
                          amendment_path=args.amendment, frozen_constants_root=args.frozen_constants_root)
    print(json.dumps(report["adjudication"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
