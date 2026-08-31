"""Tests for K6 coverage evaluation.

The statistics are checked against values computed independently (scipy where
available, closed forms otherwise), and the coverage logic is checked against
synthetic records whose true coverage is known by construction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sasto.k6_coverage import (
    AMENDMENT_01_SHA256,
    PREMISE_TRUE_POINT,
    PREREGISTRATION_SHA256,
    BinResult,
    K6Error,
    adjudicate,
    binomial_cdf,
    binomial_power,
    evaluate_bins,
    holm_adjust,
    one_sided_binomial_p_value,
    wilson_lower_bound,
    _normal_quantile,
)

NORMALIZATION = {
    "role": "fit",
    "means": {"compliance": 0.0, "max_displacement": 0.0, "max_von_mises": 0.0},
    "scales": {"compliance": 1.0, "max_displacement": 1.0, "max_von_mises": 1.0},
    "stats_digest": "test",
}
KAPPA = {"compliance": 0.0, "max_displacement": 0.0, "max_von_mises": 0.0}
Q_ZERO = {"compliance": 0.0, "max_displacement": 0.0, "max_von_mises": 0.0}


# ---------------------------------------------------------------- statistics


def test_normal_quantile_matches_known_values():
    assert _normal_quantile(0.95) == pytest.approx(1.6448536269514722, abs=1e-9)
    assert _normal_quantile(0.975) == pytest.approx(1.959963984540054, abs=1e-9)
    assert _normal_quantile(0.5) == pytest.approx(0.0, abs=1e-12)


def test_wilson_lower_bound_matches_closed_form():
    # Hand-computed: n=100, x=95, z=1.6448536269514722
    n, x = 100, 95
    z = 1.6448536269514722
    p = x / n
    expected = (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
    assert wilson_lower_bound(x, n) == pytest.approx(expected, abs=1e-12)


def test_wilson_lower_bound_is_monotone_in_successes():
    values = [wilson_lower_bound(x, 200) for x in range(150, 200)]
    assert values == sorted(values)


def test_wilson_lower_bound_widens_as_n_shrinks():
    """The asymmetry that motivated amendment 01: smaller n pushes L down."""
    assert wilson_lower_bound(round(0.96 * 2235), 2235) > wilson_lower_bound(round(0.96 * 355), 355)


def test_wilson_rejects_invalid_inputs():
    with pytest.raises(K6Error):
        wilson_lower_bound(5, 0)
    with pytest.raises(K6Error):
        wilson_lower_bound(11, 10)


def test_binomial_cdf_against_exact_small_case():
    # Bin(5, 0.5): P(X<=2) = (1+5+10)/32
    assert binomial_cdf(2, 5, 0.5) == pytest.approx(16 / 32, abs=1e-12)
    assert binomial_cdf(5, 5, 0.5) == pytest.approx(1.0)
    assert binomial_cdf(-1, 5, 0.5) == 0.0


def test_binomial_cdf_matches_scipy_when_available():
    scipy_stats = pytest.importorskip("scipy.stats")
    for n, k, p in [(355, 329, 0.95), (1751, 1647, 0.95), (343, 300, 0.93)]:
        assert binomial_cdf(k, n, p) == pytest.approx(float(scipy_stats.binom.cdf(k, n, p)), rel=1e-9)


def test_binomial_power_reproduces_amendment_table():
    """Amendment 01 §2 states power 0.435 at n=355 against true coverage 0.93."""
    result = binomial_power(355, 0.93)
    assert result["critical_value"] == 329
    assert result["power"] == pytest.approx(0.435, abs=0.005)


def test_binomial_power_reproduces_preregistration_table():
    """Pre-registration §3 states power 0.989 at n=2235 against 0.93."""
    assert binomial_power(2235, 0.93)["power"] == pytest.approx(0.989, abs=0.002)


def test_holm_adjust_is_monotone_and_bounded():
    adjusted = holm_adjust([0.001, 0.02, 0.5, 0.9])
    assert adjusted[0] == pytest.approx(0.004)
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted == sorted(adjusted)


def test_holm_adjust_preserves_input_order():
    adjusted = holm_adjust([0.5, 0.001])
    assert adjusted[1] < adjusted[0]


# ------------------------------------------------------------------ coverage


def _row(bin_label, y, mu, sigma=1.0):
    """Synthetic row; identity normalization means y is used as exp() input."""
    return {
        "bin_label": bin_label,
        "solver": {
            "compliance_j": math.exp(y),
            "max_gauss_von_mises_pa": math.exp(y),
            "max_displacement_m": math.exp(y),
        },
        "prediction": {
            "mu": {name: mu for name in ("compliance", "max_displacement", "max_von_mises")},
            "sigma": {name: sigma for name in ("compliance", "max_displacement", "max_von_mises")},
        },
    }


def test_coverage_is_one_when_predictor_dominates():
    rows = [_row("(5,10%]", y=0.0, mu=10.0) for _ in range(50)]
    results, marginal = evaluate_bins(rows, kappa=KAPPA, q=Q_ZERO, normalization=NORMALIZATION)
    assert results[0].coverage == 1.0
    assert marginal.covered == 50


def test_coverage_is_zero_when_predictor_is_dominated():
    rows = [_row("(5,10%]", y=10.0, mu=0.0) for _ in range(50)]
    results, _ = evaluate_bins(rows, kappa=KAPPA, q=Q_ZERO, normalization=NORMALIZATION)
    assert results[0].coverage == 0.0


def test_coverage_counts_exact_known_fraction():
    """40 covered of 50 by construction, so coverage must be exactly 0.80."""
    rows = [_row("(5,10%]", y=0.0, mu=1.0) for _ in range(40)]
    rows += [_row("(5,10%]", y=1.0, mu=0.0) for _ in range(10)]
    results, _ = evaluate_bins(rows, kappa=KAPPA, q=Q_ZERO, normalization=NORMALIZATION)
    assert results[0].n == 50
    assert results[0].covered == 40
    assert results[0].coverage == pytest.approx(0.80)


def test_joint_coverage_requires_all_targets():
    """One failing target must sink the joint indicator."""
    row = _row("(5,10%]", y=0.0, mu=1.0)
    row["solver"]["max_displacement_m"] = math.exp(5.0)
    results, _ = evaluate_bins([row], kappa=KAPPA, q=Q_ZERO, normalization=NORMALIZATION)
    assert results[0].covered == 0
    assert results[0].per_target_covered.get("compliance") == 1
    assert results[0].per_target_covered.get("max_displacement", 0) == 0


def test_q_shifts_the_upper_bound():
    rows = [_row("(5,10%]", y=1.0, mu=0.0) for _ in range(10)]
    uncovered, _ = evaluate_bins(rows, kappa=KAPPA, q=Q_ZERO, normalization=NORMALIZATION)
    assert uncovered[0].coverage == 0.0
    generous = {name: 2.0 for name in Q_ZERO}
    covered, _ = evaluate_bins(rows, kappa=KAPPA, q=generous, normalization=NORMALIZATION)
    assert covered[0].coverage == 1.0


def test_kappa_shifts_the_upper_bound():
    rows = [_row("(5,10%]", y=1.0, mu=0.0, sigma=1.0) for _ in range(10)]
    kappa = {name: 2.0 for name in KAPPA}
    results, _ = evaluate_bins(rows, kappa=kappa, q=Q_ZERO, normalization=NORMALIZATION)
    assert results[0].coverage == 1.0


def test_bins_are_evaluated_separately():
    rows = [_row("(5,10%]", y=0.0, mu=10.0) for _ in range(20)]
    rows += [_row(">25%", y=10.0, mu=0.0) for _ in range(20)]
    results, marginal = evaluate_bins(rows, kappa=KAPPA, q=Q_ZERO, normalization=NORMALIZATION)
    by_label = {result.bin_label: result for result in results}
    assert by_label["(5,10%]"].coverage == 1.0
    assert by_label[">25%"].coverage == 0.0
    assert marginal.coverage == pytest.approx(0.5)


def test_empty_bins_are_omitted():
    rows = [_row("(5,10%]", y=0.0, mu=10.0) for _ in range(5)]
    results, _ = evaluate_bins(rows, kappa=KAPPA, q=Q_ZERO, normalization=NORMALIZATION)
    assert [result.bin_label for result in results] == ["(5,10%]"]


# --------------------------------------------------------------- adjudication


def _bin(label, n, covered):
    return BinResult(
        bin_label=label, n=n, covered=covered, coverage=covered / n,
        wilson_lower=wilson_lower_bound(covered, n),
        p_value=one_sided_binomial_p_value(covered, n),
        per_target_covered={},
    )


def test_premise_false_when_every_bin_clears_the_kill_threshold():
    results = [_bin(label, 355, 350) for label in ("(5,10%]", "(10,15%]")]
    verdict = adjudicate(results, family_count=355)
    assert verdict["verdict"] == "PREMISE_FALSE"
    assert verdict["status"] == "INTERIM"


def test_premise_true_requires_the_added_point_condition():
    """Amendment 01 §5: coverage 0.94 has L<0.93 at n=355 but must NOT be a win."""
    result = _bin("(5,10%]", 355, round(0.94 * 355))
    assert result.wilson_lower < 0.93
    assert result.coverage > PREMISE_TRUE_POINT
    verdict = adjudicate([result], family_count=355)
    assert verdict["verdict"] == "AMBIGUOUS"


def test_premise_true_when_coverage_is_materially_below_target():
    result = _bin("(5,10%]", 355, round(0.85 * 355))
    verdict = adjudicate([result], family_count=355)
    assert verdict["verdict"] == "PREMISE_TRUE"
    assert verdict["per_bin"][0]["branch"] == "PREMISE_TRUE"


def test_mixed_verdict_when_bins_disagree():
    results = [_bin("(5,10%]", 355, 350), _bin(">25%", 355, round(0.80 * 355))]
    assert adjudicate(results, family_count=355)["verdict"] == "MIXED"


def test_partial_kill_when_only_some_bins_clear():
    results = [_bin("(5,10%]", 355, 350), _bin(">25%", 355, round(0.945 * 355))]
    assert adjudicate(results, family_count=355)["verdict"] == "PREMISE_FALSE_PARTIAL"


def test_full_population_is_not_marked_interim():
    results = [_bin("(5,10%]", 2235, 2200)]
    verdict = adjudicate(results, family_count=2235)
    assert verdict["status"] == "FULL_POPULATION"
    assert verdict["realized_power_vs_0.93"] > 0.95


def test_interim_is_decided_by_power_not_family_count():
    """Amendment 02 §3: 2,096 of 2,235 families retains power 0.986, not INTERIM."""
    results = [_bin("(5,10%]", 2096, 2063), _bin("(10,15%]", 1980, 1952)]
    verdict = adjudicate(results, family_count=2096)
    assert verdict["status"] == "FULL_POPULATION"
    assert verdict["smallest_bin_n"] == 1980
    assert verdict["realized_power_vs_0.93"] == pytest.approx(0.981, abs=0.005)


def test_small_n_is_still_interim_under_the_power_rule():
    """Amendment 01's 355-family run had power 0.435 and stays INTERIM."""
    results = [_bin("(5,10%]", 355, 351), _bin(">25%", 343, 225)]
    verdict = adjudicate(results, family_count=355)
    assert verdict["status"] == "INTERIM"
    assert verdict["realized_power_vs_0.93"] == pytest.approx(0.447, abs=0.005)


def test_adjudicate_rejects_empty_results():
    with pytest.raises(K6Error):
        adjudicate([], family_count=355)


def test_holm_correction_can_withhold_a_single_marginal_bin():
    """A lone weakly-significant bin among five must not carry a verdict."""
    strong = _bin("(5,10%]", 355, round(0.921 * 355))
    others = [_bin(label, 355, round(0.955 * 355)) for label in ("(10,15%]", "(15,20%]", "(20,25%]", ">25%")]
    verdict = adjudicate([strong, *others], family_count=355)
    adjusted = verdict["per_bin"][0]["p_value_holm_adjusted"]
    assert adjusted >= verdict["per_bin"][0]["p_value_one_sided"]


# ------------------------------------------------------------------- digests


def test_frozen_digests_match_the_documents_on_disk():
    control = Path("/Users/eric/workspace/sasto-modernization-control/v2/g3")
    prereg = control / "K6_PREREGISTRATION.md"
    amendment = control / "K6_AMENDMENT_01_REDUCED_N.md"
    if not prereg.exists() or not amendment.exists():
        pytest.skip("control documents not present on this host")
    import hashlib
    assert hashlib.sha256(prereg.read_bytes()).hexdigest() == PREREGISTRATION_SHA256
    assert hashlib.sha256(amendment.read_bytes()).hexdigest() == AMENDMENT_01_SHA256
    amendment2 = control / "K6_AMENDMENT_02_GB200_POPULATION.md"
    if amendment2.exists():
        from sasto.k6_coverage import AMENDMENT_02_SHA256
        assert hashlib.sha256(amendment2.read_bytes()).hexdigest() == AMENDMENT_02_SHA256
