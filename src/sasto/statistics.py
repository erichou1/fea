"""Exact binomial helpers; deliberately separate from conformal coverage."""

from __future__ import annotations

import math


def one_sided_zero_failure_upper_bound(n: int, *, alpha: float = 0.05) -> float:
    """Exact one-sided Clopper--Pearson upper bound for observing zero failures.

    This estimates a binomial failure-rate upper bound only under the declared
    Bernoulli/exchangeability assumptions.  It is not a conformal coverage result.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return 1.0 - alpha ** (1.0 / n)


def minimum_zero_failure_sample_size(maximum_upper_bound: float, *, alpha: float = 0.05) -> int:
    """Smallest `n` whose 0/n one-sided upper bound is at most the requested rate.

    This is planning arithmetic for binomial evidence, not conformal calibration.
    """
    if not 0.0 < maximum_upper_bound < 1.0:
        raise ValueError("maximum_upper_bound must be strictly between 0 and 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return int(math.ceil(math.log(alpha) / math.log(1.0 - maximum_upper_bound)))


def format_zero_failure_evidence(n: int, *, alpha: float = 0.05) -> dict:
    """A clearly labelled reporting record for an observed 0/n event."""
    return {
        "observation": "0/{:d}".format(n),
        "method": "one-sided Clopper-Pearson zero-failure upper bound",
        "alpha": alpha,
        "upper_bound": one_sided_zero_failure_upper_bound(n, alpha=alpha),
        "not_conformal_coverage": True,
    }
