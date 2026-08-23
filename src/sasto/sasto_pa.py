"""The sole canonical SASTO-PA admission primitive for the G0 bootstrap.

This module is intentionally narrow: an erosion candidate cannot pass a proxy
constraint before the canonical 6-foreground/26-background topology gate passes.
Full optimization, surrogate inference, and FEA verification belong to later gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from .targets import ConstraintResult, TargetRegistry
from .topology import is_simple_point_6_26


@dataclass(frozen=True)
class CandidateDecision:
    accepted: bool
    reason: str
    constraints: Mapping[str, ConstraintResult]


def evaluate_erosion_candidate(
    volume: Sequence[Sequence[Sequence[bool]]],
    point: Tuple[int, int, int],
    predicted_responses: Mapping[str, float],
    targets: TargetRegistry,
) -> CandidateDecision:
    """Apply topology first, then named proxy constraints without positional indices."""
    if not is_simple_point_6_26(volume, point):
        return CandidateDecision(False, "topology-not-simple-6-26", {})
    constraints = targets.evaluate(predicted_responses)
    if not all(result.passed for result in constraints.values()):
        return CandidateDecision(False, "proxy-constraint-failed", constraints)
    return CandidateDecision(True, "accepted", constraints)
