"""Named physical target contracts; positional target indices are forbidden."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple


@dataclass(frozen=True)
class TargetSpec:
    """A named scalar simulator target and its frozen proxy constraint."""

    name: str
    unit: str
    direction: str
    threshold: float
    normalization: str = "absolute"
    base_target: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.unit:
            raise ValueError("target name and unit must be non-empty")
        if self.direction not in {"upper", "lower"}:
            raise ValueError("direction must be 'upper' or 'lower'")
        if self.normalization == "absolute":
            if self.unit == "1":
                raise ValueError("absolute targets cannot use unit '1'")
            if self.base_target is not None:
                raise ValueError("absolute targets cannot declare base_target")
        elif self.normalization == "baseline_ratio":
            if self.unit != "1":
                raise ValueError("baseline ratios must use unit '1'")
            if not self.name.endswith("_ratio"):
                raise ValueError("baseline ratio target names must end in '_ratio'")
            if not isinstance(self.base_target, str) or not self.base_target:
                raise ValueError("baseline ratios require a non-empty base_target")
        else:
            raise ValueError("normalization must be 'absolute' or 'baseline_ratio'")

    def passes(self, value: float) -> bool:
        return value <= self.threshold if self.direction == "upper" else value >= self.threshold

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "direction": self.direction,
            "threshold": self.threshold,
            "normalization": self.normalization,
            "base_target": self.base_target,
        }


@dataclass(frozen=True)
class ConstraintResult:
    name: str
    value: float
    threshold: float
    direction: str
    passed: bool


class TargetRegistry:
    """Immutable lookup by target name, never by tensor/list position."""

    def __init__(self, targets: Iterable[TargetSpec]) -> None:
        values = tuple(targets)
        if not values:
            raise ValueError("target registry must contain at least one target")
        names = [target.name for target in values]
        if len(names) != len(set(names)):
            raise ValueError("target names must be unique")
        known_names = set(names)
        for target in values:
            if target.normalization == "baseline_ratio" and target.base_target not in known_names:
                raise ValueError("baseline ratio base_target must name a registry target")
        self._targets = values
        self._by_name = {target.name: target for target in values}

    def __iter__(self):
        return iter(self._targets)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(target.name for target in self._targets)

    def target(self, name: str) -> TargetSpec:
        try:
            return self._by_name[name]
        except KeyError as error:
            raise KeyError("unknown named target: {!r}".format(name)) from error

    def evaluate(self, responses: Mapping[str, float]) -> Mapping[str, ConstraintResult]:
        response_names = set(responses)
        target_names = set(self._by_name)
        if response_names != target_names:
            raise ValueError(
                "responses must match named registry exactly; missing={}, extra={}".format(
                    sorted(target_names - response_names), sorted(response_names - target_names)
                )
            )
        return {
            target.name: ConstraintResult(
                name=target.name,
                value=float(responses[target.name]),
                threshold=target.threshold,
                direction=target.direction,
                passed=target.passes(float(responses[target.name])),
            )
            for target in self._targets
        }

    def as_list(self) -> list:
        return [target.as_dict() for target in self._targets]
