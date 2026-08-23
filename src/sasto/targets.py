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

    def __post_init__(self) -> None:
        if not self.name or not self.unit:
            raise ValueError("target name and unit must be non-empty")
        if self.direction not in {"upper", "lower"}:
            raise ValueError("direction must be 'upper' or 'lower'")

    def passes(self, value: float) -> bool:
        return value <= self.threshold if self.direction == "upper" else value >= self.threshold

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "direction": self.direction,
            "threshold": self.threshold,
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
