"""Value objects for the scenario-driven design-space generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AxisSpec:
    """Specification for one orthogonal axis."""

    name: str
    values: tuple[Any, ...]
    description: str = ""
    provenance: str = ""


@dataclass(frozen=True)
class Constraint:
    """Conditional constraint with typed reason code."""

    name: str
    when: dict[str, tuple[Any, ...]]
    require: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    exclude_combinations: tuple[dict[str, tuple[Any, ...]], ...] = field(default_factory=tuple)
    reason: str = ""

    def matches(self, combo: dict[str, Any]) -> bool:
        """Return True if the combination triggers this constraint."""
        return all(combo.get(axis) in allowed for axis, allowed in self.when.items())

    def require_violations(self, combo: dict[str, Any]) -> list[str]:
        """Return list of axis names that violate a ``require`` rule."""
        violations: list[str] = []
        for axis, allowed in self.require.items():
            if combo.get(axis) not in allowed:
                violations.append(axis)
        return violations

    def excludes(self, combo: dict[str, Any]) -> bool:
        """Return True if combo matches any explicit exclude_combination."""
        for exclusion in self.exclude_combinations:
            if all(combo.get(axis) in allowed for axis, allowed in exclusion.items()):
                return True
        return False


@dataclass(frozen=True)
class DesignPoint:
    """A single normalized design-point candidate."""

    design_point_id: str
    hardware_config: dict[str, Any]
    scenario_ref: str
    workload_ref: str | None
    axis_values: dict[str, Any]


__all__ = ["AxisSpec", "Constraint", "DesignPoint"]
