"""Value objects for the scenario-driven design-space generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class AxisSpec:
    """Specification for one orthogonal axis."""

    name: str
    values: Tuple[Any, ...]
    description: str = ""
    provenance: str = ""


@dataclass(frozen=True)
class Constraint:
    """Conditional constraint with typed reason code."""

    name: str
    when: Dict[str, Tuple[Any, ...]]
    require: Dict[str, Tuple[Any, ...]] = field(default_factory=dict)
    exclude_combinations: Tuple[Dict[str, Tuple[Any, ...]], ...] = field(default_factory=tuple)
    reason: str = ""

    def matches(self, combo: Dict[str, Any]) -> bool:
        """Return True if the combination triggers this constraint."""
        for axis, allowed in self.when.items():
            if combo.get(axis) not in allowed:
                return False
        return True

    def require_violations(self, combo: Dict[str, Any]) -> list[str]:
        """Return list of axis names that violate a ``require`` rule."""
        violations: list[str] = []
        for axis, allowed in self.require.items():
            if combo.get(axis) not in allowed:
                violations.append(axis)
        return violations

    def excludes(self, combo: Dict[str, Any]) -> bool:
        """Return True if combo matches any explicit exclude_combination."""
        for exclusion in self.exclude_combinations:
            if all(combo.get(axis) in allowed for axis, allowed in exclusion.items()):
                return True
        return False


@dataclass(frozen=True)
class DesignPoint:
    """A single normalized design-point candidate."""

    design_point_id: str
    hardware_config: Dict[str, Any]
    scenario_ref: str
    workload_ref: Optional[str]
    axis_values: Dict[str, Any]


__all__ = ["AxisSpec", "Constraint", "DesignPoint"]
