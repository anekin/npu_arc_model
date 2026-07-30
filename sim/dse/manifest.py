"""Coverage manifest for scenario-driven design-space enumeration.

Tracks per-axis requested/generated/evaluated/successful/pruned/failed/missing
values and counts.  Conditional exclusions carry typed reason codes.

Invariants enforced:
* generated_count == evaluated_count + pruned_count
* evaluated_count == successful_count + filtered_count + failed_count
* Every requested axis value is either generated or has a structured exclusion.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from contracts.errors import CoverageError


@dataclass(frozen=True)
class ExclusionRecord:
    """Structured record explaining why a requested value was not generated."""

    axis: str
    value: Any
    reason: str
    constraint_name: str = ""


@dataclass
class AxisCoverage:
    """Per-axis coverage bookkeeping."""

    axis: str
    requested: set[Any] = field(default_factory=set)
    generated: set[Any] = field(default_factory=set)
    evaluated: set[Any] = field(default_factory=set)
    successful: set[Any] = field(default_factory=set)
    pruned: set[Any] = field(default_factory=set)
    failed: set[Any] = field(default_factory=set)
    filtered: set[Any] = field(default_factory=set)
    exclusions: list[ExclusionRecord] = field(default_factory=list)

    @property
    def excluded_values(self) -> set[Any]:
        """Values with at least one structured exclusion carrying a reason."""
        return {e.value for e in self.exclusions if e.reason}

    @property
    def missing(self) -> list[Any]:
        """Requested values that are neither generated nor excluded."""
        return sorted(self.requested - self.generated - self.excluded_values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "requested": sorted(self.requested),
            "generated": sorted(self.generated),
            "evaluated": sorted(self.evaluated),
            "successful": sorted(self.successful),
            "pruned": sorted(self.pruned),
            "failed": sorted(self.failed),
            "filtered": sorted(self.filtered),
            "excluded": sorted(self.excluded_values),
            "missing": self.missing,
            "exclusion_records": [
                {
                    "axis": e.axis,
                    "value": e.value,
                    "reason": e.reason,
                    "constraint_name": e.constraint_name,
                }
                for e in self.exclusions
            ],
        }


class CoverageManifest:
    """Coverage manifest across all design-space axes."""

    def __init__(
        self,
        axes: dict[str, AxisSpec],
        points: Iterable[DesignPoint],
        exclusions: Iterable[ExclusionRecord] | None = None,
    ) -> None:
        """Build a manifest from axis specs and generated design points.

        Args:
            axes: Ordered mapping of axis name to spec (must expose ``values``).
            points: Generated design points.
            exclusions: Optional structured exclusions from constraint filtering.
        """
        self.axes = axes
        self.points: list[DesignPoint] = list(points)
        self.exclusions: list[ExclusionRecord] = list(exclusions or [])
        self._coverage: dict[str, AxisCoverage] = {}
        self._evaluated_ids: set[str] = set()
        self._pruned_ids: set[str] = set()
        self._successful_ids: set[str] = set()
        self._filtered_ids: set[str] = set()
        self._failed_ids: set[str] = set()
        self._build()

    def _build(self) -> None:
        """Initialize per-axis coverage from requested/generated/exclusion data."""
        self._coverage = {}
        for name, spec in self.axes.items():
            self._coverage[name] = AxisCoverage(
                axis=name,
                requested=set(spec.values),
            )

        for point in self.points:
            for axis, value in point.axis_values.items():
                if axis in self._coverage:
                    self._coverage[axis].generated.add(value)

        for exclusion in self.exclusions:
            if exclusion.axis in self._coverage:
                self._coverage[exclusion.axis].exclusions.append(exclusion)

    @property
    def generated_count(self) -> int:
        return len(self.points)

    @property
    def evaluated_count(self) -> int:
        return sum(1 for p in self.points if p.design_point_id in self._evaluated_ids)

    @property
    def pruned_count(self) -> int:
        return sum(1 for p in self.points if p.design_point_id in self._pruned_ids)

    @property
    def successful_count(self) -> int:
        return sum(1 for p in self.points if p.design_point_id in self._successful_ids)

    @property
    def filtered_count(self) -> int:
        return sum(1 for p in self.points if p.design_point_id in self._filtered_ids)

    @property
    def failed_count(self) -> int:
        return sum(1 for p in self.points if p.design_point_id in self._failed_ids)

    @property
    def axis_coverage(self) -> dict[str, AxisCoverage]:
        return dict(self._coverage)

    @property
    def missing_axes(self) -> dict[str, list[Any]]:
        """Return axes with missing values."""
        return {name: cov.missing for name, cov in self._coverage.items() if cov.missing}

    @property
    def duplicate_ids(self) -> list[str]:
        """Return design-point IDs that appear more than once."""
        seen: set[str] = set()
        duplicates: set[str] = set()
        for point in self.points:
            if point.design_point_id in seen:
                duplicates.add(point.design_point_id)
            seen.add(point.design_point_id)
        return sorted(duplicates)

    def record_evaluated(self, point: DesignPoint) -> None:
        """Mark a generated point as having entered evaluation."""
        self._evaluated_ids.add(point.design_point_id)
        for axis, value in point.axis_values.items():
            if axis in self._coverage:
                self._coverage[axis].evaluated.add(value)

    def record_success(self, point: DesignPoint) -> None:
        """Mark a generated point as successful."""
        self.record_evaluated(point)
        self._successful_ids.add(point.design_point_id)
        for axis, value in point.axis_values.items():
            if axis in self._coverage:
                self._coverage[axis].successful.add(value)

    def _clear_success_state(self, point: DesignPoint) -> None:
        """Remove point from successful/evaluated sets when reclassified."""
        self._evaluated_ids.discard(point.design_point_id)
        self._successful_ids.discard(point.design_point_id)
        for axis, value in point.axis_values.items():
            cov = self._coverage.get(axis)
            if cov is not None:
                cov.evaluated.discard(value)
                cov.successful.discard(value)

    def record_pruned(self, point: DesignPoint, reason: str) -> None:
        """Mark a generated point as pre-evaluation pruned."""
        self._clear_success_state(point)
        self._pruned_ids.add(point.design_point_id)
        for axis, value in point.axis_values.items():
            if axis in self._coverage:
                self._coverage[axis].pruned.add(value)
        self._add_point_exclusion(point, reason)

    def record_failed(self, point: DesignPoint, reason: str) -> None:
        """Mark an evaluated point as failed."""
        self._successful_ids.discard(point.design_point_id)
        self.record_evaluated(point)
        self._failed_ids.add(point.design_point_id)
        for axis, value in point.axis_values.items():
            if axis in self._coverage:
                self._coverage[axis].failed.add(value)

    def record_filtered(self, point: DesignPoint, reason: str) -> None:
        """Mark an evaluated point as post-evaluation filtered."""
        self._successful_ids.discard(point.design_point_id)
        self.record_evaluated(point)
        self._filtered_ids.add(point.design_point_id)
        for axis, value in point.axis_values.items():
            if axis in self._coverage:
                self._coverage[axis].filtered.add(value)

    def _add_point_exclusion(self, point: DesignPoint, reason: str) -> None:
        """Record an exclusion reason for every axis value of a pruned point."""
        for axis, value in point.axis_values.items():
            if axis in self._coverage:
                self._coverage[axis].exclusions.append(ExclusionRecord(axis=axis, value=value, reason=reason))

    def validate(self) -> list[str]:
        """Check invariants and coverage completeness.

        Returns a list of error messages (empty if valid).
        """
        errors: list[str] = []

        # Count invariants
        if self.generated_count != self.evaluated_count + self.pruned_count:
            errors.append(
                f"invariant violated: generated({self.generated_count}) != "
                f"evaluated({self.evaluated_count}) + pruned({self.pruned_count})"
            )
        if self.evaluated_count != self.successful_count + self.filtered_count + self.failed_count:
            errors.append(
                f"invariant violated: evaluated({self.evaluated_count}) != "
                f"successful({self.successful_count}) + filtered({self.filtered_count}) + "
                f"failed({self.failed_count})"
            )

        if self.duplicate_ids:
            errors.append(f"duplicate design_point_ids detected: {self.duplicate_ids}")

        # Coverage completeness: no silently missing values
        for name, cov in self._coverage.items():
            if cov.missing:
                errors.append(f"axis {name!r} has missing values without exclusion reason: {cov.missing}")

        return errors

    def raise_if_invalid(self) -> None:
        """Raise CoverageError if the manifest is invalid."""
        errors = self.validate()
        if errors:
            raise CoverageError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": {
                "generated": self.generated_count,
                "evaluated": self.evaluated_count,
                "pruned": self.pruned_count,
                "successful": self.successful_count,
                "filtered": self.filtered_count,
                "failed": self.failed_count,
            },
            "axis_coverage": {name: cov.to_dict() for name, cov in self._coverage.items()},
            "missing_axes": self.missing_axes,
        }


# Forward-reference imports for type checking only
from dse.models import AxisSpec, DesignPoint  # noqa: E402

__all__ = [
    "AxisCoverage",
    "CoverageManifest",
    "ExclusionRecord",
]
