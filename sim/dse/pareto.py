"""Multi-objective Pareto frontier with hard gates for scenario-driven DSE.

A design point enters the Pareto frontier only after passing all hard gates.
Infeasible points keep structured gate-failure reasons but are excluded from
recommendation.

Objectives and their directions are declared per scenario via
``objectives_from_scenario``; the default set is:

* maximize admitted completed throughput
* minimize P99 latency
* minimize deadline miss count
* minimize die area
* minimize power
* minimize energy

Tie-breaking is deterministic by ``design_point_id``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from contracts.errors import ConfigError
from contracts.result import DesignPointResult, EngineMetrics, RunStatus, RunTrustLevel
from scenarios.schema import Scenario


class ObjectiveDirection(str, Enum):
    """Direction of optimization for an objective."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True)
class Objective:
    """One Pareto objective mapped to an ``EngineMetrics`` field."""

    name: str
    metric_field: str
    direction: ObjectiveDirection


class HardGateCode(str, Enum):
    """Typed reason codes for hard-gate failures."""

    COMPLETE = "complete"
    NO_CPU_FALLBACK = "no_cpu_fallback"
    CAPACITY_FIT = "capacity_fit"
    QUALITY_GATE = "quality_gate"
    P99_DEADLINE = "p99_deadline"
    POWER_THERMAL = "power_thermal"
    TERMINAL_COMPLETION = "terminal_completion"
    AUTHORITATIVE = "authoritative"


@dataclass(frozen=True)
class GateResult:
    """Result of one hard-gate check."""

    code: HardGateCode
    passed: bool
    reason: str = ""


@dataclass
class ParetoPoint:
    """A design-point result decorated with gate and objective information."""

    result: DesignPointResult
    gate_results: tuple[GateResult, ...] = field(default_factory=tuple)
    objective_values: dict[str, float] = field(default_factory=dict)

    @property
    def passed_all_gates(self) -> bool:
        return all(g.passed for g in self.gate_results)

    @property
    def gate_failure_reasons(self) -> list[str]:
        return [f"{g.code.value}: {g.reason}" for g in self.gate_results if not g.passed and g.reason]


# Default objective set used when a scenario does not declare overrides.
DEFAULT_OBJECTIVES: tuple[Objective, ...] = (
    Objective("completed_throughput_hz", "completed_throughput_hz", ObjectiveDirection.MAXIMIZE),
    Objective("latency_p99_ms", "p99_latency_s", ObjectiveDirection.MINIMIZE),
    Objective("deadline_miss_count", "deadline_miss_count", ObjectiveDirection.MINIMIZE),
    Objective("area_mm2", "area_mm2", ObjectiveDirection.MINIMIZE),
    Objective("power_w", "power_w", ObjectiveDirection.MINIMIZE),
    Objective("energy_joules", "energy_joules", ObjectiveDirection.MINIMIZE),
)


def _metric_value(metrics: EngineMetrics | None, field: str) -> float | None:
    """Return a numeric metric value or None if missing."""
    if metrics is None:
        return None
    # Try direct attribute first, then fallback names for latency fields stored in seconds.
    if hasattr(metrics, field):
        return getattr(metrics, field)
    # Map ms aliases to second fields if caller uses legacy names.
    alias_map = {
        "latency_p99_ms": "p99_latency_s",
        "avg_latency_ms": "avg_latency_s",
        "p50_latency_ms": "p50_latency_s",
        "max_latency_ms": "max_latency_s",
    }
    if field in alias_map:
        return getattr(metrics, alias_map[field], None)
    return None


def objectives_from_scenario(scenario: Scenario) -> tuple[Objective, ...]:
    """Return scenario-declared objectives, or the default set.

    Scenario metadata may contain an ``objectives`` list:

    ::

        metadata:
          objectives:
            - {name: "throughput", metric_field: "completed_throughput_hz", direction: maximize}
    """
    raw = scenario.metadata.get("objectives") if scenario.metadata else None
    if not raw:
        return DEFAULT_OBJECTIVES
    objectives: list[Objective] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ConfigError(
                f"objective entry must be a dict, got {type(item).__name__}", field_path="metadata.objectives"
            )
        name = item.get("name") or item.get("metric_field")
        metric_field = item.get("metric_field", name)
        direction = ObjectiveDirection(item.get("direction", "maximize"))
        if not name or not metric_field:
            raise ConfigError("objective must have 'name' and 'metric_field'", field_path="metadata.objectives")
        objectives.append(Objective(name=str(name), metric_field=str(metric_field), direction=direction))
    if not objectives:
        return DEFAULT_OBJECTIVES
    return tuple(objectives)


def _scenario_deadline_ms(scenario: Scenario) -> float | None:
    """Extract a representative deadline from the scenario (ms)."""
    if not scenario.classes:
        return None
    deadlines = [c.relative_deadline_ms for c in scenario.classes if c.relative_deadline_ms is not None]
    if deadlines:
        return min(deadlines)
    return None


def _scenario_thermal_limit_w(scenario: Scenario) -> float | None:
    """Extract thermal power limit from scenario metadata (W)."""
    limit = scenario.metadata.get("thermal_limit_w") if scenario.metadata else None
    if limit is None:
        # Legacy scenarios.yaml stores constraints.area_mm2_max but not thermal.
        # Use a conservative default so the gate is meaningful without breaking old runs.
        return 150.0
    return float(limit)


class MultiObjectivePareto:
    """Compute Pareto frontier with hard gates and configurable objectives."""

    def __init__(
        self,
        objectives: Sequence[Objective] | None = None,
        *,
        deadline_ms: float | None = None,
        thermal_limit_w: float | None = None,
        quality_gate_required: bool = False,
    ) -> None:
        self.objectives = tuple(objectives) if objectives else DEFAULT_OBJECTIVES
        self.deadline_ms = deadline_ms
        self.thermal_limit_w = thermal_limit_w
        self.quality_gate_required = quality_gate_required

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> MultiObjectivePareto:
        """Build a Pareto filter from scenario metadata."""
        return cls(
            objectives=objectives_from_scenario(scenario),
            deadline_ms=_scenario_deadline_ms(scenario),
            thermal_limit_w=_scenario_thermal_limit_w(scenario),
            quality_gate_required=bool(scenario.metadata.get("quality_gate_required")) if scenario.metadata else False,
        )

    def evaluate_gates(self, result: DesignPointResult) -> tuple[GateResult, ...]:
        """Apply all hard gates to a single design-point result."""
        gates: list[GateResult] = []
        metrics = result.metrics

        gates.append(
            GateResult(
                code=HardGateCode.COMPLETE,
                passed=result.status == RunStatus.complete,
                reason="" if result.status == RunStatus.complete else f"status={result.status.value}",
            )
        )

        engine = (result.engine_type or "").lower()
        gates.append(
            GateResult(
                code=HardGateCode.NO_CPU_FALLBACK,
                passed=engine != "cpu",
                reason="" if engine != "cpu" else "engine_type=cpu",
            )
        )

        fits = True
        reason = ""
        if metrics is not None:
            footprint = metrics.memory_footprint_gib
            if footprint is not None and not (0 <= footprint < float("inf")):
                fits = False
                reason = f"invalid footprint={footprint}"
        gates.append(
            GateResult(
                code=HardGateCode.CAPACITY_FIT,
                passed=fits,
                reason=reason,
            )
        )

        quality_ok = not self.quality_gate_required
        gates.append(
            GateResult(
                code=HardGateCode.QUALITY_GATE,
                passed=quality_ok,
                reason="" if quality_ok else "quality_gate_required=true not resolved",
            )
        )

        power_ok = True
        power_reason = ""
        if self.thermal_limit_w is not None and metrics is not None:
            power = metrics.power_w
            if power is not None and power > self.thermal_limit_w:
                power_ok = False
                power_reason = f"power={power:.2f}W > thermal_limit={self.thermal_limit_w}W"
        gates.append(
            GateResult(
                code=HardGateCode.POWER_THERMAL,
                passed=power_ok,
                reason=power_reason,
            )
        )

        terminal_ok = True
        terminal_reason = ""
        if metrics is not None:
            if metrics.deadline_miss_count is not None and metrics.deadline_miss_count > 0:
                terminal_ok = False
                terminal_reason = f"deadline_miss_count={metrics.deadline_miss_count}"
            if metrics.drop_count is not None and metrics.drop_count > 0:
                terminal_ok = False
                terminal_reason = (terminal_reason + f" drop_count={metrics.drop_count}").strip()
        gates.append(
            GateResult(
                code=HardGateCode.TERMINAL_COMPLETION,
                passed=terminal_ok,
                reason=terminal_reason,
            )
        )

        auth_ok = result.trust_level == RunTrustLevel.authoritative
        gates.append(
            GateResult(
                code=HardGateCode.AUTHORITATIVE,
                passed=auth_ok,
                reason="" if auth_ok else f"trust_level={result.trust_level.value}",
            )
        )

        return tuple(gates)

    def _objective_values(self, result: DesignPointResult) -> dict[str, float]:
        """Extract objective values from result metrics, substituting infinities for missing maximize/minimize."""
        values: dict[str, float] = {}
        for obj in self.objectives:
            raw = _metric_value(result.metrics, obj.metric_field)
            if raw is None:
                # Missing value is worst-case for the direction.
                raw = float("-inf") if obj.direction == ObjectiveDirection.MAXIMIZE else float("inf")
            values[obj.name] = float(raw)
        return values

    def _dominates(self, a: ParetoPoint, b: ParetoPoint) -> bool:
        """Return True if a dominates b (strictly better in at least one objective, not worse in any)."""
        strictly_better = False
        for obj in self.objectives:
            av = a.objective_values.get(obj.name, 0.0)
            bv = b.objective_values.get(obj.name, 0.0)
            if obj.direction == ObjectiveDirection.MAXIMIZE:
                if av < bv:
                    return False
                if av > bv:
                    strictly_better = True
            else:
                if av > bv:
                    return False
                if av < bv:
                    strictly_better = True
        return strictly_better

    def compute_frontier(self, results: Sequence[DesignPointResult]) -> list[ParetoPoint]:
        """Return Pareto-optimal points that pass all hard gates, sorted by design_point_id."""
        decorated: list[ParetoPoint] = []
        for result in results:
            gates = self.evaluate_gates(result)
            point = ParetoPoint(
                result=result,
                gate_results=gates,
                objective_values=self._objective_values(result),
            )
            if point.passed_all_gates:
                decorated.append(point)

        # Pareto dominance filter
        frontier: list[ParetoPoint] = []
        for candidate in decorated:
            dominated = False
            for other in decorated:
                if other is candidate:
                    continue
                if self._dominates(other, candidate):
                    dominated = True
                    break
            if not dominated:
                frontier.append(candidate)

        # Deterministic tie-break by design_point_id.
        frontier.sort(key=lambda p: p.result.design_point_id)
        return frontier

    def gate_summary(self, results: Sequence[DesignPointResult]) -> dict[str, Any]:
        """Return counts of gate failures across all results."""
        summary: dict[str, dict[str, int]] = {}
        for result in results:
            for gate in self.evaluate_gates(result):
                entry = summary.setdefault(gate.code.value, {"passed": 0, "failed": 0})
                entry["passed" if gate.passed else "failed"] += 1
        return summary


__all__ = [
    "Objective",
    "ObjectiveDirection",
    "HardGateCode",
    "GateResult",
    "ParetoPoint",
    "MultiObjectivePareto",
    "DEFAULT_OBJECTIVES",
    "objectives_from_scenario",
]
