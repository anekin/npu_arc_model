"""Tests for scenario-driven DSE Pareto frontier and hard gates."""

import pytest

from contracts.result import (
    DesignPointResult,
    EngineMetrics,
    ErrorRecord,
    RunStatus,
    RunTrustLevel,
)
from dse.pareto import (
    HardGateCode,
    MultiObjectivePareto,
    Objective,
    ObjectiveDirection,
    objectives_from_scenario,
)
from scenarios.schema import ArrivalPattern, Scenario, WorkloadClass


def _complete_result(
    design_point_id: str,
    *,
    tok_per_s: float = 10.0,
    area_mm2: float = 50.0,
    power_w: float = 5.0,
    energy_joules: float = 100.0,
    deadline_miss_count: int = 0,
    drop_count: int = 0,
    trust_level: RunTrustLevel = RunTrustLevel.authoritative,
    status: RunStatus = RunStatus.complete,
    engine_type: str = "block",
) -> DesignPointResult:
    return DesignPointResult(
        design_point_id=design_point_id,
        status=status,
        hardware_digest="",
        scenario_ref="test",
        workload_ref="test",
        engine_type=engine_type,
        trust_level=trust_level,
        metrics=EngineMetrics(
            tok_per_s=tok_per_s,
            completed_throughput_hz=tok_per_s,
            area_mm2=area_mm2,
            power_w=power_w,
            energy_joules=energy_joules,
            deadline_miss_count=deadline_miss_count,
            drop_count=drop_count,
        ),
    )


def test_default_objectives_cover_throughput_and_resources():
    pareto = MultiObjectivePareto()
    names = {obj.name for obj in pareto.objectives}
    assert "completed_throughput_hz" in names
    assert "area_mm2" in names
    assert "power_w" in names
    assert "energy_joules" in names


def test_frontier_excludes_non_authoritative_points():
    results = [
        _complete_result("auth_fast", tok_per_s=20.0, area_mm2=60.0),
        _complete_result(
            "exploratory_slow",
            tok_per_s=5.0,
            area_mm2=40.0,
            trust_level=RunTrustLevel.exploratory,
        ),
    ]
    pareto = MultiObjectivePareto()
    frontier = pareto.compute_frontier(results)
    assert [p.result.design_point_id for p in frontier] == ["auth_fast"]


def test_frontier_excludes_failed_status():
    results = [
        _complete_result("good", tok_per_s=10.0, area_mm2=50.0),
        DesignPointResult(
            design_point_id="bad",
            status=RunStatus.failed,
            scenario_ref="test",
            workload_ref="test",
            engine_type="block",
            trust_level=RunTrustLevel.authoritative,
            error=ErrorRecord(design_point_id="bad", code="X", message="boom"),
        ),
    ]
    pareto = MultiObjectivePareto()
    frontier = pareto.compute_frontier(results)
    assert [p.result.design_point_id for p in frontier] == ["good"]


def test_frontier_excludes_deadline_misses():
    results = [
        _complete_result("clean", tok_per_s=10.0, area_mm2=50.0),
        _complete_result(
            "miss",
            tok_per_s=20.0,
            area_mm2=40.0,
            deadline_miss_count=1,
        ),
    ]
    pareto = MultiObjectivePareto()
    frontier = pareto.compute_frontier(results)
    assert [p.result.design_point_id for p in frontier] == ["clean"]


def test_frontier_excludes_cpu_fallback():
    results = [
        _complete_result("block", tok_per_s=10.0, area_mm2=50.0, engine_type="block"),
        _complete_result("cpu", tok_per_s=15.0, area_mm2=40.0, engine_type="cpu"),
    ]
    pareto = MultiObjectivePareto()
    frontier = pareto.compute_frontier(results)
    assert [p.result.design_point_id for p in frontier] == ["block"]


def test_frontier_excludes_thermal_violations():
    results = [
        _complete_result("cool", tok_per_s=10.0, area_mm2=50.0, power_w=5.0),
        _complete_result("hot", tok_per_s=20.0, area_mm2=40.0, power_w=200.0),
    ]
    pareto = MultiObjectivePareto(thermal_limit_w=150.0)
    frontier = pareto.compute_frontier(results)
    assert [p.result.design_point_id for p in frontier] == ["cool"]


def test_pareto_dominance_keeps_only_non_dominated():
    results = [
        _complete_result("dominated", tok_per_s=10.0, area_mm2=50.0, power_w=5.0),
        _complete_result("better", tok_per_s=20.0, area_mm2=40.0, power_w=4.0),
    ]
    pareto = MultiObjectivePareto()
    frontier = pareto.compute_frontier(results)
    assert [p.result.design_point_id for p in frontier] == ["better"]


def test_pareto_keeps_tradeoff_points():
    results = [
        _complete_result("fast_big", tok_per_s=30.0, area_mm2=100.0, power_w=10.0),
        _complete_result("slow_small", tok_per_s=10.0, area_mm2=30.0, power_w=3.0),
    ]
    pareto = MultiObjectivePareto()
    frontier = pareto.compute_frontier(results)
    ids = {p.result.design_point_id for p in frontier}
    assert ids == {"fast_big", "slow_small"}


def test_missing_maximize_metric_treated_as_worst():
    result = DesignPointResult(
        design_point_id="missing",
        status=RunStatus.complete,
        scenario_ref="test",
        workload_ref="test",
        engine_type="block",
        trust_level=RunTrustLevel.authoritative,
        metrics=EngineMetrics(
            tok_per_s=10.0,
            completed_throughput_hz=None,
            area_mm2=50.0,
            power_w=5.0,
            energy_joules=100.0,
        ),
    )
    pareto = MultiObjectivePareto()
    values = pareto._objective_values(result)
    assert values["completed_throughput_hz"] == float("-inf")


def test_gate_summary_counts_failures():
    results = [
        _complete_result("ok", tok_per_s=10.0, area_mm2=50.0),
        _complete_result(
            "hot",
            tok_per_s=10.0,
            area_mm2=50.0,
            power_w=200.0,
        ),
    ]
    pareto = MultiObjectivePareto(thermal_limit_w=150.0)
    summary = pareto.gate_summary(results)
    assert summary[HardGateCode.POWER_THERMAL.value]["passed"] == 1
    assert summary[HardGateCode.POWER_THERMAL.value]["failed"] == 1


def _default_class() -> WorkloadClass:
    return WorkloadClass(
        id="c0",
        arrival=ArrivalPattern(mode="periodic", period_ms=10, count=10),
        work_ms=1,
        relative_deadline_ms=2,
    )


def test_objectives_from_scenario_returns_defaults_without_metadata():
    scenario = Scenario(name="empty", classes=[_default_class()])
    objectives = objectives_from_scenario(scenario)
    assert len(objectives) == len(MultiObjectivePareto().objectives)


def test_objectives_from_scenario_reads_metadata_overrides():
    scenario = Scenario(
        name="custom",
        classes=[_default_class()],
        metadata={
            "objectives": [
                {"name": "throughput", "metric_field": "tok_per_s", "direction": "maximize"},
                {"name": "area", "metric_field": "area_mm2", "direction": "minimize"},
            ]
        },
    )
    objectives = objectives_from_scenario(scenario)
    assert len(objectives) == 2
    assert objectives[0].name == "throughput"
    assert objectives[0].metric_field == "tok_per_s"
    assert objectives[0].direction == ObjectiveDirection.MAXIMIZE
    assert objectives[1].direction == ObjectiveDirection.MINIMIZE


def test_objectives_from_scenario_rejects_invalid_entry():
    scenario = Scenario(
        name="bad",
        classes=[_default_class()],
        metadata={"objectives": [{"direction": "maximize"}]},
    )
    with pytest.raises(Exception):
        objectives_from_scenario(scenario)


def test_custom_objectives_drive_dominance():
    objectives = [
        Objective("throughput", "tok_per_s", ObjectiveDirection.MAXIMIZE),
        Objective("area", "area_mm2", ObjectiveDirection.MINIMIZE),
    ]
    results = [
        _complete_result("a", tok_per_s=10.0, area_mm2=50.0),
        _complete_result("b", tok_per_s=20.0, area_mm2=40.0),
    ]
    pareto = MultiObjectivePareto(objectives=objectives)
    frontier = pareto.compute_frontier(results)
    assert [p.result.design_point_id for p in frontier] == ["b"]
