"""End-to-end scenario runner tests for queue policies, priorities, and schema."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from scenario_runner import run_scenario
from scenarios.compiler import compile_scenario
from scenarios.schema import ArrivalMode, ArrivalPattern, Scenario, WorkloadClass

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SIM_DIR = REPO_ROOT / "sim"
YAML_PATH = SIM_DIR / "config" / "temporal_scenarios.yaml"


def _load_scenario(name: str) -> Scenario:
    raw = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    return Scenario.model_validate(raw["scenarios"][name])


def _run(name: str):
    scenario = _load_scenario(name)
    compiled = compile_scenario(scenario)
    return run_scenario(compiled)


class TestMailboxLatest:
    def test_replacements_greater_than_zero(self) -> None:
        metrics = _run("mailbox_overload")
        assert metrics.replaced_count > 0

    def test_no_drops(self) -> None:
        metrics = _run("mailbox_overload")
        assert metrics.dropped_count == 0

    def test_pending_depth_bounded_by_one(self) -> None:
        # The mailbox keeps at most one pending item per stream.
        metrics = _run("mailbox_overload")
        assert metrics.peak_queue <= 1


class TestPriorityPreempt:
    def test_high_priority_meets_deadline(self) -> None:
        metrics = _run("priority_preempt")
        high = next(cm for cm in metrics.class_metrics if cm.class_id == "high")
        assert high.deadline_miss_count == 0
        assert high.completed_count > 0

    def test_low_priority_misses_or_drops(self) -> None:
        metrics = _run("priority_preempt")
        low = next(cm for cm in metrics.class_metrics if cm.class_id == "low")
        assert low.deadline_miss_count > 0

    def test_not_stable(self) -> None:
        metrics = _run("priority_preempt")
        assert metrics.stable is False


class TestScenarioSchema:
    def test_periodic_arrival_times(self) -> None:
        pattern = ArrivalPattern(
            mode=ArrivalMode.PERIODIC,
            period_ms=10.0,
            count=5,
            offset_ms=5.0,
        )
        times = pattern.release_times_ps()
        assert len(times) == 5
        assert times[0] == 5_000_000_000
        assert times[-1] == 45_000_000_000

    def test_trace_must_be_sorted(self) -> None:
        with pytest.raises(ValueError):
            ArrivalPattern(mode=ArrivalMode.TRACE, releases_ms=[10.0, 5.0])

    def test_workload_class_deadline_defaults_to_work(self) -> None:
        cls = WorkloadClass(
            id="A",
            arrival=ArrivalPattern(
                mode=ArrivalMode.PERIODIC,
                period_ms=10.0,
                count=1,
            ),
            work_ms=7.0,
        )
        assert cls.relative_deadline_ms == 7.0
        assert cls.timeout_ms == 7.0

    def test_unique_class_ids_required(self) -> None:
        with pytest.raises(ValueError):
            Scenario(
                name="dup",
                classes=[
                    WorkloadClass(
                        id="A",
                        arrival=ArrivalPattern(
                            mode=ArrivalMode.PERIODIC,
                            period_ms=10.0,
                            count=1,
                        ),
                        work_ms=1.0,
                    ),
                    WorkloadClass(
                        id="A",
                        arrival=ArrivalPattern(
                            mode=ArrivalMode.PERIODIC,
                            period_ms=10.0,
                            count=1,
                        ),
                        work_ms=1.0,
                    ),
                ],
            )


class TestTraceArrivals:
    def test_explicit_trace_run(self) -> None:
        scenario = Scenario.model_validate(
            {
                "name": "trace",
                "seed": 0,
                "warmup_count": 0,
                "measurement_count": 3,
                "drain": True,
                "max_simulation_time_ms": 100.0,
                "compute_capacity": 1,
                "classes": [
                    {
                        "id": "A",
                        "arrival": {
                            "mode": "trace",
                            "releases_ms": [0.0, 5.0, 10.0],
                        },
                        "work_ms": 2.0,
                        "relative_deadline_ms": 5.0,
                        "queue_policy": "fifo",
                        "queue_capacity": 8,
                        "resource_requirements": {"compute": 1},
                        "admission_excluded": True,
                    }
                ],
            }
        )
        compiled = compile_scenario(scenario)
        metrics = run_scenario(compiled)
        assert metrics.released_count == 3
        assert metrics.completed_count == 3
        assert metrics.deadline_miss_count == 0


class TestNonpreemptibleOverload:
    def test_nonpreemptible_overload_is_not_stable(self) -> None:
        scenario = Scenario.model_validate(
            {
                "name": "nonpreemptible",
                "seed": 0,
                "warmup_count": 0,
                "measurement_count": 20,
                "drain": True,
                "max_simulation_time_ms": 500.0,
                "compute_capacity": 1,
                "preemption_enabled": False,
                "classes": [
                    {
                        "id": "A",
                        "arrival": {
                            "mode": "periodic",
                            "period_ms": 5.0,
                            "count": 25,
                            "offset_ms": 0.0,
                        },
                        "work_ms": 6.0,
                        "relative_deadline_ms": 5.0,
                        "queue_policy": "fifo",
                        "queue_capacity": 32,
                        "resource_requirements": {"compute": 1},
                        "admission_excluded": True,
                    }
                ],
            }
        )
        compiled = compile_scenario(scenario)
        metrics = run_scenario(compiled)
        assert metrics.stable is False
        assert metrics.deadline_miss_count > 0
