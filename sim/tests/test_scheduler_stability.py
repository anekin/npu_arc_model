"""Stability, overload, recovery, and admission tests for scenario runner."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from scenarios.compiler import compile_scenario
from scenarios.schema import Scenario
from scenario_runner import run_scenario


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


class TestStableHand:
    def test_exact_oracle(self) -> None:
        metrics = _run("stable_hand")
        assert metrics.released_count == 100
        assert metrics.completed_count == 100
        assert metrics.dropped_count == 0
        assert metrics.rejected_count == 0
        assert metrics.deadline_miss_count == 0
        assert metrics.latency_p50_ms == 4.0
        assert metrics.latency_p99_ms == 4.0
        assert metrics.latency_max_ms == 4.0
        assert metrics.stable is True

    def test_utilization_matches_offered_load(self) -> None:
        metrics = _run("stable_hand")
        util = metrics.resource_utilization[0].utilization
        cm = metrics.class_metrics[0]
        assert math.isclose(util, 0.4, abs_tol=1e-9)
        assert math.isclose(cm.offered_load_ratio, 0.4, abs_tol=1e-9)
        assert math.isclose(cm.achieved_utilization, 0.4, abs_tol=1e-9)


class TestOverloadFifo:
    def test_backlog_and_misses_grow(self) -> None:
        metrics = _run("overload_fifo")
        assert metrics.released_count == 50
        assert metrics.completed_count + metrics.dropped_count == 50
        assert metrics.deadline_miss_count == metrics.released_count
        assert metrics.stable is False

    def test_offered_load_above_one(self) -> None:
        metrics = _run("overload_fifo")
        cm = metrics.class_metrics[0]
        assert cm.offered_load_ratio == pytest.approx(1.2, abs=1e-6)

    def test_achieved_utilization_capped(self) -> None:
        metrics = _run("overload_fifo")
        cm = metrics.class_metrics[0]
        assert cm.achieved_utilization <= 1.0 + 1e-12


class TestOverloadRecovery:
    def test_recovery_time_recorded(self) -> None:
        metrics = _run("overload_recovery")
        assert metrics.recovery_time_ps is not None
        # Recovery phase starts at 250 ms; backlog must drain after that.
        assert metrics.recovery_time_ps > 250_000_000_000

    def test_not_stable(self) -> None:
        metrics = _run("overload_recovery")
        assert metrics.stable is False
        assert metrics.deadline_miss_count > 0


class TestAdmission:
    def test_rejects_when_memory_full(self) -> None:
        metrics = _run("admission_reject")
        assert metrics.rejected_count > 0
        assert metrics.released_count == metrics.completed_count + metrics.rejected_count

    def test_admitted_jobs_complete_within_service_time(self) -> None:
        metrics = _run("admission_reject")
        # Admitted jobs run for 3 ms; rejected jobs should not inflate latency.
        assert metrics.latency_max_ms <= 3.0


class TestNinetyPercentStable:
    def test_high_utilization_stable(self) -> None:
        scenario = Scenario.model_validate(
            {
                "name": "ninety_percent",
                "seed": 0,
                "warmup_count": 5,
                "measurement_count": 50,
                "drain": True,
                "max_simulation_time_ms": 2000.0,
                "compute_capacity": 1,
                "classes": [
                    {
                        "id": "A",
                        "arrival": {
                            "mode": "periodic",
                            "period_ms": 5.0,
                            "count": 60,
                            "offset_ms": 0.0,
                        },
                        "work_ms": 4.5,
                        "relative_deadline_ms": 5.0,
                        "queue_policy": "fifo",
                        "queue_capacity": 16,
                        "resource_requirements": {"compute": 1},
                        "admission_excluded": True,
                    }
                ],
            }
        )
        compiled = compile_scenario(scenario)
        metrics = run_scenario(compiled)
        assert metrics.dropped_count == 0
        assert metrics.replaced_count == 0
        assert metrics.deadline_miss_count == 0
        assert math.isclose(
            metrics.class_metrics[0].offered_load_ratio, 0.9, abs_tol=1e-9
        )
