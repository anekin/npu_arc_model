"""Unit tests for deterministic scheduler metrics aggregation."""

from __future__ import annotations

import math

from scheduler.metrics import (
    MetricsCollector,
    nearest_rank_percentile,
)


class TestNearestRankPercentile:
    def test_empty_returns_zero(self) -> None:
        assert nearest_rank_percentile([], 50.0) == 0.0

    def test_single_value(self) -> None:
        assert nearest_rank_percentile([7.0], 50.0) == 7.0

    def test_p50_even_count(self) -> None:
        values = sorted([1.0, 2.0, 3.0, 4.0])
        # ceil(0.5 * 4) = 2 -> index 1
        assert nearest_rank_percentile(values, 50.0) == 2.0

    def test_p99_rank(self) -> None:
        values = list(range(1, 101))
        # ceil(0.99 * 100) = 99 -> index 98
        assert nearest_rank_percentile(values, 99.0) == 99.0


class TestMetricsCollector:
    def _windowed_collector(self) -> MetricsCollector:
        collector = MetricsCollector()
        ps = 1_000_000_000
        # Warm-up job excluded from measurement.
        collector.record_arrival(
            job_id="w1",
            class_id="A",
            arrival_ps=0,
            deadline_ps=10 * ps,
            is_warmup=True,
            is_measurement=False,
            work_ps=4 * ps,
            resource_requirements={"compute": 1},
        )
        collector.record_start("w1", 0)
        collector.record_complete("w1", 4 * ps)
        # Measurement jobs.
        for i, arrival in enumerate([10, 20, 30, 40]):
            arrival_ps = arrival * ps
            collector.record_arrival(
                job_id=f"j{i}",
                class_id="A",
                arrival_ps=arrival_ps,
                deadline_ps=arrival_ps + 10 * ps,
                is_warmup=False,
                is_measurement=True,
                work_ps=4 * ps,
                resource_requirements={"compute": 1},
            )
            collector.record_start(f"j{i}", arrival_ps)
            collector.record_complete(f"j{i}", arrival_ps + 4 * ps)
            collector.record_resource_busy("compute", arrival_ps, arrival_ps + 4 * ps)
        return collector

    def test_window_excludes_warmup(self) -> None:
        collector = self._windowed_collector()
        metrics = collector.compute(compute_capacity=1)
        assert metrics.released_count == 4
        assert metrics.completed_count == 4

    def test_latency_percentiles(self) -> None:
        collector = self._windowed_collector()
        metrics = collector.compute(compute_capacity=1)
        assert metrics.latency_p50_ms == 4.0
        assert metrics.latency_p99_ms == 4.0
        assert metrics.latency_max_ms == 4.0

    def test_utilization(self) -> None:
        collector = self._windowed_collector()
        metrics = collector.compute(
            window_start_ps=10_000_000_000,
            window_end_ps=50_000_000_000,
            compute_capacity=1,
        )
        util = metrics.resource_utilization[0]
        assert util.resource_name == "compute"
        # Window is 10 -> 50 ms (40 ms). 4 jobs * 4 ms = 16 ms busy.
        assert util.busy_time_ps == 16_000_000_000
        assert util.window_time_ps == 40_000_000_000
        assert math.isclose(util.utilization, 0.4, abs_tol=1e-9)

    def test_dropped_jobs_counted_in_misses(self) -> None:
        collector = MetricsCollector()
        for i in range(2):
            collector.record_arrival(
                job_id=f"j{i}",
                class_id="A",
                arrival_ps=i * 10,
                deadline_ps=i * 10 + 5,
                is_measurement=True,
                work_ps=4,
                resource_requirements={"compute": 1},
            )
        collector.record_complete("j0", 4)
        collector.record_drop("j1")
        metrics = collector.compute(window_start_ps=0, window_end_ps=20, compute_capacity=1)
        assert metrics.completed_count == 1
        assert metrics.dropped_count == 1
        assert metrics.deadline_miss_count == 1

    def test_replaced_jobs_counted_in_misses(self) -> None:
        collector = MetricsCollector()
        collector.record_arrival(
            job_id="j0",
            class_id="A",
            arrival_ps=0,
            deadline_ps=10,
            is_measurement=True,
            work_ps=4,
            resource_requirements={"compute": 1},
        )
        collector.record_replace("j0")
        metrics = collector.compute(window_start_ps=0, window_end_ps=10, compute_capacity=1)
        assert metrics.replaced_count == 1
        assert metrics.deadline_miss_count == 1

    def test_stable_flag_false_on_drop(self) -> None:
        collector = MetricsCollector()
        collector.record_arrival(
            job_id="j0",
            class_id="A",
            arrival_ps=0,
            deadline_ps=10,
            is_measurement=True,
            work_ps=4,
            resource_requirements={"compute": 1},
        )
        collector.record_drop("j0")
        metrics = collector.compute(window_start_ps=0, window_end_ps=10, compute_capacity=1)
        assert metrics.stable is False

    def test_offered_and_achieved_utilization(self) -> None:
        collector = MetricsCollector()
        ps = 1_000_000_000
        for i in range(5):
            collector.record_arrival(
                job_id=f"j{i}",
                class_id="A",
                arrival_ps=i * 10 * ps,
                deadline_ps=(i * 10 + 10) * ps,
                is_measurement=True,
                work_ps=4 * ps,
                resource_requirements={"compute": 1},
            )
            collector.record_start(f"j{i}", i * 10 * ps)
            if i < 4:
                end_ps = (i * 10 + 4) * ps
                collector.record_complete(f"j{i}", end_ps)
                collector.record_resource_busy("compute", i * 10 * ps, end_ps, class_id="A")
        collector.record_drop("j4")
        metrics = collector.compute(window_start_ps=0, window_end_ps=50 * ps, compute_capacity=1)
        cm = metrics.class_metrics[0]
        assert cm.offered_load_ratio == 5 * 4 / 50
        assert cm.achieved_utilization == 4 * 4 / 50

    def test_observation_age_recorded(self) -> None:
        collector = MetricsCollector()
        collector.record_arrival(
            job_id="j0",
            class_id="A",
            arrival_ps=0,
            deadline_ps=10,
            is_measurement=True,
            work_ps=4,
            resource_requirements={"compute": 1},
        )
        collector.record_observation_age("j0", 1_000_000_000)
        metrics = collector.compute(window_start_ps=0, window_end_ps=10, compute_capacity=1)
        assert metrics.observation_age_p50_ms == 1.0

    def test_recovery_time_preserved(self) -> None:
        collector = MetricsCollector()
        collector.record_arrival(
            job_id="j0",
            class_id="A",
            arrival_ps=0,
            deadline_ps=10,
            is_measurement=True,
            work_ps=4,
            resource_requirements={"compute": 1},
        )
        collector.record_complete("j0", 4)
        collector.record_recovery_time(123_000_000_000)
        metrics = collector.compute(window_start_ps=0, window_end_ps=10, compute_capacity=1)
        assert metrics.recovery_time_ps == 123_000_000_000
