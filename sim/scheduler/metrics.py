"""Deterministic temporal metrics aggregation for the discrete-event scheduler.

All internal times are integer picoseconds.  Latency percentiles use the
nearest-rank method (no interpolation) so results are deterministic and
auditable by hand.

Metrics follow the scenario-driven DSE contract:

* ``arrival-to-start`` and ``arrival-to-complete`` latency distributions
* ``observation_age`` for mailbox_latest queues
* nearest-rank P50/P99/max
* completed throughput, deadline/timeout miss, drop/replacement/admission
  reject/underflow counters
* per-resource ``busy_time / window_time`` utilization
* peak queue depth, bytes, energy

Dropped/replaced jobs are kept in denominators for miss-rate calculations
so that overload cannot be hidden by discarding jobs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from contracts.errors import ConfigError

# ── Unit conversions ─────────────────────────────────────────────────────────

_PS_PER_MS = 1_000_000_000
_PS_PER_US = 1_000_000
_PS_PER_S = 1_000_000_000_000


def ps_to_ms(ps: int) -> float:
    """Convert picoseconds to milliseconds."""
    return ps / _PS_PER_MS


def ps_to_us(ps: int) -> float:
    """Convert picoseconds to microseconds."""
    return ps / _PS_PER_US


def ms_to_ps(ms: float) -> int:
    """Convert milliseconds to integer picoseconds."""
    return int(round(ms * _PS_PER_MS))


# ── Per-job record ───────────────────────────────────────────────────────────


class JobOutcome:
    """Terminal outcome of a job in the metrics window."""

    COMPLETED = "completed"
    DROPPED = "dropped"
    REPLACED = "replaced"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class JobMetricRecord:
    """Mutable record used by the collector while a simulation runs."""

    job_id: str
    class_id: str
    arrival_ps: int
    deadline_ps: int
    is_warmup: bool = False
    is_measurement: bool = False
    work_ps: int = 0
    resource_requirements: dict[str, int] = field(default_factory=dict)
    start_ps: int | None = None
    complete_ps: int | None = None
    outcome: str = JobOutcome.COMPLETED
    observation_age_ps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ps(self) -> int | None:
        """Arrival-to-complete latency; None if not completed."""
        if self.complete_ps is None:
            return None
        return self.complete_ps - self.arrival_ps

    @property
    def start_latency_ps(self) -> int | None:
        """Arrival-to-start latency; None if never started."""
        if self.start_ps is None:
            return None
        return self.start_ps - self.arrival_ps


# ── Aggregated metrics ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClassMetrics:
    """Metrics for a single workload class within the measurement window."""

    class_id: str
    released_count: int
    completed_count: int
    dropped_count: int
    replaced_count: int
    rejected_count: int
    timeout_count: int
    deadline_miss_count: int
    completed_throughput_hz: float
    latency_p50_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    start_latency_p50_ms: float
    start_latency_p99_ms: float
    observation_age_p50_ms: float
    observation_age_p99_ms: float
    offered_load_ratio: float
    achieved_utilization: float


@dataclass(frozen=True)
class ResourceUtilization:
    """Utilization of a single resource over the measurement window."""

    resource_name: str
    busy_time_ps: int
    window_time_ps: int
    utilization: float


@dataclass(frozen=True)
class ScenarioMetrics:
    """Final deterministic metrics for a scenario run."""

    window_start_ps: int
    window_end_ps: int
    window_time_ps: int
    released_count: int
    completed_count: int
    dropped_count: int
    replaced_count: int
    rejected_count: int
    timeout_count: int
    deadline_miss_count: int
    completed_throughput_hz: float
    latency_p50_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    start_latency_p50_ms: float
    start_latency_p99_ms: float
    observation_age_p50_ms: float
    observation_age_p99_ms: float
    peak_queue: int
    bytes_total: int
    energy_joules: float
    resource_utilization: tuple[ResourceUtilization, ...]
    class_metrics: tuple[ClassMetrics, ...]
    recovery_time_ps: int | None = None
    stable: bool = False


# ── Percentile helpers ───────────────────────────────────────────────────────


def nearest_rank_percentile(sorted_values: list[float], percentile: float) -> float:
    """Return nearest-rank percentile (1-indexed, no interpolation).

    Args:
        sorted_values: ascending list of values
        percentile: value in (0, 100]

    Returns:
        The value at rank ``ceil(percentile/100 * N)``.  Returns ``0.0`` for
        an empty input so that empty distributions do not crash aggregation.
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    rank = math.ceil((percentile / 100.0) * n)
    rank = max(1, min(rank, n))
    return sorted_values[rank - 1]


def _percentile_ms(values_ps: list[int]) -> tuple[float, float, float]:
    """Return (p50, p99, max) in milliseconds for a list of picosecond values."""
    sorted_ms = sorted(ps_to_ms(v) for v in values_ps)
    p50 = nearest_rank_percentile(sorted_ms, 50.0)
    p99 = nearest_rank_percentile(sorted_ms, 99.0)
    max_ms = sorted_ms[-1] if sorted_ms else 0.0
    return p50, p99, max_ms


# ── Interval union for utilization ───────────────────────────────────────────


def _union_intervals_length(intervals: list[tuple[int, int]]) -> int:
    """Return total covered length of a list of [start, end) intervals."""
    if not intervals:
        return 0
    sorted_intervals = sorted(intervals)
    total = 0
    cur_start, cur_end = sorted_intervals[0]
    for start, end in sorted_intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


# ── Collector ────────────────────────────────────────────────────────────────


class MetricsCollector:
    """Collect per-job and per-resource events and produce ``ScenarioMetrics``."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobMetricRecord] = {}
        self._resource_intervals: dict[str, list[tuple[int, int]]] = {}
        self._class_resource_intervals: dict[str, list[tuple[int, int]]] = {}
        self._queue_sizes: dict[str, list[tuple[int, int]]] = {}
        self._bytes_total: int = 0
        self._energy_joules: float = 0.0
        self._recovery_time_ps: int | None = None

    # ── Job lifecycle ────────────────────────────────────────────────────────

    def record_arrival(
        self,
        job_id: str,
        class_id: str,
        arrival_ps: int,
        deadline_ps: int,
        *,
        is_warmup: bool = False,
        is_measurement: bool = False,
        work_ps: int = 0,
        resource_requirements: dict[str, int] | None = None,
    ) -> JobMetricRecord:
        """Record a job arrival and return its metric record."""
        if job_id in self._jobs:
            raise ConfigError(f"duplicate metric job_id {job_id!r}", field_path="metrics.job_id")
        rec = JobMetricRecord(
            job_id=job_id,
            class_id=class_id,
            arrival_ps=arrival_ps,
            deadline_ps=deadline_ps,
            is_warmup=is_warmup,
            is_measurement=is_measurement,
            work_ps=work_ps,
            resource_requirements=resource_requirements or {},
        )
        self._jobs[job_id] = rec
        return rec

    def record_start(self, job_id: str, start_ps: int) -> None:
        """Record when a job started executing."""
        rec = self._jobs[job_id]
        if rec.start_ps is None or start_ps < rec.start_ps:
            rec.start_ps = start_ps

    def record_complete(self, job_id: str, complete_ps: int) -> None:
        """Record a job completion."""
        rec = self._jobs[job_id]
        rec.complete_ps = complete_ps
        rec.outcome = JobOutcome.COMPLETED

    def record_drop(self, job_id: str) -> None:
        """Record a job dropped by a full FIFO."""
        self._jobs[job_id].outcome = JobOutcome.DROPPED

    def record_replace(self, job_id: str) -> None:
        """Record a job replaced in a mailbox_latest queue."""
        self._jobs[job_id].outcome = JobOutcome.REPLACED

    def record_reject(self, job_id: str) -> None:
        """Record a job rejected by admission control."""
        self._jobs[job_id].outcome = JobOutcome.REJECTED

    def record_timeout(self, job_id: str) -> None:
        """Record a job that hit a hard timeout."""
        self._jobs[job_id].outcome = JobOutcome.TIMEOUT

    def record_observation_age(self, job_id: str, age_ps: int) -> None:
        """Record observation age for a mailbox_latest item."""
        self._jobs[job_id].observation_age_ps = age_ps

    # ── Resource and queue ───────────────────────────────────────────────────

    def record_resource_busy(
        self,
        resource_name: str,
        start_ps: int,
        end_ps: int,
        class_id: str | None = None,
    ) -> None:
        """Record a busy interval for ``resource_name`` and optionally ``class_id``."""
        if end_ps <= start_ps:
            return
        self._resource_intervals.setdefault(resource_name, []).append((start_ps, end_ps))
        if class_id is not None:
            key = f"{class_id}:{resource_name}"
            self._class_resource_intervals.setdefault(key, []).append((start_ps, end_ps))

    def record_queue_size(self, queue_name: str, size: int, now_ps: int) -> None:
        """Record a queue size sample at ``now_ps``."""
        self._queue_sizes.setdefault(queue_name, []).append((now_ps, size))

    def record_bytes(self, bytes_: int) -> None:
        """Add to total transferred bytes."""
        if bytes_ > 0:
            self._bytes_total += bytes_

    def record_energy(self, joules: float) -> None:
        """Add to total energy consumption."""
        if joules > 0:
            self._energy_joules += joules

    def record_recovery_time(self, time_ps: int) -> None:
        """Record the time at which backlog drained to zero after overload."""
        if self._recovery_time_ps is None or time_ps < self._recovery_time_ps:
            self._recovery_time_ps = time_ps

    # ── Computation ──────────────────────────────────────────────────────────

    def compute(
        self,
        window_start_ps: int | None = None,
        window_end_ps: int | None = None,
        compute_capacity: int = 1,
    ) -> ScenarioMetrics:
        """Compute final metrics from collected records.

        If ``window_start_ps`` or ``window_end_ps`` are not provided, they are
        derived from the earliest and latest measurement-job arrivals.
        """
        measurement_jobs = [j for j in self._jobs.values() if j.is_measurement and not j.is_warmup]
        if not measurement_jobs:
            measurement_jobs = [j for j in self._jobs.values() if not j.is_warmup]

        if window_start_ps is None:
            window_start_ps = min((j.arrival_ps for j in measurement_jobs), default=0)
        if window_end_ps is None:
            window_end_ps = max(
                (j.arrival_ps for j in measurement_jobs),
                default=window_start_ps,
            )

        window_time_ps = max(0, window_end_ps - window_start_ps)

        # Global counters over measurement window.
        released = list(measurement_jobs)
        completed = [j for j in released if j.outcome == JobOutcome.COMPLETED]
        dropped = [j for j in released if j.outcome == JobOutcome.DROPPED]
        replaced = [j for j in released if j.outcome == JobOutcome.REPLACED]
        rejected = [j for j in released if j.outcome == JobOutcome.REJECTED]
        timed_out = [j for j in released if j.outcome == JobOutcome.TIMEOUT]

        deadline_misses = [
            j
            for j in released
            if (
                (j.outcome == JobOutcome.COMPLETED and j.complete_ps is not None and j.complete_ps > j.deadline_ps)
                or j.outcome in (JobOutcome.DROPPED, JobOutcome.REPLACED, JobOutcome.TIMEOUT)
            )
        ]

        completed_latencies_ps = [j.latency_ps for j in completed if j.latency_ps is not None]
        start_latencies_ps = [j.start_latency_ps for j in released if j.start_latency_ps is not None]
        observation_ages_ps = [j.observation_age_ps for j in released if j.observation_age_ps > 0]

        lat_p50, lat_p99, lat_max = _percentile_ms(completed_latencies_ps)
        start_p50, start_p99, _ = _percentile_ms(start_latencies_ps)
        age_p50, age_p99, _ = _percentile_ms(observation_ages_ps)

        throughput_hz = 0.0
        if window_time_ps > 0:
            throughput_hz = len(completed) * _PS_PER_S / window_time_ps

        # Per-class metrics.
        class_ids = sorted({j.class_id for j in released})
        class_metrics_list: list[ClassMetrics] = []
        for cid in class_ids:
            class_jobs = [j for j in released if j.class_id == cid]
            class_completed = [j for j in class_jobs if j.outcome == JobOutcome.COMPLETED]
            class_dropped = [j for j in class_jobs if j.outcome == JobOutcome.DROPPED]
            class_replaced = [j for j in class_jobs if j.outcome == JobOutcome.REPLACED]
            class_rejected = [j for j in class_jobs if j.outcome == JobOutcome.REJECTED]
            class_timeouts = [j for j in class_jobs if j.outcome == JobOutcome.TIMEOUT]
            class_misses = [
                j
                for j in class_jobs
                if (
                    (j.outcome == JobOutcome.COMPLETED and j.complete_ps is not None and j.complete_ps > j.deadline_ps)
                    or j.outcome in (JobOutcome.DROPPED, JobOutcome.REPLACED, JobOutcome.TIMEOUT)
                )
            ]
            class_latencies = [j.latency_ps for j in class_completed if j.latency_ps is not None]
            class_starts = [j.start_latency_ps for j in class_jobs if j.start_latency_ps is not None]
            class_ages = [j.observation_age_ps for j in class_jobs if j.observation_age_ps > 0]
            c_lat_p50, c_lat_p99, c_lat_max = _percentile_ms(class_latencies)
            c_start_p50, c_start_p99, _ = _percentile_ms(class_starts)
            c_age_p50, c_age_p99, _ = _percentile_ms(class_ages)

            class_metrics_list.append(
                ClassMetrics(
                    class_id=cid,
                    released_count=len(class_jobs),
                    completed_count=len(class_completed),
                    dropped_count=len(class_dropped),
                    replaced_count=len(class_replaced),
                    rejected_count=len(class_rejected),
                    timeout_count=len(class_timeouts),
                    deadline_miss_count=len(class_misses),
                    completed_throughput_hz=(
                        len(class_completed) * _PS_PER_S / window_time_ps if window_time_ps > 0 else 0.0
                    ),
                    latency_p50_ms=c_lat_p50,
                    latency_p99_ms=c_lat_p99,
                    latency_max_ms=c_lat_max,
                    start_latency_p50_ms=c_start_p50,
                    start_latency_p99_ms=c_start_p99,
                    observation_age_p50_ms=c_age_p50,
                    observation_age_p99_ms=c_age_p99,
                    offered_load_ratio=(
                        sum(j.work_ps for j in class_jobs)
                        * max(j.resource_requirements.get("compute", 1) for j in class_jobs)
                        / (window_time_ps * compute_capacity)
                        if window_time_ps > 0 and class_jobs
                        else 0.0
                    ),
                    achieved_utilization=(
                        _union_intervals_length(
                            [
                                (max(start, window_start_ps), min(end, window_end_ps))
                                for start, end in self._class_resource_intervals.get(f"{cid}:compute", [])
                                if end > window_start_ps and start < window_end_ps
                            ]
                        )
                        / (window_time_ps * compute_capacity)
                        if window_time_ps > 0
                        else 0.0
                    ),
                )
            )

        # Resource utilization.
        resource_utils: list[ResourceUtilization] = []
        for name in sorted(self._resource_intervals):
            intervals = self._resource_intervals[name]
            clipped = [
                (max(start, window_start_ps), min(end, window_end_ps))
                for start, end in intervals
                if end > window_start_ps and start < window_end_ps
            ]
            busy = _union_intervals_length(clipped)
            util = busy / window_time_ps if window_time_ps > 0 else 0.0
            resource_utils.append(
                ResourceUtilization(
                    resource_name=name,
                    busy_time_ps=busy,
                    window_time_ps=window_time_ps,
                    utilization=util,
                )
            )

        # Peak queue across all queues.
        peak_queue = 0
        for samples in self._queue_sizes.values():
            for _, size in samples:
                if size > peak_queue:
                    peak_queue = size

        stable = len(dropped) == 0 and len(replaced) == 0 and len(timed_out) == 0 and len(deadline_misses) == 0

        return ScenarioMetrics(
            window_start_ps=window_start_ps,
            window_end_ps=window_end_ps,
            window_time_ps=window_time_ps,
            released_count=len(released),
            completed_count=len(completed),
            dropped_count=len(dropped),
            replaced_count=len(replaced),
            rejected_count=len(rejected),
            timeout_count=len(timed_out),
            deadline_miss_count=len(deadline_misses),
            completed_throughput_hz=throughput_hz,
            latency_p50_ms=lat_p50,
            latency_p99_ms=lat_p99,
            latency_max_ms=lat_max,
            start_latency_p50_ms=start_p50,
            start_latency_p99_ms=start_p99,
            observation_age_p50_ms=age_p50,
            observation_age_p99_ms=age_p99,
            peak_queue=peak_queue,
            bytes_total=self._bytes_total,
            energy_joules=self._energy_joules,
            resource_utilization=tuple(resource_utils),
            class_metrics=tuple(class_metrics_list),
            recovery_time_ps=self._recovery_time_ps,
            stable=stable,
        )


__all__ = [
    "JobMetricRecord",
    "JobOutcome",
    "ClassMetrics",
    "ResourceUtilization",
    "ScenarioMetrics",
    "MetricsCollector",
    "nearest_rank_percentile",
    "ps_to_ms",
    "ps_to_us",
    "ms_to_ps",
]
