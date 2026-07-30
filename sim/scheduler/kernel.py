"""Discrete-event scheduler kernel.

The kernel maintains a deterministic event queue, tracks job state, and
guarantees that every processed event advances simulation time, consumes
work, or terminates a job.  All times are integer picoseconds; cycle
inputs are converted to picoseconds with ``ceil(cycles * ps_per_cycle)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from contracts.errors import ConfigError
from scheduler.events import Event, EventPhase, EventQueue


class SchedulerError(RuntimeError):
    """Raised for scheduling invariants violated at runtime."""

    def __init__(self, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.detail = detail


class JobState(IntEnum):
    """Lifecycle states of a scheduled job."""

    PENDING = 0
    READY = 1
    RUNNING = 2
    COMPLETED = 3
    REJECTED = 4


@dataclass
class JobHandle:
    """A schedulable unit of work.

    ``work_ps`` is the total service time required; ``remaining_ps`` is
    updated by the kernel as the job executes.  ``release_ps`` and
    ``deadline_ps`` are absolute picosecond timestamps.
    """

    job_id: str
    release_ps: int
    deadline_ps: int
    work_ps: int
    remaining_ps: int = field(init=False)
    state: JobState = field(default=JobState.PENDING)
    priority_class: int = 0
    relative_deadline_ps: int = 0
    depends_on: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.work_ps < 0:
            raise SchedulerError(f"job {self.job_id!r} work_ps must be non-negative, got {self.work_ps}")
        self.remaining_ps = self.work_ps

    @property
    def is_ready(self) -> bool:
        return self.state == JobState.READY

    @property
    def is_running(self) -> bool:
        return self.state == JobState.RUNNING

    @property
    def is_completed(self) -> bool:
        return self.state == JobState.COMPLETED


class DiscreteEventKernel:
    """Deterministic discrete-event scheduler kernel.

    Args:
        frequency_mhz: Core clock frequency used for cycle→picosecond
            conversion.  ``ps_per_cycle = 1_000_000 / frequency_mhz``.
        dispatch_callback: Optional callable invoked on every ``DISPATCH``
            event; receives the kernel instance and the list of currently
            ready jobs.  If ``None``, the kernel still processes events but
            does not dispatch work automatically.
    """

    def __init__(
        self,
        frequency_mhz: int,
        dispatch_callback: Callable[[DiscreteEventKernel, list[JobHandle]], None] | None = None,
    ) -> None:
        if frequency_mhz <= 0:
            raise ConfigError(
                f"frequency_mhz must be positive, got {frequency_mhz}",
                field_path="frequency_mhz",
            )
        self.frequency_mhz = frequency_mhz
        self.ps_per_cycle = 1_000_000 / frequency_mhz
        self.now_ps: int = 0
        self.queue = EventQueue()
        self.jobs: dict[str, JobHandle] = {}
        self.completed: list[JobHandle] = []
        self.rejected: list[JobHandle] = []
        self._dispatch_callback = dispatch_callback
        self._completed_ids: set[str] = set()
        self._rejected_ids: set[str] = set()
        self._next_dispatch_time: int | None = None

    def cycles_to_ps(self, cycles: int) -> int:
        """Convert core cycles to picoseconds with ceiling."""
        if cycles < 0:
            raise SchedulerError(
                f"cycles must be non-negative, got {cycles}",
                detail=cycles,
            )
        if cycles == 0:
            return 0
        return math.ceil(cycles * self.ps_per_cycle)

    def microseconds_to_ps(self, us: float) -> int:
        """Convert microseconds to picoseconds."""
        if us < 0:
            raise SchedulerError(f"us must be non-negative, got {us}")
        return math.ceil(us * 1_000_000)

    def register_job(
        self,
        job_id: str,
        work_ps: int,
        release_ps: int = 0,
        relative_deadline_ps: int | None = None,
        priority_class: int = 0,
        depends_on: set[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> JobHandle:
        """Register a new job and schedule its release event.

        If ``release_ps`` equals the current time and all dependencies are
        satisfied, the job becomes ``READY`` immediately and a dispatch event
        is scheduled for the same timestamp.
        """
        if not job_id:
            raise ConfigError("job_id must be non-empty", field_path="job_id")
        if job_id in self.jobs:
            raise ConfigError(
                f"duplicate job_id {job_id!r}",
                field_path="job_id",
            )
        if work_ps < 0:
            raise SchedulerError(f"job {job_id!r} work_ps must be non-negative, got {work_ps}")
        if release_ps < self.now_ps:
            raise SchedulerError(f"job {job_id!r} release_ps {release_ps} is in the past (now={self.now_ps})")

        rel_deadline = relative_deadline_ps if relative_deadline_ps is not None else work_ps
        if rel_deadline < 0:
            raise SchedulerError(f"job {job_id!r} relative_deadline_ps must be non-negative, got {rel_deadline}")

        dep_set = set(depends_on or ())
        if job_id in dep_set:
            raise SchedulerError(
                f"job {job_id!r} depends on itself",
                detail=dep_set,
            )

        job = JobHandle(
            job_id=job_id,
            release_ps=release_ps,
            deadline_ps=release_ps + rel_deadline,
            work_ps=work_ps,
            state=JobState.PENDING,
            priority_class=priority_class,
            relative_deadline_ps=rel_deadline,
            depends_on=dep_set,
            metadata=metadata or {},
        )
        self.jobs[job_id] = job

        self._detect_cycle(job_id)

        if release_ps == self.now_ps and self._dependencies_satisfied(job):
            self._make_ready(job)
            self._schedule_dispatch(release_ps)
        elif release_ps == self.now_ps:
            # Dependencies not yet satisfied; will be checked on completion.
            pass
        else:
            self.queue.push(release_ps, EventPhase.RELEASE, self._release_payload(job_id))
        return job

    def reject_job(self, job_id: str) -> None:
        """Mark a pending/ready job as rejected and remove it from scheduling."""
        job = self._get_job(job_id)
        job.state = JobState.REJECTED
        self.rejected.append(job)
        self._rejected_ids.add(job_id)

    def schedule(
        self,
        time_ps: int,
        phase: EventPhase,
        payload: Callable[[DiscreteEventKernel], None],
    ) -> Event:
        """Schedule an event at ``time_ps`` with the given phase/payload."""
        if time_ps < self.now_ps:
            raise SchedulerError(f"cannot schedule event at {time_ps} ps (now={self.now_ps} ps)")
        return self.queue.push(time_ps, phase, payload)

    def cancel_event(self, event: Event) -> bool:
        """Cancel a previously scheduled event."""
        return self.queue.cancel(event)

    def _release_payload(self, job_id: str) -> Callable[[DiscreteEventKernel], None]:
        def _release(kernel: DiscreteEventKernel) -> None:
            job = kernel._get_job(job_id)
            if job.state != JobState.PENDING:
                return
            if kernel._dependencies_satisfied(job):
                kernel._make_ready(job)
                kernel._schedule_dispatch(kernel.now_ps)

        return _release

    def _make_ready(self, job: JobHandle) -> None:
        if job.state == JobState.PENDING:
            job.state = JobState.READY

    def _dependencies_satisfied(self, job: JobHandle) -> bool:
        return all(self.jobs.get(dep, job).state == JobState.COMPLETED for dep in job.depends_on)

    def _detect_cycle(self, start_job_id: str) -> None:
        """Raise SchedulerError if ``start_job_id`` participates in a cycle."""
        visited: set[str] = set()
        stack: list[str] = [start_job_id]
        while stack:
            current = stack.pop()
            if current in visited:
                if current == start_job_id:
                    raise SchedulerError(
                        f"cyclic dependency detected involving {start_job_id!r}",
                        detail=start_job_id,
                    )
                continue
            visited.add(current)
            job = self.jobs.get(current)
            if job is None:
                continue
            for dep in job.depends_on:
                stack.append(dep)

    def _schedule_dispatch(self, time_ps: int) -> None:
        """Schedule a dispatch event at ``time_ps`` if a callback is set.

        Multiple dispatch requests at the same timestamp are coalesced so
        that serial dispatchers do not receive redundant no-op events.
        """
        if self._dispatch_callback is None:
            return
        if self._next_dispatch_time is not None and self._next_dispatch_time <= time_ps:
            return
        self._next_dispatch_time = time_ps
        self.queue.push(time_ps, EventPhase.DISPATCH, self._dispatch_payload())

    def _dispatch_payload(self) -> Callable[[DiscreteEventKernel], None]:
        def _dispatch(kernel: DiscreteEventKernel) -> None:
            kernel._next_dispatch_time = None
            if kernel._dispatch_callback is None:
                return
            ready = kernel.ready_jobs()
            if ready:
                kernel._dispatch_callback(kernel, ready)

        return _dispatch

    def ready_jobs(self) -> list[JobHandle]:
        """Return all jobs in ``READY`` state, sorted by job_id for determinism."""
        return sorted(
            (j for j in self.jobs.values() if j.state == JobState.READY),
            key=lambda j: j.job_id,
        )

    def running_jobs(self) -> list[JobHandle]:
        """Return all jobs in ``RUNNING`` state."""
        return [j for j in self.jobs.values() if j.state == JobState.RUNNING]

    def consume_work(self, job_id: str, duration_ps: int) -> None:
        """Consume ``duration_ps`` from ``job_id``'s remaining work.

        Raises ``SchedulerError`` if the job is not running or if the
        duration exceeds the remaining work.
        """
        job = self._get_job(job_id)
        if job.state != JobState.RUNNING:
            raise SchedulerError(f"cannot consume work for job {job_id!r} in state {job.state.name}")
        if duration_ps < 0:
            raise SchedulerError(f"consume_work duration must be non-negative, got {duration_ps}")
        if duration_ps > job.remaining_ps:
            raise SchedulerError(f"job {job_id!r} over-consumed: duration {duration_ps} > remaining {job.remaining_ps}")
        job.remaining_ps -= duration_ps

    def complete_job(self, job_id: str) -> None:
        """Mark a running job as completed and wake dependent jobs.

        Idempotent: calling on an already-completed job is a no-op.
        """
        job = self._get_job(job_id)
        if job.state == JobState.COMPLETED:
            return
        if job.state != JobState.RUNNING:
            raise SchedulerError(f"cannot complete job {job_id!r} in state {job.state.name}")
        self._complete_job_internal(job)

    def _complete_job_internal(self, job: JobHandle) -> None:
        job.remaining_ps = 0
        job.state = JobState.COMPLETED
        if job not in self.completed:
            self.completed.append(job)
        self._completed_ids.add(job.job_id)

        for other in self.jobs.values():
            if (
                other.state == JobState.PENDING
                and self._dependencies_satisfied(other)
                and other.release_ps <= self.now_ps
            ):
                self._make_ready(other)
        if any(j.state == JobState.READY for j in self.jobs.values()):
            self._schedule_dispatch(self.now_ps)

    def _consume_elapsed_work(self, previous_time_ps: int) -> None:
        delta = self.now_ps - previous_time_ps
        if delta <= 0:
            return
        for job in list(self.jobs.values()):
            if job.state != JobState.RUNNING:
                continue
            consumed = min(delta, job.remaining_ps)
            job.remaining_ps -= consumed
            if job.remaining_ps == 0:
                self._complete_job_internal(job)

    def _get_job(self, job_id: str) -> JobHandle:
        try:
            return self.jobs[job_id]
        except KeyError as exc:
            raise SchedulerError(f"unknown job_id {job_id!r}") from exc

    def run_until(
        self,
        end_time_ps: int | None = None,
        max_events: int | None = None,
    ) -> None:
        """Process events until completion, ``end_time_ps``, or ``max_events``.

        Raises ``SchedulerError`` if an event fails to advance time,
        consume work, or terminate a job (zero-time livelock).
        """
        event_count = 0
        while True:
            ev = self.queue.peek()
            if ev is None:
                break
            if end_time_ps is not None and ev.time_ps > end_time_ps:
                break
            if max_events is not None and event_count >= max_events:
                break

            ev = self.queue.pop()
            old_time = self.now_ps
            old_completed = len(self.completed)
            old_remaining = {jid: j.remaining_ps for jid, j in self.jobs.items() if j.state == JobState.RUNNING}
            old_running = {jid for jid, j in self.jobs.items() if j.state == JobState.RUNNING}
            old_ready = {jid for jid, j in self.jobs.items() if j.state == JobState.READY}

            if ev.time_ps < self.now_ps:
                raise SchedulerError(
                    f"event at {ev.time_ps} ps is before current time {self.now_ps} ps",
                    detail=ev,
                )

            self.now_ps = ev.time_ps
            # Running jobs consume work as time advances.
            self._consume_elapsed_work(old_time)
            if callable(ev.payload):
                ev.payload(self)

            new_running = {jid for jid, j in self.jobs.items() if j.state == JobState.RUNNING}
            newly_running = new_running - old_running
            new_completed = len(self.completed)
            # A DISPATCH event that finds no ready jobs is a legitimate no-op.
            idle_dispatch = ev.phase == EventPhase.DISPATCH and not old_ready
            progress = (
                self.now_ps > old_time
                or new_completed > old_completed
                or newly_running
                or idle_dispatch
                or any(self.jobs[jid].remaining_ps < old_remaining[jid] for jid in old_remaining)
            )
            if not progress:
                raise SchedulerError(
                    "zero-time livelock detected: event did not advance time, consume work, or terminate a job",
                    detail=ev,
                )
            event_count += 1

    def run_until_complete(
        self,
        max_time_ps: int = 10**18,
        max_events: int = 1_000_000,
    ) -> None:
        """Run until all non-rejected jobs are completed or limits are hit."""
        for _ in range(max_events):
            pending_or_ready = [
                j for j in self.jobs.values() if j.state in (JobState.PENDING, JobState.READY, JobState.RUNNING)
            ]
            if not pending_or_ready:
                break
            ev = self.queue.peek()
            if ev is None or ev.time_ps > max_time_ps:
                break
            self.run_until(max_events=1)
