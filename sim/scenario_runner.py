"""End-to-end scenario runner producing deterministic temporal metrics.

``ScenarioRunner`` takes a ``CompiledScenario``, drives the discrete-event
scheduler kernel, and returns a ``ScenarioMetrics``.  It supports FIFO and
mailbox_latest queues, admission control, service-class priority, EDF,
stable job-id tie-breaking, and optional preemption.

Kernel jobs are registered only when they begin executing; runner queues
hold admitted jobs until dispatch.  This keeps arrival events making
progress even when no resource is free: the scheduled dispatch becomes an
idle dispatch, which the kernel accepts as progress.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from contracts.errors import ConfigError
from scheduler.admission import AdmissionController
from scheduler.events import Event, EventPhase
from scheduler.kernel import DiscreteEventKernel, JobState, SchedulerError
from scheduler.metrics import MetricsCollector, ScenarioMetrics, ms_to_ps
from scheduler.policies import policy_key
from scheduler.queues import BoundedFIFO, MailboxLatest, QueueFullError
from scheduler.resources import CapacityResource
from scenarios.compiler import CompiledScenario, JobRelease
from scenarios.schema import QueuePolicy, Scenario, WorkloadClass


@dataclass
class _Candidate:
    """Internal dispatch candidate."""

    job_id: str
    source: str  # "preempted", "fifo", "mailbox"


class ScenarioRunner:
    """Run a compiled scenario and produce metrics."""

    def __init__(self, compiled: CompiledScenario) -> None:
        self.compiled = compiled
        self.scenario = compiled.scenario
        self.collector = MetricsCollector()
        self.kernel = DiscreteEventKernel(frequency_mhz=1000)

        # Resource tracking
        self._resource_used: Dict[str, int] = {name: 0 for name in compiled.resources}
        self._running: Dict[str, Event] = {}
        self._preempted: Set[str] = set()
        self._preempted_remaining: Dict[str, int] = {}
        self._start_times: Dict[str, int] = {}

        # Queue handles
        self._fifo_queues: Dict[str, BoundedFIFO[str]] = {}
        self._mailboxes: Dict[str, MailboxLatest[str]] = {}
        for key, q in compiled.queues.items():
            if isinstance(q, BoundedFIFO):
                self._fifo_queues[key] = q
            elif isinstance(q, MailboxLatest):
                self._mailboxes[key] = q

        # Release bookkeeping
        self._release_by_id: Dict[str, JobRelease] = {
            r.job_id: r for r in compiled.releases
        }
        self._admitted: Set[str] = set()

        self._recovery_phase_start_ps: int = (
            ms_to_ps(self.scenario.recovery_phase_start_ms)
            if self.scenario.recovery_phase_start_ms is not None
            else math.inf
        )
        self._recovery_recorded = False

    # ── Resource helpers ───────────────────────────────────────────────────────

    def _can_allocate(self, job_id: str) -> bool:
        release = self._release_by_id[job_id]
        cls = self.scenario.get_class(release.class_id)
        for name, units in cls.resource_requirements.items():
            resource = self.compiled.resources.get(name)
            if resource is None:
                return False
            if resource.capacity - self._resource_used[name] < units:
                return False
        return True

    def _allocate(self, job_id: str) -> None:
        release = self._release_by_id[job_id]
        cls = self.scenario.get_class(release.class_id)
        for name, units in cls.resource_requirements.items():
            self._resource_used[name] += units

    def _release_resources(self, job_id: str) -> None:
        release = self._release_by_id[job_id]
        cls = self.scenario.get_class(release.class_id)
        for name, units in cls.resource_requirements.items():
            self._resource_used[name] = max(0, self._resource_used[name] - units)

    # ── Policy helpers ─────────────────────────────────────────────────────────

    def _policy_key(self, job_id: str) -> Tuple[int, int, int, str]:
        release = self._release_by_id[job_id]
        cls = self.scenario.get_class(release.class_id)
        return policy_key(
            cls.priority,
            release.deadline_ps,
            release.arrival_ps,
            job_id,
        )

    def _class_for(self, job_id: str) -> WorkloadClass:
        return self.scenario.get_class(self._release_by_id[job_id].class_id)

    # ── Scheduling arrival events ──────────────────────────────────────────────

    def _schedule_arrivals(self) -> None:
        """Schedule arrival events as DISPATCH events so the kernel sees progress."""
        for release in self.compiled.releases:
            self.kernel.schedule(
                release.arrival_ps,
                EventPhase.DISPATCH,
                lambda k, r=release: self._on_arrival_and_dispatch(r),
            )
        self.kernel.schedule(0, EventPhase.DISPATCH, lambda k: self._dispatch(k))

    def _on_arrival(self, release: JobRelease) -> None:
        """Handle a job arrival: admission, queueing, and metrics."""
        cls = self.scenario.get_class(release.class_id)
        self.collector.record_arrival(
            job_id=release.job_id,
            class_id=release.class_id,
            arrival_ps=release.arrival_ps,
            deadline_ps=release.deadline_ps,
            is_warmup=release.is_warmup,
            is_measurement=release.is_measurement,
            work_ps=release.work_ps,
            resource_requirements=cls.resource_requirements,
        )

        # Admission check (stateless)
        if not cls.admission_excluded:
            lower_priority_holders = self._lower_priority_holders(cls.priority)
            result = self.compiled.admission.check(
                job_id=release.job_id,
                memory_bytes=cls.memory_bytes,
                bandwidth_fraction=cls.bandwidth_fraction,
                priority=cls.priority,
                lower_priority_holders=lower_priority_holders,
            )
            if not result.admitted:
                self.collector.record_reject(release.job_id)
                self._maybe_record_recovery()
                self._schedule_dispatch()
                return

        # Queue policy
        if cls.queue_policy == QueuePolicy.FIFO:
            q = self._fifo_queues[f"fifo:{cls.id}"]
            try:
                q.enqueue(release.job_id)
                self.collector.record_queue_size(q.name, q.size, self.kernel.now_ps)
            except QueueFullError:
                self.collector.record_drop(release.job_id)
                self._maybe_record_recovery()
                self._schedule_dispatch()
                return
        elif cls.queue_policy == QueuePolicy.MAILBOX_LATEST:
            stream = cls.stream_id or cls.id
            mb = self._mailboxes[f"mailbox:{stream}"]
            old = mb.peek(stream, self.kernel.now_ps)
            replaced = mb.put(stream, release.job_id, self.kernel.now_ps)
            if replaced and old is not None:
                old_job_id = old[1]
                self.collector.record_replace(old_job_id)
                if old_job_id in self._admitted:
                    self.compiled.admission.release(old_job_id)
                    self._admitted.discard(old_job_id)
                if old_job_id in self._running:
                    self._preempt_job(old_job_id)
                elif old_job_id in self._preempted:
                    self._preempted.discard(old_job_id)
                    self._preempted_remaining.pop(old_job_id, None)

        # Commit admission reservation now that the job is queued.
        if not cls.admission_excluded:
            lower_priority_holders = self._lower_priority_holders(cls.priority)
            self.compiled.admission.admit(
                job_id=release.job_id,
                memory_bytes=cls.memory_bytes,
                bandwidth_fraction=cls.bandwidth_fraction,
                priority=cls.priority,
                lower_priority_holders=lower_priority_holders,
            )
            self._admitted.add(release.job_id)

        self._maybe_record_recovery()
        self._schedule_dispatch()

    def _lower_priority_holders(self, priority: int) -> Dict[str, int]:
        """Return resources held by running jobs with lower priority."""
        holders: Dict[str, int] = {}
        for job_id in self._running:
            cls = self._class_for(job_id)
            if cls.priority < priority:
                for name in cls.resource_requirements:
                    holders[name] = min(
                        holders.get(name, math.inf),  # type: ignore[arg-type]
                        cls.priority,
                    )
        return holders

    def _on_arrival_and_dispatch(self, release: JobRelease) -> None:
        """Handle an arrival and then run the dispatcher."""
        self._on_arrival(release)
        self._dispatch(self.kernel)

    def _schedule_dispatch(self) -> None:
        """Schedule a dispatch event at the current time."""
        self.kernel.schedule(
            self.kernel.now_ps, EventPhase.DISPATCH, lambda k: self._dispatch(k)
        )

    # ── Dispatch ───────────────────────────────────────────────────────────────

    def _dispatch(self, kernel: DiscreteEventKernel, ready: list | None = None) -> None:
        """Dispatch ready jobs to resources, with optional preemption."""
        candidates = self._collect_candidates()
        if not candidates:
            self._maybe_record_recovery()
            return

        candidates.sort(key=lambda c: self._policy_key(c.job_id))
        seen: Set[str] = set()
        for candidate in candidates:
            if candidate.job_id in seen or candidate.job_id in self._running:
                continue
            seen.add(candidate.job_id)
            if self._try_start(candidate.job_id):
                continue
            if not self.scenario.preemption_enabled:
                continue
            if self._preempt_for(candidate.job_id):
                self._try_start(candidate.job_id)

        self._maybe_record_recovery()

    def _collect_candidates(self) -> List[_Candidate]:
        """Collect dispatch candidates from preempted, FIFO, and mailbox sources."""
        candidates: List[_Candidate] = []

        for job_id in list(self._preempted):
            if job_id not in self._running:
                candidates.append(_Candidate(job_id, "preempted"))

        for q in self._fifo_queues.values():
            head = q.peek()
            if head is not None:
                candidates.append(_Candidate(head[1], "fifo"))

        for mb_key, mb in self._mailboxes.items():
            stream = mb_key.split(":", 1)[1]
            pending = mb.peek(stream, self.kernel.now_ps)
            if pending is not None:
                candidates.append(_Candidate(pending[1], "mailbox"))

        return candidates

    def _try_start(self, job_id: str) -> bool:
        """Start ``job_id`` if resources are available."""
        if not self._can_allocate(job_id):
            return False
        self._start_job(job_id)
        return True

    def _start_job(self, job_id: str) -> None:
        """Allocate resources and begin/resume executing a job."""
        release = self._release_by_id[job_id]
        cls = self.scenario.get_class(release.class_id)
        self._allocate(job_id)
        self._preempted.discard(job_id)

        # Remove from runner queue if present (FIFO head / mailbox pending).
        self._dequeue_from_runner(job_id)

        remaining = self._preempted_remaining.pop(job_id, release.work_ps)
        if job_id in self.kernel.jobs:
            kernel_job = self.kernel.jobs[job_id]
            kernel_job.state = JobState.RUNNING
            kernel_job.remaining_ps = remaining
        else:
            rel_deadline = max(0, release.deadline_ps - self.kernel.now_ps)
            self.kernel.register_job(
                job_id=job_id,
                work_ps=remaining,
                release_ps=self.kernel.now_ps,
                relative_deadline_ps=rel_deadline,
                priority_class=cls.priority,
            )
            self.kernel.jobs[job_id].state = JobState.RUNNING

        self._start_times[job_id] = self.kernel.now_ps
        self.collector.record_start(job_id, self.kernel.now_ps)

        completion = self.kernel.schedule(
            self.kernel.now_ps + remaining,
            EventPhase.TIMER,
            lambda k, jid=job_id: self._on_complete(jid),
        )
        self._running[job_id] = completion

    def _dequeue_from_runner(self, job_id: str) -> None:
        """Remove a job from its runner queue/mailbox if present."""
        cls = self._class_for(job_id)
        if cls.queue_policy == QueuePolicy.FIFO:
            q = self._fifo_queues.get(f"fifo:{cls.id}")
            if q is not None and not q.is_empty and q.peek()[1] == job_id:
                q.dequeue()
        elif cls.queue_policy == QueuePolicy.MAILBOX_LATEST:
            stream = cls.stream_id or cls.id
            mb = self._mailboxes.get(f"mailbox:{stream}")
            if mb is not None and mb.has_pending(stream):
                age_item = mb.take(stream, self.kernel.now_ps)
                if age_item is not None:
                    self.collector.record_observation_age(job_id, age_item[0])

    def _preempt_for(self, job_id: str) -> bool:
        """Preempt lower-priority running jobs until ``job_id`` can start.

        Returns True if at least one running job was preempted.
        """
        candidate_key = self._policy_key(job_id)
        running = list(self._running.keys())
        running.sort(key=lambda jid: self._policy_key(jid), reverse=True)

        preempted_any = False
        for victim in running:
            if self._can_allocate(job_id):
                break
            victim_key = self._policy_key(victim)
            if candidate_key < victim_key:
                self._preempt_job(victim)
                preempted_any = True
        return preempted_any

    def _preempt_job(self, job_id: str) -> None:
        """Move a running job back to the preempted set."""
        if job_id not in self._running:
            return
        event = self._running.pop(job_id)
        self.kernel.cancel_event(event)
        kernel_job = self.kernel.jobs.get(job_id)
        if kernel_job is not None and kernel_job.state == JobState.RUNNING:
            # Hide the job from the kernel's ready set until the runner resumes
            # it; this avoids no-op DISPATCH events seeing a ready job.
            kernel_job.state = JobState.PENDING
            kernel_job.release_ps = ms_to_ps(self.scenario.max_simulation_time_ms) + 1
        start_ps = self._start_times.pop(job_id, self.kernel.now_ps)
        self.collector.record_resource_busy(
            "compute", start_ps, self.kernel.now_ps, class_id=self._class_for(job_id).id
        )
        remaining = kernel_job.remaining_ps if kernel_job is not None else 0
        self._preempted_remaining[job_id] = remaining
        self._release_resources(job_id)
        self._preempted.add(job_id)

        # Put FIFO-preempted jobs back at the head of their queue.
        cls = self._class_for(job_id)
        if cls.queue_policy == QueuePolicy.FIFO:
            q = self._fifo_queues.get(f"fifo:{cls.id}")
            if q is not None:
                q._items.insert(0, (q._sequence, job_id))
                q._sequence += 1

    # ── Completion ─────────────────────────────────────────────────────────────

    def _on_complete(self, job_id: str) -> None:
        """Handle a job completion."""
        if job_id not in self._running:
            return
        self._running.pop(job_id)
        try:
            self.kernel.complete_job(job_id)
        except SchedulerError:
            pass
        start_ps = self._start_times.pop(job_id, self.kernel.now_ps)
        self.collector.record_resource_busy(
            "compute", start_ps, self.kernel.now_ps, class_id=self._class_for(job_id).id
        )
        self.collector.record_complete(job_id, self.kernel.now_ps)
        self._release_resources(job_id)
        if job_id in self._admitted:
            self.compiled.admission.release(job_id)
            self._admitted.discard(job_id)
        self._maybe_record_recovery()
        self._schedule_dispatch()

    # ── Recovery tracking ──────────────────────────────────────────────────────

    def _backlog_count(self) -> int:
        """Return current number of queued or running jobs."""
        queued = sum(q.size for q in self._fifo_queues.values())
        pending_mailbox = sum(
            1 for stream, mb in self._mailboxes.items() if mb.has_pending(stream)
        )
        return queued + pending_mailbox + len(self._running) + len(self._preempted)

    def _maybe_record_recovery(self) -> None:
        """Record recovery time if backlog drains after recovery phase start."""
        if self._recovery_recorded:
            return
        if self.kernel.now_ps < self._recovery_phase_start_ps:
            return
        if self._backlog_count() == 0:
            self.collector.record_recovery_time(self.kernel.now_ps)
            self._recovery_recorded = True

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self) -> ScenarioMetrics:
        """Run the compiled scenario and return metrics."""
        self._schedule_arrivals()
        max_time_ps = ms_to_ps(self.scenario.max_simulation_time_ms)

        while True:
            ev = self.kernel.queue.peek()
            if ev is None or ev.time_ps > max_time_ps:
                break
            self.kernel.run_until(end_time_ps=max_time_ps, max_events=1000)

        # Ensure any jobs still in queues are counted appropriately.
        for q in self._fifo_queues.values():
            while not q.is_empty:
                seq, job_id = q.dequeue()
                self.collector.record_drop(job_id)

        return self.collector.compute(
            window_start_ps=self.compiled.window_start_ps,
            window_end_ps=self.compiled.window_end_ps,
            compute_capacity=self.scenario.compute_capacity,
        )


def run_scenario(compiled: CompiledScenario) -> ScenarioMetrics:
    """Convenience wrapper: create a runner and run it."""
    return ScenarioRunner(compiled).run()


__all__ = [
    "ScenarioRunner",
    "run_scenario",
]
