"""Tests for the discrete-event scheduler kernel."""

from __future__ import annotations

import math

import pytest

from contracts.errors import ConfigError
from scheduler.events import EventPhase
from scheduler.kernel import DiscreteEventKernel, JobState, SchedulerError
from scheduler.policies import PriorityClass, SchedulePolicy, edf_release_tie_break


def _noop_dispatch(kernel: DiscreteEventKernel, ready: list) -> None:
    pass


class TestCycleConversion:
    def test_cycles_to_ps_at_1000mhz(self) -> None:
        kernel = DiscreteEventKernel(1000)
        # 1 cycle = 1 ns = 1000 ps at 1000 MHz
        assert kernel.cycles_to_ps(1) == 1000
        assert kernel.cycles_to_ps(10) == 10000

    def test_cycles_to_ps_at_1200mhz(self) -> None:
        kernel = DiscreteEventKernel(1200)
        # 1 cycle = ceil(1_000_000 / 1200) = 834 ps
        assert kernel.cycles_to_ps(1) == 834

    def test_cycles_to_ps_ceiling(self) -> None:
        kernel = DiscreteEventKernel(1200)
        # 5 cycles = ceil(5 * 833.333...) = 4167 ps
        assert kernel.cycles_to_ps(5) == 4167

    def test_zero_cycles_zero_ps(self) -> None:
        kernel = DiscreteEventKernel(1000)
        assert kernel.cycles_to_ps(0) == 0

    def test_negative_cycles_rejected(self) -> None:
        kernel = DiscreteEventKernel(1000)
        with pytest.raises(SchedulerError):
            kernel.cycles_to_ps(-1)

    def test_invalid_frequency_rejected(self) -> None:
        with pytest.raises(ConfigError):
            DiscreteEventKernel(0)


class TestJobLifecycle:
    def test_register_job_and_release(self) -> None:
        kernel = DiscreteEventKernel(1000)
        job = kernel.register_job("j1", work_ps=10_000, release_ps=5_000)
        assert job.state == JobState.PENDING
        assert job.release_ps == 5_000
        assert job.deadline_ps == 15_000

        kernel.run_until(end_time_ps=5_000)
        assert kernel.now_ps == 5_000
        assert job.state == JobState.READY

    def test_register_at_time_zero_becomes_ready(self) -> None:
        kernel = DiscreteEventKernel(1000)
        job = kernel.register_job("j1", work_ps=5_000)
        assert job.state == JobState.READY

    def test_duplicate_job_id_rejected(self) -> None:
        kernel = DiscreteEventKernel(1000)
        kernel.register_job("j1", work_ps=100)
        with pytest.raises(ConfigError):
            kernel.register_job("j1", work_ps=100)

    def test_negative_work_rejected(self) -> None:
        kernel = DiscreteEventKernel(1000)
        with pytest.raises(SchedulerError):
            kernel.register_job("j1", work_ps=-1)

    def test_release_in_past_rejected(self) -> None:
        kernel = DiscreteEventKernel(1000)
        kernel.register_job("j0", work_ps=100, release_ps=1_000)
        kernel.run_until(end_time_ps=1_000)
        with pytest.raises(SchedulerError):
            kernel.register_job("j1", work_ps=100, release_ps=0)

    def test_consume_work_and_complete(self) -> None:
        kernel = DiscreteEventKernel(1000)
        job = kernel.register_job("j1", work_ps=5_000)
        job.state = JobState.RUNNING

        kernel.consume_work("j1", 2_000)
        assert job.remaining_ps == 3_000

        kernel.consume_work("j1", 3_000)
        assert job.remaining_ps == 0

        kernel.complete_job("j1")
        assert job.state == JobState.COMPLETED
        assert job in kernel.completed

    def test_cannot_consume_from_ready_job(self) -> None:
        kernel = DiscreteEventKernel(1000)
        kernel.register_job("j1", work_ps=5_000)
        with pytest.raises(SchedulerError):
            kernel.consume_work("j1", 1_000)

    def test_over_consume_rejected(self) -> None:
        kernel = DiscreteEventKernel(1000)
        job = kernel.register_job("j1", work_ps=1_000)
        job.state = JobState.RUNNING
        with pytest.raises(SchedulerError):
            kernel.consume_work("j1", 2_000)

    def test_reject_job(self) -> None:
        kernel = DiscreteEventKernel(1000)
        job = kernel.register_job("j1", work_ps=1_000)
        kernel.reject_job("j1")
        assert job.state == JobState.REJECTED
        assert job in kernel.rejected


class TestDispatchAndCompletion:
    def test_dispatch_callback_runs(self) -> None:
        called = []

        def dispatch(kernel: DiscreteEventKernel, ready: list) -> None:
            called.append(len(ready))
            for job in ready:
                job.state = JobState.RUNNING
                kernel.complete_job(job.job_id)

        kernel = DiscreteEventKernel(1000, dispatch_callback=dispatch)
        kernel.register_job("j1", work_ps=5_000)
        kernel.run_until_complete()

        assert called == [1]
        assert kernel.jobs["j1"].state == JobState.COMPLETED

    def test_run_to_completion(self) -> None:
        completed_times = []

        def dispatch(kernel: DiscreteEventKernel, ready: list) -> None:
            for job in ready:
                job.state = JobState.RUNNING
                finish = kernel.now_ps + job.remaining_ps
                completed_times.append((job.job_id, finish))
                kernel.schedule(
                    finish,
                    EventPhase.TIMER,
                    lambda k, jid=job.job_id: k.complete_job(jid),
                )

        kernel = DiscreteEventKernel(1000, dispatch_callback=dispatch)
        kernel.register_job("j1", work_ps=5_000)
        kernel.run_until_complete()

        assert completed_times == [("j1", 5_000)]
        assert kernel.now_ps == 5_000

    def test_multiple_jobs_finish_in_order(self) -> None:
        finishes = []

        def dispatch(kernel: DiscreteEventKernel, ready: list) -> None:
            for job in ready:
                if job.state == JobState.READY:
                    job.state = JobState.RUNNING
                    kernel.schedule(
                        kernel.now_ps + job.remaining_ps,
                        EventPhase.TIMER,
                        lambda k, jid=job.job_id: (finishes.append((jid, k.now_ps)), k.complete_job(jid)),
                    )

        kernel = DiscreteEventKernel(1000, dispatch_callback=dispatch)
        kernel.register_job("short", work_ps=3_000)
        kernel.register_job("long", work_ps=7_000, release_ps=1_000)
        kernel.run_until_complete()

        # Dispatcher starts every ready job immediately, so they overlap.
        assert finishes == [("short", 3_000), ("long", 8_000)]


class TestLivelockPrevention:
    def test_zero_time_no_progress_raises(self) -> None:
        def bad_dispatch(kernel: DiscreteEventKernel, ready: list) -> None:
            pass  # no progress

        kernel = DiscreteEventKernel(1000, dispatch_callback=bad_dispatch)
        kernel.register_job("j1", work_ps=5_000)
        with pytest.raises(SchedulerError, match="zero-time livelock"):
            kernel.run_until_complete()

    def test_event_advancing_time_is_progress(self) -> None:
        kernel = DiscreteEventKernel(1000)
        kernel.schedule(1_000, EventPhase.TIMER, lambda k: None)
        kernel.run_until(end_time_ps=1_000)  # advances time, OK
        assert kernel.now_ps == 1_000

    def test_scheduling_past_time_rejected(self) -> None:
        kernel = DiscreteEventKernel(1000)
        kernel.schedule(1_000, EventPhase.TIMER, lambda k: None)
        kernel.run_until(end_time_ps=1_000)
        with pytest.raises(SchedulerError):
            kernel.schedule(500, EventPhase.TIMER, lambda k: None)


class TestStableJobIds:
    def test_shuffle_order_same_results(self) -> None:
        def dispatch(kernel: DiscreteEventKernel, ready: list) -> None:
            if kernel.running_jobs():
                return  # serial dispatcher: one job at a time
            if ready:
                job = ready[0]
                job.state = JobState.RUNNING
                kernel.schedule(
                    kernel.now_ps + job.remaining_ps,
                    EventPhase.TIMER,
                    lambda k, jid=job.job_id: k.complete_job(jid),
                )

        def run(order: tuple[str, ...]) -> dict[str, int]:
            kernel = DiscreteEventKernel(1000, dispatch_callback=dispatch)
            for jid in order:
                kernel.register_job(jid, work_ps=5_000)
            kernel.run_until_complete()
            return {j.job_id: kernel.now_ps for j in kernel.completed}

        r1 = run(("a", "b"))
        r2 = run(("b", "a"))
        assert r1 == r2


class TestDependencies:
    def test_self_dependency_rejected(self) -> None:
        kernel = DiscreteEventKernel(1000)
        with pytest.raises(SchedulerError):
            kernel.register_job("j1", work_ps=100, depends_on={"j1"})

    def test_cyclic_dependency_rejected(self) -> None:
        kernel = DiscreteEventKernel(1000)
        kernel.register_job("a", work_ps=100, depends_on={"b"})
        with pytest.raises(SchedulerError):
            kernel.register_job("b", work_ps=100, depends_on={"a"})

    def test_dependency_waits_for_completion(self) -> None:
        finishes = []

        def dispatch(kernel: DiscreteEventKernel, ready: list) -> None:
            if ready:
                job = ready[0]
                job.state = JobState.RUNNING
                kernel.schedule(
                    kernel.now_ps + job.remaining_ps,
                    EventPhase.TIMER,
                    lambda k, jid=job.job_id: (finishes.append((jid, k.now_ps)), k.complete_job(jid)),
                )

        kernel = DiscreteEventKernel(1000, dispatch_callback=dispatch)
        kernel.register_job("parent", work_ps=3_000)
        kernel.register_job("child", work_ps=2_000, depends_on={"parent"})
        kernel.run_until_complete()

        assert ("parent", 3_000) in finishes
        assert ("child", 5_000) in finishes


class TestPreemption:
    def test_priority_preempts_running_job(self) -> None:
        log = []
        preempted: dict[str, int] = {}
        completion_events: dict[str, "Event"] = {}

        def dispatch(kernel: DiscreteEventKernel, ready: list) -> None:
            ordered = edf_release_tie_break(ready + kernel.running_jobs())
            if not ordered:
                return
            chosen = ordered[0]
            running = kernel.running_jobs()
            for job in running:
                if job.job_id != chosen.job_id:
                    job.state = JobState.READY
                    preempted[job.job_id] = job.remaining_ps
                    if job.job_id in completion_events:
                        kernel.cancel_event(completion_events.pop(job.job_id))
                    log.append(("preempt", kernel.now_ps, job.job_id))
            if chosen.state == JobState.READY:
                chosen.state = JobState.RUNNING
            if chosen.state == JobState.RUNNING and chosen.remaining_ps > 0:
                # Run until next preemption point or completion.
                run_for = chosen.remaining_ps
                completion_events[chosen.job_id] = kernel.schedule(
                    kernel.now_ps + run_for,
                    EventPhase.TIMER,
                    lambda k, jid=chosen.job_id: (
                        log.append(("complete", k.now_ps, jid)),
                        k.complete_job(jid),
                    ),
                )

        kernel = DiscreteEventKernel(1000, dispatch_callback=dispatch)
        low_pc = PriorityClass("be", 0)
        high_pc = PriorityClass("critical", 1)
        kernel.register_job("low", work_ps=10_000_000, priority_class=low_pc.level)
        kernel.register_job(
            "high",
            work_ps=2_000_000,
            release_ps=3_000_000,
            priority_class=high_pc.level,
        )
        kernel.run_until_complete()

        # high arrives at 3us, preempts low, finishes at 5us; low resumes and
        # finishes at 5us + remaining 7us = 12us.
        assert ("complete", 5_000_000, "high") in log
        assert ("complete", 12_000_000, "low") in log
