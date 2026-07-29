"""Tests for scheduling policies."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from scheduler.policies import (
    PriorityClass,
    SchedulePolicy,
    deadline_missed_at,
    edf_release_tie_break,
    feasible_at,
    policy_key,
)


@dataclass
class _StubJob:
    job_id: str
    priority_class: PriorityClass
    release_ps: int
    deadline_ps: int


class TestPolicyKey:
    def test_service_class_dominates_deadline(self) -> None:
        high = policy_key(5, 100, 0, "a")
        low = policy_key(1, 50, 0, "b")
        assert high < low

    def test_edf_within_class(self) -> None:
        early = policy_key(0, 50, 0, "a")
        late = policy_key(0, 100, 0, "b")
        assert early < late

    def test_release_tie_break(self) -> None:
        first = policy_key(0, 100, 10, "a")
        second = policy_key(0, 100, 20, "b")
        assert first < second

    def test_job_id_tie_break(self) -> None:
        a = policy_key(0, 100, 10, "a")
        b = policy_key(0, 100, 10, "b")
        assert a < b


class TestEdfReleaseTieBreak:
    def test_sort_by_class_then_deadline(self) -> None:
        critical = PriorityClass("critical", 2)
        best_effort = PriorityClass("be", 0)
        jobs = [
            _StubJob("late_critical", critical, 0, 200),
            _StubJob("early_be", best_effort, 0, 50),
            _StubJob("early_critical", critical, 0, 100),
        ]
        ordered = edf_release_tie_break(jobs)
        assert [j.job_id for j in ordered] == [
            "early_critical",
            "late_critical",
            "early_be",
        ]

    def test_release_time_tie_break(self) -> None:
        pc = PriorityClass("be", 0)
        jobs = [
            _StubJob("b", pc, 20, 100),
            _StubJob("a", pc, 10, 100),
        ]
        ordered = edf_release_tie_break(jobs)
        assert [j.job_id for j in ordered] == ["a", "b"]

    def test_job_id_tie_break(self) -> None:
        pc = PriorityClass("be", 0)
        jobs = [
            _StubJob("z", pc, 0, 100),
            _StubJob("a", pc, 0, 100),
        ]
        ordered = edf_release_tie_break(jobs)
        assert [j.job_id for j in ordered] == ["a", "z"]


class TestAbsoluteDeadline:
    def test_deadline_from_release_and_relative(self) -> None:
        pc = PriorityClass("rt", 1)
        policy = SchedulePolicy(pc, relative_deadline_ps=1_000)
        assert policy.absolute_deadline(5_000) == 6_000

    def test_pass_at_exactly_deadline(self) -> None:
        assert feasible_at(100, 100) is True
        assert deadline_missed_at(100, 100) is False

    def test_miss_after_deadline(self) -> None:
        assert feasible_at(101, 100) is False
        assert deadline_missed_at(101, 100) is True
