"""Scheduling policies: service-class priority + intra-class EDF.

The canonical ordering key is:

  1. higher service-class priority first
  2. earlier absolute deadline first (EDF)
  3. earlier release time first
  4. stable job_id for deterministic tie-breaking

``relative_deadline_ps`` plus ``release_ps`` produces the absolute deadline.
A job is feasible at time ``t`` when ``t <= deadline_ps`` (pass-at-exactly-deadline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PriorityClass:
    """Service class with a priority level.

    Higher ``level`` means more urgent.  Negative levels are allowed for
    background traffic.
    """

    name: str
    level: int


@dataclass(frozen=True)
class SchedulePolicy:
    """Policy attached to a job."""

    priority_class: PriorityClass
    relative_deadline_ps: int

    def absolute_deadline(self, release_ps: int) -> int:
        return release_ps + self.relative_deadline_ps


def policy_key(
    priority_class_level: int,
    absolute_deadline_ps: int,
    release_ps: int,
    job_id: str,
) -> tuple[int, int, int, str]:
    """Return the canonical ordering key for a job.

    Lower tuple = scheduled earlier.
    """
    return (
        -priority_class_level,  # higher level first
        absolute_deadline_ps,   # earliest deadline first
        release_ps,             # earliest release first
        job_id,                 # stable deterministic tie-break
    )


def _priority_level(priority_class: Any) -> int:
    """Extract priority level from a PriorityClass or raw int."""
    if isinstance(priority_class, PriorityClass):
        return int(priority_class.level)
    return int(priority_class)


def edf_release_tie_break(jobs: list[Any]) -> list[Any]:
    """Sort ``jobs`` by service class, EDF, release time, and job ID.

    Each job is expected to expose ``priority_class`` (a ``PriorityClass``
    or an int level), ``release_ps``, ``deadline_ps``, and ``job_id``.
    """

    def _key(job: Any) -> tuple[int, int, int, str]:
        level = _priority_level(job.priority_class)
        deadline = int(job.deadline_ps)
        release = int(job.release_ps)
        jid = str(job.job_id)
        return policy_key(level, deadline, release, jid)

    return sorted(jobs, key=_key)


def deadline_missed_at(now_ps: int, deadline_ps: int) -> bool:
    """Return True iff ``now_ps`` is strictly past ``deadline_ps``."""
    return now_ps > deadline_ps


def feasible_at(now_ps: int, deadline_ps: int) -> bool:
    """Return True iff the job passes its deadline at ``now_ps``."""
    return now_ps <= deadline_ps
