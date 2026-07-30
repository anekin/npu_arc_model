"""Admission control for the scheduler.

Checks four independent dimensions before a job is admitted:
  1. memory: required resident bytes <= available bytes
  2. context/inflight: current inflight jobs < max inflight
  3. peak bandwidth fraction: sum of requested fractions <= 1.0
  4. lower-priority blocking: required exclusive resources are not held by
     lower-priority jobs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AdmissionResult:
    """Result of an admission check."""

    admitted: bool
    reason: str = ""
    checks: dict[str, bool] = field(default_factory=dict)


class AdmissionController:
    """Stateful admission controller."""

    def __init__(
        self,
        memory_available_bytes: int,
        max_inflight_jobs: int,
        max_bandwidth_fraction: float = 1.0,
    ) -> None:
        if memory_available_bytes < 0:
            raise ValueError(f"memory_available_bytes must be non-negative, got {memory_available_bytes}")
        if max_inflight_jobs <= 0:
            raise ValueError(f"max_inflight_jobs must be positive, got {max_inflight_jobs}")
        if not 0.0 < max_bandwidth_fraction <= 1.0:
            raise ValueError(f"max_bandwidth_fraction must be in (0, 1], got {max_bandwidth_fraction}")
        self.memory_available_bytes = memory_available_bytes
        self.max_inflight_jobs = max_inflight_jobs
        self.max_bandwidth_fraction = max_bandwidth_fraction
        self._memory_used_bytes: int = 0
        self._inflight: dict[str, dict[str, Any]] = {}
        self._bandwidth_fraction_used: float = 0.0

    @property
    def memory_used_bytes(self) -> int:
        return self._memory_used_bytes

    @property
    def memory_free_bytes(self) -> int:
        return self.memory_available_bytes - self._memory_used_bytes

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    @property
    def bandwidth_fraction_used(self) -> float:
        return self._bandwidth_fraction_used

    def check(
        self,
        job_id: str,
        memory_bytes: int = 0,
        bandwidth_fraction: float = 0.0,
        priority: int = 0,
        exclusive_resources: set[str] | None = None,
        lower_priority_holders: dict[str, int] | None = None,
    ) -> AdmissionResult:
        """Return an admission result without modifying state."""
        exclusive_resources = exclusive_resources or set()
        lower_priority_holders = lower_priority_holders or {}

        memory_ok = self.memory_free_bytes >= memory_bytes
        inflight_ok = self.inflight_count < self.max_inflight_jobs
        bw_ok = self._bandwidth_fraction_used + bandwidth_fraction <= self.max_bandwidth_fraction + 1e-12

        # A higher-priority job is blocked if any required exclusive resource
        # is held by a job with strictly lower priority.
        blocking = any(lower_priority_holders.get(res, float("inf")) < priority for res in exclusive_resources)
        blocking_ok = not blocking

        checks = {
            "memory": memory_ok,
            "inflight": inflight_ok,
            "bandwidth": bw_ok,
            "lower_priority_blocking": blocking_ok,
        }
        admitted = all(checks.values())

        reason = ""
        if not admitted:
            failed = [name for name, ok in checks.items() if not ok]
            reason = f"admission rejected for {job_id}: {', '.join(failed)}"

        return AdmissionResult(admitted=admitted, reason=reason, checks=checks)

    def admit(
        self,
        job_id: str,
        memory_bytes: int = 0,
        bandwidth_fraction: float = 0.0,
        priority: int = 0,
        exclusive_resources: set[str] | None = None,
        lower_priority_holders: dict[str, int] | None = None,
    ) -> AdmissionResult:
        """Admit ``job_id`` if all checks pass and record the reservation."""
        result = self.check(
            job_id=job_id,
            memory_bytes=memory_bytes,
            bandwidth_fraction=bandwidth_fraction,
            priority=priority,
            exclusive_resources=exclusive_resources,
            lower_priority_holders=lower_priority_holders,
        )
        if not result.admitted:
            return result

        self._memory_used_bytes += memory_bytes
        self._bandwidth_fraction_used += bandwidth_fraction
        self._inflight[job_id] = {
            "memory_bytes": memory_bytes,
            "bandwidth_fraction": bandwidth_fraction,
            "priority": priority,
            "exclusive_resources": set(exclusive_resources or ()),
        }
        return AdmissionResult(admitted=True, checks=result.checks)

    def release(self, job_id: str) -> bool:
        """Release the reservation for ``job_id``.  Return True if found."""
        info = self._inflight.pop(job_id, None)
        if info is None:
            return False
        self._memory_used_bytes -= info["memory_bytes"]
        self._bandwidth_fraction_used -= info["bandwidth_fraction"]
        return True

    def is_admitted(self, job_id: str) -> bool:
        return job_id in self._inflight
