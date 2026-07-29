"""Scheduler resources: capacity, bounded ports/FIFO, and byte servers.

Resources are intentionally stateless-ish per request: a member's
completion time is recomputed deterministically whenever the active
membership changes.  This makes bandwidth sharing and strict-priority QoS
easy to audit by hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Literal

from contracts.errors import ConfigError


class ResourceError(RuntimeError):
    """Raised for resource allocation errors."""

    def __init__(self, message: str, resource: str = "") -> None:
        super().__init__(message)
        self.resource = resource


@dataclass
class CapacityResource:
    """A reusable capacity resource such as a compute engine or DMA channel.

    ``capacity`` units are available; each allocation consumes one or more
    units.  The resource is work-conserving: allocations may be partially
    satisfied if ``allow_partial`` is True.
    """

    name: str
    capacity: int = 1
    allow_partial: bool = False
    _allocated: Dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ResourceError(
                f"capacity resource {self.name!r} must have positive capacity, "
                f"got {self.capacity}",
                resource=self.name,
            )

    @property
    def used(self) -> int:
        return sum(self._allocated.values())

    @property
    def available(self) -> int:
        return self.capacity - self.used

    def allocate(self, member_id: str, units: int = 1) -> int:
        """Allocate ``units`` to ``member_id``.  Return actually allocated."""
        if units <= 0:
            raise ResourceError(
                f"allocate units must be positive, got {units}",
                resource=self.name,
            )
        available = self.available
        if available <= 0:
            return 0
        granted = min(units, available)
        if granted <= 0:
            return 0
        self._allocated[member_id] = self._allocated.get(member_id, 0) + granted
        return granted

    def release(self, member_id: str, units: int | None = None) -> int:
        """Release ``units`` from ``member_id``; default releases all."""
        held = self._allocated.get(member_id, 0)
        if units is None:
            units = held
        if units <= 0:
            return 0
        released = min(units, held)
        if released == held:
            del self._allocated[member_id]
        else:
            self._allocated[member_id] = held - released
        return released

    def release_all(self) -> int:
        """Release all allocations and return the freed units."""
        freed = self.used
        self._allocated.clear()
        return freed


@dataclass
class BoundedResource:
    """A bounded resource such as an SRAM port or FIFO slot.

    Unlike ``CapacityResource``, ``BoundedResource`` rejects requests that
    cannot be fully satisfied.
    """

    name: str
    capacity: int = 1
    _holders: Dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ResourceError(
                f"bounded resource {self.name!r} must have positive capacity, "
                f"got {self.capacity}",
                resource=self.name,
            )

    @property
    def used(self) -> int:
        return sum(self._holders.values())

    @property
    def available(self) -> int:
        return self.capacity - self.used

    def acquire(self, member_id: str, units: int = 1) -> bool:
        """Acquire ``units``; return True on success."""
        if units <= 0:
            raise ResourceError(
                f"acquire units must be positive, got {units}",
                resource=self.name,
            )
        if self.available < units:
            return False
        self._holders[member_id] = self._holders.get(member_id, 0) + units
        return True

    def release(self, member_id: str, units: int | None = None) -> int:
        """Release ``units`` from ``member_id``; default releases all."""
        held = self._holders.get(member_id, 0)
        if units is None:
            units = held
        if units <= 0:
            return 0
        released = min(units, held)
        if released == held:
            del self._holders[member_id]
        else:
            self._holders[member_id] = held - released
        return released


@dataclass
class ByteServer:
    """Work-conserving byte server for shared bandwidth resources.

    Models LPDDR/HBM/3D-DRAM/NoC as a server with a fixed total bandwidth
    (bytes per picosecond).  Active members either share bandwidth equally
    (``equal_share``) or are served in strict priority order
    (``strict_priority``).  Adding or removing a member triggers a
    deterministic recomputation of every active member's completion time.

    ``bandwidth_bytes_per_ps`` is the raw physical bandwidth.  QoS
    derating (if any) is applied by the caller.
    """

    name: str
    bandwidth_bytes_per_ps: float
    qos_mode: Literal["equal_share", "strict_priority"] = "equal_share"
    _members: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.bandwidth_bytes_per_ps <= 0:
            raise ResourceError(
                f"byte server {self.name!r} bandwidth must be positive, "
                f"got {self.bandwidth_bytes_per_ps}",
                resource=self.name,
            )
        if self.qos_mode not in ("equal_share", "strict_priority"):
            raise ConfigError(
                f"unknown qos_mode {self.qos_mode!r}",
                field_path=f"{self.name}.qos_mode",
            )

    @property
    def member_count(self) -> int:
        return len(self._members)

    @property
    def total_bytes_remaining(self) -> int:
        return sum(int(m["bytes_remaining"]) for m in self._members.values())

    def add_member(
        self,
        member_id: str,
        bytes_requested: int,
        priority: int = 0,
        now_ps: int = 0,
    ) -> int:
        """Add a new member request and return its absolute completion time."""
        if bytes_requested < 0:
            raise ResourceError(
                f"bytes_requested must be non-negative, got {bytes_requested}",
                resource=self.name,
            )
        self._members[member_id] = {
            "bytes_remaining": bytes_requested,
            "priority": priority,
        }
        self._recompute(now_ps)
        return self.completion_time_ps(member_id)

    def remove_member(self, member_id: str, now_ps: int = 0) -> None:
        """Remove a member and recompute remaining members."""
        if member_id in self._members:
            del self._members[member_id]
            self._recompute(now_ps)

    def update_bytes(
        self,
        member_id: str,
        bytes_remaining: int,
        now_ps: int = 0,
    ) -> int:
        """Update remaining bytes for a member and return new completion time."""
        if member_id not in self._members:
            raise ResourceError(
                f"unknown member {member_id!r}",
                resource=self.name,
            )
        if bytes_remaining < 0:
            raise ResourceError(
                f"bytes_remaining must be non-negative, got {bytes_remaining}",
                resource=self.name,
            )
        self._members[member_id]["bytes_remaining"] = bytes_remaining
        self._recompute(now_ps)
        return self.completion_time_ps(member_id)

    def completion_time_ps(self, member_id: str) -> int:
        """Return absolute completion time in ps for ``member_id``."""
        if member_id not in self._members:
            raise ResourceError(
                f"unknown member {member_id!r}",
                resource=self.name,
            )
        return self._members[member_id]["completion_ps"]

    def _recompute(self, now_ps: int) -> None:
        """Recompute absolute completion times for all active members.

        The algorithm is deterministic:
          * equal_share: split bandwidth evenly among all members, ordered by
            member_id for stable tie-breaking.
          * strict_priority: serve highest priority first; ties broken by
            member_id.  Each member runs at full bandwidth until done, then
            bandwidth shifts to the next member.
        """
        if not self._members:
            return

        elapsed = 0
        if self.qos_mode == "equal_share":
            share = self.bandwidth_bytes_per_ps / len(self._members)
            if share <= 0:
                raise ResourceError(
                    f"equal-share bandwidth per member became non-positive",
                    resource=self.name,
                )
            for member_id in sorted(self._members):
                remaining = int(self._members[member_id]["bytes_remaining"])
                duration = math.ceil(remaining / share) if remaining > 0 else 0
                self._members[member_id]["completion_ps"] = now_ps + elapsed + duration
        else:
            # strict_priority: sort by (-priority, member_id)
            ordered = sorted(
                self._members,
                key=lambda mid: (-self._members[mid]["priority"], mid),
            )
            for member_id in ordered:
                remaining = int(self._members[member_id]["bytes_remaining"])
                duration = (
                    math.ceil(remaining / self.bandwidth_bytes_per_ps)
                    if remaining > 0 else 0
                )
                self._members[member_id]["completion_ps"] = now_ps + elapsed + duration
                elapsed += duration

    def clear(self) -> None:
        """Remove all members."""
        self._members.clear()
