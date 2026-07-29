"""Scenario schema for deterministic temporal simulation.

A ``Scenario`` describes one or more workload classes, each with a periodic or
explicit trace arrival pattern, service time, deadline, queue policy, and
resource requirements.  The schema is intentionally free of random
distributions; arrivals are either periodic or drawn from an explicit sorted
trace.

Measurement windows:
* ``warmup_count`` releases per class are excluded from metrics.
* ``measurement_count`` subsequent releases per class form the measurement
  window.
* After the last measurement arrival the runner drains all admitted jobs
  before finalizing metrics to avoid right-censor bias.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.errors import ConfigError
from scheduler.metrics import ms_to_ps


class QueuePolicy(str, Enum):
    """Queueing policy for a workload class."""

    FIFO = "fifo"
    MAILBOX_LATEST = "mailbox_latest"


class ArrivalMode(str, Enum):
    """Arrival pattern mode."""

    PERIODIC = "periodic"
    TRACE = "trace"


class ArrivalPattern(BaseModel):
    """Arrival pattern for a workload class."""

    model_config = ConfigDict(extra="forbid")

    mode: ArrivalMode = Field(..., description="periodic or trace")
    period_ms: Optional[float] = Field(
        default=None,
        description="Period for periodic arrivals (ms)",
    )
    count: Optional[int] = Field(
        default=None,
        description="Number of releases for periodic arrivals",
    )
    offset_ms: float = Field(default=0.0, description="First release offset (ms)")
    releases_ms: List[float] = Field(
        default_factory=list,
        description="Sorted explicit release times for trace arrivals (ms)",
    )

    @field_validator("count")
    @classmethod
    def _count_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("count must be positive")
        return v

    @model_validator(mode="after")
    def _mode_consistency(self) -> "ArrivalPattern":
        if self.mode == ArrivalMode.PERIODIC:
            if self.period_ms is None or self.period_ms <= 0:
                raise ValueError("periodic mode requires positive period_ms")
            if self.count is None or self.count <= 0:
                raise ValueError("periodic mode requires positive count")
            if self.offset_ms < 0:
                raise ValueError("offset_ms must be non-negative")
        elif self.mode == ArrivalMode.TRACE:
            if not self.releases_ms:
                raise ValueError("trace mode requires non-empty releases_ms")
            if any(r < 0 for r in self.releases_ms):
                raise ValueError("release times must be non-negative")
            if self.releases_ms != sorted(self.releases_ms):
                raise ValueError("releases_ms must be sorted ascending")
        return self

    @property
    def period_ps(self) -> int:
        """Return the inter-arrival period in picoseconds for periodic patterns."""
        if self.mode != ArrivalMode.PERIODIC:
            raise ValueError("period_ps only defined for periodic arrivals")
        return ms_to_ps(self.period_ms)

    def release_times_ps(self) -> List[int]:
        """Return all release times in picoseconds."""
        if self.mode == ArrivalMode.PERIODIC:
            assert self.period_ms is not None and self.count is not None
            return [
                ms_to_ps(self.offset_ms + i * self.period_ms)
                for i in range(self.count)
            ]
        return [ms_to_ps(r) for r in self.releases_ms]


class WorkloadClass(BaseModel):
    """A single workload class within a scenario."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Stable class identifier")
    arrival: ArrivalPattern = Field(..., description="Arrival pattern")
    work_ms: float = Field(..., gt=0, description="Service time (ms)")
    relative_deadline_ms: Optional[float] = Field(
        default=None,
        description="Relative deadline (ms); defaults to work_ms",
    )
    timeout_ms: Optional[float] = Field(
        default=None,
        description="Hard timeout (ms); defaults to relative_deadline_ms",
    )
    priority: int = Field(default=0, description="Higher value = higher priority")
    queue_policy: QueuePolicy = Field(default=QueuePolicy.FIFO)
    queue_capacity: int = Field(default=1000, ge=1, description="Max queued items")
    stream_id: Optional[str] = Field(
        default=None,
        description="Stream identifier for mailbox_latest replacement",
    )
    resource_requirements: Dict[str, int] = Field(
        default_factory=lambda: {"compute": 1},
        description="Resource units required while running",
    )
    memory_bytes: int = Field(default=0, ge=0)
    bandwidth_fraction: float = Field(default=0.0, ge=0.0)
    admission_excluded: bool = Field(
        default=False,
        description="If True, bypass admission control (for baseline hand cases)",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("class id must be non-empty")
        return v

    @model_validator(mode="after")
    def _deadline_defaults(self) -> "WorkloadClass":
        if self.relative_deadline_ms is None:
            self.relative_deadline_ms = self.work_ms
        if self.timeout_ms is None:
            self.timeout_ms = self.relative_deadline_ms
        return self

    @property
    def work_ps(self) -> int:
        return ms_to_ps(self.work_ms)

    @property
    def relative_deadline_ps(self) -> int:
        return ms_to_ps(self.relative_deadline_ms or self.work_ms)

    @property
    def timeout_ps(self) -> int:
        return ms_to_ps(self.timeout_ms or self.relative_deadline_ms or self.work_ms)



class Scenario(BaseModel):
    """A deterministic temporal scenario."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Scenario identifier")
    description: str = Field(default="")
    seed: int = Field(default=0, description="Deterministic seed for future stochastic adapters")
    warmup_count: int = Field(default=10, ge=0, description="Releases per class to exclude")
    measurement_count: int = Field(default=1000, ge=1, description="Releases per class to measure")
    drain: bool = Field(default=True, description="Drain admitted jobs before finalizing metrics")
    max_simulation_time_ms: float = Field(
        default=1_000_000.0,
        description="Absolute simulation wall-clock limit (ms)",
    )
    recovery_phase_start_ms: Optional[float] = Field(
        default=None,
        description="Time after which backlog draining to zero is recorded as recovery (ms)",
    )

    workload_ref: Optional[str] = Field(
        default=None,
        description="Optional workload fixture name for memory/resource planning",
    )
    classes: List[WorkloadClass] = Field(..., min_length=1)

    # Global resources and admission
    compute_capacity: int = Field(default=1, ge=1)
    resources: Dict[str, int] = Field(
        default_factory=dict,
        description="Additional named resource capacities",
    )
    memory_available_bytes: int = Field(default=8_000_000_000, ge=0)
    max_inflight_jobs: int = Field(default=128, ge=1)
    max_bandwidth_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    preemption_enabled: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("scenario name must be non-empty")
        return v

    @model_validator(mode="after")
    def _unique_class_ids(self) -> "Scenario":
        ids = [c.id for c in self.classes]
        if len(ids) != len(set(ids)):
            raise ValueError("workload class ids must be unique")
        return self

    def get_class(self, class_id: str) -> WorkloadClass:
        """Look up a workload class by id."""
        for cls in self.classes:
            if cls.id == class_id:
                return cls
        raise ConfigError(f"unknown class {class_id!r}", field_path="classes")


__all__ = [
    "QueuePolicy",
    "ArrivalMode",
    "ArrivalPattern",
    "WorkloadClass",
    "Scenario",
]
