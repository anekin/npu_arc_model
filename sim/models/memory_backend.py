"""Replaceable parametric memory PPA/energy backend protocol.

Defines ``MemoryBackend`` as an abstract protocol (ABC) plus request/response
Pydantic models.  A backend accepts a memory topology, capacity, bandwidth,
access pattern and returns latency, area, leakage/dynamic energy, active power,
a thermal proxy, and a validity envelope.

The protocol is intentionally minimal so that a first-principles closed-form
backend (``Parametric3DMemoryBackend``), a Ramulator/DRAMSim adapter, or a pure
fake implementation can all satisfy the same contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from contracts.errors import ConfigError


class MemoryAccessPattern(BaseModel):
    """Description of the memory traffic pattern under evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    read_bytes: int = Field(..., ge=0, description="Total read bytes in the window")
    write_bytes: int = Field(..., ge=0, description="Total write bytes in the window")
    # read-write mix is derived from bytes; kept explicit for backends that use it.
    read_fraction: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of accesses that are reads by byte",
    )
    active_time_seconds: float = Field(
        default=1e-6,
        gt=0,
        description="Time window over which energy is averaged into active power",
    )

    @field_validator("active_time_seconds")
    @classmethod
    def _active_time_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("active_time_seconds must be positive")
        return v


class MemoryTopology(BaseModel):
    """Physical packaging/tier context for the memory under evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: Literal["on_chip_3d_dram", "hbm2e", "hbm3", "lpddr5", "lpddr5x", "sram"] = Field(
        ...,
        description="Canonical memory tier name",
    )
    process_node_nm: float = Field(default=12.0, gt=0, description="Memory process node in nm")
    stack_count: int = Field(default=1, ge=1, description="Number of memory die stacks")
    include_phy: bool = Field(
        default=True,
        description="Whether an external DRAM PHY is required (HBM/LPDDR)",
    )
    include_tsv: bool = Field(
        default=False,
        description="Whether TSV/interface area should be included",
    )
    include_package: bool = Field(
        default=True,
        description="Whether package/interposer area is required",
    )


class ValidityEnvelope(BaseModel):
    """Range over which the backend's output is calibrated/authoritative."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capacity_gb_min: float = Field(..., ge=0)
    capacity_gb_max: float = Field(..., ge=0)
    bandwidth_gbps_min: float = Field(..., ge=0)
    bandwidth_gbps_max: float = (...)
    read_bytes_max: int = Field(default=1 << 40, ge=0)
    write_bytes_max: int = Field(default=1 << 40, ge=0)
    trust_level: Literal["T0", "T1", "T2", "T3"] = Field(
        default="T0",
        description="Trust level of this estimate",
    )
    status: Literal["engineering_assumption", "calibrated_estimate", "authoritative"] = Field(
        default="engineering_assumption",
        description="Provenance status of this estimate",
    )
    reason: Optional[str] = Field(default=None, description="Reason for exploratory marking")

    @field_validator("bandwidth_gbps_max")
    @classmethod
    def _bandwidth_max_nonnegative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("bandwidth_gbps_max must be non-negative")
        return v


class MemoryRequest(BaseModel):
    """Input to a ``MemoryBackend``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    topology: MemoryTopology
    capacity_gb: float = Field(..., gt=0, description="Requested memory capacity in GB")
    bandwidth_gbps: float = Field(..., gt=0, description="Requested interface bandwidth in GB/s")
    access: MemoryAccessPattern

    @field_validator("capacity_gb", "bandwidth_gbps")
    @classmethod
    def _finite_positive(cls, v: float) -> float:
        import math

        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"must be finite and positive, got {v}")
        return v


class MemoryResponse(BaseModel):
    """Output of a ``MemoryBackend``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    latency_seconds: float = Field(..., ge=0, description="Average access latency")
    memory_die_area_mm2: float = Field(..., ge=0, description="Area of memory die(s)")
    interface_area_mm2: float = Field(..., ge=0, description="Area of TSV/PHY/interface")
    total_area_mm2: float = Field(..., ge=0, description="Total memory-related area")
    static_power_w: float = Field(..., ge=0, description="Leakage/static power")
    dynamic_energy_j: float = Field(..., ge=0, description="Energy for the access window")
    active_power_w: float = Field(..., ge=0, description="dynamic_energy_j / active_time")
    thermal_proxy_c: float = Field(..., ge=0, description="Proxy for thermal stress")
    validity: ValidityEnvelope
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of area/power/energy by component",
    )
    notes: List[str] = Field(default_factory=list, description="Human-readable caveats")


class MemoryBackend(ABC):
    """Abstract protocol for memory PPA/energy estimation.

    Implementations must be stateless with respect to a single request: calling
    ``estimate`` with the same ``MemoryRequest`` must return an equivalent
    ``MemoryResponse``.
    """

    @abstractmethod
    def estimate(self, request: MemoryRequest) -> MemoryResponse:
        """Return PPA/energy estimate for ``request``.

        The response must be deterministic for equivalent requests.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def validity_envelope(self) -> ValidityEnvelope:
        """Return the default calibrated range for this backend."""
        raise NotImplementedError


def validate_component_manifest(
    topology: MemoryTopology,
    required: Optional[List[str]] = None,
    excluded: Optional[List[str]] = None,
) -> None:
    """Validate that a topology obeys a component manifest.

    Raises ``ConfigError`` for illegal PHY/TSV combinations.
    """
    required = required or []
    excluded = excluded or []
    missing = [c for c in required if not _has_component(topology, c)]
    if missing:
        raise ConfigError(
            f"topology missing required components: {missing}",
            field_path="topology.components",
        )
    illegal = [c for c in excluded if _has_component(topology, c)]
    if illegal:
        raise ConfigError(
            f"topology includes excluded components: {illegal}",
            field_path="topology.components",
        )


def _has_component(topology: MemoryTopology, component: str) -> bool:
    mapping: Dict[str, bool] = {
        "dram_phy": topology.include_phy,
        "tsv": topology.include_tsv,
        "package": topology.include_package,
    }
    return mapping.get(component, False)
