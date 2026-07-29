"""Typed memory hierarchy for unified residency and spill planning.

Defines ``MemoryTier`` and ``MemoryHierarchy`` with capacity, read/write
bandwidth, latency, alignment, and reserve fraction.  Supports the six
currently modeled memory technologies:

* ``sram``              — on-die L1/L2 scratchpad
* ``on_chip_3d_dram``   — 3D-stacked or monolithic on-die DRAM
* ``lpddr5``            — external LPDDR5
* ``lpddr5x``           — external LPDDR5X
* ``hbm2e``             — external HBM2e
* ``hbm3``              — external HBM3
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from contracts.errors import ConfigError


class MemoryTierName(str, Enum):
    """Canonical memory tier names."""

    SRAM = "sram"
    ON_CHIP_3D_DRAM = "on_chip_3d_dram"
    LPDDR5 = "lpddr5"
    LPDDR5X = "lpddr5x"
    HBM2E = "hbm2e"
    HBM3 = "hbm3"


SUPPORTED_TIER_NAMES: Tuple[str, ...] = tuple(t.value for t in MemoryTierName)
EXTERNAL_TIER_NAMES: Tuple[str, ...] = (
    MemoryTierName.LPDDR5.value,
    MemoryTierName.LPDDR5X.value,
    MemoryTierName.HBM2E.value,
    MemoryTierName.HBM3.value,
)


# Typical SRAM read/write bandwidth when not explicitly configured (GB/s).
# Derived from 256-bit interface × 16 banks × 1 GHz ≈ 512 GB/s; we round up
# to leave headroom for near-memory compute scenarios.
_DEFAULT_SRAM_BW_GBPS = 1000.0

# Default external memory capacity when the YAML does not specify one.
# 32 GiB is large enough to host any model in the current workload fixtures.
_DEFAULT_EXTERNAL_CAPACITY_GB = 32.0


def _align_up(value: int, alignment: int) -> int:
    """Return the smallest multiple of ``alignment`` >= ``value``."""
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


class MemoryTier(BaseModel):
    """A single memory tier in the hierarchy.

    ``reserve_fraction`` expresses how much of the tier is reserved for the
    runtime allocator / OS / debug buffers and is therefore unavailable for
    tensor placement.  The reserved bytes are computed as
    ``ceil(capacity_bytes * reserve_fraction)``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., description="Canonical tier name")
    capacity_bytes: int = Field(..., ge=0, description="Total capacity in bytes")
    read_bw_gbps: float = Field(..., gt=0, description="Read bandwidth in GB/s")
    write_bw_gbps: float = Field(..., gt=0, description="Write bandwidth in GB/s")
    read_latency_cycles: int = Field(default=1, ge=0, description="Read latency in core cycles")
    write_latency_cycles: int = Field(default=1, ge=0, description="Write latency in core cycles")
    alignment_bytes: int = Field(default=256, ge=1, description="Allocation alignment in bytes")
    reserve_fraction: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of capacity reserved")
    read_efficiency: float = Field(default=1.0, ge=0.0, le=1.0, description="Effective read bandwidth factor")
    write_efficiency: float = Field(default=1.0, ge=0.0, le=1.0, description="Effective write bandwidth factor")

    @field_validator("name")
    @classmethod
    def _name_must_be_supported(cls, v: str) -> str:
        if v not in SUPPORTED_TIER_NAMES:
            raise ValueError(f"unsupported memory tier name: {v!r}")
        return v

    @property
    def reserve_bytes(self) -> int:
        """Bytes reserved from this tier."""
        return int(self.capacity_bytes * self.reserve_fraction)

    @property
    def usable_bytes(self) -> int:
        """Capacity available for tensor placement after reserve."""
        return max(0, self.capacity_bytes - self.reserve_bytes)

    def align(self, size_bytes: int) -> int:
        """Return ``size_bytes`` rounded up to this tier's alignment."""
        return _align_up(size_bytes, self.alignment_bytes)

    def effective_read_bw_gbps(self) -> float:
        """Read bandwidth after efficiency derating."""
        return self.read_bw_gbps * self.read_efficiency

    def effective_write_bw_gbps(self) -> float:
        """Write bandwidth after efficiency derating."""
        return self.write_bw_gbps * self.write_efficiency


class MemoryHierarchy(BaseModel):
    """Ordered list of memory tiers from fastest to slowest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tiers: Tuple[MemoryTier, ...] = Field(..., description="Ordered tiers, fastest first")
    default_alignment_bytes: int = Field(default=256, ge=1)

    def get_tier(self, name: str) -> MemoryTier:
        """Look up a tier by name."""
        for tier in self.tiers:
            if tier.name == name:
                return tier
        raise KeyError(f"tier not found: {name!r}")

    def has_tier(self, name: str) -> bool:
        """Return True if the named tier exists."""
        return any(tier.name == name for tier in self.tiers)

    def total_capacity_bytes(self) -> int:
        """Sum capacity across all tiers."""
        return sum(tier.capacity_bytes for tier in self.tiers)

    def total_usable_bytes(self) -> int:
        """Sum usable capacity across all tiers."""
        return sum(tier.usable_bytes for tier in self.tiers)

    def fastest_tier(self) -> MemoryTier:
        """Return the first (fastest) tier."""
        if not self.tiers:
            raise ConfigError("memory hierarchy has no tiers", field_path="hierarchy.tiers")
        return self.tiers[0]

    def external_tier(self) -> Optional[MemoryTier]:
        """Return the first external DRAM tier, if any."""
        for tier in self.tiers:
            if tier.name in EXTERNAL_TIER_NAMES:
                return tier
        return None


def _gb_to_bytes(gb: float) -> int:
    """Convert decimal gigabytes to bytes."""
    return int(gb * 1_000_000_000)


def _kb_to_bytes(kb: float) -> int:
    """Convert decimal kilobytes to bytes."""
    return int(kb * 1000)


def _infer_external_tier_name(mem_type: str) -> str:
    """Map a memory type string to a canonical external tier name."""
    mem_type_lower = mem_type.lower()
    if "lpddr5x" in mem_type_lower:
        return MemoryTierName.LPDDR5X.value
    if "lpddr5" in mem_type_lower:
        return MemoryTierName.LPDDR5.value
    if "hbm3" in mem_type_lower:
        return MemoryTierName.HBM3.value
    if "hbm2e" in mem_type_lower or "hbm2" in mem_type_lower:
        return MemoryTierName.HBM2E.value
    return MemoryTierName.LPDDR5.value


def build_hierarchy_from_config(config: Dict[str, Any]) -> MemoryHierarchy:
    """Build a ``MemoryHierarchy`` from a legacy-style config dict.

    The config is expected to contain ``sram``, ``memory``, and optionally
    ``on_chip_memory`` sections.  Unknown sections are ignored.
    """
    tiers: list[MemoryTier] = []

    sram = config.get("sram", {})
    if sram:
        l1_kb = float(sram.get("l1_per_core_kb", 512))
        l2_kb = float(sram.get("l2_shared_kb", 2048))
        sram_capacity_bytes = _kb_to_bytes(l1_kb + l2_kb)
        tiers.append(
            MemoryTier(
                name=MemoryTierName.SRAM.value,
                capacity_bytes=sram_capacity_bytes,
                read_bw_gbps=_DEFAULT_SRAM_BW_GBPS,
                write_bw_gbps=_DEFAULT_SRAM_BW_GBPS,
                read_latency_cycles=2,
                write_latency_cycles=2,
                alignment_bytes=256,
                reserve_fraction=0.0,
                read_efficiency=1.0,
                write_efficiency=1.0,
            )
        )

    onchip = config.get("on_chip_memory", {})
    onchip_capacity_gb = float(onchip.get("capacity_gb", 0))
    if onchip_capacity_gb > 0:
        onchip_bw = float(onchip.get("bandwidth_gbps", 500))
        tiers.append(
            MemoryTier(
                name=MemoryTierName.ON_CHIP_3D_DRAM.value,
                capacity_bytes=_gb_to_bytes(onchip_capacity_gb),
                read_bw_gbps=onchip_bw,
                write_bw_gbps=onchip_bw,
                read_latency_cycles=10,
                write_latency_cycles=10,
                alignment_bytes=256,
                reserve_fraction=0.0,
                read_efficiency=1.0,
                write_efficiency=1.0,
            )
        )

    memory = config.get("memory", {})
    mem_type = str(memory.get("type", "LPDDR5-6400"))
    external_bw = float(memory.get("bandwidth_gbps", 51.2))
    external_capacity_gb = float(memory.get("capacity_gb", _DEFAULT_EXTERNAL_CAPACITY_GB))
    dram_efficiency = float(memory.get("dram_efficiency", 0.85))
    external_name = _infer_external_tier_name(mem_type)

    tiers.append(
        MemoryTier(
            name=external_name,
            capacity_bytes=_gb_to_bytes(external_capacity_gb),
            read_bw_gbps=external_bw,
            write_bw_gbps=external_bw,
            read_latency_cycles=80,
            write_latency_cycles=80,
            alignment_bytes=256,
            reserve_fraction=0.0,
            read_efficiency=dram_efficiency,
            write_efficiency=dram_efficiency,
        )
    )

    return MemoryHierarchy(tiers=tuple(tiers))
