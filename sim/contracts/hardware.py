"""Pydantic v2 hardware and memory schema for the Arc Model.

This module defines the v2 canonical schema.  Key rules:

* The canonical hardware key is ``mac_engine``.
* Legacy ``mxu`` is accepted by the adapter but never stored.
* ``mxu`` alone → migrate silently.
* ``mxu`` + ``mac_engine`` both present and consistent → accept with warning.
* ``mxu`` + ``mac_engine`` both present and inconsistent → fail-closed.
* Unknown fields are forbidden at every level (extra='forbid').
* Every physical parameter carries a ``provenance`` record.

Reference: ``.omo/plans/arc-model-ppa-corrections.md``
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

# ── provenance ──────────────────────────────────────────────────────────────


class TrustLevel(str, Enum):
    """Trust level of a physical parameter.

    T0: engineering assumption — exploratory sensitivity only.
    T1: published proxy or architectural reasoning — feasibility bounds.
    T2: reproduced from verified source — relative decisions with intervals.
    T3: signed-off reference — numeric predictions with intervals.
    """

    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"


class Provenance(BaseModel):
    """Audit trail for a physical parameter."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="Human-readable source description")
    trust_level: TrustLevel = Field(..., description="T0-T3 as defined above")
    calibration_range: str | None = Field(
        default=None,
        description="Optional range this value was calibrated within",
    )
    reference_uri: str | None = Field(
        default=None,
        description="Optional URI/DOI/path to source document",
    )


# ── MAC Engine ───────────────────────────────────────────────────────────────


def _positive_int(v: Any) -> int:
    """Coerce to int and reject non-positive values."""
    if isinstance(v, bool):
        raise ValueError("bool value is not allowed as integer")
    n = int(v)
    if n <= 0:
        raise ValueError(f"must be positive, got {n}")
    return n


def _positive_float(v: Any) -> float:
    """Coerce to float and reject non-positive or non-finite values."""
    if isinstance(v, bool):
        raise ValueError("bool value is not allowed as numeric")
    n = float(v)
    if n <= 0:
        raise ValueError(f"must be positive, got {n}")
    if not _isfinite(n):
        raise ValueError(f"must be finite, got {n}")
    return n


def _nonnegative_float(v: Any) -> float:
    """Coerce to float and reject negative or non-finite values."""
    if isinstance(v, bool):
        raise ValueError("bool value is not allowed as numeric")
    n = float(v)
    if n < 0:
        raise ValueError(f"must be non-negative, got {n}")
    if not _isfinite(n):
        raise ValueError(f"must be finite, got {n}")
    return n


def _nonnegative_int(v: Any) -> int:
    """Coerce to int and reject negative values."""
    if isinstance(v, bool):
        raise ValueError("bool value is not allowed as integer")
    n = int(v)
    if n < 0:
        raise ValueError(f"must be non-negative, got {n}")
    return n


def _isfinite(v: float) -> bool:
    """Return True if v is finite (not NaN or Inf)."""
    import math

    return math.isfinite(v)


PositiveInt = Annotated[int, AfterValidator(_positive_int)]
NonNegativeInt = Annotated[int, AfterValidator(_nonnegative_int)]
PositiveFloat = Annotated[float, AfterValidator(_positive_float)]
NonNegativeFloat = Annotated[float, AfterValidator(_nonnegative_float)]


class MACEngineConfig(BaseModel):
    """v2 canonical MAC engine block."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description="Engine architecture: systolic, block, etc.")
    array_height: PositiveInt
    array_width: PositiveInt
    frequency_mhz: PositiveFloat = Field(
        default=1000.0,
        description="Core clock frequency in MHz",
    )
    weight_precision_bits: PositiveInt = Field(default=4)
    activation_precision_bits: PositiveInt = Field(default=8)
    accumulate_precision_bits: PositiveInt = Field(default=32)
    dataflow: str = Field(default="weight_stationary")
    double_buffer: bool = Field(default=True)
    ops_per_mac: PositiveInt = Field(default=2, description="multiply + accumulate = 2 ops/MAC")

    provenance: Provenance | None = None

    @field_validator(
        "array_height",
        "array_width",
        "frequency_mhz",
        "weight_precision_bits",
        "activation_precision_bits",
        "accumulate_precision_bits",
        "ops_per_mac",
        mode="before",
    )
    @classmethod
    def _reject_bool_for_numeric(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("bool value is not allowed as integer")
        return v


# ── Memory ───────────────────────────────────────────────────────────────────


class MemoryConfig(BaseModel):
    """v2 memory subsystem configuration.

    bandwidth_gbps is the canonical decimal-GB/s field.
    bandwidth_bytes_per_cycle is computed, not configured.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default="LPDDR5-6400")
    bandwidth_gbps: PositiveFloat = Field(default=51.2, description="Memory bandwidth in decimal GB/s")
    dram_efficiency: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.85,
        description=(
            "Per-bank refresh only ~3.6% (JEDEC tRFCpb=140ns/tREFI=3900ns @ 16Gb LPDDR5); "
            "extra overhead from controller scheduling/command bus/bank conflicts. "
            "After Todo 7 this field serves as the sequential-access baseline "
            "(weights and activations, AccessType.SEQUENTIAL)."
        ),
    )
    dram_efficiency_random_bw: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.50,
        description=(
            "Random-access bandwidth efficiency for KV cache reads (AccessType.RANDOM). "
            "Lower than dram_efficiency because scattered token positions cause "
            "row-buffer conflicts and poor burst utilization. "
            "Final random eff. BW = bw_raw * dram_efficiency_random_bw * _kv_dram_efficiency(kv_bytes)."
        ),
    )
    random_latency_penalty_cycles: NonNegativeInt = Field(
        default=40,
        description=(
            "Extra fixed latency (cycles) added on each random KV cache miss, "
            "independent of bandwidth.  Not added on SRAM hit."
        ),
    )
    dram_width_bits: PositiveInt = Field(default=64)
    tRC_cycles: PositiveInt = Field(default=48)
    tRAS_cycles: PositiveInt = Field(default=42)
    refresh_overhead_percent: NonNegativeFloat = Field(default=3.0)

    provenance: Provenance | None = None

    @field_validator("bandwidth_gbps", "dram_width_bits", "tRC_cycles", "tRAS_cycles", mode="before")
    @classmethod
    def _reject_bool_for_numeric(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("bool value is not allowed as numeric")
        return v

    @field_validator("bandwidth_gbps")
    @classmethod
    def _check_bandwidth_finite_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"bandwidth_gbps must be positive, got {v}")
        if not _isfinite(v):
            raise ValueError(f"bandwidth_gbps must be finite, got {v}")
        return v


# ── SRAM ─────────────────────────────────────────────────────────────────────


class SRAMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    l1_per_core_kb: PositiveInt = Field(default=512)
    l2_shared_kb: PositiveInt = Field(default=2048)
    banks: PositiveInt = Field(default=16)
    read_width_bits: PositiveInt = Field(default=256)
    write_width_bits: PositiveInt = Field(default=256)


# ── Root Hardware Config ─────────────────────────────────────────────────────


class HardwareConfigV2(BaseModel):
    """v2 canonical hardware configuration.

    The root model enforces:

    1. ``mac_engine`` is the canonical key.
    2. ``memory`` stores ``bandwidth_gbps`` (not raw ``bandwidth_bytes_per_cycle``).
    3. Unknown fields are forbidden everywhere.
    4. The ``version`` field defaults to "2" and must not be anything else.
    """

    model_config = ConfigDict(extra="forbid")

    version: Literal["2"] = Field(default="2")

    mac_engine: MACEngineConfig
    memory: MemoryConfig
    sram: SRAMConfig = Field(default_factory=SRAMConfig)

    cores: PositiveInt = Field(default=1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_version(cls, data: Any) -> Any:
        """Accept "2" or int 2; reject everything else.  Defaults to "2" when missing."""
        if not isinstance(data, dict):
            raise ValueError("root config must be a dictionary/mapping")
        version = data.get("version")
        if version is not None and version != "2" and version != 2:
            raise ValueError(f"unsupported schema version: {version!r} (expected '2')")
        if version is None:
            data["version"] = "2"
        return data

    def bandwidth_bytes_per_cycle(self) -> float:
        """Compute bytes/cycle from canonical fields.

        Uses the plan formula: bytes_per_cycle = bandwidth_gbps * 1000 / frequency_mhz
        """
        from contracts.units import bandwidth_gbps_to_bytes_per_cycle

        return bandwidth_gbps_to_bytes_per_cycle(
            self.memory.bandwidth_gbps,
            self.mac_engine.frequency_mhz,
        )

    def effective_bytes_per_cycle(self) -> float:
        """Compute effective bytes/cycle accounting for DRAM efficiency."""
        return self.bandwidth_bytes_per_cycle() * self.memory.dram_efficiency


# ── Default provenance for PPA parameters ────────────────────────────────────

# Per .omo/plans/arc-model-ppa-corrections.md

DEFAULT_DRAM_EFFICIENCY_PROVENANCE = Provenance(
    source=(
        "per-bank refresh ~3.6% (JEDEC tRFCpb=140ns/tREFI=3900ns); "
        "extra from controller scheduling/command bus/bank conflicts; "
        "0.85 is conservative sequential decode value; "
        "after Todo 7 this is the sequential-access baseline"
    ),
    trust_level=TrustLevel.T1,
    calibration_range="0.80–0.90",
    reference_uri=".omo/plans/arc-model-ppa-corrections.md#修正-1",
)

DEFAULT_DRAM_EFFICIENCY_RANDOM_BW_PROVENANCE = Provenance(
    source=(
        "Random KV cache access: scattered token positions cause row-buffer misses; "
        "0.50 is an architectural rule-of-thumb for LPDDR5 random-access effective BW"
    ),
    trust_level=TrustLevel.T0,
    calibration_range="0.40–0.60",
    reference_uri=".omo/plans/engine-selection-p0-improvements.md",
)

DEFAULT_RANDOM_LATENCY_PENALTY_PROVENANCE = Provenance(
    source=(
        "Row-buffer miss / precharge-activate latency proxy for LPDDR5 random access; "
        "independent of bandwidth, applied only on KV miss"
    ),
    trust_level=TrustLevel.T0,
    calibration_range="30–60 cycles",
    reference_uri=".omo/plans/engine-selection-p0-improvements.md",
)

DEFAULT_NODE_SCALE_PROVENANCE = Provenance(
    source=(
        "TSMC 12FFC is an optical shrink of 16FFC, not true 12nm geometry; "
        "density ratio 91.2/33.8 = 2.70× (not (12/7)² = 2.94×)"
    ),
    trust_level=TrustLevel.T1,
    calibration_range="2.5–2.9×",
    reference_uri=".omo/plans/arc-model-ppa-corrections.md#修正-2",
)

DEFAULT_PE_AREA_RATIO_PROVENANCE = Provenance(
    source=(
        "TPUv1 ISCA 2017 die-shot for systolic PE baseline; "
        "block/systolic = 2.0× is architectural reasoning; "
        "systolic=1.79× vector per Gemmini DAC 2021"
    ),
    trust_level=TrustLevel.T1,
    calibration_range="1.8–2.2×",
    reference_uri=".omo/plans/arc-model-ppa-corrections.md#修正-3",
)

DEFAULT_GMMA_PIPELINE_PROVENANCE = Provenance(
    source="H100 architectural assumption; uncalibrated",
    trust_level=TrustLevel.T0,
    calibration_range=None,
    reference_uri=".omo/plans/arc-model-ppa-corrections.md#修正-3",
)

DEFAULT_TSV_PROVENANCE = Provenance(
    source="Industry rule-of-thumb; no published proxy yet",
    trust_level=TrustLevel.T1,
    calibration_range="0.05–0.15",
    reference_uri=".omo/plans/arc-model-ppa-corrections.md#修正-5",
)
