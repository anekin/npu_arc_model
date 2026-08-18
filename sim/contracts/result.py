"""Result schema v2 — typed run outcomes with stable identities.

The v2 result schema normalises every design-point evaluation into a
deterministic, self-describing record that carries:

* Stable :class:`DesignPointResult` identified by ``design_point_id`` (SHA-256
  of canonical config JSON) rather than by position.
* :class:`RunStatus` and :class:`RunTrustLevel` that encode partial/failed runs.
* Full hardware, scenario, workload, and calibration references.
* Per-engine metrics (avg/P50/P99/max latency, throughput, deadline
  miss/drop, resource utilisation, memory/PPA/energy).
* A :class:`ResultSummary` with generated/evaluated/pruned/failed/error counts.

Partial runs are forced to ``non_authoritative``; the ``release_recommendation``
API rejects any result set that contains non-authoritative points.

Reference: ``.omo/plans/arc-model-scenario-driven-dse-development.md`` Todo 9.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "RunStatus",
    "RunTrustLevel",
    "ErrorRecord",
    "DesignPointResult",
    "ResultSummary",
    "DesignSpaceResultV2",
    "release_recommendation",
    "result_standalone_from_ppa",
]


# ── Enums ────────────────────────────────────────────────────────────────────


class RunStatus(str, Enum):
    """Per-design-point run status."""

    complete = "complete"
    partial = "partial"
    failed = "failed"
    filtered = "filtered"


class RunTrustLevel(str, Enum):
    """Trust level of a design-point result.

    ``authoritative`` — all required axes covered, no failures, calibration
    gate passed.
    ``calibrated_estimate`` — calibrated model, but missing one or more
    secondary coverage axes.
    ``exploratory`` — uncalibrated or extrapolated parameters; sensitivity only.
    ``non_authoritative`` — partial run, failure, or coverage gap on a
    required axis.
    """

    authoritative = "authoritative"
    calibrated_estimate = "calibrated_estimate"
    exploratory = "exploratory"
    non_authoritative = "non_authoritative"


# ── Error record ──────────────────────────────────────────────────────────────


class ErrorRecord(BaseModel):
    """Structured error for a failed/filtered design point.

    Avoids using raw error text as the sole classification; always carries
    a typed ``code`` plus bounded ``details``.
    """

    model_config = ConfigDict(extra="forbid")

    design_point_id: str = Field(..., description="Stable design-point ID from contracts.identity")
    code: str = Field(..., description="Typed error code (e.g. ConfigError, RuntimeError)")
    message: str = Field(default="", description="Sanitised short message (≤ 200 chars)")
    details: dict[str, Any] = Field(default_factory=dict, description="Structured context (bounded)")

    @field_validator("message", mode="after")
    @classmethod
    def _truncate_message(cls, v: str) -> str:
        return v[:200]


# ── Per-engine metrics ────────────────────────────────────────────────────────


class EngineMetrics(BaseModel):
    """Latency and throughput metrics for a single engine evaluation."""

    model_config = ConfigDict(extra="forbid")

    tok_per_s: float = Field(..., description="Throughput in tokens/s (LLM) or FPS (CV)")
    area_mm2: float = Field(..., description="Die area in mm²")
    power_w: float = Field(..., description="Power consumption in watts")
    ttft_ms: float = Field(default=0.0, description="Time-to-first-token for batch_m prompt tokens")
    efficiency_tok_per_watt: float = Field(default=0.0)
    efficiency_tok_per_mm2: float = Field(default=0.0)

    # Optional per-engine drill-down
    completed_throughput_hz: float | None = Field(default=None)
    mac_count: int | None = Field(default=None, description="Total MAC operations")
    op_count: int | None = Field(default=None, description="Total arithmetic operations (2× MAC)")
    total_cycles: int | None = Field(default=None)
    utilization: float | None = Field(default=None, ge=0.0, le=1.0)

    # CV-specific
    sram_spill_mb: float | None = Field(default=None)
    depthwise_util_pct: float | None = Field(default=None)

    # Latency breakdown (seconds) — filled when temporal data is available
    avg_latency_s: float | None = Field(default=None)
    p50_latency_s: float | None = Field(default=None)
    p99_latency_s: float | None = Field(default=None)
    max_latency_s: float | None = Field(default=None)
    deadline_miss_count: int | None = Field(default=None)
    drop_count: int | None = Field(default=None)

    memory_footprint_gib: float | None = Field(default=None)
    spill_bytes: int | None = Field(default=None)
    energy_joules: float | None = Field(default=None)


class CalibrationRef(BaseModel):
    """Calibration parameters in effect for this result."""

    model_config = ConfigDict(extra="forbid")

    process_node_nm: float = Field(default=12.0)
    node_scale: float = Field(default=2.70)
    dram_efficiency: float = Field(default=0.85)
    pe_area_ratio_block_systolic: float = Field(default=2.0)
    trust_level: RunTrustLevel = Field(default=RunTrustLevel.exploratory)


class DesignPointResult(BaseModel):
    """Single design-point result in schema v2."""

    model_config = ConfigDict(extra="forbid")

    design_point_id: str = Field(..., description="Stable SHA-256 identity of the config")
    status: RunStatus = Field(..., description="complete | partial | failed | filtered")

    # References (normalised)
    hardware_digest: str = Field(default="", description="SHA-256 of hardware config")
    scenario_ref: str = Field(default="", description="Scenario identifier")
    workload_ref: str = Field(default="", description="Workload identifier")
    calibration: CalibrationRef = Field(default_factory=CalibrationRef)

    # Config that produced this result (for traceability)
    config_label: str = Field(default="", description="Human-readable config summary")
    engine_type: str = Field(default="", description="Engine architecture type")

    # Trust
    trust_level: RunTrustLevel = Field(default=RunTrustLevel.exploratory)

    # Metrics populated on success; None on failure/filter
    metrics: EngineMetrics | None = Field(default=None)

    # Error populated on failure; None on success
    error: ErrorRecord | None = Field(default=None)


class ResultSummary(BaseModel):
    """Aggregate counts for a DSE run."""

    model_config = ConfigDict(extra="forbid")

    generated: int = Field(default=0, description="Total configs generated")
    evaluated: int = Field(default=0, description="Configs that entered evaluation")
    pruned: int = Field(default=0, description="Configs pruned pre-evaluation")
    failed: int = Field(default=0, description="Configs that raised during evaluation")
    filtered: int = Field(default=0, description="Configs filtered post-evaluation (e.g. area > 200)")
    complete: int = Field(default=0, description="Configs that completed successfully")
    partial: int = Field(default=0, description="Configs evaluated under --allow-partial")


class DesignSpaceResultV2(BaseModel):
    """Top-level v2 DSE result container."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="2", description="Schema version for this output")
    trust_level: RunTrustLevel = Field(default=RunTrustLevel.exploratory)
    summary: ResultSummary = Field(default_factory=ResultSummary)
    results: list[DesignPointResult] = Field(default_factory=list)
    errors: list[ErrorRecord] = Field(default_factory=list)

    # Reproducibility digests
    input_digest: str = Field(default="", description="SHA-256 of input config")
    workload_digest: str = Field(default="", description="SHA-256 of workload spec")
    calibration_digest: str = Field(default="", description="SHA-256 of calibration params")

    frontier_design_point_ids: list[str] = Field(
        default_factory=list,
        description="IDs of design points on the Pareto frontier",
    )


# ── Release recommendation gate ───────────────────────────────────────────────


def release_recommendation(result_set: DesignSpaceResultV2) -> list[DesignPointResult]:
    """Return recommended design points from an authoritative result set.

    Raises :class:`NonAuthoritativeRunError` if the result set is partial or
    contains any non-authoritative / failed / filtered entries.
    """
    from contracts.errors import NonAuthoritativeRunError

    # Whole-set trust
    if result_set.trust_level != RunTrustLevel.authoritative:
        raise NonAuthoritativeRunError(
            f"Cannot produce release recommendation: result-set trust level is {result_set.trust_level.value}",
            reason=f"trust_level={result_set.trust_level.value}",
        )

    # Check individual results
    non_auth = [r for r in result_set.results if r.trust_level != RunTrustLevel.authoritative]
    if non_auth:
        ids = [r.design_point_id[:12] for r in non_auth[:5]]
        raise NonAuthoritativeRunError(
            f"Found {len(non_auth)} non-authoritative result(s): {', '.join(ids)}...",
            reason=f"{len(non_auth)} non_authoritative points",
        )

    return [r for r in result_set.results if r.status == RunStatus.complete]


# ── PPA → v2 bridge ───────────────────────────────────────────────────────────


def result_standalone_from_ppa(
    ppa: Any,  # PPA instance
    config: dict[str, Any],
    *,
    status: RunStatus = RunStatus.complete,
    trust_level: RunTrustLevel = RunTrustLevel.exploratory,
) -> DesignPointResult:
    """Create a standalone DesignPointResult from a legacy PPA object and config.

    Uses ``contracts.identity`` to derive a stable ID from *config*.
    """
    from contracts.identity import digest_sha256
    from engine.ppa_model import AreaModel, _node_scale_factor

    dp_id = digest_sha256(config)
    hw_digest = digest_sha256(config)

    area_model = AreaModel(config)
    node_scale = _node_scale_factor(area_model.process_node_nm)

    metrics = EngineMetrics(
        tok_per_s=ppa.tok_s,
        area_mm2=ppa.area_mm2,
        power_w=ppa.power_w,
        ttft_ms=ppa.ttft_ms,
        efficiency_tok_per_watt=ppa.efficiency_tok_per_watt,
        efficiency_tok_per_mm2=ppa.efficiency_tok_per_mm2,
        sram_spill_mb=ppa.sram_spill_mb if ppa.sram_spill_mb else None,
        depthwise_util_pct=ppa.depthwise_util_pct if ppa.depthwise_util_pct else None,
    )

    return DesignPointResult(
        design_point_id=dp_id,
        status=status,
        hardware_digest=hw_digest,
        config_label=ppa.config_label,
        engine_type=config.get("mac_engine", {}).get("type", "unknown"),
        trust_level=trust_level,
        calibration=CalibrationRef(
            process_node_nm=area_model.process_node_nm,
            node_scale=node_scale,
            dram_efficiency=config.get("memory", {}).get("dram_efficiency", 0.85),
            pe_area_ratio_block_systolic=2.0,
            trust_level=trust_level,
        ),
        metrics=metrics,
    )
