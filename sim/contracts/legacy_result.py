"""Legacy result projection — preserve Todo 1 snapshot fields.

Projects :class:`~contracts.result.DesignSpaceResultV2` back to the legacy LLM
and CV JSON shapes that are snapshotted in
``sim/tests/golden/legacy_cli_contract.json`` (Todo 1).  Every
v2-only field is recorded in a ``LossReport`` so downstream consumers know
what was dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "LegacyLossReport",
    "project_v2_to_legacy_llm",
    "project_v2_to_legacy_cv",
    "legacy_result_dict_from_ppa",
]


@dataclass
class LegacyLossReport:
    """Structured report of v2-only data dropped during legacy projection."""

    dropped_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_loss(self) -> bool:
        return bool(self.dropped_fields) or bool(self.warnings)


# ── Per-point projection ──────────────────────────────────────────────────────


def legacy_result_dict_from_ppa(
    ppa: Any,
    *,
    on_pareto: bool = False,
    cv_mode: bool = False,
) -> dict[str, Any]:
    """Build a legacy result dict from a PPA object.

    Preserves the exact field set from Todo 1 snapshots:
    ``label``, ``tok_s``, ``area_mm2``, ``power_w``, plus CV-only fields.
    """
    d: dict[str, Any] = {
        "label": ppa.config_label,
        "tok_s": ppa.tok_s,
        "area_mm2": ppa.area_mm2,
        "power_w": ppa.power_w,
    }
    if cv_mode:
        d["sram_spill_mb"] = ppa.sram_spill_mb
        d["depthwise_util_pct"] = ppa.depthwise_util_pct
        prefix = (ppa.config_label or "").split()[0]
        engine_map = {
            "syst": "systolic",
            "os_s": "os_systolic",
            "bloc": "block",
            "tens": "tensor_core",
            "wmma": "wmma",
            "gmma": "gmma",
            "inpu": "input_stationary",
            "fsa ": "fsa",
        }
        d["engine_type"] = engine_map.get(prefix, prefix)
        d["pareto"] = on_pareto
    return d


# ── Top-level projection ──────────────────────────────────────────────────────


def project_v2_to_legacy_llm(
    v2_result: Any,  # DesignSpaceResultV2
    *,
    model_spec: str = "qwen2.5-3b",
    batch_m: int = 1,
    total_configs: int = 0,
    top_n: int = 20,
) -> tuple[dict[str, Any], LegacyLossReport]:
    """Project a v2 result set to the legacy LLM JSON shape.

    Returns ``(legacy_dict, loss_report)``.
    """
    loss = LegacyLossReport()

    # Collect successful results
    complete = [r for r in v2_result.results if r.status.value in ("complete", "partial") and r.metrics is not None]
    pareto_ids = _pareto_ids_from_results(v2_result.results)

    # Pareto frontier — keep top results by tok/s
    complete.sort(key=lambda r: r.metrics.tok_per_s, reverse=True)
    pareto = [r for r in complete if r.design_point_id in pareto_ids]
    top = complete[:top_n]

    error_details = []
    for err in v2_result.errors:
        error_details.append(
            {
                "engine_type": err.details.get("engine_type", "unknown"),
                "dims": err.details.get("dims", "?"),
                "memory_mode": err.details.get("memory_mode", "unknown"),
                "error": err.message or err.code,
            }
        )

    # Build legacy dict
    legacy: dict[str, Any] = {
        "cv_model": "",
        "model_spec": model_spec,
        "batch_m": batch_m,
        "total_configs": total_configs,
        "valid_results": len(complete),
        "generated": v2_result.summary.generated,
        "evaluated": v2_result.summary.evaluated,
        "filtered_by_area": v2_result.summary.filtered,
        "errors": v2_result.summary.failed,
        "error_details": error_details,
        "pareto_frontier": [_legacy_point(r, pareto_ids) for r in pareto],
        "top_results": [_legacy_point(r, pareto_ids) for r in top],
    }

    # Loss report for v2-only data
    if v2_result.input_digest:
        loss.dropped_fields.append("input_digest")
    if v2_result.workload_digest:
        loss.dropped_fields.append("workload_digest")
    if v2_result.calibration_digest:
        loss.dropped_fields.append("calibration_digest")
    if v2_result.trust_level.value != "exploratory":
        loss.dropped_fields.append("trust_level")
    if any(r.calibration.process_node_nm != 12.0 for r in v2_result.results):
        loss.dropped_fields.append("calibration")
    if any(r.hardware_digest for r in v2_result.results):
        loss.dropped_fields.append("hardware_digest")
    if any(r.design_point_id for r in v2_result.results):
        loss.dropped_fields.append("design_point_id")
    if v2_result.summary.pruned:
        loss.dropped_fields.append("summary.pruned")
    if v2_result.summary.complete:
        loss.dropped_fields.append("summary.complete")
    if v2_result.summary.partial:
        loss.dropped_fields.append("summary.partial")

    return legacy, loss


def project_v2_to_legacy_cv(
    v2_result: Any,  # DesignSpaceResultV2
    *,
    cv_model: str = "",
    top_n: int = 20,
) -> tuple[dict[str, Any], LegacyLossReport]:
    """Project a v2 result set to the legacy CV JSON shape.

    Returns ``(legacy_dict, loss_report)``.
    """
    loss = LegacyLossReport()

    complete = [r for r in v2_result.results if r.status.value in ("complete", "partial") and r.metrics is not None]
    complete.sort(key=lambda r: r.metrics.tok_per_s, reverse=True)
    pareto_ids = _pareto_ids_from_results(v2_result.results)

    points = []
    seen: set[str] = set()
    for r in complete:
        is_pareto = r.design_point_id in pareto_ids
        if is_pareto:
            points.append(_cv_legacy_point(r, True))
            seen.add(r.design_point_id)
    for r in complete[:top_n]:
        if r.design_point_id not in seen:
            points.append(_cv_legacy_point(r, False))
            seen.add(r.design_point_id)

    error_details = []
    for err in v2_result.errors:
        error_details.append(
            {
                "engine_type": err.details.get("engine_type", "unknown"),
                "dims": err.details.get("dims", "?"),
                "memory_mode": err.details.get("memory_mode", "unknown"),
                "error": err.message or err.code,
            }
        )

    legacy: dict[str, Any] = {
        "cv_model": cv_model,
        "metadata": {
            "cv_model": cv_model,
            "valid_results": len(complete),
            "generated": v2_result.summary.generated,
            "evaluated": v2_result.summary.evaluated,
            "filtered_by_area": v2_result.summary.filtered,
            "errors": v2_result.summary.failed,
            "error_details": error_details,
        },
        "points": points,
    }

    # Loss report
    if v2_result.input_digest:
        loss.dropped_fields.append("input_digest")
    loss.dropped_fields.append("schema_version")
    if any(r.calibration.process_node_nm != 12.0 for r in v2_result.results):
        loss.dropped_fields.append("calibration")

    return legacy, loss


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pareto_ids_from_results(results: list[Any]) -> set[str]:
    """Find Pareto-optimal design-point IDs (max tok/s, min area)."""
    pareto_ids: set[str] = set()
    for i, r in enumerate(results):
        if r.metrics is None:
            continue
        dominated = False
        for j, other in enumerate(results):
            if i == j or other.metrics is None:
                continue
            if (
                other.metrics.tok_per_s >= r.metrics.tok_per_s
                and other.metrics.area_mm2 <= r.metrics.area_mm2
                and (other.metrics.tok_per_s > r.metrics.tok_per_s or other.metrics.area_mm2 < r.metrics.area_mm2)
            ):
                dominated = True
                break
        if not dominated:
            pareto_ids.add(r.design_point_id)
    return pareto_ids


def _legacy_point(r: Any, pareto_ids: set[str]) -> dict[str, Any]:
    """Convert a DesignPointResult to legacy LLM point dict."""
    m = r.metrics
    return {
        "label": r.config_label,
        "tok_s": m.tok_per_s,
        "area_mm2": m.area_mm2,
        "power_w": m.power_w,
    }


def _cv_legacy_point(r: Any, on_pareto: bool) -> dict[str, Any]:
    """Convert a DesignPointResult to legacy CV point dict."""
    m = r.metrics
    prefix = (r.config_label or "").split()[0]
    engine_map = {
        "syst": "systolic",
        "os_s": "os_systolic",
        "bloc": "block",
        "tens": "tensor_core",
        "wmma": "wmma",
        "gmma": "gmma",
        "inpu": "input_stationary",
        "fsa ": "fsa",
    }
    return {
        "label": r.config_label,
        "tok_s": m.tok_per_s,
        "area_mm2": m.area_mm2,
        "power_w": m.power_w,
        "sram_spill_mb": m.sram_spill_mb or 0.0,
        "depthwise_util_pct": m.depthwise_util_pct or 0.0,
        "engine_type": engine_map.get(prefix, r.engine_type),
        "pareto": on_pareto,
    }
