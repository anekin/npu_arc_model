"""Calibration digest computation and trust gate.

``TrustGate`` takes a set of calibration IDs used by a design point (plus the
actual config values for those parameters) and returns ``(ok, max_trust,
violations)``.  Digest helpers bind calibration state into stable result
identities.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from calibration.registry import CalibrationRegistry
from contracts.hardware import TrustLevel
from contracts.identity import digest_sha256


def calibration_digest(registry: CalibrationRegistry) -> str:
    """Return SHA-256 digest of the entire registry contents."""
    return digest_sha256(registry.to_dict())


def result_digest(
    input_digest: str,
    workload_digest: str,
    calibration_digest: str,
    *,
    extra: dict[str, Any] | None = None,
) -> str:
    """Return a stable digest combining input, workload, and calibration."""
    source = {
        "input_digest": input_digest,
        "workload_digest": workload_digest,
        "calibration_digest": calibration_digest,
    }
    if extra:
        source["extra"] = extra
    return digest_sha256(source)


def _node_scale_factor(process_node_nm: float) -> float:
    """Mirror of ppa_model._node_scale_factor for 12nm density ratio."""
    if process_node_nm == 12.0:
        return 2.70
    return (process_node_nm / 7.0) ** 2


def calibration_ids_for_design_point(hw_config: dict[str, Any]) -> set[str]:
    """Return the calibration IDs consumed by a hardware config.

    The mapping mirrors the parameters consumed by ``engine.ppa_model`` and
    the MAC engines: PE baselines, engine-specific ratios, pipeline/descriptor
    constants, memory PHY/TSV, and power-density anchors.
    """
    ids: set[str] = set()

    # Always-used power/energy anchors.
    ids.add("pj_per_mac_12nm_int8")
    ids.add("power_density_12nm")

    engine_type = (hw_config.get("mac_engine", {}).get("type", "block")).lower()

    # PE area baseline and engine-specific ratios.
    ids.add("systolic_pe_area_7nm")
    if engine_type in {"block", "os_systolic", "input_stationary", "tensor_core", "wmma", "gmma", "fsa"}:
        ids.add("block_systolic_pe_ratio")
    if engine_type == "wmma":
        ids.add("wmma_pe_ratio")
    if engine_type == "gmma":
        ids.add("gmma_pe_ratio")
        ids.add("gmma_pipeline_scale")
    if engine_type == "tensor_core":
        ids.add("tensor_core_descriptor_overhead")

    # Memory subsystem.
    mem_type = str(hw_config.get("memory", {}).get("type", "LPDDR5-6400")).lower()
    onchip = hw_config.get("on_chip_memory", {})
    uses_tsv = "hbm" in mem_type or "onchip" in mem_type or float(onchip.get("capacity_gb", 0)) > 0
    if uses_tsv:
        ids.add("tsv_overhead_pct")
    # External DRAM PHY is only relevant for external memory tiers.
    if not (float(onchip.get("capacity_gb", 0)) > 0):
        ids.add("dram_phy_area_12nm")

    return ids


def _actual_value(calibration_id: str, hw_config: dict[str, Any]) -> float | None:
    """Extract the actual config value for a calibration ID."""
    area_model = hw_config.get("area_model", {})

    if calibration_id == "systolic_pe_area_7nm":
        return float(area_model.get("systolic_pe_area_mm2", 2.0))

    if calibration_id == "block_systolic_pe_ratio":
        systolic = float(area_model.get("systolic_pe_area_mm2", 2.0))
        block = float(area_model.get("block_pe_area_mm2", 4.0))
        if systolic <= 0:
            return None
        return block / systolic

    if calibration_id == "wmma_pe_ratio":
        block = float(area_model.get("block_pe_area_mm2", 4.0))
        wmma = float(area_model.get("wmma_pe_area_mm2", 6.0))
        if block <= 0:
            return None
        return wmma / block

    if calibration_id == "gmma_pe_ratio":
        block = float(area_model.get("block_pe_area_mm2", 4.0))
        gmma = float(area_model.get("gmma_pe_area_mm2", 7.0))
        if block <= 0:
            return None
        return gmma / block

    if calibration_id == "gmma_pipeline_scale":
        return float(hw_config.get("gmma", {}).get("pipeline_scale", 0.05))

    if calibration_id == "tensor_core_descriptor_overhead":
        return float(hw_config.get("dma", {}).get("descriptor_overhead_cycles", 5))

    if calibration_id == "tsv_overhead_pct":
        return float(area_model.get("tsv_overhead_pct", 0.10))

    if calibration_id == "dram_phy_area_12nm":
        node = float(area_model.get("process_node_nm", area_model.get("process_node", 7.0)))
        baseline = float(area_model.get("dram_phy_area_mm2", 5.0))
        return baseline * _node_scale_factor(node)

    if calibration_id == "power_density_12nm":
        # Logic power density is the primary ranking driver.
        return 0.5

    if calibration_id == "pj_per_mac_12nm_int8":
        # Not yet a configurable field; model uses proxy values.
        return 0.15

    return None


class TrustGate:
    """Evaluate whether a set of calibration IDs meets a trust threshold."""

    def __init__(self, registry: CalibrationRegistry) -> None:
        self.registry = registry

    def check(
        self,
        calibration_ids: Iterable[str],
        *,
        values: dict[str, float] | None = None,
        require_trust: TrustLevel = TrustLevel.T0,
        hw_config: dict[str, Any] | None = None,
    ) -> tuple[bool, TrustLevel, list[dict[str, Any]]]:
        """Return ``(ok, max_trust, violations)``.

        * ``ok`` is True when every ID is known, its trust level is at least
          *require_trust*, and (when values/hw_config are supplied) the actual
          value lies inside the calibration range.
        * ``max_trust`` is the minimum trust level among the IDs; a single T0
          caps the whole point at T0.
        * ``violations`` is a list of structured records with ``calibration_id``,
          ``reason``, and relevant context.
        """
        violations: list[dict[str, Any]] = []
        max_trust = TrustLevel.T3
        values = dict(values) if values else {}

        for cid in sorted(set(calibration_ids)):
            entry = self.registry.lookup(cid)
            if entry is None:
                violations.append(
                    {
                        "calibration_id": cid,
                        "reason": "unknown_calibration_id",
                    }
                )
                max_trust = min(max_trust, TrustLevel.T0, key=_trust_rank)
                continue

            if _trust_rank(entry.trust_level) < _trust_rank(require_trust):
                violations.append(
                    {
                        "calibration_id": cid,
                        "reason": "trust_level_too_low",
                        "actual_trust": entry.trust_level.value,
                        "required_trust": require_trust.value,
                    }
                )

            actual: float | None
            if cid in values:
                actual = values[cid]
            elif hw_config is not None:
                actual = _actual_value(cid, hw_config)
            else:
                actual = None

            if actual is not None and not entry.is_in_range(actual):
                violations.append(
                    {
                        "calibration_id": cid,
                        "reason": "out_of_calibration_range",
                        "actual_value": actual,
                        "calibration_range": entry.calibration_range,
                    }
                )
                # Out-of-range is capped at T1 even if the entry itself is T2+.
                max_trust = min(max_trust, TrustLevel.T1, key=_trust_rank)
                continue

            max_trust = min(max_trust, entry.trust_level, key=_trust_rank)

        return (not violations), max_trust, violations


def _trust_rank(level: TrustLevel) -> int:
    """Numeric rank for trust level comparison (T3 highest)."""
    return {"T0": 0, "T1": 1, "T2": 2, "T3": 3}[level.value]


__all__ = [
    "TrustGate",
    "calibration_digest",
    "calibration_ids_for_design_point",
    "result_digest",
]
