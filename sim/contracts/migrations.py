"""v1 ↔ v2 migration and v2 → legacy projection.

The key rules:

* All functions are pure — inputs are never mutated in place.
* v1 → v2: rename ``mxu`` → ``mac_engine``, drop ``bandwidth_bytes_per_cycle``.
* v2 → legacy: project back to v1 shape for backward compatibility.
* Inexpressible fields (e.g. provenance, new v2-only keys) are reported in
  a ``LossReport``, never silently dropped.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "LossReport",
    "migrate_v1_to_v2",
    "project_v2_to_legacy",
]


@dataclass
class LossReport:
    """Structured report of fields that cannot be expressed in the target schema."""

    dropped_keys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_loss(self) -> bool:
        return bool(self.dropped_keys) or bool(self.warnings)

    def __repr__(self) -> str:
        return f"LossReport(dropped={self.dropped_keys}, warnings={self.warnings})"


def migrate_v1_to_v2(config: dict[str, Any]) -> tuple[dict[str, Any], LossReport]:
    """Migrate a legacy v1 config dict to v2 shape.

    Returns (v2_config, loss_report).  The input dict is NOT mutated.

    Rules:
      - ``mxu`` → ``mac_engine`` (rename).
      - ``mxu`` + ``mac_engine`` both present and consistent → keep ``mac_engine``
        with structured warning.
      - ``mxu`` + ``mac_engine`` both present and inconsistent → raise ConfigError.
      - ``bandwidth_bytes_per_cycle`` in ``memory`` → dropped; derived from
        ``bandwidth_gbps``.
      - ``version`` set to "2".
      - Unknown top-level keys preserved only if they pass through validation.
    """
    from sim.contracts.errors import ConfigError

    data = deepcopy(config)
    loss = LossReport()

    data["version"] = "2"

    # ── Handle mac_engine / mxu ──
    has_mac = "mac_engine" in data
    has_mxu = "mxu" in data

    if has_mac and has_mxu:
        mac = data["mac_engine"]
        mxu = data["mxu"]
        # Normalise both to compare
        mac_type = mac.get("type") if isinstance(mac, dict) else None
        mxu_type = mxu.get("type") if isinstance(mxu, dict) else None
        if mac_type != mxu_type:
            raise ConfigError(
                f"Conflicting 'mac_engine' and 'mxu' keys: "
                f"mac_engine.type={mac_type!r}, mxu.type={mxu_type!r}. "
                f"Remove one or ensure they are consistent.",
                field_path="mac_engine|mxu",
            )
        loss.warnings.append(
            "Both 'mac_engine' and 'mxu' present and consistent; using 'mac_engine'. "
            "Remove 'mxu' to silence this warning."
        )
        del data["mxu"]
    elif has_mxu and not has_mac:
        data["mac_engine"] = data.pop("mxu")
        loss.warnings.append("Renamed legacy 'mxu' → 'mac_engine'.")
    elif not has_mac:
        # Neither present — let validation catch it
        pass

    # ── Clean memory section ──
    mem = data.get("memory")
    if isinstance(mem, dict):
        if "bandwidth_bytes_per_cycle" in mem:
            del mem["bandwidth_bytes_per_cycle"]
            loss.dropped_keys.append("memory.bandwidth_bytes_per_cycle")
            loss.warnings.append(
                "Dropped 'memory.bandwidth_bytes_per_cycle' — "
                "use bandwidth_gbps + frequency_mhz in v2."
            )

    return data, loss


def project_v2_to_legacy(config: dict[str, Any]) -> tuple[dict[str, Any], LossReport]:
    """Project a v2 config back to the legacy v1 shape.

    Returns (legacy_config, loss_report).  The input dict is NOT mutated.

    This is a lossy projection — v2-only fields (provenance, etc.) are dropped
    and recorded in the loss report.
    """
    data = deepcopy(config)
    loss = LossReport()

    # ── version: v2 sets "2"; legacy has no version field ──
    if "version" in data:
        del data["version"]

    # ── mac_engine → mxu ──
    if "mac_engine" in data:
        data["mxu"] = data.pop("mac_engine")
    if "cores" in data:
        # cores was present in v1, keep it but note
        pass

    # ── Drop provenance from mac_engine ──
    mxu_section = data.get("mxu")
    if isinstance(mxu_section, dict) and "provenance" in mxu_section:
        del mxu_section["provenance"]
        loss.dropped_keys.append("mxu.provenance")

    # ── memory: add back bandwidth_bytes_per_cycle (same as bandwidth_gbps as legacy did) ──
    mem = data.get("memory")
    if isinstance(mem, dict):
        bw_gbps = mem.get("bandwidth_gbps")
        if bw_gbps is not None:
            mem["bandwidth_bytes_per_cycle"] = float(bw_gbps)
        if "provenance" in mem:
            del mem["provenance"]
            loss.dropped_keys.append("memory.provenance")

    # ── Drop any provenance fields from top-level or other sections ──
    for key in list(data.keys()):
        val = data[key]
        if isinstance(val, dict) and "provenance" in val:
            del val["provenance"]
            loss.dropped_keys.append(f"{key}.provenance")

    return data, loss
