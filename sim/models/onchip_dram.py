"""Parametric 3D on-chip DRAM PPA/energy backend.

``Parametric3DMemoryBackend`` implements the ``MemoryBackend`` protocol using
closed-form macro models:

* memory die area grows monotonically with capacity;
* TSV/interface area grows monotonically with bandwidth and lane count;
* leakage/static power grows with capacity;
* dynamic energy grows with read/write bytes;
* active power = dynamic energy / active time;
* parameters outside the calibration envelope are marked exploratory.

The model is intentionally simple: it is meant for DSE sensitivity analysis,
not for signed-off silicon predictions.  Out-of-range parameters produce a
response with ``validity.status='engineering_assumption'`` and ``trust_level='T0'``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, ConfigDict, Field

from contracts.errors import ConfigError
from contracts.hardware import Provenance, TrustLevel
from models.memory_backend import (
    MemoryAccessPattern,
    MemoryBackend,
    MemoryRequest,
    MemoryResponse,
    MemoryTopology,
    ValidityEnvelope,
    validate_component_manifest,
)


class MacroParameter(BaseModel):
    """A single physical parameter with provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float = Field(..., description="Numerical value")
    unit: str = Field(..., description="Physical unit")
    provenance: Provenance
    status: str = Field(
        default="engineering_assumption",
        pattern=r"^(engineering_assumption|calibrated_estimate|authoritative)$",
    )
    range: Dict[str, float] = Field(default_factory=dict, description="min/max sweep range")


class MemoryMacroTable(BaseModel):
    """Loaded table of memory macro parameters."""

    model_config = ConfigDict(extra="forbid")

    process_node_nm: float = Field(default=12.0, gt=0)
    memory_die_area_per_gb_mm2: MacroParameter
    tsv_area_per_gbps_mm2: MacroParameter
    phy_area_fixed_mm2: MacroParameter
    package_area_fixed_mm2: MacroParameter
    leakage_power_per_gb_w: MacroParameter
    read_energy_per_byte_j: MacroParameter
    write_energy_per_byte_j: MacroParameter
    access_latency_ns: MacroParameter
    thermal_proxy_per_w_per_mm2_c: MacroParameter


class Parametric3DMemoryBackend(MemoryBackend):
    """Closed-form 3D DRAM backend with monotonic component costs."""

    def __init__(self, macros: MemoryMacroTable | None = None):
        self.macros = macros or _default_macros()

    @property
    def validity_envelope(self) -> ValidityEnvelope:
        return ValidityEnvelope(
            capacity_gb_min=0.1,
            capacity_gb_max=16.0,
            bandwidth_gbps_min=100.0,
            bandwidth_gbps_max=1000.0,
            read_bytes_max=1 << 30,
            write_bytes_max=1 << 30,
            trust_level="T0",
            status="engineering_assumption",
            reason="uncalibrated parametric macro; for exploratory sensitivity only",
        )

    def estimate(self, request: MemoryRequest) -> MemoryResponse:
        topology = request.topology
        capacity_gb = request.capacity_gb
        bandwidth_gbps = request.bandwidth_gbps
        access = request.access

        # On-chip 3D DRAM must NOT add an external PHY.
        if topology.tier == "on_chip_3d_dram" and topology.include_phy:
            raise ConfigError(
                "on-chip 3D DRAM topology must not include external dram_phy",
                field_path="topology.include_phy",
            )

        m = self.macros
        env = self.validity_envelope

        # Memory die area is monotonic in capacity.
        memory_die_area = capacity_gb * m.memory_die_area_per_gb_mm2.value

        # Interface area is monotonic in bandwidth (TSV + PHY when required).
        tsv_area = bandwidth_gbps * m.tsv_area_per_gbps_mm2.value if topology.include_tsv else 0.0
        phy_area = m.phy_area_fixed_mm2.value if topology.include_phy else 0.0
        package_area = m.package_area_fixed_mm2.value if topology.include_package else 0.0
        interface_area = tsv_area + phy_area + package_area

        total_area = memory_die_area + interface_area

        # Static power grows with capacity.
        static_power = capacity_gb * m.leakage_power_per_gb_w.value

        # Dynamic energy grows with read/write bytes.
        read_energy = access.read_bytes * m.read_energy_per_byte_j.value
        write_energy = access.write_bytes * m.write_energy_per_byte_j.value
        dynamic_energy = read_energy + write_energy

        # Active power = energy / time.
        active_power = dynamic_energy / access.active_time_seconds

        # Latency is a constant macro proxy plus a bandwidth-dependent term.
        base_latency_s = m.access_latency_ns.value * 1e-9
        bw_latency_s = 1.0 / max(bandwidth_gbps * 1e9, 1.0)
        latency = base_latency_s + bw_latency_s

        # Thermal proxy scales with power density.
        thermal_proxy = (
            (static_power + active_power)
            / max(total_area, 1.0)
            * m.thermal_proxy_per_w_per_mm2_c.value
        )

        # Determine trust/status from calibration envelope.
        validity = self._validity_for_request(env, capacity_gb, bandwidth_gbps, access)

        components: Dict[str, float] = {
            "memory_die_area_mm2": memory_die_area,
            "tsv_area_mm2": tsv_area,
            "phy_area_mm2": phy_area,
            "package_area_mm2": package_area,
            "static_power_w": static_power,
            "read_energy_j": read_energy,
            "write_energy_j": write_energy,
            "dynamic_energy_j": dynamic_energy,
            "active_power_w": active_power,
        }

        notes: list[str] = []
        if validity.status == "engineering_assumption":
            notes.append(
                f"Parameters outside calibrated envelope: {validity.reason}"
            )

        return MemoryResponse(
            latency_seconds=latency,
            memory_die_area_mm2=memory_die_area,
            interface_area_mm2=interface_area,
            total_area_mm2=total_area,
            static_power_w=static_power,
            dynamic_energy_j=dynamic_energy,
            active_power_w=active_power,
            thermal_proxy_c=thermal_proxy,
            validity=validity,
            components=components,
            notes=notes,
        )

    def _validity_for_request(
        self,
        env: ValidityEnvelope,
        capacity_gb: float,
        bandwidth_gbps: float,
        access: MemoryAccessPattern,
    ) -> ValidityEnvelope:
        reasons: list[str] = []
        if capacity_gb < env.capacity_gb_min or capacity_gb > env.capacity_gb_max:
            reasons.append(
                f"capacity_gb={capacity_gb} outside [{env.capacity_gb_min}, {env.capacity_gb_max}]"
            )
        if bandwidth_gbps < env.bandwidth_gbps_min or bandwidth_gbps > env.bandwidth_gbps_max:
            reasons.append(
                f"bandwidth_gbps={bandwidth_gbps} outside "
                f"[{env.bandwidth_gbps_min}, {env.bandwidth_gbps_max}]"
            )
        if access.read_bytes > env.read_bytes_max:
            reasons.append(f"read_bytes={access.read_bytes} > {env.read_bytes_max}")
        if access.write_bytes > env.write_bytes_max:
            reasons.append(f"write_bytes={access.write_bytes} > {env.write_bytes_max}")

        if reasons:
            return env.model_copy(
                update={
                    "trust_level": "T0",
                    "status": "engineering_assumption",
                    "reason": "; ".join(reasons),
                }
            )
        return env.model_copy(
            update={
                "trust_level": "T0",
                "status": "engineering_assumption",
                "reason": "calibrated-range parameters, but macro is still uncalibrated",
            }
        )


def _default_macros() -> MemoryMacroTable:
    """Return conservative engineering-assumption macro parameters."""
    return MemoryMacroTable(
        process_node_nm=12.0,
        memory_die_area_per_gb_mm2=MacroParameter(
            value=2.5,
            unit="mm2/GB",
            provenance=Provenance(
                source="Conservative 3D DRAM cell-area rule-of-thumb; uncalibrated",
                trust_level=TrustLevel.T0,
                calibration_range="1.0–4.0 mm2/GB",
                reference_uri=".omo/plans/arc-model-ppa-corrections.md",
            ),
            status="engineering_assumption",
            range={"min": 1.0, "max": 4.0},
        ),
        tsv_area_per_gbps_mm2=MacroParameter(
            value=0.02,
            unit="mm2/GB/s",
            provenance=Provenance(
                source="Industry rule-of-thumb for TSV keep-out + SerDes area",
                trust_level=TrustLevel.T1,
                calibration_range="0.01–0.05 mm2/GB/s",
                reference_uri=".omo/plans/arc-model-ppa-corrections.md#修正-5",
            ),
            status="engineering_assumption",
            range={"min": 0.01, "max": 0.05},
        ),
        phy_area_fixed_mm2=MacroParameter(
            value=5.0,
            unit="mm2",
            provenance=Provenance(
                source="External DRAM PHY area placeholder; scales with node",
                trust_level=TrustLevel.T1,
                calibration_range="3.0–8.0 mm2",
                reference_uri=".omo/plans/arc-model-ppa-corrections.md#修正-5",
            ),
            status="engineering_assumption",
            range={"min": 3.0, "max": 8.0},
        ),
        package_area_fixed_mm2=MacroParameter(
            value=2.0,
            unit="mm2",
            provenance=Provenance(
                source="Package/interposer overhead placeholder",
                trust_level=TrustLevel.T0,
                calibration_range="1.0–4.0 mm2",
                reference_uri=".omo/plans/arc-model-ppa-corrections.md",
            ),
            status="engineering_assumption",
            range={"min": 1.0, "max": 4.0},
        ),
        leakage_power_per_gb_w=MacroParameter(
            value=0.05,
            unit="W/GB",
            provenance=Provenance(
                source="DRAM retention leakage rule-of-thumb; uncalibrated",
                trust_level=TrustLevel.T0,
                calibration_range="0.02–0.10 W/GB",
                reference_uri=".omo/plans/arc-model-ppa-corrections.md",
            ),
            status="engineering_assumption",
            range={"min": 0.02, "max": 0.10},
        ),
        read_energy_per_byte_j=MacroParameter(
            value=2e-12,
            unit="J/byte",
            provenance=Provenance(
                source="3D DRAM access energy rule-of-thumb; uncalibrated",
                trust_level=TrustLevel.T0,
                calibration_range="1e-12–5e-12 J/byte",
                reference_uri=".omo/plans/arc-model-ppa-corrections.md",
            ),
            status="engineering_assumption",
            range={"min": 1e-12, "max": 5e-12},
        ),
        write_energy_per_byte_j=MacroParameter(
            value=3e-12,
            unit="J/byte",
            provenance=Provenance(
                source="3D DRAM write energy rule-of-thumb; uncalibrated",
                trust_level=TrustLevel.T0,
                calibration_range="1e-12–6e-12 J/byte",
                reference_uri=".omo/plans/arc-model-ppa-corrections.md",
            ),
            status="engineering_assumption",
            range={"min": 1e-12, "max": 6e-12},
        ),
        access_latency_ns=MacroParameter(
            value=10.0,
            unit="ns",
            provenance=Provenance(
                source="On-chip 3D DRAM latency placeholder",
                trust_level=TrustLevel.T0,
                calibration_range="5–30 ns",
                reference_uri=".omo/plans/arc-model-ppa-corrections.md",
            ),
            status="engineering_assumption",
            range={"min": 5.0, "max": 30.0},
        ),
        thermal_proxy_per_w_per_mm2_c=MacroParameter(
            value=0.5,
            unit="C/(W/mm2)",
            provenance=Provenance(
                source="Thermal proxy scaling factor; uncalibrated",
                trust_level=TrustLevel.T0,
                calibration_range="0.2–1.0",
                reference_uri=".omo/plans/arc-model-ppa-corrections.md",
            ),
            status="engineering_assumption",
            range={"min": 0.2, "max": 1.0},
        ),
    )


def load_macro_table(path: str | Path) -> MemoryMacroTable:
    """Load a ``MemoryMacroTable`` from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"macro file {path!r} must contain a mapping", field_path="path")
    return MemoryMacroTable.model_validate(data)


def backend_from_macros(path: str | Path) -> Parametric3DMemoryBackend:
    """Build a backend from a YAML macro table."""
    return Parametric3DMemoryBackend(load_macro_table(path))
