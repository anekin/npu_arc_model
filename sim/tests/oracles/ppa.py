"""Independent closed-form PPA/energy oracle for 3D/HBM/LPDDR memory.

This oracle recomputes component costs from first principles without calling
any production estimator.  It is intentionally separate from
``models.onchip_dram`` and ``engine.ppa_model`` so it can serve as an
independent checker for the new backend protocol.

The oracle models:
* memory die area ∝ capacity;
* TSV/interface area ∝ bandwidth;
* external PHY/package as fixed costs when required;
* leakage ∝ capacity;
* dynamic energy ∝ read/write bytes;
* active power = energy / time;
* TSMC 12nm density ratio of 2.70× (not the old 2.94×).
"""

from __future__ import annotations

from typing import Dict, List, Literal


NODE_DENSITY_RATIO_12NM = 2.70  # TSMC 12FFC density ratio vs 7nm baseline

# Baseline macro parameters @ 12nm (engineering assumptions — see memory_macros.yaml).
_MEMORY_DIE_AREA_PER_GB_MM2 = 2.5
_TSV_AREA_PER_GBPS_MM2 = 0.02
_PHY_AREA_FIXED_MM2 = 5.0
_PACKAGE_AREA_FIXED_MM2 = 2.0
_LEAKAGE_PER_GB_W = 0.05
_READ_ENERGY_PER_BYTE_J = 2e-12
_WRITE_ENERGY_PER_BYTE_J = 3e-12
_ACCESS_LATENCY_NS = 10.0
_THERMAL_PROXY_FACTOR = 0.5


def memory_ppa_oracle(
    tier: Literal["on_chip_3d_dram", "hbm2e", "hbm3", "lpddr5", "lpddr5x"],
    capacity_gb: float,
    bandwidth_gbps: float,
    read_bytes: int,
    write_bytes: int,
    active_time_seconds: float = 1e-6,
    process_node_nm: float = 12.0,
) -> Dict[str, float]:
    """Return an oracle-estimated PPA/energy dict.

    The returned dict mirrors the fields in ``models.memory_backend.MemoryResponse``
    so tests can compare directly.
    """
    if capacity_gb <= 0:
        raise ValueError(f"capacity_gb must be positive, got {capacity_gb}")
    if bandwidth_gbps <= 0:
        raise ValueError(f"bandwidth_gbps must be positive, got {bandwidth_gbps}")
    if read_bytes < 0 or write_bytes < 0:
        raise ValueError("read_bytes and write_bytes must be non-negative")
    if active_time_seconds <= 0:
        raise ValueError("active_time_seconds must be positive")

    # Node scale relative to 12nm baseline; oracle anchors at 12nm = 1.0.
    node_scale = _node_scale(process_node_nm)

    # Component inclusion depends on tier type.
    include_phy, include_tsv, include_package = _component_flags(tier)

    # Memory die area monotonic with capacity.
    memory_die_area = capacity_gb * _MEMORY_DIE_AREA_PER_GB_MM2 * node_scale

    # Interface area monotonic with bandwidth; PHY/package fixed when present.
    tsv_area = bandwidth_gbps * _TSV_AREA_PER_GBPS_MM2 * node_scale if include_tsv else 0.0
    phy_area = _PHY_AREA_FIXED_MM2 * node_scale if include_phy else 0.0
    package_area = _PACKAGE_AREA_FIXED_MM2 * node_scale if include_package else 0.0
    interface_area = tsv_area + phy_area + package_area

    total_area = memory_die_area + interface_area

    # Leakage monotonic with capacity.
    static_power = capacity_gb * _LEAKAGE_PER_GB_W

    # Dynamic energy monotonic with bytes.
    read_energy = read_bytes * _READ_ENERGY_PER_BYTE_J
    write_energy = write_bytes * _WRITE_ENERGY_PER_BYTE_J
    dynamic_energy = read_energy + write_energy

    active_power = dynamic_energy / active_time_seconds

    latency = _ACCESS_LATENCY_NS * 1e-9 + 1.0 / max(bandwidth_gbps * 1e9, 1.0)

    thermal_proxy = (
        (static_power + active_power) / max(total_area, 1.0) * _THERMAL_PROXY_FACTOR
    )

    return {
        "latency_seconds": latency,
        "memory_die_area_mm2": memory_die_area,
        "interface_area_mm2": interface_area,
        "total_area_mm2": total_area,
        "static_power_w": static_power,
        "dynamic_energy_j": dynamic_energy,
        "active_power_w": active_power,
        "thermal_proxy_c": thermal_proxy,
    }


def _node_scale(process_node_nm: float) -> float:
    """Return area scale relative to 12nm baseline.

    Uses the density ratio 2.70× for 12nm vs 7nm; other nodes keep the
    conventional quadratic scaling so the oracle remains closed-form.
    """
    if process_node_nm == 12.0:
        return 1.0
    # 12nm vs 7nm density ratio is 2.70, so area at 7nm = area_at_12nm / 2.70.
    # Scale = (node / 12nm)^2 adjusted so node=7 gives 1/2.70.
    return (process_node_nm / 12.0) ** 2 / NODE_DENSITY_RATIO_12NM


def _component_flags(
    tier: Literal["on_chip_3d_dram", "hbm2e", "hbm3", "lpddr5", "lpddr5x"],
) -> tuple[bool, bool, bool]:
    """Return (include_phy, include_tsv, include_package) for a tier."""
    if tier == "on_chip_3d_dram":
        return False, True, True
    if tier in {"hbm2e", "hbm3"}:
        return True, True, True
    if tier in {"lpddr5", "lpddr5x"}:
        return True, False, True
    raise ValueError(f"unsupported tier: {tier}")


def component_manifest_ok(
    tier: Literal["on_chip_3d_dram", "hbm2e", "hbm3", "lpddr5", "lpddr5x"],
    components: List[str],
) -> bool:
    """Return True if ``components`` satisfies the tier manifest rules."""
    required, excluded = _manifest_rules(tier)
    return (
        all(c in components for c in required) and not any(c in components for c in excluded)
    )


def _manifest_rules(
    tier: Literal["on_chip_3d_dram", "hbm2e", "hbm3", "lpddr5", "lpddr5x"],
) -> tuple[List[str], List[str]]:
    """Return (required, excluded) component lists for a tier."""
    if tier == "on_chip_3d_dram":
        return ["pcie", "tsv"], ["dram_phy"]
    if tier in {"hbm2e", "hbm3"}:
        return ["dram_phy", "pcie", "tsv"], []
    if tier in {"lpddr5", "lpddr5x"}:
        return ["dram_phy", "pcie"], ["tsv"]
    raise ValueError(f"unsupported tier: {tier}")
