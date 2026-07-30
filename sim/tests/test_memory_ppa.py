"""Parametric 3D memory PPA monotonicity, energy, and invalid/extrapolation tests.

Compares the production ``Parametric3DMemoryBackend`` against the independent
closed-form oracle in ``tests.oracles.ppa``.  All monotonicity assertions are
relative: larger capacity/bandwidth/bytes must not produce smaller area,
power, or energy.
"""

from __future__ import annotations

from typing import Literal

import pytest
from contracts.errors import ConfigError
from models.memory_backend import (
    MemoryAccessPattern,
    MemoryRequest,
    MemoryTopology,
)
from models.onchip_dram import Parametric3DMemoryBackend
from tests.oracles.ppa import (
    component_manifest_ok,
    memory_ppa_oracle,
)


def _request(
    tier: Literal["on_chip_3d_dram", "hbm2e", "hbm3", "lpddr5", "lpddr5x"] = "on_chip_3d_dram",
    capacity_gb: float = 5.0,
    bandwidth_gbps: float = 500.0,
    read_bytes: int = 1_000_000,
    write_bytes: int = 500_000,
    include_phy: bool | None = None,
    include_tsv: bool | None = None,
    process_node_nm: float = 12.0,
) -> MemoryRequest:
    if tier == "on_chip_3d_dram":
        include_phy = False if include_phy is None else include_phy
        include_tsv = True if include_tsv is None else include_tsv
    elif tier in {"hbm2e", "hbm3"}:
        include_phy = True if include_phy is None else include_phy
        include_tsv = True if include_tsv is None else include_tsv
    else:
        include_phy = True if include_phy is None else include_phy
        include_tsv = False if include_tsv is None else include_tsv

    return MemoryRequest(
        topology=MemoryTopology(
            tier=tier,
            process_node_nm=process_node_nm,
            include_phy=include_phy,
            include_tsv=include_tsv,
            include_package=True,
        ),
        capacity_gb=capacity_gb,
        bandwidth_gbps=bandwidth_gbps,
        access=MemoryAccessPattern(
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            read_fraction=read_bytes / max(read_bytes + write_bytes, 1),
            active_time_seconds=1e-6,
        ),
    )


@pytest.mark.parametrize("process_node_nm", [7.0, 12.0, 22.0, 28.0])
def test_topology_process_node_parameterized(process_node_nm):
    """MemoryTopology carries the requested process node for all supported nodes."""
    request = _request(process_node_nm=process_node_nm)
    assert request.topology.process_node_nm == process_node_nm


@pytest.mark.parametrize("capacity_gb", [0.1, 5.0, 16.0])
@pytest.mark.parametrize("bandwidth_gbps", [100.0, 500.0, 1000.0])
def test_capacity_bandwidth_matrix_distinct_ppa(capacity_gb, bandwidth_gbps):
    """0.1/5/16 GB × 100/500/1000 GB/s produce distinct PPA numbers."""
    backend = Parametric3DMemoryBackend()
    base = backend.estimate(_request(capacity_gb=0.1, bandwidth_gbps=100.0))
    current = backend.estimate(_request(capacity_gb=capacity_gb, bandwidth_gbps=bandwidth_gbps))

    assert current.memory_die_area_mm2 >= base.memory_die_area_mm2
    assert current.total_area_mm2 >= base.total_area_mm2
    assert current.static_power_w >= base.static_power_w
    if capacity_gb > 0.1 or bandwidth_gbps > 100.0:
        assert (current.total_area_mm2, current.static_power_w) != (
            base.total_area_mm2,
            base.static_power_w,
        )


@pytest.mark.parametrize("capacity_gb", [0.1, 1.0, 5.0, 16.0])
@pytest.mark.parametrize("process_node_nm", [7.0, 12.0, 22.0, 28.0])
def test_capacity_monotonic_area_and_leakage(capacity_gb, process_node_nm):
    """Capacity↑ → memory die area and leakage non-decreasing across nodes."""
    backend = Parametric3DMemoryBackend()
    base = backend.estimate(_request(capacity_gb=0.1, process_node_nm=process_node_nm))
    current = backend.estimate(_request(capacity_gb=capacity_gb, process_node_nm=process_node_nm))

    assert current.memory_die_area_mm2 >= base.memory_die_area_mm2
    assert current.static_power_w >= base.static_power_w


@pytest.mark.parametrize("bandwidth_gbps", [100.0, 500.0, 1000.0])
def test_bandwidth_monotonic_interface_area_and_active_power(bandwidth_gbps):
    """Bandwidth↑ → interface area and active power non-decreasing."""
    backend = Parametric3DMemoryBackend()
    base = backend.estimate(_request(bandwidth_gbps=100.0))
    current = backend.estimate(_request(bandwidth_gbps=bandwidth_gbps))

    assert current.interface_area_mm2 >= base.interface_area_mm2
    assert current.active_power_w >= base.active_power_w


@pytest.mark.parametrize("read_bytes", [0, 1_000_000, 10_000_000])
@pytest.mark.parametrize("write_bytes", [0, 500_000, 5_000_000])
def test_access_bytes_monotonic_energy(read_bytes, write_bytes):
    """Access bytes↑ → dynamic energy non-decreasing."""
    backend = Parametric3DMemoryBackend()
    base = backend.estimate(_request(read_bytes=0, write_bytes=0))
    current = backend.estimate(_request(read_bytes=read_bytes, write_bytes=write_bytes))

    assert current.dynamic_energy_j >= base.dynamic_energy_j
    assert current.active_power_w >= base.active_power_w


@pytest.mark.parametrize("capacity_gb", [0.1, 5.0, 16.0])
@pytest.mark.parametrize("bandwidth_gbps", [100.0, 500.0, 1000.0])
@pytest.mark.parametrize("read_bytes", [1_000_000, 10_000_000])
def test_oracle_reproduces_production_ppa(capacity_gb, bandwidth_gbps, read_bytes):
    """Oracle recomputes production backend numbers for 3D DRAM at 12nm."""
    backend = Parametric3DMemoryBackend()
    response = backend.estimate(
        _request(
            capacity_gb=capacity_gb,
            bandwidth_gbps=bandwidth_gbps,
            read_bytes=read_bytes,
            process_node_nm=12.0,
        )
    )
    oracle = memory_ppa_oracle(
        tier="on_chip_3d_dram",
        capacity_gb=capacity_gb,
        bandwidth_gbps=bandwidth_gbps,
        read_bytes=read_bytes,
        write_bytes=500_000,
        active_time_seconds=1e-6,
        process_node_nm=12.0,
    )

    assert response.memory_die_area_mm2 == pytest.approx(oracle["memory_die_area_mm2"], rel=1e-9)
    assert response.total_area_mm2 == pytest.approx(oracle["total_area_mm2"], rel=1e-9)
    assert response.dynamic_energy_j == pytest.approx(oracle["dynamic_energy_j"], rel=1e-9)


@pytest.mark.parametrize(
    "tier,expected_required,expected_excluded",
    [
        ("on_chip_3d_dram", ["pcie", "tsv"], ["dram_phy"]),
        ("hbm2e", ["dram_phy", "pcie", "tsv"], []),
        ("hbm3", ["dram_phy", "pcie", "tsv"], []),
        ("lpddr5", ["dram_phy", "pcie"], ["tsv"]),
        ("lpddr5x", ["dram_phy", "pcie"], ["tsv"]),
    ],
)
def test_tier_component_manifests(tier, expected_required, expected_excluded):
    """On-chip/HBM/LPDDR component manifests satisfy rules."""
    assert component_manifest_ok(tier, expected_required + ["package"])
    assert not component_manifest_ok(tier, expected_excluded[:1])


@pytest.mark.parametrize("tier", ["on_chip_3d_dram", "hbm2e", "lpddr5"])
def test_tier_ppa_component_breakdown(tier):
    """Each tier produces a component breakdown consistent with its manifest."""
    backend = Parametric3DMemoryBackend()
    response = backend.estimate(_request(tier=tier))
    assert response.components["memory_die_area_mm2"] >= 0
    assert response.components["tsv_area_mm2"] >= 0
    if tier in {"on_chip_3d_dram", "lpddr5"}:
        assert response.components["phy_area_mm2"] == (
            0.0 if tier == "on_chip_3d_dram" else pytest.approx(5.0, rel=1e-9)
        )


@pytest.mark.parametrize("process_node_nm", [7.0, 12.0, 22.0, 28.0])
def test_invalid_onchip_with_phy_rejects(process_node_nm):
    """Illegal PHY/TSV combo for on-chip 3D DRAM fails across nodes."""
    backend = Parametric3DMemoryBackend()
    request = _request(
        tier="on_chip_3d_dram",
        include_phy=True,
        include_tsv=False,
        process_node_nm=process_node_nm,
    )
    with pytest.raises(ConfigError):
        backend.estimate(request)


def test_invalid_negative_capacity_rejects():
    """Negative capacity is rejected at the request boundary."""
    with pytest.raises((ConfigError, ValueError)):
        _request(capacity_gb=-1.0)


def test_invalid_zero_bandwidth_rejects():
    """Zero bandwidth is rejected at the request boundary."""
    with pytest.raises((ConfigError, ValueError)):
        _request(bandwidth_gbps=0.0)


def test_extrapolation_beyond_capacity_envelope_is_exploratory():
    """Capacity outside the envelope is marked exploratory, not authoritative."""
    backend = Parametric3DMemoryBackend()
    response = backend.estimate(_request(capacity_gb=100.0))
    assert response.validity.status == "engineering_assumption"
    assert response.validity.trust_level == "T0"
    assert response.validity.reason is not None


def test_extrapolation_beyond_bandwidth_envelope_is_exploratory():
    """Bandwidth outside the envelope is marked exploratory, not authoritative."""
    backend = Parametric3DMemoryBackend()
    response = backend.estimate(_request(bandwidth_gbps=2000.0))
    assert response.validity.status == "engineering_assumption"
    assert response.validity.trust_level == "T0"


def test_calibrated_range_still_engineering_assumption():
    """Even within the calibrated range the macro remains uncalibrated."""
    backend = Parametric3DMemoryBackend()
    response = backend.estimate(_request(capacity_gb=5.0, bandwidth_gbps=500.0))
    assert response.validity.status == "engineering_assumption"
    assert response.validity.trust_level == "T0"


def test_active_power_equals_energy_over_time():
    """active_power_w == dynamic_energy_j / active_time_seconds."""
    backend = Parametric3DMemoryBackend()
    response = backend.estimate(_request())
    assert response.active_power_w == pytest.approx(response.dynamic_energy_j / 1e-6, rel=1e-9)
