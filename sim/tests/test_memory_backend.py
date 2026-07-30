"""MemoryBackend protocol and fake-implementation substitution conformance tests.

Ensures that:
* ``MemoryRequest``/``MemoryResponse``/``MemoryTopology`` forbid unknown fields;
* ``Parametric3DMemoryBackend`` satisfies the ``MemoryBackend`` ABC;
* a fake backend implementation can substitute and run the same conformance suite.
"""

from __future__ import annotations

import pytest
from contracts.errors import ConfigError
from models.memory_backend import (
    MemoryAccessPattern,
    MemoryBackend,
    MemoryRequest,
    MemoryResponse,
    MemoryTopology,
    ValidityEnvelope,
    validate_component_manifest,
)
from models.onchip_dram import Parametric3DMemoryBackend
from pydantic import ValidationError


def _onchip_request(
    capacity_gb: float = 5.0,
    bandwidth_gbps: float = 500.0,
    read_bytes: int = 1_000_000,
    write_bytes: int = 500_000,
) -> MemoryRequest:
    return MemoryRequest(
        topology=MemoryTopology(
            tier="on_chip_3d_dram",
            process_node_nm=12.0,
            include_phy=False,
            include_tsv=True,
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


class FakeMemoryBackend(MemoryBackend):
    """Fake backend used to prove the protocol is substitutable."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    @property
    def validity_envelope(self) -> ValidityEnvelope:
        return ValidityEnvelope(
            capacity_gb_min=0.1,
            capacity_gb_max=100.0,
            bandwidth_gbps_min=1.0,
            bandwidth_gbps_max=2000.0,
            trust_level="T0",
            status="engineering_assumption",
        )

    def estimate(self, request: MemoryRequest) -> MemoryResponse:
        area = request.capacity_gb * request.bandwidth_gbps * 1e-3 * self.scale
        energy = (request.access.read_bytes + request.access.write_bytes) * 1e-12
        return MemoryResponse(
            latency_seconds=1e-9,
            memory_die_area_mm2=area,
            interface_area_mm2=0.0,
            total_area_mm2=area,
            static_power_w=0.01,
            dynamic_energy_j=energy,
            active_power_w=energy / request.access.active_time_seconds,
            thermal_proxy_c=0.1,
            validity=self.validity_envelope,
            components={"fake_area": area, "fake_energy": energy},
        )


@pytest.mark.parametrize("backend", [Parametric3DMemoryBackend(), FakeMemoryBackend()])
def test_backend_is_abstract_instance(backend):
    """Both production and fake backends are ``MemoryBackend`` instances."""
    assert isinstance(backend, MemoryBackend)


@pytest.mark.parametrize("backend", [Parametric3DMemoryBackend(), FakeMemoryBackend()])
def test_backend_estimate_returns_response(backend):
    """Every backend returns a ``MemoryResponse`` for a valid request."""
    request = _onchip_request()
    response = backend.estimate(request)
    assert isinstance(response, MemoryResponse)
    assert response.total_area_mm2 >= 0
    assert response.dynamic_energy_j >= 0
    assert response.active_power_w >= 0
    assert response.validity is not None


@pytest.mark.parametrize("backend", [Parametric3DMemoryBackend(), FakeMemoryBackend()])
def test_backend_estimate_is_deterministic(backend):
    """Equivalent requests produce equivalent responses."""
    request = _onchip_request()
    r1 = backend.estimate(request)
    r2 = backend.estimate(request)
    assert r1.total_area_mm2 == pytest.approx(r2.total_area_mm2, rel=1e-12)
    assert r1.dynamic_energy_j == pytest.approx(r2.dynamic_energy_j, rel=1e-12)


@pytest.mark.parametrize("backend", [Parametric3DMemoryBackend(), FakeMemoryBackend()])
def test_backend_validity_envelope_nonnegative(backend):
    """Validity ranges must be non-negative and well-formed."""
    env = backend.validity_envelope
    assert env.capacity_gb_min >= 0
    assert env.capacity_gb_max >= env.capacity_gb_min
    assert env.bandwidth_gbps_min >= 0
    assert env.bandwidth_gbps_max >= env.bandwidth_gbps_min


def test_request_forbids_unknown_fields():
    """``MemoryRequest`` rejects extra fields."""
    data = _onchip_request().model_dump()
    data["extra_field"] = 123
    with pytest.raises(ValidationError):
        MemoryRequest.model_validate(data)


def test_response_forbids_unknown_fields():
    """``MemoryResponse`` rejects extra fields."""
    response = Parametric3DMemoryBackend().estimate(_onchip_request())
    data = response.model_dump()
    data["extra_field"] = 123
    with pytest.raises(ValidationError):
        MemoryResponse.model_validate(data)


def test_topology_forbids_unknown_fields():
    """``MemoryTopology`` rejects extra fields."""
    data = _onchip_request().topology.model_dump()
    data["unknown"] = True
    with pytest.raises(ValidationError):
        MemoryTopology.model_validate(data)


@pytest.mark.parametrize("backend", [Parametric3DMemoryBackend(), FakeMemoryBackend()])
def test_backend_response_components_are_nonnegative(backend):
    """Area/power/energy breakdown components are non-negative."""
    response = backend.estimate(_onchip_request())
    assert response.memory_die_area_mm2 >= 0
    assert response.interface_area_mm2 >= 0
    assert response.total_area_mm2 >= response.memory_die_area_mm2
    assert response.static_power_w >= 0
    assert response.dynamic_energy_j >= 0
    assert response.active_power_w >= 0
    assert response.thermal_proxy_c >= 0


@pytest.mark.parametrize("backend", [Parametric3DMemoryBackend(), FakeMemoryBackend()])
def test_substitute_backend_same_conformance_shape(backend):
    """A fake backend passes the same structural conformance assertions."""
    request = _onchip_request()
    response = backend.estimate(request)
    assert response.total_area_mm2 > 0 or response.memory_die_area_mm2 > 0
    assert response.validity.trust_level in {"T0", "T1", "T2", "T3"}


@pytest.mark.parametrize(
    "required,excluded,expect_ok",
    [
        (["tsv"], ["dram_phy"], True),
        (["dram_phy"], ["tsv"], False),
        (["tsv"], ["tsv"], False),
    ],
)
def test_component_manifest_validation(required, excluded, expect_ok):
    """Component manifests reject illegal PHY/TSV combinations."""
    topology = MemoryTopology(
        tier="on_chip_3d_dram",
        include_phy=False,
        include_tsv=True,
        include_package=True,
    )
    if expect_ok:
        validate_component_manifest(topology, required=required, excluded=excluded)
    else:
        with pytest.raises(ConfigError):
            validate_component_manifest(topology, required=required, excluded=excluded)


def test_onchip_topology_rejects_external_phy():
    """On-chip 3D DRAM must not include an external PHY."""
    backend = Parametric3DMemoryBackend()
    request = _onchip_request().model_copy(
        update={
            "topology": MemoryTopology(
                tier="on_chip_3d_dram",
                include_phy=True,
                include_tsv=True,
                include_package=True,
            )
        }
    )
    with pytest.raises(ConfigError):
        backend.estimate(request)


@pytest.mark.parametrize("capacity_gb", [0.1, 5.0, 16.0])
@pytest.mark.parametrize("bandwidth_gbps", [100.0, 500.0, 1000.0])
def test_parametric_backend_varies_with_capacity_and_bandwidth(capacity_gb, bandwidth_gbps):
    """Different capacity × bandwidth produce different PPA numbers."""
    backend = Parametric3DMemoryBackend()
    base = backend.estimate(_onchip_request(capacity_gb=0.1, bandwidth_gbps=100.0))
    current = backend.estimate(_onchip_request(capacity_gb=capacity_gb, bandwidth_gbps=bandwidth_gbps))
    assert current.total_area_mm2 >= base.total_area_mm2
    assert current.memory_die_area_mm2 >= base.memory_die_area_mm2
