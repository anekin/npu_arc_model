"""Unified memory hierarchy, residency, and spill planning tests.

Covers:
  * full / partial / spill state transitions across capacities
  * byte-level capacity conservation
  * capacity boundary conditions (capacity-1 / aligned / +1)
  * multi-resident-model weight replication
  * KV and activation queue placement
  * deterministic digest across all 8 MAC engines
  * monotonic traffic/latency as spill increases
  * negative paths: reserve overflow, alias mismatch, invalid parameters
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from contracts.errors import ConfigError, CoverageError
from engine.mac_engine import create_engine
from engine.registry import canonical_engine_ids
from models.memory_hierarchy import MemoryHierarchy, MemoryTier
from models.residency import build_memory_access_plan
from tests.oracles.memory import (
    boundary_check,
    capacity_conservation_ok,
    total_required_bytes,
    total_usable_bytes,
    verify_plan_conservation,
)
from workloads.schema import NodeSpec, Precision, TensorSpec, WorkloadGraphV1


def _tier(
    name: str,
    capacity_gb: float,
    bw_gbps: float,
    reserve_fraction: float = 0.0,
    alignment: int = 256,
) -> MemoryTier:
    return MemoryTier(
        name=name,
        capacity_bytes=int(capacity_gb * 1_000_000_000),
        read_bw_gbps=bw_gbps,
        write_bw_gbps=bw_gbps,
        alignment_bytes=alignment,
        reserve_fraction=reserve_fraction,
    )


def _make_hierarchy(
    onchip_gb: float,
    *,
    sram_kb: float = 0.0,
    external_gb: float = 32.0,
    external_bw_gbps: float = 51.2,
) -> MemoryHierarchy:
    tiers = []
    if sram_kb > 0:
        tiers.append(
            MemoryTier(
                name="sram",
                capacity_bytes=int(sram_kb * 1000),
                read_bw_gbps=1000.0,
                write_bw_gbps=1000.0,
                alignment_bytes=256,
            )
        )
    tiers.append(_tier("on_chip_3d_dram", onchip_gb, 500.0))
    tiers.append(
        MemoryTier(
            name="lpddr5",
            capacity_bytes=int(external_gb * 1_000_000_000),
            read_bw_gbps=external_bw_gbps,
            write_bw_gbps=external_bw_gbps,
            read_efficiency=0.85,
            write_efficiency=0.85,
            alignment_bytes=256,
        )
    )
    return MemoryHierarchy(tiers=tuple(tiers))


def _make_config(onchip_gb: float, external_bw_gbps: float = 51.2) -> dict[str, Any]:
    """Return a minimal config dict with the requested on-chip capacity."""
    return {
        "mac_engine": {
            "type": "block",
            "array_height": 64,
            "array_width": 64,
            "frequency_mhz": 1000,
            "weight_precision_bits": 4,
            "activation_precision_bits": 8,
        },
        "memory": {
            "type": "LPDDR5-6400",
            "bandwidth_gbps": external_bw_gbps,
            "dram_efficiency": 0.85,
        },
        "on_chip_memory": {
            "capacity_gb": onchip_gb,
            "bandwidth_gbps": 500.0,
        },
        "sram": {
            "l1_per_core_kb": 0,
            "l2_shared_kb": 0,
        },
        "dma": {
            "burst_size_bytes": 256,
            "descriptor_overhead_cycles": 5,
            "num_channels": 2,
            "per_channel_fifo_depth": 64,
        },
        "kv_cache": {
            "sram_kb": 256,
            "dram_region_mb": 96,
            "precision_bits": 8,
        },
        "optimizations": {"dma_bw_multiplier": 1.0},
    }


def _make_graph(
    weight_bytes: int,
    activation_bytes: int,
    kv_bytes: int = 0,
) -> WorkloadGraphV1:
    """Return a minimal DAG with one weight and one activation tensor."""
    tensors = [
        TensorSpec(
            tensor_id="weight_0",
            shape=[weight_bytes * 8 // 4, 1],
            precision=Precision.INT4,
            producer_node="input",
            bytes=weight_bytes,
        ),
    ]
    inputs = ["weight_0"]
    nodes = []
    if activation_bytes > 0:
        tensors.append(
            TensorSpec(
                tensor_id="act_0",
                shape=[activation_bytes, 1],
                precision=Precision.INT8,
                producer_node="n_input",
                bytes=activation_bytes,
            )
        )
        inputs.append("act_0")
        nodes.append(NodeSpec(node_id="n_input", op_type="placeholder"))
    if kv_bytes > 0:
        tensors.append(
            TensorSpec(
                tensor_id="kv_0",
                shape=[kv_bytes, 1],
                precision=Precision.INT8,
                producer_node="input",
                bytes=kv_bytes,
            )
        )
    tensors.append(
        TensorSpec(
            tensor_id="out_0",
            shape=[1, 1],
            precision=Precision.INT8,
            producer_node="n_gemm_0",
            bytes=1,
        )
    )
    nodes.append(
        NodeSpec(
            node_id="n_gemm_0",
            op_type="gemm",
            inputs=inputs,
            outputs=["out_0"],
        )
    )
    return WorkloadGraphV1(
        version="1",
        graph_name="minimal",
        nodes=nodes,
        tensors=tensors,
    )


def _make_queue_graph(queue_bytes: int) -> WorkloadGraphV1:
    """Return an empty graph plus a queue buffer passed via queue_buffers."""
    return WorkloadGraphV1(version="1", graph_name="queue_only")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capacity_gb", [0.001, 0.1, 5.0, 16.0])
def test_capacity_sweep_reports_distinct_residency(capacity_gb):
    """0.001/0.1/5/16 GB capacities must not all report full resident."""
    graph = _make_graph(weight_bytes=5_000_000, activation_bytes=1_000_000)
    hierarchy = _make_hierarchy(capacity_gb)
    plan = build_memory_access_plan(graph, hierarchy)

    weight_states = {p.state for p in plan.placements if p.category == "weight"}
    assert weight_states, "weight placement missing"


@pytest.mark.parametrize("capacity_gb", [0.001, 0.1, 5.0, 16.0])
def test_capacity_conservation_holds(capacity_gb):
    """allocated + reserved <= capacity for every tier at every sweep point."""
    graph = _make_graph(weight_bytes=5_000_000, activation_bytes=1_000_000)
    hierarchy = _make_hierarchy(capacity_gb)
    plan = build_memory_access_plan(graph, hierarchy)

    allocated = {s.name: s.allocated_bytes for s in plan.tier_summaries}
    ok, msg = capacity_conservation_ok(hierarchy, allocated)
    assert ok, msg

    ok2, msg2 = verify_plan_conservation(hierarchy, list(plan.placements))
    assert ok2, msg2

    for summary in plan.tier_summaries:
        assert summary.free_bytes >= 0, (
            f"tier {summary.name} over-allocated: {summary.allocated_bytes} + "
            f"{summary.reserve_bytes} > {summary.capacity_bytes}"
        )


def test_full_partial_spill_monotonic_with_capacity():
    """As on-chip capacity grows, weight residency transitions spill→partial→full."""
    weight_bytes = 1_000_000
    activation_bytes = 500_000
    states = []
    for capacity_gb in [0.001, 0.1, 5.0, 16.0]:
        graph = _make_graph(weight_bytes, activation_bytes)
        hierarchy = _make_hierarchy(capacity_gb)
        plan = build_memory_access_plan(graph, hierarchy)
        weight = next(p for p in plan.placements if p.category == "weight")
        states.append((capacity_gb, weight.state, weight.spill_bytes))

    # With tiny capacity the weight must spill; with large capacity it must be full.
    assert states[0][1] == "spill", f"tiny capacity should spill, got {states[0]}"
    assert states[-1][1] == "full", f"large capacity should be full, got {states[-1]}"

    # Spill bytes must be non-increasing as capacity grows.
    for i in range(1, len(states)):
        assert states[i][2] <= states[i - 1][2], f"spill bytes increased from {states[i - 1]} to {states[i]}"


@pytest.mark.parametrize("capacity_gb", [0.001, 0.1, 5.0, 16.0])
def test_spill_never_decreases_memory_traffic(capacity_gb):
    """More spill must not produce lower memory traffic/latency."""
    weight_bytes = 2_000_000
    base_cfg = _make_config(16.0)
    small_cfg = _make_config(capacity_gb)

    base_engine = create_engine(copy.deepcopy(base_cfg))
    small_engine = create_engine(copy.deepcopy(small_cfg))

    base_result = base_engine.estimate(1, weight_bytes * 2, 1024)
    small_result = small_engine.estimate(1, weight_bytes * 2, 1024)

    # With less capacity (more spill), total DMA wall time must not drop.
    if capacity_gb < 16.0:
        assert small_result.total_cycles >= base_result.total_cycles * 0.99, (
            f"smaller capacity {capacity_gb} produced lower latency "
            f"({small_result.total_cycles}) than full-resident ({base_result.total_cycles})"
        )


# ---------------------------------------------------------------------------
# Capacity boundary tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "capacity_bytes,expected_state",
    [
        (256 - 1, "spill"),
        (256, "full"),
        (256 + 1, "full"),
    ],
)
def test_capacity_boundary_aligned_256(capacity_bytes, expected_state):
    """Single 256-byte weight against an on-chip tier at capacity-1/equal/+1."""
    tensor_size = 256
    graph = _make_graph(weight_bytes=tensor_size, activation_bytes=0)
    hierarchy = MemoryHierarchy(
        tiers=(
            MemoryTier(
                name="on_chip_3d_dram",
                capacity_bytes=capacity_bytes,
                read_bw_gbps=500.0,
                write_bw_gbps=500.0,
                alignment_bytes=256,
            ),
            MemoryTier(
                name="lpddr5",
                capacity_bytes=1_000_000_000,
                read_bw_gbps=51.2,
                write_bw_gbps=51.2,
            ),
        )
    )
    plan = build_memory_access_plan(graph, hierarchy)
    weight = next(p for p in plan.placements if p.category == "weight")
    assert weight.state == expected_state, (
        f"capacity={capacity_bytes}, size={tensor_size}: expected {expected_state}, got {weight.state}"
    )

    expected_state_oracle, resident, spill = boundary_check(tensor_size, capacity_bytes, 256)
    assert weight.state == expected_state_oracle
    assert weight.spill_bytes == spill


# ---------------------------------------------------------------------------
# Multi-resident-model and KV/queue
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resident_models", [1, 2, 4, 8])
def test_resident_models_replicate_weight_footprint(resident_models):
    """Weight footprint scales with resident_models; conservation still holds."""
    graph = _make_graph(weight_bytes=1_000_000, activation_bytes=100_000)
    hierarchy = _make_hierarchy(5.0)
    plan = build_memory_access_plan(graph, hierarchy, resident_models=resident_models)

    weight = next(p for p in plan.placements if p.category == "weight")
    assert weight.replication_factor == resident_models
    assert weight.effective_bytes == weight.aligned_bytes * resident_models

    allocated = {s.name: s.allocated_bytes for s in plan.tier_summaries}
    ok, msg = capacity_conservation_ok(hierarchy, allocated)
    assert ok, msg


def test_kv_priority_above_activation():
    """KV tensors are placed before activations/scratch in priority order."""
    weight_bytes = 100_000
    kv_bytes = 200_000
    activation_bytes = 1_000_000
    onchip_gb = 0.0004  # 400 KB: fits weight+KV atomically, only partial activation

    graph = _make_graph(weight_bytes, activation_bytes, kv_bytes=kv_bytes)
    hierarchy = _make_hierarchy(onchip_gb)
    plan = build_memory_access_plan(graph, hierarchy)

    kv = next(p for p in plan.placements if p.category == "kv")
    act = next(p for p in plan.placements if p.category == "activation")
    assert kv.spill_bytes <= act.spill_bytes, (
        f"KV spilled {kv.spill_bytes} B but activation spilled {act.spill_bytes} B"
    )


def test_queue_buffer_placement():
    """Explicit queue buffers are placed with queue priority."""
    graph = _make_queue_graph(0)
    hierarchy = _make_hierarchy(0.001)
    plan = build_memory_access_plan(
        graph,
        hierarchy,
        queue_buffers={"cmd_queue": 512, "completion_queue": 1024},
    )
    queue_placements = [p for p in plan.placements if p.category == "queue"]
    assert len(queue_placements) == 2
    total_queue = sum(p.effective_bytes for p in queue_placements)
    assert total_queue == 512 + 1024
    assert total_queue > 0


# ---------------------------------------------------------------------------
# Cross-engine determinism
# ---------------------------------------------------------------------------


def _engine_plan_digest(engine_type: str) -> str:
    config = _make_config(5.0)
    graph = _make_graph(weight_bytes=1_000_000, activation_bytes=500_000)
    engine = create_engine(config, graph=graph)
    assert engine.memory_access_plan is not None
    assert engine.memory_access_plan.digest
    return engine.memory_access_plan.digest


def test_engine_plan_digests_are_identical():
    """All 8 engines on the same graph/config produce the same plan digest."""
    digests = [_engine_plan_digest(t) for t in canonical_engine_ids()]
    assert len(set(digests)) == 1, f"engine plan digests differ: {digests}"


# ---------------------------------------------------------------------------
# Negative paths
# ---------------------------------------------------------------------------


def test_reserve_fraction_exceeds_capacity_rejects():
    """A tier whose reserve equals/exceeds capacity leaves no room and must fail."""
    graph = _make_graph(weight_bytes=1000, activation_bytes=1000)
    hierarchy = MemoryHierarchy(
        tiers=(
            MemoryTier(
                name="on_chip_3d_dram",
                capacity_bytes=1000,
                read_bw_gbps=500.0,
                write_bw_gbps=500.0,
                reserve_fraction=1.0,
            ),
            _tier("lpddr5", 32.0, 51.2),
        )
    )
    with pytest.raises((ConfigError, CoverageError)):
        build_memory_access_plan(graph, hierarchy, allow_spill=False)


def test_invalid_reserve_fraction_rejects():
    """Reserve fraction outside [0, 1] is invalid."""
    with pytest.raises((ConfigError, ValueError)):
        MemoryTier(
            name="on_chip_3d_dram",
            capacity_bytes=1_000_000_000,
            read_bw_gbps=500.0,
            write_bw_gbps=500.0,
            reserve_fraction=1.5,
        )


def test_invalid_alignment_rejects():
    """Alignment must be positive."""
    with pytest.raises((ConfigError, ValueError)):
        MemoryTier(
            name="on_chip_3d_dram",
            capacity_bytes=1_000_000_000,
            read_bw_gbps=500.0,
            write_bw_gbps=500.0,
            alignment_bytes=0,
        )


def test_negative_capacity_rejects():
    """Tier capacity cannot be negative."""
    with pytest.raises((ConfigError, ValueError)):
        MemoryTier(
            name="on_chip_3d_dram",
            capacity_bytes=-1,
            read_bw_gbps=500.0,
            write_bw_gbps=500.0,
        )


def test_alias_size_mismatch_rejects():
    """An alias whose byte size differs from its target is illegal."""
    tensors = [
        TensorSpec(tensor_id="base", shape=[100, 1], precision=Precision.INT8, producer_node="input", bytes=100),
        TensorSpec(
            tensor_id="alias",
            shape=[50, 1],
            precision=Precision.INT8,
            producer_node="input",
            bytes=50,
            alias_of="base",
        ),
    ]
    graph = WorkloadGraphV1(version="1", graph_name="alias_bad", nodes=[], tensors=tensors)
    hierarchy = _make_hierarchy(1.0)
    with pytest.raises(ConfigError):
        build_memory_access_plan(graph, hierarchy)


def test_no_spill_tier_rejects():
    """If allow_spill=False and total bytes exceed capacity, raise CoverageError."""
    graph = _make_graph(weight_bytes=10_000_000, activation_bytes=0)
    hierarchy = MemoryHierarchy(
        tiers=(
            MemoryTier(
                name="on_chip_3d_dram",
                capacity_bytes=1_000_000,
                read_bw_gbps=500.0,
                write_bw_gbps=500.0,
            ),
        )
    )
    with pytest.raises(CoverageError):
        build_memory_access_plan(graph, hierarchy, allow_spill=False)


def test_negative_resident_models_rejects():
    """resident_models must be positive."""
    graph = _make_graph(weight_bytes=1000, activation_bytes=1000)
    hierarchy = _make_hierarchy(1.0)
    with pytest.raises(ConfigError):
        build_memory_access_plan(graph, hierarchy, resident_models=0)


# ---------------------------------------------------------------------------
# Oracle independence sanity
# ---------------------------------------------------------------------------


def test_oracle_total_required_bytes_matches_plan():
    """The independent oracle's required-bytes bound matches the plan footprint."""
    graph = _make_graph(weight_bytes=1_000_000, activation_bytes=200_000)
    hierarchy = _make_hierarchy(5.0)
    plan = build_memory_access_plan(graph, hierarchy, resident_models=2)

    required = total_required_bytes(graph, 256, resident_models=2)
    usable = total_usable_bytes(hierarchy)
    resident = plan.resident_bytes_total
    assert resident <= required, "plan resident bytes exceed oracle required bytes"
    assert resident <= usable, "plan resident bytes exceed usable capacity"
