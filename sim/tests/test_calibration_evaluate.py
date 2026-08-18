"""Calibration evaluation and trust gate tests.

Covers:
  - TrustGate returns ok/max_trust/violations
  - T0 parameters fail decision-grade mode
  - Exploratory mode succeeds with T0
  - Extrapolation (out-of-range) detection
  - Registry change changes calibration/result digest
  - calibration_ids_for_design_point covers engine-specific parameters
"""

from __future__ import annotations

from pathlib import Path

import pytest
from calibration.evaluate import (
    TrustGate,
    calibration_digest,
    calibration_ids_for_design_point,
    result_digest,
)
from calibration.registry import CalibrationRegistry
from calibration.schema import CalibrationEntry, CalibrationStatus
from contracts.hardware import TrustLevel

# Synthetic T0 calibration entry for trust-gate / runner tests.
# After tensor_core_descriptor_overhead was upgraded to T1, no real T0
# entries exist in parameters.yaml.  Tests that need a T0 parameter
# inject this synthetic entry instead.
SYNTHETIC_T0 = CalibrationEntry(
    calibration_id="synthetic_t0_param",
    value=1.0,
    unit="ratio",
    source_uri=None,
    source_hash=None,
    trust_level=TrustLevel.T0,
    calibration_range="0.5–1.5",
    range_min=0.5,
    range_max=1.5,
    status=CalibrationStatus.assumption,
    description="Synthetic T0 entry for trust-gate testing.",
)


def _base_hw_config(engine_type: str = "block") -> dict:
    return {
        "version": "2",
        "mac_engine": {
            "type": engine_type,
            "array_height": 64,
            "array_width": 64,
            "frequency_mhz": 1000,
            "weight_precision_bits": 4,
            "activation_precision_bits": 8,
            "dataflow": "weight_stationary",
            "double_buffer": True,
            "ops_per_mac": 2,
        },
        "memory": {
            "type": "LPDDR5-6400",
            "bandwidth_gbps": 51.2,
            "dram_efficiency": 0.85,
            "dram_width_bits": 64,
        },
        "sram": {"l1_per_core_kb": 512, "l2_shared_kb": 2048},
        "optimizations": {"weight_cache": False, "dma_bw_multiplier": 1.0},
        "dma": {"descriptor_overhead_cycles": 5},
        "gmma": {"pipeline_scale": 0.05},
        "area_model": {
            "process_node": 7,
            "process_node_nm": 7.0,
            "systolic_pe_area_mm2": 2.0,
            "block_pe_area_mm2": 4.0,
            "os_pe_area_mm2": 4.0,
            "is_pe_area_mm2": 4.0,
            "tc_pe_area_mm2": 4.0,
            "wmma_pe_area_mm2": 6.0,
            "gmma_pe_area_mm2": 7.0,
            "fsa_pe_area_mm2": 2.2,
            "dram_phy_area_mm2": 5.0,
            "tsv_overhead_pct": 0.10,
        },
        "on_chip_memory": {"capacity_gb": 0, "bandwidth_gbps": 0},
    }


def test_calibration_ids_for_block_engine():
    """Block config consumes expected calibration IDs."""
    cfg = _base_hw_config("block")
    ids = calibration_ids_for_design_point(cfg)
    assert "systolic_pe_area_7nm" in ids
    assert "block_pe_area_7nm" in ids
    assert "block_systolic_pe_ratio" in ids
    assert "pj_per_mac_12nm_int8" in ids
    assert "power_density_12nm" in ids
    assert "dram_phy_area_12nm" in ids
    # GMMA/TensorCore/WMMA specific IDs are not used by block.
    assert "gmma_pipeline_scale" not in ids
    assert "tensor_core_descriptor_overhead" not in ids
    assert "wmma_pe_ratio" not in ids


def test_calibration_ids_use_configured_process_node():
    """PE area calibration IDs follow the configured process node."""
    cfg = _base_hw_config("block")
    cfg["area_model"]["process_node_nm"] = 28.0
    ids = calibration_ids_for_design_point(cfg)
    assert "systolic_pe_area_28nm" in ids
    assert "block_pe_area_28nm" in ids
    assert "systolic_pe_area_7nm" not in ids


def test_actual_value_per_node_pe_area():
    """_actual_value derives scaled PE area from AreaModel for the requested node."""
    from calibration.evaluate import _actual_value

    cfg = _base_hw_config("block")
    cfg["area_model"]["process_node_nm"] = 28.0
    assert _actual_value("systolic_pe_area_28nm", cfg) == pytest.approx(32.0, rel=1e-4)
    assert _actual_value("block_pe_area_28nm", cfg) == pytest.approx(64.0, rel=1e-4)

    cfg["area_model"]["process_node_nm"] = 12.0
    assert _actual_value("systolic_pe_area_12nm", cfg) == pytest.approx(5.4, rel=1e-4)
    assert _actual_value("block_pe_area_12nm", cfg) == pytest.approx(10.8, rel=1e-4)


def test_calibration_ids_for_gmma_engine():
    """GMMA config consumes GMMA-specific calibration IDs."""
    cfg = _base_hw_config("gmma")
    ids = calibration_ids_for_design_point(cfg)
    assert "gmma_pipeline_scale" in ids
    assert "gmma_pe_ratio" in ids
    assert "block_systolic_pe_ratio" in ids


def test_calibration_ids_for_tensor_core_engine():
    """TensorCore config consumes descriptor overhead ID."""
    cfg = _base_hw_config("tensor_core")
    ids = calibration_ids_for_design_point(cfg)
    assert "tensor_core_descriptor_overhead" in ids


def test_calibration_ids_for_hbm_memory():
    """HBM config consumes TSV overhead ID."""
    cfg = _base_hw_config("block")
    cfg["memory"]["type"] = "HBM3"
    ids = calibration_ids_for_design_point(cfg)
    assert "tsv_overhead_pct" in ids


def test_calibration_ids_for_onchip_memory():
    """On-chip 3D DRAM consumes TSV overhead and drops external DRAM PHY."""
    cfg = _base_hw_config("block")
    cfg["on_chip_memory"] = {"capacity_gb": 1, "bandwidth_gbps": 500}
    ids = calibration_ids_for_design_point(cfg)
    assert "tsv_overhead_pct" in ids
    assert "dram_phy_area_12nm" not in ids


def test_trust_gate_t2_required_with_synthetic_t0_fails():
    """Decision-grade (T2+) fails when a synthetic T0 parameter is present."""
    real_registry = CalibrationRegistry.from_yaml()
    t1_entry = real_registry.get("block_systolic_pe_ratio")
    synthetic_registry = CalibrationRegistry([SYNTHETIC_T0, t1_entry])
    gate = TrustGate(synthetic_registry)
    ids = {SYNTHETIC_T0.calibration_id, "block_systolic_pe_ratio"}
    ok, max_trust, violations = gate.check(ids, require_trust=TrustLevel.T2)
    assert not ok
    assert max_trust == TrustLevel.T0
    assert any(v["calibration_id"] == SYNTHETIC_T0.calibration_id for v in violations)


def test_trust_gate_exploratory_allows_synthetic_t0():
    """Exploratory mode (default T0 requirement) succeeds with synthetic T0 parameters."""
    real_registry = CalibrationRegistry.from_yaml()
    t1_entry = real_registry.get("block_systolic_pe_ratio")
    synthetic_registry = CalibrationRegistry([SYNTHETIC_T0, t1_entry])
    gate = TrustGate(synthetic_registry)
    ids = {SYNTHETIC_T0.calibration_id, "block_systolic_pe_ratio"}
    ok, max_trust, violations = gate.check(ids)
    assert ok
    assert max_trust == TrustLevel.T0
    assert violations == []


def test_trust_gate_blocks_unknown_id():
    """Unknown calibration ID is reported as a violation."""
    registry = CalibrationRegistry.from_yaml()
    gate = TrustGate(registry)
    ok, _max_trust, violations = gate.check({"unknown_param"}, require_trust=TrustLevel.T0)
    assert not ok
    assert any(v["reason"] == "unknown_calibration_id" for v in violations)


def test_trust_gate_detects_out_of_range():
    """Actual value outside calibration range is flagged as out_of_calibration_range."""
    registry = CalibrationRegistry.from_yaml()
    gate = TrustGate(registry)
    cfg = _base_hw_config("gmma")
    cfg["gmma"]["pipeline_scale"] = 0.99  # well above 0.10 upper bound
    ok, max_trust, violations = gate.check(
        {"gmma_pipeline_scale"},
        hw_config=cfg,
        require_trust=TrustLevel.T0,
    )
    assert not ok
    assert any(v["reason"] == "out_of_calibration_range" for v in violations)
    assert max_trust == TrustLevel.T1


def test_trust_gate_in_range_passes_t1():
    """In-range T1 parameter passes T1 requirement."""
    registry = CalibrationRegistry.from_yaml()
    gate = TrustGate(registry)
    cfg = _base_hw_config("block")
    ok, max_trust, violations = gate.check(
        {"block_systolic_pe_ratio"},
        hw_config=cfg,
        require_trust=TrustLevel.T1,
    )
    assert ok
    assert max_trust == TrustLevel.T1
    assert violations == []


def test_calibration_digest_changes_on_registry_change():
    """Changing a registry entry changes the calibration digest."""
    registry = CalibrationRegistry.from_yaml()
    d1 = calibration_digest(registry)

    modified_data = registry.to_dict()
    modified_data["parameters"]["gmma_pipeline_scale"]["value"] = 0.99
    modified = CalibrationRegistry.from_dict(modified_data["parameters"])
    d2 = calibration_digest(modified)
    assert d1 != d2


def test_result_digest_changes_on_calibration_digest_change():
    """Result digest changes when calibration digest changes."""
    d1 = result_digest("input_a", "workload_a", "cal_a")
    d2 = result_digest("input_a", "workload_a", "cal_b")
    assert d1 != d2


def test_result_digest_stable_for_same_inputs():
    """Result digest is deterministic for identical inputs."""
    d1 = result_digest("input_a", "workload_a", "cal_a")
    d2 = result_digest("input_a", "workload_a", "cal_a")
    assert d1 == d2


def test_runner_exploratory_marks_t0_points_exploratory(monkeypatch):
    """ScenarioDseRunner in exploratory mode marks synthetic-T0-affected points exploratory."""
    from dse.runner import DseRunConfig, ScenarioDseRunner
    from dse.space import load_design_space_from_yaml
    from scenarios.schema import (
        ArrivalMode,
        ArrivalPattern,
        QueuePolicy,
        Scenario,
        WorkloadClass,
    )

    # Inject synthetic T0 into the registry so tests don't depend on real T0 entries.
    real_registry = CalibrationRegistry.from_yaml()
    synthetic_registry = CalibrationRegistry(
        [SYNTHETIC_T0, *real_registry.entries()],
    )
    monkeypatch.setattr(CalibrationRegistry, "from_yaml", lambda: synthetic_registry)

    # Also monkeypatch calibration_ids_for_design_point so the synthetic T0 is consumed.
    _original_ids = calibration_ids_for_design_point
    monkeypatch.setattr(
        "calibration.evaluate.calibration_ids_for_design_point",
        lambda hw: _original_ids(hw) | {SYNTHETIC_T0.calibration_id},
    )

    scenario = Scenario(
        name="test_exploratory",
        workload_ref="llm-qwen25-3b",
        classes=[
            WorkloadClass(
                id="inference",
                arrival=ArrivalPattern(mode=ArrivalMode.PERIODIC, period_ms=10.0, count=10),
                work_ms=1.0,
                relative_deadline_ms=10.0,
                queue_policy=QueuePolicy.FIFO,
                queue_capacity=64,
                resource_requirements={"compute": 1},
            ),
        ],
        compute_capacity=1,
        memory_available_bytes=8_000_000_000,
        max_inflight_jobs=128,
        max_bandwidth_fraction=1.0,
        preemption_enabled=True,
        warmup_count=0,
        measurement_count=1,
    )
    design_space = load_design_space_from_yaml(
        scenario,
        Path(__file__).resolve().parent.parent / "config" / "dse_axes.yaml",
        mode="ci_all_axes",
    )
    config = DseRunConfig(scenario=scenario, design_space=design_space, trust_mode="exploratory")
    result_set, _manifest, _frontier = ScenarioDseRunner(config).run()
    # At least one complete point should be marked exploratory because the
    # synthetic T0 caps trust level below authoritative.
    exploratory = [r for r in result_set.results if r.trust_level.value == "exploratory"]
    assert exploratory, "expected at least one exploratory point in exploratory mode"


def test_runner_decision_grade_fails_on_t0(monkeypatch):
    """ScenarioDseRunner in decision-grade mode raises ConfigError with synthetic T0 present."""
    from contracts.errors import ConfigError
    from dse.runner import DseRunConfig, ScenarioDseRunner
    from dse.space import load_design_space_from_yaml
    from scenarios.schema import (
        ArrivalMode,
        ArrivalPattern,
        QueuePolicy,
        Scenario,
        WorkloadClass,
    )

    # Inject synthetic T0 into the registry.
    real_registry = CalibrationRegistry.from_yaml()
    synthetic_registry = CalibrationRegistry(
        [SYNTHETIC_T0, *real_registry.entries()],
    )
    monkeypatch.setattr(CalibrationRegistry, "from_yaml", lambda: synthetic_registry)

    # Monkeypatch calibration_ids so the synthetic T0 is consumed.
    _original_ids = calibration_ids_for_design_point
    monkeypatch.setattr(
        "calibration.evaluate.calibration_ids_for_design_point",
        lambda hw: _original_ids(hw) | {SYNTHETIC_T0.calibration_id},
    )

    scenario = Scenario(
        name="test_decision_grade",
        workload_ref="llm-qwen25-3b",
        classes=[
            WorkloadClass(
                id="inference",
                arrival=ArrivalPattern(mode=ArrivalMode.PERIODIC, period_ms=10.0, count=10),
                work_ms=1.0,
                relative_deadline_ms=10.0,
                queue_policy=QueuePolicy.FIFO,
                queue_capacity=64,
                resource_requirements={"compute": 1},
            ),
        ],
        compute_capacity=1,
        memory_available_bytes=8_000_000_000,
        max_inflight_jobs=128,
        max_bandwidth_fraction=1.0,
        preemption_enabled=True,
        warmup_count=0,
        measurement_count=1,
    )
    design_space = load_design_space_from_yaml(
        scenario,
        Path(__file__).resolve().parent.parent / "config" / "dse_axes.yaml",
        mode="ci_all_axes",
    )
    config = DseRunConfig(scenario=scenario, design_space=design_space, trust_mode="decision_grade")
    with pytest.raises(ConfigError, match="decision-grade trust gate failed"):
        ScenarioDseRunner(config).run()


def test_tensor_core_overhead_trust():
    """tensor_core_descriptor_overhead is T1 with non-empty source_uri and range [0, 10]."""
    registry = CalibrationRegistry.from_yaml()
    entry = registry.get("tensor_core_descriptor_overhead")
    assert entry.trust_level == TrustLevel.T1, f"expected T1, got {entry.trust_level}"
    assert entry.source_uri, f"source_uri is empty or None: {entry.source_uri!r}"
    assert entry.range_min == 0.0
    assert entry.range_max == 10.0
