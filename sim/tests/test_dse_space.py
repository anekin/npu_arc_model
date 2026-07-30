"""Test: process_node axis in DSE space generation.

Given: a design space with the process_node axis and old_node_sram_l2_limit constraint.
When:  design points are generated in full and CI modes.
Then:  4 process_node variants appear, older nodes cap SRAM at 4096 KB, and
       build_hardware_config propagates the value correctly.
"""

from __future__ import annotations

from typing import Any

import pytest
from dse.hardware_builder import build_hardware_config
from dse.space import DesignSpace
from scenarios.schema import ArrivalPattern, Scenario, WorkloadClass

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_scenario() -> Scenario:
    """A minimal scenario for process_node tests."""
    return Scenario(
        name="process-node-test",
        description="Minimal scenario for process_node tests",
        workload_ref="llm-qwen25-3b",
        classes=[
            WorkloadClass(
                id="main",
                arrival=ArrivalPattern(mode="periodic", period_ms=10, count=10),
                work_ms=1.0,
            ),
        ],
    )


@pytest.fixture
def sram_base_config() -> dict[str, Any]:
    """Base config with area_model.process_node=7."""
    return {
        "area_model": {"process_node": 7},
    }


@pytest.fixture
def fix_axes_with_node() -> dict[str, Any]:
    """Axes config with all other axes fixed to single values and
    process_node having 4 values + SRAM L2 having 5 values."""
    return {
        "axes": {
            "engine": {"values": ["block"]},
            "array_height": {"values": [128]},
            "array_width": {"values": [128]},
            "frequency_mhz": {"values": [1000]},
            "weight_precision_bits": {"values": [4]},
            "activation_precision_bits": {"values": [8]},
            "memory_type": {"values": ["lpddr5"]},
            "bandwidth_gbps": {"values": [51.2]},
            "dram_width_bits": {"values": [64]},
            "sram_l2_kb": {"values": [512, 1024, 2048, 4096, 8192]},
            "weight_cache": {"values": [False]},
            "on_chip_capacity_gb": {"values": [0]},
            "on_chip_bandwidth_gbps": {"values": [0]},
            "process_node": {"values": [28, 22, 12, 7]},
            "queue_policy": {"values": ["fifo"]},
            "nonpreemptible_quantum_ms": {"values": [0.0]},
            "partition": {"values": ["none"]},
            "request_batch": {"values": [1]},
            "active_sequences": {"values": [1]},
            "token_block": {"values": [16]},
            "image_count": {"values": [1]},
            "action_horizon": {"values": [8]},
            "flow_steps": {"values": [4]},
            "resident_models": {"values": [1]},
            "inflight_jobs": {"values": [4]},
        },
        "defaults": {
            "engine": "block",
            "array_height": 128,
            "array_width": 128,
            "frequency_mhz": 1000,
            "weight_precision_bits": 4,
            "activation_precision_bits": 8,
            "memory_type": "lpddr5",
            "bandwidth_gbps": 51.2,
            "dram_width_bits": 64,
            "sram_l2_kb": 2048,
            "weight_cache": False,
            "on_chip_capacity_gb": 0,
            "on_chip_bandwidth_gbps": 0,
            "process_node": 7,
            "queue_policy": "fifo",
            "nonpreemptible_quantum_ms": 0.0,
            "partition": "none",
            "request_batch": 1,
            "active_sequences": 1,
            "token_block": 16,
            "image_count": 1,
            "action_horizon": 8,
            "flow_steps": 4,
            "resident_models": 1,
            "inflight_jobs": 4,
        },
        "constraints": [
            {
                "name": "old_node_sram_l2_limit",
                "when": {"process_node": [28, 22]},
                "require": {"sram_l2_kb": [512, 1024, 2048, 4096]},
                "reason": "old_node_sram_l2_limit",
            },
        ],
        "reason_codes": {
            "old_node_sram_l2_limit": "SRAM L2 limited to 4096 KB on 28nm and 22nm",
        },
    }


# ── Full-mode generation tests ────────────────────────────────────────────────


class TestProcessNodeGenerationFull:
    """full-mode DSE generation with process_node axis."""

    def test_four_process_node_variants(self, minimal_scenario, fix_axes_with_node):
        """Given fixed other axes, full mode generates exactly 4 process_node values."""
        space = DesignSpace(minimal_scenario, axes_config=fix_axes_with_node, mode="full")
        points = space.generate()
        nodes = sorted({p.axis_values.get("process_node") for p in points})
        assert nodes == [7, 12, 22, 28], f"Expected [7, 12, 22, 28], got {nodes}"

    def test_old_node_sram_capacity_limited(self, minimal_scenario, fix_axes_with_node):
        """28nm and 22nm combos with sram_l2_kb=8192 are excluded."""
        space = DesignSpace(minimal_scenario, axes_config=fix_axes_with_node, mode="full")
        result = space.generate_with_exclusions()
        for point in result.points:
            pn: int | None = point.axis_values.get("process_node")  # type: ignore[assignment]
            sram: int | None = point.axis_values.get("sram_l2_kb")  # type: ignore[assignment]
            if pn is not None and pn in (28, 22) and sram is not None:
                assert sram <= 4096, f"{pn}nm with sram_l2_kb={sram} must be excluded"
        exclusions = [e for e in result.exclusions if e.reason == "old_node_sram_l2_limit"]
        assert len(exclusions) > 0, "Expected old_node_sram_l2_limit exclusions"

    def test_new_node_allows_large_sram(self, minimal_scenario, fix_axes_with_node):
        """12nm and 7nm nodes can use sram_l2_kb=8192."""
        space = DesignSpace(minimal_scenario, axes_config=fix_axes_with_node, mode="full")
        points = space.generate()
        has_8192_new = any(
            p.axis_values.get("process_node") in (7, 12) and p.axis_values.get("sram_l2_kb") == 8192
            for p in points
        )
        assert has_8192_new, "12nm and 7nm must allow sram_l2_kb=8192"

    def test_full_total_count_after_constraint(self, minimal_scenario, fix_axes_with_node):
        """Total points = 4 variants × 5 sram, minus 2 excluded (28/22+8192) = 18."""
        space = DesignSpace(minimal_scenario, axes_config=fix_axes_with_node, mode="full")
        points = space.generate()
        assert len(points) == 18, f"Expected 18 points after constraint, got {len(points)}"


# ── CI-mode generation tests ──────────────────────────────────────────────────


class TestProcessNodeGenerationCi:
    """ci_all_axes mode generates process_node variants correctly."""

    def test_ci_all_axes_includes_four_nodes(self, minimal_scenario, fix_axes_with_node):
        """ci_all_axes mode includes all 4 process_node values."""
        space = DesignSpace(minimal_scenario, axes_config=fix_axes_with_node, mode="ci_all_axes")
        points = space.generate()
        nodes = sorted({p.axis_values.get("process_node") for p in points})
        assert nodes == [7, 12, 22, 28], f"Expected [7, 12, 22, 28], got {nodes}"

    def test_ci_all_axes_count_with_node(self, minimal_scenario, fix_axes_with_node):
        """ci_all_axes generates process_node + sram_l2_kb sweeps."""
        space = DesignSpace(minimal_scenario, axes_config=fix_axes_with_node, mode="ci_all_axes")
        points = space.generate()
        # With all other axes fixed, ci_all_axes creates combos by varying each
        # axis against defaults: 4 process_node variants + 5 sram variants,
        # minus 1 duplicate (defaults combo), giving 8 distinct points.
        assert len(points) == 8, (
            f"Expected 8 ci_all_axes points, got {len(points)}"
        )


# ── hardware_builder process_node propagation tests ───────────────────────────


class TestBuildHardwareConfigNode:
    """build_hardware_config propagates process_node to area_model."""

    def test_propagates_28nm(self, sram_base_config):
        combo = _fixed_combo()
        combo["process_node"] = 28
        cfg = build_hardware_config(sram_base_config, combo)
        assert cfg["area_model"]["process_node"] == 28

    def test_propagates_7nm(self, sram_base_config):
        combo = _fixed_combo()
        combo["process_node"] = 7
        cfg = build_hardware_config(sram_base_config, combo)
        assert cfg["area_model"]["process_node"] == 7

    def test_propagates_12nm(self, sram_base_config):
        combo = _fixed_combo()
        combo["process_node"] = 12
        cfg = build_hardware_config(sram_base_config, combo)
        assert cfg["area_model"]["process_node"] == 12

    def test_propagates_22nm(self, sram_base_config):
        combo = _fixed_combo()
        combo["process_node"] = 22
        cfg = build_hardware_config(sram_base_config, combo)
        assert cfg["area_model"]["process_node"] == 22

    def test_backward_compat_no_process_node(self, sram_base_config):
        """Without process_node, area_model.process_node stays at base config default (7)."""
        combo = _fixed_combo()
        assert "process_node" not in combo
        cfg = build_hardware_config(sram_base_config, combo)
        assert cfg["area_model"]["process_node"] == 7

    def test_preserves_existing_area_model_struct(self, sram_base_config):
        """Existing area_model structure is preserved during propagation."""
        from copy import deepcopy
        base = deepcopy(sram_base_config)
        combo = _fixed_combo()
        combo["process_node"] = 28
        cfg = build_hardware_config(base, combo)
        assert "area_model" in cfg
        # The original key survives alongside the new one
        assert cfg["area_model"]["process_node"] == 28


# ── AreaModel process_node propagation tests ──────────────────────────────────


class TestAreaModelProcessNodePropagation:
    """_evaluate_ppa merges the design point's process_node into the config
    used to construct AreaModel, so cross-node DSE uses correct area physics."""

    def test_merged_config_28nm_propagates_node_scale(self, sram_base_config):
        """AreaModel built from config with process_node=28 gets node_scale=16."""
        from copy import deepcopy

        from engine.ppa_model import AreaModel

        cfg = deepcopy(sram_base_config)
        cfg["area_model"]["process_node"] = 28
        am = AreaModel(cfg)
        assert am.process_node_nm == 28.0
        assert am.node_scale == 16.0  # (28/7)^2

    def test_merged_config_7nm_propagates_node_scale(self, sram_base_config):
        """AreaModel built from config with process_node=7 gets node_scale=1."""
        from copy import deepcopy

        from engine.ppa_model import AreaModel

        cfg = deepcopy(sram_base_config)
        cfg["area_model"]["process_node"] = 7
        am = AreaModel(cfg)
        assert am.process_node_nm == 7.0
        assert am.node_scale == 1.0

    def test_merged_config_12nm_propagates_node_scale(self, sram_base_config):
        """AreaModel gets node_scale=2.70 (density ratio) for 12nm."""
        from copy import deepcopy

        from engine.ppa_model import AreaModel

        cfg = deepcopy(sram_base_config)
        cfg["area_model"]["process_node"] = 12
        am = AreaModel(cfg)
        assert am.process_node_nm == 12.0
        assert am.node_scale == 2.70

    def test_default_without_point_node_stays_7nm(self, sram_base_config):
        """Without an override, AreaModel defaults to process_node=7."""
        from engine.ppa_model import AreaModel

        am = AreaModel(sram_base_config)
        assert am.process_node_nm == 7.0
        assert am.node_scale == 1.0

    def test_28nm_area_greater_than_7nm_for_same_config(self, sram_base_config):
        """28nm produces larger area than 7nm for the same block engine config."""
        from copy import deepcopy

        from engine.ppa_model import AreaModel

        cfg_28 = deepcopy(sram_base_config)
        cfg_28["area_model"]["process_node"] = 28
        cfg_28["mac_engine"] = {"type": "block", "array_height": 128, "array_width": 128,
                                "frequency_mhz": 1000, "weight_precision_bits": 4,
                                "activation_precision_bits": 8, "dataflow": "weight_stationary",
                                "double_buffer": True, "ops_per_mac": 2}
        cfg_28["memory"] = {"type": "LPDDR5-6400", "bandwidth_gbps": 51.2, "dram_efficiency": 0.85,
                            "dram_width_bits": 64}
        cfg_28["sram"] = {"l2_shared_kb": 2048, "l1_per_core_kb": 512}
        cfg_28["on_chip_memory"] = {"capacity_gb": 0, "bandwidth_gbps": 0}
        cfg_28["optimizations"] = {"weight_cache": False}

        cfg_7 = deepcopy(cfg_28)
        cfg_7["area_model"]["process_node"] = 7

        am28 = AreaModel(cfg_28)
        am7 = AreaModel(cfg_7)

        area28 = am28.estimate(cfg_28, "block")["total_mm2"]
        area7 = am7.estimate(cfg_7, "block")["total_mm2"]

        assert area28 > area7, f"28nm area ({area28:.1f}) must exceed 7nm area ({area7:.1f})"
        ratio = area28 / area7
        # SRAM uses sub-quadratic bitcell scaling (~3.4× from 7nm to 28nm),
        # logic uses node_scale=16×. The total ratio is a blend — >2.0 is expected.
        assert ratio > 2.0, f"28nm area should exceed 7nm (got {ratio:.1f}x)"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fixed_combo() -> dict[str, Any]:
    """Return a minimal combo dict without process_node."""
    return {
        "engine": "block",
        "array_height": 128,
        "array_width": 128,
        "frequency_mhz": 1000,
        "weight_precision_bits": 4,
        "activation_precision_bits": 8,
        "memory_type": "lpddr5",
        "bandwidth_gbps": 51.2,
        "dram_width_bits": 64,
        "sram_l2_kb": 2048,
        "weight_cache": False,
        "on_chip_capacity_gb": 0,
        "on_chip_bandwidth_gbps": 0,
        "queue_policy": "fifo",
        "nonpreemptible_quantum_ms": 0.0,
        "partition": "none",
        "request_batch": 1,
        "active_sequences": 1,
        "token_block": 16,
        "image_count": 1,
        "action_horizon": 8,
        "flow_steps": 4,
        "resident_models": 1,
        "inflight_jobs": 4,
    }
