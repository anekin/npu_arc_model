"""Tests for scenario-driven design-space generation.

Given: a Scenario and declarative orthogonal axes
When:  DesignSpace generates candidates
Then:  constraints are enforced, every requested value is generated or
       excluded with a reason, and design-point IDs are deterministic.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import pytest

from contracts.errors import ConfigError
from dse.manifest import CoverageManifest
from dse.space import (
    AXES_PATH,
    DesignPoint,
    DesignSpace,
    GenerationResult,
    load_design_space_from_yaml,
)
from scenarios.schema import ArrivalPattern, Scenario, WorkloadClass


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_scenario() -> Scenario:
    """A minimal valid Scenario for fast tests."""
    return Scenario(
        name="minimal-test",
        description="Minimal scenario for unit tests",
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
def tiny_axes_config() -> Dict[str, Any]:
    """Small axis configuration for fast deterministic tests."""
    return {
        "axes": {
            "engine": {"values": ["block", "systolic"]},
            "array_height": {"values": [64, 128]},
            "array_width": {"values": [64, 128]},
            "frequency_mhz": {"values": [1000]},
            "weight_precision_bits": {"values": [4]},
            "activation_precision_bits": {"values": [8]},
            "memory_type": {"values": ["lpddr5", "on_chip_3d_dram"]},
            "bandwidth_gbps": {"values": [51.2, 500]},
            "dram_width_bits": {"values": [0, 64]},
            "sram_l2_kb": {"values": [2048]},
            "weight_cache": {"values": [False]},
            "on_chip_capacity_gb": {"values": [0, 5]},
            "on_chip_bandwidth_gbps": {"values": [0, 500]},
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
            "array_height": 64,
            "array_width": 64,
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
        },
        "constraints": [
            {
                "name": "lpddr5_no_onchip",
                "when": {"memory_type": ["lpddr5"]},
                "require": {
                    "on_chip_capacity_gb": [0],
                    "on_chip_bandwidth_gbps": [0],
                    "dram_width_bits": [64],
                },
                "reason": "lpddr5_no_onchip",
            },
            {
                "name": "onchip_requires_3d",
                "when": {"memory_type": ["on_chip_3d_dram"]},
                "require": {
                    "dram_width_bits": [0],
                    "on_chip_capacity_gb": [5],
                    "on_chip_bandwidth_gbps": [500],
                },
                "reason": "onchip_dram_3d",
            },
        ],
        "reason_codes": {
            "lpddr5_no_onchip": "LPDDR5 is external DRAM; on-chip 3D DRAM must be disabled",
            "onchip_dram_3d": "On-chip 3D DRAM requires positive capacity/bandwidth and no external DRAM PHY",
        },
    }


# ── Space generation and constraints ─────────────────────────────────────────


class TestSpaceGeneration:
    """Given: declarative axes. When: DesignSpace generates. Then: constraints hold."""

    def test_tiny_full_generates_points(self, minimal_scenario, tiny_axes_config):
        space = DesignSpace(minimal_scenario, axes_config=tiny_axes_config, mode="full")
        result = space.generate_with_exclusions()
        assert len(result.points) > 0

    def test_full_mode_3d_onchip_generated(self, minimal_scenario, tiny_axes_config):
        space = DesignSpace(minimal_scenario, axes_config=tiny_axes_config, mode="full")
        points = space.generate()
        onchip = [p for p in points if p.axis_values["memory_type"] == "on_chip_3d_dram"]
        assert len(onchip) > 0
        assert all(p.axis_values["on_chip_capacity_gb"] > 0 for p in onchip)

    def test_lpddr5_points_have_no_onchip(self, minimal_scenario, tiny_axes_config):
        space = DesignSpace(minimal_scenario, axes_config=tiny_axes_config, mode="full")
        points = space.generate()
        for p in points:
            if p.axis_values["memory_type"] == "lpddr5":
                assert p.axis_values["on_chip_capacity_gb"] == 0
                assert p.axis_values["dram_width_bits"] != 0

    def test_onchip_points_have_no_dram_width(self, minimal_scenario, tiny_axes_config):
        space = DesignSpace(minimal_scenario, axes_config=tiny_axes_config, mode="full")
        points = space.generate()
        for p in points:
            if p.axis_values["memory_type"] == "on_chip_3d_dram":
                assert p.axis_values["dram_width_bits"] == 0
                assert p.axis_values["on_chip_capacity_gb"] > 0

    def test_exclusions_carry_reason_codes(self, minimal_scenario, tiny_axes_config):
        space = DesignSpace(minimal_scenario, axes_config=tiny_axes_config, mode="full")
        result = space.generate_with_exclusions()
        assert all(e.reason for e in result.exclusions)

    def test_unknown_mode_rejects(self, minimal_scenario, tiny_axes_config):
        with pytest.raises(ConfigError):
            DesignSpace(minimal_scenario, axes_config=tiny_axes_config, mode="invalid")


# ── CI all-axes coverage ─────────────────────────────────────────────────────


class TestCiAllAxesCoverage:
    """Given: full YAML axes. When: ci_all_axes mode runs. Then: every value covered or excluded."""

    def test_all_axes_coverage(self, minimal_scenario):
        space = load_design_space_from_yaml(minimal_scenario, AXES_PATH, mode="ci_all_axes")
        result = space.generate_with_exclusions()
        manifest = CoverageManifest(space.axes, result.points, result.exclusions)
        for point in result.points:
            manifest.record_success(point)
        manifest.raise_if_invalid()

    def test_ci_all_axes_3d_onchip_generated(self, minimal_scenario):
        space = load_design_space_from_yaml(minimal_scenario, AXES_PATH, mode="ci_all_axes")
        points = space.generate()
        onchip = [p for p in points if p.axis_values["memory_type"] == "on_chip_3d_dram"]
        assert len(onchip) > 0
        assert all(p.axis_values["on_chip_capacity_gb"] > 0 for p in onchip)

    def test_ci_all_axes_touches_every_engine(self, minimal_scenario):
        space = load_design_space_from_yaml(minimal_scenario, AXES_PATH, mode="ci_all_axes")
        result = space.generate_with_exclusions()
        generated_engines = {p.axis_values["engine"] for p in result.points}
        excluded_engines = {e.value for e in result.exclusions if e.axis == "engine"}
        requested_engines = set(space.axes["engine"].values)
        assert requested_engines == generated_engines | excluded_engines


# ── Determinism and identity ─────────────────────────────────────────────────


class TestDeterminism:
    """Given: same scenario and axes. When: input order is shuffled. Then: ID set is stable."""

    def test_shuffled_axis_order_same_id_set(self, minimal_scenario, tiny_axes_config):
        ordered = tiny_axes_config
        shuffled = copy.deepcopy(ordered)
        items = list(shuffled["axes"].items())
        # Reverse the axis declaration order
        shuffled["axes"] = dict(reversed(items))

        space_ordered = DesignSpace(minimal_scenario, axes_config=ordered, mode="full")
        space_shuffled = DesignSpace(minimal_scenario, axes_config=shuffled, mode="full")

        ids_ordered = {p.design_point_id for p in space_ordered.generate()}
        ids_shuffled = {p.design_point_id for p in space_shuffled.generate()}
        assert ids_ordered == ids_shuffled

    def test_axis_change_changes_id(self, minimal_scenario, tiny_axes_config):
        space = DesignSpace(minimal_scenario, axes_config=tiny_axes_config, mode="full")
        points = space.generate()
        base = points[0]
        # Modify one axis value and rebuild the ID source for the same hardware
        changed_axis_values = dict(base.axis_values)
        changed_axis_values["frequency_mhz"] = 1200
        changed_point = DesignPoint(
            design_point_id="",
            hardware_config=base.hardware_config,
            scenario_ref=base.scenario_ref,
            workload_ref=base.workload_ref,
            axis_values=changed_axis_values,
        )
        assert changed_point.design_point_id != base.design_point_id


# ── YAML loading ─────────────────────────────────────────────────────────────


class TestYamlLoading:
    """Given: the real dse_axes.yaml. When: loaded. Then: all axes have values and reason codes."""

    def test_real_yaml_loads(self, minimal_scenario):
        space = load_design_space_from_yaml(minimal_scenario, AXES_PATH, mode="ci_all_axes")
        assert len(space.axes) > 0
        assert all(spec.values for spec in space.axes.values())

    def test_reason_codes_defined_for_constraints(self, minimal_scenario):
        space = load_design_space_from_yaml(minimal_scenario, AXES_PATH, mode="ci_all_axes")
        for constraint in space.constraints:
            assert constraint.reason
            assert space.exclusion_reason(constraint.reason)
