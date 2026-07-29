"""Tests for the DSE coverage manifest.

Given: generated design points and structured exclusions
When:  the CoverageManifest tracks evaluation outcomes
Then:  invariants hold, silent omissions are rejected, and duplicate IDs
       are reported.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import pytest

from contracts.errors import CoverageError
from dse.manifest import AxisCoverage, CoverageManifest, ExclusionRecord
from dse.space import DesignPoint, DesignSpace
from scenarios.schema import ArrivalPattern, Scenario, WorkloadClass


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_scenario() -> Scenario:
    return Scenario(
        name="manifest-test",
        description="Minimal scenario for manifest tests",
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
    return {
        "axes": {
            "engine": {"values": ["block"]},
            "array_height": {"values": [64, 128]},
            "array_width": {"values": [64]},
            "frequency_mhz": {"values": [1000]},
            "weight_precision_bits": {"values": [4]},
            "activation_precision_bits": {"values": [8]},
            "memory_type": {"values": ["lpddr5"]},
            "bandwidth_gbps": {"values": [51.2]},
            "dram_width_bits": {"values": [64]},
            "sram_l2_kb": {"values": [2048]},
            "weight_cache": {"values": [False]},
            "on_chip_capacity_gb": {"values": [0]},
            "on_chip_bandwidth_gbps": {"values": [0]},
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
        "constraints": [],
        "reason_codes": {},
    }


@pytest.fixture
def manifest(minimal_scenario, tiny_axes_config) -> CoverageManifest:
    space = DesignSpace(minimal_scenario, axes_config=tiny_axes_config, mode="full")
    result = space.generate_with_exclusions()
    return CoverageManifest(space.axes, result.points, result.exclusions)


# ── Count invariants ─────────────────────────────────────────────────────────


class TestManifestInvariants:
    """Given: a manifest. When: outcomes are recorded. Then: invariants hold."""

    def test_all_success_invariant(self, manifest):
        for point in manifest.points:
            manifest.record_success(point)
        assert manifest.generated_count == 2
        assert manifest.evaluated_count == 2
        assert manifest.successful_count == 2
        assert manifest.pruned_count == 0
        assert manifest.filtered_count == 0
        assert manifest.failed_count == 0
        assert manifest.validate() == []

    def test_pruned_invariant(self, manifest):
        points = list(manifest.points)
        manifest.record_pruned(points[0], "area_budget")
        manifest.record_success(points[1])
        assert manifest.generated_count == 2
        assert manifest.pruned_count == 1
        assert manifest.evaluated_count == 1
        assert manifest.validate() == []

    def test_failed_and_filtered_invariant(self, manifest):
        points = list(manifest.points)
        manifest.record_failed(points[0], "runtime_error")
        manifest.record_filtered(points[1], "area_too_large")
        assert manifest.generated_count == 2
        assert manifest.evaluated_count == 2
        assert manifest.failed_count == 1
        assert manifest.filtered_count == 1
        assert manifest.successful_count == 0
        assert manifest.validate() == []

    def test_invariant_violation_detected(self, manifest):
        # Manually corrupt state: mark only one evaluated but none pruned
        points = list(manifest.points)
        # Reset tracking sets
        manifest._evaluated_ids = {points[0].design_point_id}
        manifest._pruned_ids = set()
        manifest._successful_ids = set()
        manifest._filtered_ids = set()
        manifest._failed_ids = set()
        errors = manifest.validate()
        assert any("generated" in e and "evaluated" in e for e in errors)


# ── Coverage completeness ────────────────────────────────────────────────────


class TestCoverageCompleteness:
    """Given: requested values. When: one is silently omitted. Then: validate fails."""

    def test_missing_value_fails_validation(self, manifest):
        coverage = manifest.axis_coverage["array_height"]
        # Simulate a generator bug: remove 128 from generated without reason
        coverage.generated.discard(128)
        errors = manifest.validate()
        assert any("array_height" in e and "128" in e for e in errors)
        with pytest.raises(CoverageError):
            manifest.raise_if_invalid()

    def test_exclusion_prevents_missing_error(self, manifest):
        coverage = manifest.axis_coverage["array_height"]
        coverage.generated.discard(128)
        coverage.exclusions.append(
            ExclusionRecord(
                axis="array_height",
                value=128,
                reason="area_budget",
                constraint_name="area",
            )
        )
        assert manifest.axis_coverage["array_height"].missing == []


# ── Duplicate ID detection ───────────────────────────────────────────────────


class TestDuplicateId:
    """Given: design points. When: duplicate IDs exist. Then: validate reports them."""

    def test_duplicate_id_detected(self, manifest):
        points = list(manifest.points)
        duplicate = DesignPoint(
            design_point_id=points[0].design_point_id,
            hardware_config=points[1].hardware_config,
            scenario_ref=points[1].scenario_ref,
            workload_ref=points[1].workload_ref,
            axis_values=points[1].axis_values,
        )
        bad_manifest = CoverageManifest(
            manifest.axes,
            [points[0], duplicate],
            manifest.exclusions,
        )
        for p in bad_manifest.points:
            bad_manifest.record_success(p)
        errors = bad_manifest.validate()
        assert any("duplicate" in e for e in errors)


# ── Missing reason ───────────────────────────────────────────────────────────


class TestMissingReason:
    """Given: exclusions. When: a reason is empty. Then: the value remains uncovered."""

    def test_empty_reason_does_not_cover_value(self, manifest):
        coverage = manifest.axis_coverage["array_height"]
        coverage.generated.discard(128)
        coverage.exclusions.append(
            ExclusionRecord(
                axis="array_height",
                value=128,
                reason="",
                constraint_name="area",
            )
        )
        # The value is still considered missing because empty reason is not a
        # valid structured exclusion.  We treat the record as present but the
        # test below asserts validation fails.
        errors = manifest.validate()
        assert any("array_height" in e for e in errors)


# ── Manifest serialization ───────────────────────────────────────────────────


class TestManifestSerialization:
    """Given: a manifest. When: serialized to dict. Then: structure is preserved."""

    def test_to_dict_contains_counts_and_coverage(self, manifest):
        for point in manifest.points:
            manifest.record_success(point)
        data = manifest.to_dict()
        assert data["counts"]["generated"] == 2
        assert "axis_coverage" in data
        assert "array_height" in data["axis_coverage"]
        assert data["axis_coverage"]["array_height"]["missing"] == []


# ── AxisCoverage helpers ─────────────────────────────────────────────────────


class TestAxisCoverage:
    """Given: an AxisCoverage. When: queried. Then: missing is computed correctly."""

    def test_missing_excludes_generated_and_excluded(self):
        cov = AxisCoverage(axis="x", requested={1, 2, 3})
        cov.generated.add(1)
        cov.exclusions.append(ExclusionRecord(axis="x", value=2, reason="ok"))
        assert cov.missing == [3]
