"""Tests for reproducible scenario-driven DSE and replay bundles."""

import shutil
import tempfile
from pathlib import Path

import pytest

from contracts.identity import canonical_json_bytes
from contracts.result import DesignSpaceResultV2
from dse.runner import DseRunConfig, ScenarioDseRunner
from dse.serialization import (
    read_replay_bundle,
    replay_bundle_canonical_digest,
    write_replay_bundle,
)
from dse.space import DesignSpace
from scenarios.schema import (
    ArrivalMode,
    ArrivalPattern,
    Scenario,
    WorkloadClass,
)


def _tiny_axes_config() -> dict:
    return {
        "base_config_source": "config/design_space.yaml",
        "axes": {
            "engine": {"values": ["block", "systolic"], "description": "MAC engine type"},
            "array_height": {"values": [64, 128], "description": "MAC array height"},
            "array_width": {"values": [64, 128], "description": "MAC array width"},
            "frequency_mhz": {"values": [1000], "description": "Core clock frequency"},
            "weight_precision_bits": {"values": [4], "description": "Weight precision"},
            "activation_precision_bits": {"values": [8], "description": "Activation precision"},
            "memory_type": {"values": ["lpddr5"], "description": "Memory type"},
            "bandwidth_gbps": {"values": [51.2], "description": "Memory bandwidth"},
            "dram_width_bits": {"values": [64], "description": "DRAM interface width"},
            "sram_l2_kb": {"values": [2048], "description": "Shared L2 SRAM capacity"},
            "weight_cache": {"values": [False], "description": "PE weight cache"},
            "on_chip_capacity_gb": {"values": [0], "description": "On-chip 3D DRAM capacity"},
            "on_chip_bandwidth_gbps": {"values": [0], "description": "On-chip 3D DRAM bandwidth"},
            "queue_policy": {"values": ["fifo"], "description": "Queueing policy"},
            "nonpreemptible_quantum_ms": {"values": [0.0], "description": "Non-preemptible quantum"},
            "partition": {"values": ["none"], "description": "Resource partition"},
            "request_batch": {"values": [1], "description": "Request batch size"},
            "active_sequences": {"values": [1], "description": "Concurrent active sequences"},
            "token_block": {"values": [16], "description": "Token block size"},
            "image_count": {"values": [1], "description": "Images per inference"},
            "action_horizon": {"values": [8], "description": "VLA action horizon"},
            "flow_steps": {"values": [4], "description": "Flow-model steps"},
            "resident_models": {"values": [1], "description": "Resident models"},
            "inflight_jobs": {"values": [4], "description": "Concurrent executing jobs"},
        },
        "constraints": [],
    }


def _tiny_scenario() -> Scenario:
    return Scenario(
        name="repro_test",
        seed=7,
        warmup_count=1,
        measurement_count=5,
        workload_ref="smolvla-class",
        classes=[
            WorkloadClass(
                id="vla",
                arrival=ArrivalPattern(
                    mode=ArrivalMode.PERIODIC,
                    period_ms=100.0,
                    count=8,
                ),
                work_ms=20.0,
                relative_deadline_ms=80.0,
            )
        ],
        compute_capacity=1,
        memory_available_bytes=8_000_000_000,
        max_inflight_jobs=128,
        max_bandwidth_fraction=1.0,
        preemption_enabled=True,
    )


def _run_once(tmp_path: Path, seed: int = 42, bundle_name: str = "bundle") -> tuple[DesignSpaceResultV2, Path]:
    scenario = _tiny_scenario()
    scenario = scenario.model_copy(update={"seed": seed})
    axes = _tiny_axes_config()
    design_space = DesignSpace(scenario, axes_config=axes, mode="full")
    generation_result = design_space.generate_with_exclusions()
    run_config = DseRunConfig(
        scenario=scenario,
        design_space=design_space,
        seed=seed,
    )
    runner = ScenarioDseRunner(run_config)
    result_set, manifest, frontier = runner.run(generation_result)
    bundle_path = tmp_path / bundle_name
    write_replay_bundle(
        bundle_path,
        result_set=result_set,
        manifest=manifest,
        scenario_dict=scenario.model_dump(mode="json"),
        axes_dict=axes,
        seed=seed,
        run_config={"space": "full"},
        generation_result=generation_result,
    )
    return result_set, bundle_path


def test_same_seed_produces_identical_result_digests(tmp_path: Path):
    result_a, _bundle_a = _run_once(tmp_path, seed=123, bundle_name="a")
    result_b, _bundle_b = _run_once(tmp_path, seed=123, bundle_name="b")
    bytes_a = canonical_json_bytes(result_a.model_dump(mode="json"))
    bytes_b = canonical_json_bytes(result_b.model_dump(mode="json"))
    assert bytes_a == bytes_b
    assert result_a.frontier_design_point_ids == result_b.frontier_design_point_ids


def test_different_seed_produces_different_inputs(tmp_path: Path):
    _result_a, bundle_a = _run_once(tmp_path, seed=1, bundle_name="a")
    _result_b, bundle_b = _run_once(tmp_path, seed=2, bundle_name="b")
    inputs_a = read_replay_bundle(bundle_a)["inputs"]
    inputs_b = read_replay_bundle(bundle_b)["inputs"]
    assert inputs_a["seed"] != inputs_b["seed"]


def test_replay_bundle_contains_inputs_result_coverage_manifest(tmp_path: Path):
    _result, bundle_path = _run_once(tmp_path)
    bundle = read_replay_bundle(bundle_path)
    assert "inputs" in bundle
    assert "result" in bundle
    assert "coverage" in bundle
    assert "metadata" in bundle
    assert bundle["metadata"]["schema_version"] == "1"


def test_replay_bundle_canonical_digest_matches_payload(tmp_path: Path):
    _result, bundle_path = _run_once(tmp_path)
    digest = replay_bundle_canonical_digest(bundle_path)
    bundle = read_replay_bundle(bundle_path)
    inputs_bytes = canonical_json_bytes(bundle["inputs"])
    result_bytes = canonical_json_bytes(bundle["result"])
    coverage_bytes = canonical_json_bytes(bundle["coverage"])
    expected = __import__("hashlib").sha256(inputs_bytes + result_bytes + coverage_bytes).hexdigest()
    assert digest == expected


def test_replay_bundle_refuses_to_overwrite_existing_directory(tmp_path: Path):
    _result, bundle_path = _run_once(tmp_path)
    scenario = _tiny_scenario()
    axes = _tiny_axes_config()
    design_space = DesignSpace(scenario, axes_config=axes, mode="full")
    generation_result = design_space.generate_with_exclusions()
    with pytest.raises(Exception):
        write_replay_bundle(
            bundle_path,
            result_set=_result,
            manifest=design_space.build_manifest(generation_result),
            scenario_dict=scenario.model_dump(mode="json"),
            axes_dict=axes,
            seed=0,
            run_config={"space": "full"},
            generation_result=generation_result,
        )


def test_replayed_run_matches_original_digest(tmp_path: Path):
    _result, bundle_path = _run_once(tmp_path, seed=99)
    original_digest = replay_bundle_canonical_digest(bundle_path)
    bundle = read_replay_bundle(bundle_path)
    inputs = bundle["inputs"]

    scenario = Scenario.model_validate(inputs["scenario"])
    design_space = DesignSpace(scenario, axes_config=inputs["axes"], mode="full")
    generation_result = design_space.generate_with_exclusions()

    run_config = DseRunConfig(
        scenario=scenario,
        design_space=design_space,
        seed=inputs["seed"],
    )
    runner = ScenarioDseRunner(run_config)
    new_result, manifest, _frontier = runner.run(generation_result)

    inputs_bytes = canonical_json_bytes(inputs)
    result_bytes = canonical_json_bytes(new_result.model_dump(mode="json"))
    coverage_bytes = canonical_json_bytes(manifest.to_dict())
    new_digest = __import__("hashlib").sha256(inputs_bytes + result_bytes + coverage_bytes).hexdigest()
    assert new_digest == original_digest


def test_design_space_result_v2_has_frontier_ids(tmp_path: Path):
    result, _bundle = _run_once(tmp_path)
    assert isinstance(result.frontier_design_point_ids, list)
    if result.summary.complete > 0:
        assert len(result.frontier_design_point_ids) <= result.summary.complete
