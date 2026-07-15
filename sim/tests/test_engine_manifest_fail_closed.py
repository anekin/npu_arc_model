import copy

import pytest
import yaml

from dse.evaluator import estimate_layer
from dse.memory import bandwidth_bytes_per_cycle
from dse.reporting import build_engine_comparison, print_engine_comparison
from dse.types import DSEPoint
from dse.workload import load_workload
from engine.mac_engine import create_engine
from engine.manifest import (
    engine_names,
    get_engine_manifest,
    load_engine_manifests,
    validate_manifest_set,
)


def _valid_payload():
    return {
        "schema_version": 1,
        "engines": {
            "test": {
                "display_name": "Test Engine",
                "module": "engine.test",
                "class_name": "TestEngine",
                "role": "dse_candidate",
                "maturity": "M1",
                "evidence": {
                    "performance": {"level": 1, "kind": "analytical"},
                    "ppa": {"level": 1, "kind": "analytical"},
                    "functional_scope": {"level": 1, "kind": "kernel"},
                    "system_integration": {"level": 1, "kind": "workload"},
                },
                "scope": {"dataflow": "test", "workload": "full_model"},
                "fallbacks": {},
                "uncertainty": {
                    "performance_pct": 20,
                    "area_pct": 30,
                    "power_pct": 35,
                },
                "calibration_dataset": None,
                "known_gaps": ["not calibrated"],
                "sources": ["test source"],
            }
        },
    }


def _write_manifest(tmp_path, payload, name="manifest.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    load_engine_manifests.cache_clear()
    return path


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda p: [], "manifest root must be a mapping"),
        (lambda p: {**p, "schema_version": 2}, "schema_version"),
        (lambda p: {**p, "engines": []}, "engines must be a mapping"),
        (lambda p: {**p, "engines": {}}, "at least one engine"),
        (lambda p: {**p, "engines": {"": p["engines"]["test"]}}, "engine name"),
        (lambda p: {**p, "engines": {"test": []}}, "test must be a mapping"),
    ],
)
def test_manifest_root_and_engine_shape_fail_closed(tmp_path, mutate, match):
    payload = mutate(_valid_payload())
    with pytest.raises(ValueError, match=match):
        load_engine_manifests(_write_manifest(tmp_path, payload))


def _mutate_engine(payload, function):
    item = payload["engines"]["test"]
    function(item)
    return payload


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda e: e["evidence"].pop("performance"), "evidence.performance"),
        (lambda e: e["evidence"].update({"performance": []}), "must be a mapping"),
        (lambda e: e["evidence"]["performance"].update({"level": 5}), "level must be 0..4"),
        (lambda e: e["evidence"]["performance"].update({"level": "1"}), "level must be 0..4"),
        (lambda e: e["evidence"]["performance"].update({"kind": ""}), "kind must be non-empty"),
        (lambda e: e.update({"maturity": "M9"}), "maturity must be one of"),
        (lambda e: e["evidence"]["ppa"].update({"level": 0}), "M1 requires"),
        (lambda e: (e.update({"maturity": "M2"}), e["evidence"]["performance"].update({"level": 1})), "M2 requires performance"),
        (lambda e: (e.update({"maturity": "M2"}), e["evidence"]["performance"].update({"level": 2})), "full workload/system"),
        (lambda e: (e.update({"maturity": "M3"}), e["evidence"]["performance"].update({"level": 3}), e["evidence"]["functional_scope"].update({"level": 2}), e["evidence"]["system_integration"].update({"level": 2})), "calibrated performance and PPA"),
        (lambda e: e["uncertainty"].update({"performance_pct": 0}), "must be in"),
        (lambda e: e["uncertainty"].update({"area_pct": "unknown"}), "must be in"),
        (lambda e: e.update({"display_name": ""}), "display_name must be non-empty"),
        (lambda e: e.update({"scope": []}), "scope must be a mapping"),
        (lambda e: e["scope"].pop("dataflow"), "scope must define"),
        (lambda e: e.update({"known_gaps": []}), "known_gaps"),
        (lambda e: e.update({"sources": []}), "sources"),
        (lambda e: e.update({"fallbacks": []}), "fallbacks must be a mapping"),
    ],
)
def test_manifest_evidence_and_contract_fail_closed(tmp_path, mutation, match):
    payload = _mutate_engine(_valid_payload(), mutation)
    with pytest.raises(ValueError, match=match):
        load_engine_manifests(_write_manifest(tmp_path, payload))


def test_manifest_lookup_and_set_validation_fail_closed():
    with pytest.raises(ValueError, match="unknown maturity"):
        engine_names("M9")
    with pytest.raises(ValueError, match="has no validated manifest"):
        get_engine_manifest("does_not_exist")
    with pytest.raises(ValueError, match="manifest mismatch"):
        validate_manifest_set({"block"})


def _config(engine="block"):
    return {
        "mac_engine": {
            "type": engine,
            "array_height": 64,
            "array_width": 64,
            "frequency_mhz": 1000,
            "weight_precision_bits": 4,
            "activation_precision_bits": 8,
            "ops_per_mac": 2,
        },
        "memory": {"bandwidth_gbps": 51.2, "dram_efficiency": 0.85},
        "sram": {"l2_shared_kb": 2048},
        "optimizations": {"dma_bw_multiplier": 1.0},
    }


def test_engine_factory_and_engine_specific_invalid_inputs_fail_closed():
    with pytest.raises(ValueError, match="Unknown engine type"):
        create_engine(_config("unknown"))

    bad_os = _config("os_systolic")
    bad_os["mac_engine"]["os_tile_control_cycles"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        create_engine(bad_os)

    bad_fused = _config("block_fused_attention")
    bad_fused["mac_engine"]["fused_attention_overlap_factor"] = 0
    with pytest.raises(ValueError, match="must be in"):
        create_engine(bad_fused)

    fsa = create_engine(_config("fsa"))
    with pytest.raises(ValueError, match="dimensions must be positive"):
        fsa.estimate_attention(0, 128, 128)


def test_workload_mode_and_legacy_bandwidth_fallback_branches():
    workload = load_workload("qwen2.5-3b", 128)
    with pytest.raises(ValueError, match="unsupported workload mode"):
        estimate_layer(_config(), workload, "invalid")
    assert bandwidth_bytes_per_cycle({"bandwidth_bytes_per_cycle": 64}, 1000) == 64


def test_cv_reporting_branch_and_dsepoint_helpers(capsys):
    point = DSEPoint(
        tok_s=20,
        area_mm2=40,
        power_w=10,
        maturity="M2",
        raw_exploration_eligible=True,
        comparison_eligible=True,
        recommendation_eligible=True,
        config={"engine": "block"},
        config_label="block test",
    )
    rows = build_engine_comparison([point], {"objectives": ["area_mm2"]})
    print_engine_comparison(rows, cv_mode=True)
    assert "FPS" in capsys.readouterr().out
    assert "DSEPoint" in repr(point)
    assert point.to_dict()["maturity"] == "M2"
