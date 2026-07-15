"""Repeatable microbenchmark and admission audit for every DSE engine."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict

from design_space_explorer import _load_base_config
from dse.evaluator import evaluate_candidate
from engine.mac_engine import ENGINE_TYPES, create_engine
from engine.manifest import get_engine_manifest, validate_manifest_set
from engine.ppa_model import AreaModel, PowerModel


MICROBENCH_SHAPES = {
    "decode": (1, 256, 256),
    "prefill": (128, 256, 256),
    "agent_append": (875, 128, 256),
}


def _base_config(engine: str, bandwidth_gbps: float = 51.2) -> Dict[str, Any]:
    config = _load_base_config()
    config["mac_engine"].update({
        "type": engine,
        "array_height": 64,
        "array_width": 64,
        "frequency_mhz": 1000,
        "weight_precision_bits": 4,
        "activation_precision_bits": 8,
        "ops_per_mac": 2,
    })
    config["memory"].update({
        "type": "lpddr5",
        "bandwidth_gbps": bandwidth_gbps,
        "dram_efficiency": 0.85,
        "dram_width_bits": 64,
    })
    config.setdefault("optimizations", {})["weight_cache"] = False
    config["_model_name"] = "qwen2.5-3b"
    config["_seq_len"] = 128
    return config


def _result_is_physical(result, shape: tuple[int, int, int]) -> bool:
    m_dim, k_dim, n_dim = shape
    expected_ops = m_dim * k_dim * n_dim * 2
    return all((
        result.total_cycles > 0,
        result.compute_cycles >= 0,
        result.dma_cycles >= 0,
        result.total_cycles >= result.dma_cycles,
        0 <= result.utilization <= 1,
        result.ops == expected_ops,
        result.weight_bytes >= 0,
        result.bottleneck in {"compute", "dma"},
    ))


def audit_engine(name: str) -> Dict[str, Any]:
    manifest = get_engine_manifest(name)
    config = _base_config(name)
    engine = create_engine(config)
    checks: Dict[str, bool] = {"factory_identity": engine.engine_type == name}
    samples: Dict[str, Any] = {}
    for label, shape in MICROBENCH_SHAPES.items():
        result = engine.estimate(*shape)
        checks[f"physical_{label}"] = _result_is_physical(result, shape)
        samples[label] = {
            "shape": list(shape),
            "total_cycles": result.total_cycles,
            "compute_cycles": result.compute_cycles,
            "dma_cycles": result.dma_cycles,
            "utilization": result.utilization,
            "ops": result.ops,
            "bottleneck": result.bottleneck,
        }

    shape = (32, 256, 256)
    baseline = engine.estimate(*shape)
    preloaded = engine.estimate(*shape, weight_preloaded=True)
    high_bw = create_engine(_base_config(name, 102.4)).estimate(*shape)
    pair = engine.estimate_weight_cache_pair(*shape)
    checks["bandwidth_non_regression"] = high_bw.total_cycles <= baseline.total_cycles
    checks["preload_non_regression"] = preloaded.total_cycles <= baseline.total_cycles
    checks["pair_scheduler_non_regression"] = pair.total_cycles <= 2 * baseline.total_cycles

    small_area = AreaModel(config).estimate(config, name)["total_mm2"]
    large_config = copy.deepcopy(config)
    large_config["mac_engine"]["array_width"] = 128
    large_area = AreaModel(large_config).estimate(large_config, name)["total_mm2"]
    power = PowerModel(config).estimate(AreaModel(config), config, name)
    checks["ppa_positive"] = small_area > 0 and power > 0
    checks["area_monotonic"] = large_area >= small_area

    scenario = {
        "model": "qwen2.5-3b",
        "workload": {"prompt_tokens": 128, "output_tokens": 32},
        "constraints": {},
    }
    point = evaluate_candidate(config, AreaModel(config), PowerModel(config), scenario)
    checks["dse_candidate_evaluates"] = all((
        point.decode_tps > 0,
        point.prefill_tps > 0,
        point.ttft_ms > 0,
        math.isfinite(point.area_mm2),
        math.isfinite(point.power_w),
    ))
    return {
        "engine": name,
        "maturity": manifest.maturity,
        "calibration_tier": manifest.calibration_tier,
        "comparison_ready": manifest.comparison_ready,
        "product_qualified": manifest.product_qualified,
        "uncertainty": manifest.uncertainty,
        "checks": checks,
        "passed": all(checks.values()),
        "samples": samples,
        "dse_smoke": {
            "decode_tps": point.decode_tps,
            "prefill_tps": point.prefill_tps,
            "ttft_ms": point.ttft_ms,
            "area_mm2": point.area_mm2,
            "power_w": point.power_w,
        },
        "known_gaps": list(manifest.known_gaps),
    }


def run_audit() -> Dict[str, Any]:
    validate_manifest_set(ENGINE_TYPES)
    engines = [audit_engine(name) for name in ENGINE_TYPES]
    return {
        "schema_version": 1,
        "microbench_shapes": {
            name: list(shape) for name, shape in MICROBENCH_SHAPES.items()
        },
        "engine_count": len(engines),
        "passed_engines": sum(1 for engine in engines if engine["passed"]),
        "all_passed": all(engine["passed"] for engine in engines),
        "engines": engines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_audit()
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"Saved engine audit to {args.output}")
    else:
        print(payload)
    raise SystemExit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
