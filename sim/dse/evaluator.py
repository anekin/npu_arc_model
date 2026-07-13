"""Canonical scenario-driven Arc Model performance evaluator."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict
from typing import Any, Dict

from dse.constraints import evaluate_constraints
from dse.memory import (
    bandwidth_bytes_per_cycle,
    couple_on_chip_bandwidth,
    estimate_memory_footprint,
)
from dse.types import DSEPoint, LayerEstimate, WorkloadSpec
from dse.workload import load_workload, projection_trace
from engine.mac_engine import create_engine
from models.sfu import SFUModel
from models.vector import VectorModel

ARC_VERSION = "v3.2-performance-contract"


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def tops_int8(config: Dict[str, Any]) -> float:
    mac = config.get("mac_engine", {})
    return round(
        int(mac.get("array_height", 0))
        * int(mac.get("array_width", 0))
        * int(mac.get("ops_per_mac", 2))
        * int(mac.get("frequency_mhz", 0))
        / 1_000_000,
        2,
    )


def _attention(
    engine,
    config: Dict[str, Any],
    workload: WorkloadSpec,
    token_rows: int,
    context_len: int,
    kv_instances: int = 1,
) -> LayerEstimate:
    """Attention with explicit QK, softmax, PV and K/V traffic."""
    if engine.engine_type == "fsa":
        result = engine.estimate_attention(
            seq_q=token_rows,
            seq_kv=context_len,
            head_dim=workload.head_dim,
            num_heads=workload.num_heads,
            num_kv_heads=workload.kv_heads,
            kv_batch_size=kv_instances,
        )
        return LayerEstimate(
            total_cycles=result.total_cycles,
            compute_cycles=result.compute_cycles,
            memory_cycles=result.dma_cycles,
            attention_cycles=result.total_cycles,
            transferred_bytes=int(result.details.get("attention_bytes", 0)),
        )

    qk = engine.estimate(token_rows, workload.head_dim, context_len)
    pv = engine.estimate(token_rows, context_len, workload.head_dim)
    compute_cycles = workload.num_heads * (qk.compute_cycles + pv.compute_cycles)

    elem_bytes = max(1, engine.a_bits // 8)
    attention_bytes = (
        workload.num_heads * token_rows * workload.head_dim * elem_bytes
        + 2 * workload.kv_heads * context_len * workload.head_dim
        * elem_bytes * kv_instances
    )
    memory_cycles = math.ceil(attention_bytes / max(engine.eff_bw, 1e-9))
    matrix_cycles = max(compute_cycles, memory_cycles)

    elements = workload.num_heads * token_rows * context_len
    sfu = SFUModel(config)
    vector = VectorModel(config)
    sfu_parts = sfu.estimate_softmax_decomposed(elements)
    vec_parts = vector.estimate_softmax_vector_parts(elements)
    softmax_cycles = sum(sfu_parts.values()) + sum(vec_parts.values())

    return LayerEstimate(
        total_cycles=matrix_cycles + softmax_cycles,
        compute_cycles=compute_cycles,
        memory_cycles=memory_cycles,
        attention_cycles=matrix_cycles,
        sfu_cycles=softmax_cycles,
        transferred_bytes=attention_bytes,
    )


def estimate_layer(
    config: Dict[str, Any], workload: WorkloadSpec, mode: str,
) -> LayerEstimate:
    if mode not in ("decode", "prefill"):
        raise ValueError(f"unsupported workload mode: {mode}")
    token_rows = workload.decode_batch_size if mode == "decode" else workload.prompt_tokens
    context_len = workload.prompt_tokens
    kv_instances = workload.decode_batch_size if mode == "decode" else 1
    engine = create_engine(config)
    weight_cache = bool(config.get("optimizations", {}).get("weight_cache", False))

    projection_total = projection_compute = projection_memory = 0
    transferred_bytes = 0
    trace = projection_trace(workload, token_rows)
    i = 0
    while i < len(trace):
        m, k, n, name = trace[i]
        if weight_cache and name == "FFN_gate" and i + 1 < len(trace):
            result = engine.estimate_weight_cache_pair(m, k, n)
            i += 2
        else:
            result = engine.estimate(m, k, n)
            i += 1
        projection_total += result.total_cycles
        projection_compute += result.compute_cycles
        projection_memory += result.dma_cycles
        transferred_bytes += result.weight_bytes

    attention = _attention(
        engine, config, workload, token_rows, context_len, kv_instances,
    )

    sfu = SFUModel(config)
    vector = VectorModel(config)
    h_elems = workload.hidden * token_rows
    q_elems = workload.num_heads * workload.head_dim * token_rows
    i_elems = workload.intermediate * token_rows
    post_cycles = (
        2 * sfu.estimate("rmsnorm", h_elems)
        + sfu.estimate("rope", q_elems)
        + sfu.estimate("silu", i_elems)
        + 2 * vector.estimate_residual_add(h_elems)
    )

    kv_write_bytes = 0
    kv_write_cycles = 0
    if mode == "prefill":
        kv_write_bytes = (
            2 * workload.kv_heads * token_rows * workload.head_dim
            * max(1, engine.a_bits // 8)
        )
        kv_write_cycles = math.ceil(kv_write_bytes / max(engine.eff_bw, 1e-9))

    total = projection_total + attention.total_cycles + post_cycles + kv_write_cycles
    return LayerEstimate(
        total_cycles=int(total),
        compute_cycles=int(projection_compute + attention.compute_cycles + post_cycles),
        memory_cycles=int(projection_memory + attention.memory_cycles + kv_write_cycles),
        attention_cycles=int(attention.attention_cycles),
        sfu_cycles=int(attention.sfu_cycles + post_cycles),
        kv_cycles=int(kv_write_cycles),
        transferred_bytes=int(transferred_bytes + attention.transferred_bytes + kv_write_bytes),
    )



def estimate_output_head(
    config: Dict[str, Any], workload: WorkloadSpec, token_rows: int,
) -> LayerEstimate:
    """Estimate the final vocabulary projection, which runs once per step."""
    if workload.vocab_size <= 0:
        return LayerEstimate()
    result = create_engine(config).estimate(
        max(1, token_rows), workload.hidden, workload.vocab_size,
    )
    return LayerEstimate(
        total_cycles=int(result.total_cycles),
        compute_cycles=int(result.compute_cycles),
        memory_cycles=int(result.dma_cycles),
        transferred_bytes=int(result.weight_bytes),
    )

def evaluate_candidate(config, area_model, power_model, scenario=None) -> DSEPoint:
    cfg = copy.deepcopy(config)
    engine_type = cfg["mac_engine"]["type"]

    area_result = area_model.estimate(cfg, engine_type)
    couple_on_chip_bandwidth(cfg, area_result.get("logic_die_mm2", area_result["total_mm2"]))
    area_result = area_model.estimate(cfg, engine_type)
    area = float(area_result["total_mm2"])

    scenario_payload = scenario or {}
    model_name = cfg.get("_model_name", scenario_payload.get("model", "qwen2.5-3b"))
    workload_cfg = copy.deepcopy(scenario_payload.get("workload", {}))
    workload_cfg.update(cfg.get("_workload", {}))
    seq_len = int(cfg.get(
        "_seq_len",
        workload_cfg.get("prompt_tokens", scenario_payload.get("seq_len", 128)),
    ))
    mac = cfg["mac_engine"]
    workload = load_workload(
        model_name,
        seq_len,
        workload_config=workload_cfg,
        weight_bits=int(mac.get("weight_precision_bits", 4)),
    )
    decode = estimate_layer(cfg, workload, "decode")
    prefill = estimate_layer(cfg, workload, "prefill")
    decode_head = estimate_output_head(cfg, workload, workload.decode_batch_size)
    prefill_head = estimate_output_head(cfg, workload, 1)

    freq_mhz = int(mac.get("frequency_mhz", 1000))
    decode_total_cycles = decode.total_cycles * workload.layers + decode_head.total_cycles
    prefill_total_cycles = prefill.total_cycles * workload.layers + prefill_head.total_cycles
    engine = create_engine(cfg)
    model_weight_bytes = (
        workload.parameters_b * 1e9 * workload.weight_bits / 8.0
    )
    decode_weight_floor = math.ceil(
        model_weight_bytes / max(engine.eff_bw, 1e-9)
    )
    decode_total_cycles = max(decode_total_cycles, decode_weight_floor)
    decode_batch_us = decode_total_cycles / freq_mhz
    prefill_ms = prefill_total_cycles / (freq_mhz * 1000.0)
    decode_batch_ms = decode_batch_us / 1000.0
    service_rounds = math.ceil(
        workload.concurrent_requests / workload.decode_batch_size
    )
    itl_ms = decode_batch_ms * service_rounds
    decode_tps = 1000.0 / itl_ms if itl_ms > 0 else 0.0
    aggregate_tps = workload.concurrent_requests * decode_tps
    prefill_tps = (
        workload.prompt_tokens * 1000.0 / prefill_ms if prefill_ms > 0 else 0.0
    )
    ttft_ms = prefill_ms + decode_batch_ms
    e2e_latency_ms = ttft_ms + max(0, workload.output_tokens - 1) * itl_ms
    power = power_model.estimate(area_model, cfg, engine_type)
    footprint = estimate_memory_footprint(cfg, workload)

    raw_bpc = bandwidth_bytes_per_cycle(cfg.get("memory", {}), freq_mhz)
    decode_total_bytes = max(
        decode.transferred_bytes * workload.layers + decode_head.transferred_bytes,
        model_weight_bytes,
    )
    achieved_bpc = decode_total_bytes / max(decode_total_cycles, 1)
    bw_util = min(100.0, achieved_bpc / max(raw_bpc, 1e-9) * 100.0)
    metrics = {
        "tok_s": decode_tps,
        "decode_tps": decode_tps,
        "aggregate_tps": aggregate_tps,
        "prefill_tps": prefill_tps,
        "ttft_ms": ttft_ms,
        "itl_ms": itl_ms,
        "e2e_latency_ms": e2e_latency_ms,
        "area_mm2": area,
        "power_w": power,
        "memory_required_gb": footprint.required_gb,
        "memory_available_gb": footprint.usable_gb,
        "memory_capacity_specified": footprint.capacity_specified,
        "concurrent_requests": workload.concurrent_requests,
    }
    constraint_result = evaluate_constraints(metrics, scenario_payload)

    label = (
        f"{engine_type[:4]} {mac['array_height']}x{mac['array_width']} "
        f"INT{mac['weight_precision_bits']} {freq_mhz}MHz "
        f"B{workload.decode_batch_size} "
        f"{'WC ' if cfg.get('optimizations', {}).get('weight_cache') else ''}"
        f"{cfg.get('_dram_label', '')}"
    ).strip()
    point_config = {
        "scenario": cfg.get("_scenario_name", ""),
        "engine": engine_type,
        "array_height": int(mac["array_height"]),
        "array_width": int(mac["array_width"]),
        "frequency_mhz": freq_mhz,
        "weight_precision_bits": int(mac["weight_precision_bits"]),
        "weight_cache": bool(cfg.get("optimizations", {}).get("weight_cache", False)),
        "memory_type": cfg.get("memory", {}).get("type", ""),
        "memory_bandwidth_gbps": round(float(cfg.get("memory", {}).get("bandwidth_gbps", 0)), 2),
        "memory_capacity_gb": footprint.installed_gb,
        "sram_l2_kb": int(cfg.get("sram", {}).get("l2_shared_kb", 0)),
        "model": model_name,
        "prompt_tokens": workload.prompt_tokens,
        "output_tokens": workload.output_tokens,
        "concurrent_requests": workload.concurrent_requests,
        "decode_batch_size": workload.decode_batch_size,
    }
    return DSEPoint(
        tok_s=round(decode_tps, 2),
        decode_tps=round(decode_tps, 2),
        aggregate_tps=round(aggregate_tps, 2),
        prefill_tps=round(prefill_tps, 2),
        itl_ms=round(itl_ms, 3),
        e2e_latency_ms=round(e2e_latency_ms, 3),
        area_mm2=round(area, 2),
        power_w=round(power, 2),
        config_label=label,
        config=point_config,
        ttft_ms=round(ttft_ms, 3),
        prefill_ms=round(prefill_ms, 3),
        decode_ms=round(decode_batch_ms, 3),
        tops_int8=tops_int8(cfg),
        bandwidth_gbps=point_config["memory_bandwidth_gbps"],
        bandwidth_util_pct=round(bw_util, 2),
        memory_required_gb=footprint.required_gb,
        memory_available_gb=footprint.usable_gb,
        memory_margin_gb=footprint.margin_gb,
        memory_fits=footprint.fits,
        constraints_passed=constraint_result.passed,
        failed_reasons=constraint_result.failed_reasons,
        warnings=constraint_result.warnings,
        breakdown={
            "decode": decode.to_dict(),
            "prefill": prefill.to_dict(),
            "decode_output_head": decode_head.to_dict(),
            "prefill_output_head": prefill_head.to_dict(),
            "memory": footprint.to_dict(),
            "workload": asdict(workload),
            "service_rounds": service_rounds,
            "decode_total_cycles": decode_total_cycles,
            "prefill_total_cycles": prefill_total_cycles,
            "decode_weight_floor_cycles": decode_weight_floor,
        },
        provenance={
            "arc_version": ARC_VERSION,
            "scenario_hash": _hash(scenario_payload),
            "config_hash": _hash(point_config),
            "model_spec": model_name,
        },
    )


def ranking_key(point: DSEPoint, scenario: Dict[str, Any] | None):
    """Prefer design-target compliance, then apply product objectives."""
    objectives = (scenario or {}).get(
        "objectives", ["area_mm2", "power_w", "-decode_tps"],
    )
    values = [target_violation_score(point, scenario)]
    for objective in objectives:
        descending = str(objective).startswith("-")
        name = str(objective)[1:] if descending else str(objective)
        value = float(getattr(point, name))
        values.append(-value if descending else value)
    preference = ((scenario or {}).get("tie_breakers", {}) or {}).get(
        "engine_preference", [],
    )
    engine = str(point.config.get("engine", ""))
    values.append(
        preference.index(engine) if engine in preference else len(preference)
    )
    return tuple(values)


def target_violation_score(
    point: DSEPoint, scenario: Dict[str, Any] | None,
) -> float:
    """Normalized distance from non-hard product design targets."""
    targets = (scenario or {}).get("targets", {})
    lower_bounds = {
        "tps_min": "decode_tps",
        "decode_tps_min": "decode_tps",
        "aggregate_tps_min": "aggregate_tps",
        "prefill_tps_min": "prefill_tps",
    }
    upper_bounds = {
        "ttft_ms_max": "ttft_ms",
        "itl_ms_max": "itl_ms",
        "e2e_latency_ms_max": "e2e_latency_ms",
        "area_mm2_max": "area_mm2",
        "power_w_max": "power_w",
    }
    score = 0.0
    for key, attr in lower_bounds.items():
        if key in targets:
            limit = float(targets[key])
            score += max(0.0, limit - float(getattr(point, attr))) / max(limit, 1e-9)
    for key, attr in upper_bounds.items():
        if key in targets:
            limit = float(targets[key])
            score += max(0.0, float(getattr(point, attr)) - limit) / max(limit, 1e-9)
    return round(score, 8)


def violation_score(point: DSEPoint, scenario: Dict[str, Any] | None) -> float:
    """Normalized hard-constraint distance used only when no point is feasible."""
    constraints = (scenario or {}).get("constraints", {})
    lower_bounds = {
        "tps_min": "decode_tps",
        "decode_tps_min": "decode_tps",
        "aggregate_tps_min": "aggregate_tps",
        "prefill_tps_min": "prefill_tps",
    }
    upper_bounds = {
        "ttft_ms_max": "ttft_ms",
        "itl_ms_max": "itl_ms",
        "e2e_latency_ms_max": "e2e_latency_ms",
        "area_mm2_max": "area_mm2",
        "power_w_max": "power_w",
    }
    score = 0.0
    for key, attr in lower_bounds.items():
        if key in constraints:
            limit = float(constraints[key])
            score += max(0.0, limit - float(getattr(point, attr))) / max(limit, 1e-9)
    for key, attr in upper_bounds.items():
        if key in constraints:
            limit = float(constraints[key])
            score += max(0.0, float(getattr(point, attr)) - limit) / max(limit, 1e-9)
    if not point.memory_fits:
        score += max(0.0, -point.memory_margin_gb) / max(point.memory_required_gb, 1e-9)
    return round(score, 8)
