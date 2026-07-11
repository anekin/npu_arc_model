"""Canonical scenario-driven Arc Model performance evaluator."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Dict

from dse.constraints import evaluate_constraints
from dse.memory import (
    bandwidth_bytes_per_cycle,
    couple_on_chip_bandwidth,
)
from dse.types import DSEPoint, LayerEstimate, WorkloadSpec
from dse.workload import load_workload, projection_trace
from engine.mac_engine import create_engine
from models.sfu import SFUModel
from models.vector import VectorModel

ARC_VERSION = "v3.1-physics-baseline"


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
    engine, config: Dict[str, Any], workload: WorkloadSpec,
    token_rows: int, context_len: int,
) -> LayerEstimate:
    """Attention with explicit QK^T, softmax, PV and K/V traffic."""
    if engine.engine_type == "fsa":
        result = engine.estimate_attention(
            seq_q=token_rows,
            seq_kv=context_len,
            head_dim=workload.head_dim,
            num_heads=workload.num_heads,
            num_kv_heads=workload.kv_heads,
        )
        return LayerEstimate(
            total_cycles=result.total_cycles,
            compute_cycles=result.compute_cycles,
            memory_cycles=result.dma_cycles,
            attention_cycles=result.total_cycles,
            transferred_bytes=int(result.details.get("attention_bytes", 0)),
        )

    # Use each engine's compute mapping, but account for shared GQA K/V data
    # only once.  Calling estimate().total_cycles per head would multiply the
    # shared K/V memory traffic by num_heads.
    qk = engine.estimate(token_rows, workload.head_dim, context_len)
    pv = engine.estimate(token_rows, context_len, workload.head_dim)
    compute_cycles = workload.num_heads * (qk.compute_cycles + pv.compute_cycles)

    elem_bytes = max(1, engine.a_bits // 8)
    attention_bytes = (
        workload.num_heads * token_rows * workload.head_dim * elem_bytes
        + 2 * workload.kv_heads * context_len * workload.head_dim * elem_bytes
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
    token_rows = 1 if mode == "decode" else workload.seq_len
    context_len = workload.seq_len
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

    attention = _attention(engine, config, workload, token_rows, context_len)

    # Non-attention elementwise work: two RMSNorms, RoPE, SiLU and residuals.
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

    # Prefill writes newly generated K/V once. Decode attention_bytes already
    # includes reading the existing cache and the current Q vector.
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


def evaluate_candidate(config, area_model, power_model, scenario=None) -> DSEPoint:
    cfg = copy.deepcopy(config)
    engine_type = cfg["mac_engine"]["type"]

    # Area is evaluated first because on-chip stack bandwidth is physically
    # coupled to the compute-die footprint in the current product contract.
    area_result = area_model.estimate(cfg, engine_type)
    couple_on_chip_bandwidth(cfg, area_result.get("logic_die_mm2", area_result["total_mm2"]))
    area_result = area_model.estimate(cfg, engine_type)
    area = float(area_result["total_mm2"])

    model_name = cfg.get("_model_name", (scenario or {}).get("model", "qwen2.5-3b"))
    seq_len = int(cfg.get("_seq_len", (scenario or {}).get("seq_len", 128)))
    workload = load_workload(model_name, seq_len)
    decode = estimate_layer(cfg, workload, "decode")
    prefill = estimate_layer(cfg, workload, "prefill")

    freq_mhz = int(cfg["mac_engine"].get("frequency_mhz", 1000))
    decode_us = decode.total_cycles * workload.layers / freq_mhz
    prefill_ms = prefill.total_cycles * workload.layers / (freq_mhz * 1000.0)
    decode_ms = decode_us / 1000.0
    tok_s = 1e6 / decode_us if decode_us > 0 else 0.0
    ttft_ms = prefill_ms + decode_ms
    power = power_model.estimate(area_model, cfg, engine_type)

    raw_bpc = bandwidth_bytes_per_cycle(cfg.get("memory", {}), freq_mhz)
    achieved_bpc = decode.transferred_bytes / max(decode.total_cycles, 1)
    bw_util = min(100.0, achieved_bpc / max(raw_bpc, 1e-9) * 100.0)
    metrics = {
        "tok_s": tok_s,
        "ttft_ms": ttft_ms,
        "area_mm2": area,
        "power_w": power,
    }
    constraint_result = evaluate_constraints(metrics, scenario)

    mac = cfg["mac_engine"]
    label = (
        f"{engine_type[:4]} {mac['array_height']}x{mac['array_width']} "
        f"INT{mac['weight_precision_bits']} {freq_mhz}MHz "
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
        "sram_l2_kb": int(cfg.get("sram", {}).get("l2_shared_kb", 0)),
        "model": model_name,
        "seq_len": seq_len,
    }
    scenario_payload = scenario or {}
    return DSEPoint(
        tok_s=round(tok_s, 2),
        area_mm2=round(area, 2),
        power_w=round(power, 2),
        config_label=label,
        config=point_config,
        ttft_ms=round(ttft_ms, 3),
        prefill_ms=round(prefill_ms, 3),
        decode_ms=round(decode_ms, 3),
        tops_int8=tops_int8(cfg),
        bandwidth_gbps=point_config["memory_bandwidth_gbps"],
        bandwidth_util_pct=round(bw_util, 2),
        constraints_passed=constraint_result.passed,
        failed_reasons=constraint_result.failed_reasons,
        warnings=constraint_result.warnings,
        breakdown={"decode": decode.to_dict(), "prefill": prefill.to_dict()},
        provenance={
            "arc_version": ARC_VERSION,
            "scenario_hash": _hash(scenario_payload),
            "config_hash": _hash(point_config),
            "model_spec": model_name,
        },
    )


def ranking_key(point: DSEPoint, scenario: Dict[str, Any] | None):
    """Lexicographic product ranking after hard constraints pass."""
    objectives = (scenario or {}).get("objectives", ["area_mm2", "power_w", "-tok_s"])
    values = []
    for objective in objectives:
        descending = str(objective).startswith("-")
        name = str(objective)[1:] if descending else str(objective)
        value = float(getattr(point, name))
        values.append(-value if descending else value)
    return tuple(values)


def violation_score(point: DSEPoint, scenario: Dict[str, Any] | None) -> float:
    """Normalized hard-constraint distance used only when no point is feasible."""
    constraints = (scenario or {}).get("constraints", {})
    score = 0.0
    if "tps_min" in constraints:
        limit = float(constraints["tps_min"])
        score += max(0.0, limit - point.tok_s) / max(limit, 1e-9)
    if "ttft_ms_max" in constraints:
        limit = float(constraints["ttft_ms_max"])
        score += max(0.0, point.ttft_ms - limit) / max(limit, 1e-9)
    if "area_mm2_max" in constraints:
        limit = float(constraints["area_mm2_max"])
        score += max(0.0, point.area_mm2 - limit) / max(limit, 1e-9)
    if "power_w_max" in constraints:
        limit = float(constraints["power_w_max"])
        score += max(0.0, point.power_w - limit) / max(limit, 1e-9)
    return round(score, 8)
