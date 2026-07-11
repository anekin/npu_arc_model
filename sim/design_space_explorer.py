#!/usr/bin/env python3
"""设计空间搜索器 — 多引擎多配置对比，输出 Pareto 前沿

用法:
  python3 design_space_explorer.py              # 默认搜索
  python3 design_space_explorer.py --quick      # 快速扫描（减少组合）
  python3 design_space_explorer.py --output results/pareto.json
"""

import sys, json, copy, math, itertools
from pathlib import Path
from typing import Dict, Any, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from engine.ppa_model import AreaModel, PowerModel, PPA
from engine.mac_engine import create_engine
from model_specs import get_spec, all_aliases
from dse.evaluator import evaluate_candidate, ranking_key, violation_score

import yaml

SIM_DIR = Path(__file__).parent

_CV_MODEL: str = ""
_CV_TRACE: List[Any] = []
_CV_ONNX_PATH: str = ""
_CUSTOM_SCENARIOS: Dict[str, Dict[str, Any]] = {}

_NUM_LAYERS: int = 28
_LLM_TRACE: List[Tuple] = []
_SEQ_KV: int = 2048      # KV cache sequence length for decode
_KV_HEADS: int = 2        # num_kv_heads from model spec
_HEAD_DIM: int = 128      # head_dim from model spec


def generate_trace_from_spec(alias: str, batch_m: int = 1) -> List[Tuple]:
    global _KV_HEADS, _HEAD_DIM
    spec = get_spec(alias)
    H = spec.hidden
    I = spec.intermediate
    qkv = spec.qkv_dim
    kv = spec.kv_heads * spec.head_dim
    _KV_HEADS = spec.kv_heads
    _HEAD_DIM = spec.head_dim
    trace = []
    m_attn = batch_m  # attention projections batch all tokens
    m_ffn = batch_m if batch_m > 1 else 1  # prefill: batch tokens; decode: single token
    trace.append((m_attn, H, qkv, 0, "Q_proj"))
    trace.append((m_attn, H, kv,  0, "K_proj"))
    trace.append((m_attn, H, kv,  0, "V_proj"))
    trace.append((m_attn, qkv, H, 0, "O_proj"))
    trace.append((m_ffn, H, I,    0, "FFN_gate"))
    trace.append((m_ffn, H, I,    0, "FFN_up"))
    trace.append((m_ffn, I, H,    0, "FFN_down"))
    return trace


_LLM_TRACE = generate_trace_from_spec("qwen2.5-3b", batch_m=1)

SFU_CYCLES_PER_LAYER = {
    "attn": 33,   # softmax + layernorm + rope (simplified)
    "ffn": 8,     # gelu + layernorm
}


def _load_base_config() -> Dict[str, Any]:
    with open(SIM_DIR / "config" / "design_space.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_scenario(scenario_name: str | None) -> Dict[str, Any] | None:
    if not scenario_name:
        return None
    if scenario_name in _CUSTOM_SCENARIOS:
        return copy.deepcopy(_CUSTOM_SCENARIOS[scenario_name])
    with open(SIM_DIR / "config" / "scenarios.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    scenario = data.get("scenarios", {}).get(scenario_name)
    if scenario is None:
        known = ", ".join(sorted(data.get("scenarios", {}).keys()))
        raise ValueError(f"Unknown scenario '{scenario_name}'. Known scenarios: {known}")
    return scenario


def _apply_scenario(base: Dict[str, Any], scenario_name: str | None) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    scenario = _load_scenario(scenario_name)
    if scenario is None:
        return cfg

    cfg["_scenario_name"] = scenario_name
    cfg["_model_name"] = scenario.get("model", "qwen2.5-3b")
    cfg["_seq_len"] = int(scenario.get("seq_len", 128))
    if "process_nm" in scenario:
        cfg.setdefault("area_model", {})["process_node"] = int(scenario["process_nm"])

    memory = scenario.get("memory", {})
    memory_type = memory.get("type")
    if memory_type:
        cfg.setdefault("memory", {})["type"] = memory_type
    if "bandwidth_gbps" in memory:
        bw = float(memory["bandwidth_gbps"])
        cfg.setdefault("memory", {})["bandwidth_gbps"] = bw
        cfg.setdefault("memory", {})["bandwidth_bytes_per_cycle"] = bw
    if "dram_efficiency" in memory:
        cfg.setdefault("memory", {})["dram_efficiency"] = float(memory["dram_efficiency"])
    if "capacity_gb" in memory:
        cfg["on_chip_memory"] = {
            "capacity_gb": float(memory["capacity_gb"]),
            "bandwidth_gbps": float(memory.get("bandwidth_gbps", 0.0)),
            "bw_per_mm2_gbps": float(memory.get("bw_per_mm2_gbps", 0.0)),
            "stack_area_mm2": float(memory.get("stack_area_mm2", 0.0)),
            "stack_power_per_gbps_w": float(memory.get("stack_power_per_gbps_w", 0.015)),
        }

    constraints = scenario.get("constraints", {})
    if constraints:
        cfg.setdefault("constraints", {}).update(constraints)
    return cfg


def _scenario_dram_configs(scenario: Dict[str, Any] | None, quick: bool) -> List[Tuple[float, int, str]]:
    if scenario is None:
        if quick:
            return [
                (51.2, 64, "LPDDR5-64b"),
                (102.4, 128, "LPDDR5-128b"),
            ]
        return [
            (25.6, 32, "LPDDR5-32b"),
            (51.2, 64, "LPDDR5-64b"),
            (102.4, 128, "LPDDR5-128b"),
            (204.8, 256, "LPDDR5-256b"),
            (460.0, 1024, "HBM2e-1024b"),
            (819.2, 1024, "HBM3-1024b"),
        ]

    memory = scenario.get("memory", {})
    bw = float(memory.get("bandwidth_gbps", 51.2))
    memory_type = str(memory.get("type", "lpddr5"))
    width_bits = 64 if memory_type.startswith("lpddr") else 1024
    label = "on-chip-3D" if memory_type == "on_chip_3d_dram" else f"{memory_type.upper()}-{width_bits}b"
    return [(bw, width_bits, label)]


def _scenario_dims(scenario: Dict[str, Any] | None, quick: bool) -> List[Tuple[int, int]]:
    if scenario is None:
        if quick:
            return [(128, 128), (128, 256), (256, 256)]
        return [(64, 64), (96, 96), (128, 128), (128, 192),
                (128, 256), (192, 256), (256, 256)]

    memory_type = scenario.get("memory", {}).get("type")
    if memory_type == "on_chip_3d_dram":
        return [(32, 1536), (48, 1536), (64, 1536), (80, 1536), (96, 1536), (128, 1536)]
    if quick:
        return [(64, 128), (128, 128), (128, 256)]
    return [(64, 128), (64, 256), (128, 128), (128, 256), (192, 128)]


def _compute_kv_cycles(config: Dict[str, Any], batch_m: int = 1) -> int:
    """Dynamic KV cache DRAM read cycles per layer.

    - For decode (batch_m=1): K,V read from memory
    - For prefill (batch_m>1): KV written (not read), negligible cost
    - On-chip mode: KV also on-chip, uses on_chip_bw
    """
    if batch_m > 1:
        return 0  # Prefill: KV is being written, not a read bottleneck

    # K + V: 2 × seq_kv × kv_heads × head_dim × 1 byte (INT8)
    kv_bytes = 2 * _SEQ_KV * _KV_HEADS * _HEAD_DIM * 1

    onchip = config.get("on_chip_memory", {})
    onchip_bw = float(onchip.get("bandwidth_gbps", 0))

    if onchip_bw > 0:
        # On-chip mode: KV cache also on-chip
        return int(kv_bytes / onchip_bw) if onchip_bw > 0 else 0

    sram = config.get("sram", {})
    l2_kb = int(sram.get("l2_shared_kb", 2048))
    kvbuf_kb = int(l2_kb * 0.4)

    mem = config.get("memory", {})
    bw_raw = float(mem.get("bandwidth_bytes_per_cycle", 51.2))
    dram_eff = float(mem.get("dram_efficiency", 0.85))
    eff_bw = bw_raw * dram_eff

    kv_mb = kv_bytes / (1024 * 1024.0)
    kvbuf_mb = kvbuf_kb / 1024.0
    ratio = kvbuf_mb / max(kv_mb, 0.001)
    kv_dram_eff = 0.55 + 0.40 * ratio / (0.3 + ratio)

    if eff_bw <= 0 or kv_dram_eff <= 0:
        return 0

    return int(kv_bytes / (eff_bw * kv_dram_eff))


def simulate_layer(config: Dict[str, Any], batch_m: int = None) -> tuple:
    """Simulate one transformer layer. Returns (total_cycles, weight_bytes).

    batch_m=1 for decode, >1 for prefill. If None, inferred from trace.
    """
    if batch_m is None:
        batch_m = _LLM_TRACE[0][0] if _LLM_TRACE else 1
    engine = create_engine(config)
    opts = config.get("optimizations", {})
    weight_cache = opts.get("weight_cache", False)

    total = 0
    weight_bytes = 0
    i = 0
    ops = _LLM_TRACE

    while i < len(ops):
        M, K, N, _, name = ops[i]

        # Weight cache merge
        if (weight_cache and name == "FFN_gate" and i + 1 < len(ops)
                and ops[i + 1][4] == "FFN_up"):
            r = engine.estimate_weight_cache_pair(M, K, N)
            i += 2
        else:
            r = engine.estimate(M, K, N)
            i += 1

        total += r.total_cycles
        weight_bytes += r.weight_bytes

        # SFU
        if name == "O_proj":
            total += SFU_CYCLES_PER_LAYER["attn"]
        elif name == "FFN_down":
            total += SFU_CYCLES_PER_LAYER["ffn"]

    # KV cache: dynamic read cost based on SRAM + bandwidth
    kv_cycles = _compute_kv_cycles(config, batch_m)
    total += kv_cycles

    return total, weight_bytes


def tok_s_from_layer(layer_cycles: int, num_layers: int, f_mhz: int = 1000) -> float:
    total_us = layer_cycles * num_layers / f_mhz
    return round(1e6 / total_us, 1) if total_us > 0 else 0


def _depthwise_util_from_cv_result(cv_result: Dict[str, Any]) -> float:
    utils = [
        layer.get("mxu_util_pct", 0.0)
        for layer in cv_result.get("layers", [])
        if layer.get("type") == "depthwise_conv"
    ]
    return sum(utils) / len(utils) if utils else 0.0


def generate_configs(quick: bool = False, scenario_name: str | None = None,
                     base_config: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Generate design space configurations to sweep."""
    raw_base = copy.deepcopy(base_config) if base_config is not None else _load_base_config()
    base = _apply_scenario(raw_base, scenario_name)
    scenario = _load_scenario(scenario_name)

    configs = []

    # Default heuristic search space, optionally overridden by the
    # application-requirements YAML.
    if quick:
        engines = ["systolic", "block", "gmma"]
    else:
        engines = ["systolic", "os_systolic", "block",
                   "tensor_core", "wmma", "gmma", "input_stationary", "fsa"]
    dims = _scenario_dims(scenario, quick)
    dram_configs = _scenario_dram_configs(scenario, quick)
    precisions = [4] if quick else [4, 2]
    freqs = [1000] if quick else [800, 1000, 1200]
    sram_l2_sizes = [2048] if quick else [1024, 2048, 4096, 6144, 8192]

    search = (scenario or {}).get("search_space", {})
    engines = list(search.get("engines", engines))
    if search.get("arrays"):
        dims = [tuple(map(int, value)) for value in search["arrays"]]
    precisions = [int(v) for v in search.get("weight_precision_bits", precisions)]
    freqs = [int(v) for v in search.get("frequencies_mhz", freqs)]
    sram_l2_sizes = [int(v) for v in search.get("sram_l2_kb", sram_l2_sizes)]

    for engine_type in engines:
        for H, W in dims:
            # Area constraints
            area_gate = 400 if scenario is not None else 200
            if engine_type in ("block", "os_systolic") and H * W / (128 * 128) * 32 > area_gate:
                continue
            if engine_type == "systolic" and H * W / (128 * 128) * 8 > 80:
                continue
            if engine_type in ("tensor_core", "wmma") and H * W / (128 * 128) * 37 > 200:
                continue
            if engine_type == "gmma" and H * W / (128 * 128) * 40 > 200:
                continue
            if engine_type == "input_stationary" and H * W / (128 * 128) * 24 > 150:
                continue

            for bw_gbps, dw_bits, dram_label in dram_configs:
                for w_bits in precisions:
                    for freq in freqs:
                        for l2_kb in sram_l2_sizes:
                            # weight_cache only for systolic
                            wc_options = [False]
                            if engine_type in ("systolic", "block", "gmma"):
                                wc_options = [False, True]

                            for wc in wc_options:
                                # Block/GMMA with weight_cache skip if bandwidth too low
                                if wc and engine_type != "systolic" and bw_gbps < 51.2:
                                    continue

                                cfg = copy.deepcopy(base)
                                cfg["mac_engine"]["type"] = engine_type
                                cfg["mac_engine"]["array_height"] = H
                                cfg["mac_engine"]["array_width"] = W
                                cfg["mac_engine"]["weight_precision_bits"] = w_bits
                                cfg["mac_engine"]["frequency_mhz"] = freq
                                cfg["memory"]["bandwidth_gbps"] = bw_gbps
                                cfg["memory"]["bandwidth_bytes_per_cycle"] = bw_gbps * 1000.0 / freq
                                cfg["memory"]["dram_width_bits"] = dw_bits
                                cfg["memory"]["dram_efficiency"] = float(cfg["memory"].get("dram_efficiency", 0.85))
                                cfg["sram"]["l2_shared_kb"] = l2_kb
                                cfg["optimizations"]["weight_cache"] = wc
                                cfg["optimizations"]["dma_bw_multiplier"] = 1.0
                                cfg["_dram_label"] = dram_label

                                configs.append(cfg)

    return configs


def evaluate_config(cfg: Dict[str, Any], area_model: AreaModel,
                    power_model: PowerModel) -> PPA:
    """Evaluate one configuration through the canonical DSE evaluator."""
    if not _CV_MODEL:
        scenario = _load_scenario(cfg.get("_scenario_name"))
        return evaluate_candidate(cfg, area_model, power_model, scenario)

    cfg["_cv_workload"] = True
    from cv.cv_sim import simulate_cv
    cv_result = simulate_cv(_CV_TRACE, cfg)
    fps = 1e9 / cv_result["total_cycles"] if cv_result["total_cycles"] > 0 else 0.0
    engine_type = cfg["mac_engine"]["type"]
    area = area_model.estimate(cfg, engine_type)["total_mm2"]
    power = power_model.estimate(area_model, cfg, engine_type)
    mac = cfg["mac_engine"]
    return PPA(
        tok_s=fps, area_mm2=area, power_w=power,
        config_label=f"{engine_type[:4]} {mac['array_height']}x{mac['array_width']}",
        sram_spill_mb=cv_result.get("sram_spill_mb", 0.0),
        depthwise_util_pct=_depthwise_util_from_cv_result(cv_result),
        config={"engine": engine_type, "array_height": mac["array_height"],
                "array_width": mac["array_width"]},
    )


def _tops_int8_from_config(config: Dict[str, Any]) -> float:
    mac = config.get("mac_engine", {})
    h = int(mac.get("array_height", 0))
    w = int(mac.get("array_width", 0))
    freq = int(mac.get("frequency_mhz", 0))
    ops_per_mac = int(mac.get("ops_per_mac", 2))
    return round(h * w * ops_per_mac * freq / 1_000_000, 1)


def find_pareto(ppas: List[PPA]) -> List[PPA]:
    """Find Pareto-optimal points (max tok/s, min area)."""
    pareto = []
    for p in ppas:
        dominated = False
        for q in ppas:
            if (q.tok_s >= p.tok_s and q.area_mm2 <= p.area_mm2 and
                    (q.tok_s > p.tok_s or q.area_mm2 < p.area_mm2)):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    return sorted(pareto, key=lambda x: x.area_mm2)


# ═══════════════════════════════════════════════════════════════
# Sensitivity Analysis — generalized parameter impact ranking
# ═══════════════════════════════════════════════════════════════

import re
from statistics import mean, stdev
from collections import defaultdict


def _parse_label(label: str) -> Dict[str, Any]:
    """Parse config_label into structured params.
    
    Format: "eng H×W INT{w} {freq}MHz {WC} {DRAM_label}"
    Example: "fsa  128×128 INT4 1000MHz  LPDDR5-64b"
    """
    params = {}
    # Engine type (first token, possibly truncated)
    m = re.match(r'(\S+)', label)
    if m:
        eng_map = {'syst': 'systolic', 'os_s': 'os_systolic', 'bloc': 'block',
                   'tens': 'tensor_core', 'wmma': 'wmma', 'gmma': 'gmma',
                   'inpu': 'input_stationary', 'fsa': 'fsa', 'fsa ': 'fsa'}
        params['engine'] = eng_map.get(m.group(1), m.group(1))
    
    # Array dims: H×W
    m = re.search(r'(\d+)×(\d+)', label)
    if m:
        params['H'] = int(m.group(1))
        params['W'] = int(m.group(2))
        params['MACs'] = params['H'] * params['W']
    
    # Weight precision
    m = re.search(r'INT(\d+)', label)
    if m:
        params['w_bits'] = int(m.group(1))
    
    # Frequency
    m = re.search(r'(\d+)MHz', label)
    if m:
        params['freq_mhz'] = int(m.group(1))
    
    # Weight cache
    params['weight_cache'] = 'WC' in label
    
    # SRAM size — handles "SRAM  8MB", "SRAM512KB", "SRAM 1MB"
    m = re.search(r'SRAM\s*(\d+)\s*(MB|KB)', label)
    if m:
        val = int(m.group(1))
        params['sram_mb'] = val if m.group(2) == 'MB' else val / 1024
    
    # DRAM label
    m = re.search(r'(LPDDR\S+|HBM\S+|DDR\S+|on-chip)', label)
    if m:
        params['dram'] = m.group(1)
    
    return params


def analyze_sensitivity(results: List[PPA], 
                         metrics: List[str] = None) -> Dict[str, Any]:
    """Compute per-parameter sensitivity across a DSE result set.
    
    Args:
        results: List of PPA results from evaluate_config
        metrics: Which metrics to analyze (default: ['tok_s', 'area_mm2'])
    
    Returns:
        {
            'parameters': {
                '<param_name>': {
                    'impact_tps_pct': float,   # max TPS variation / mean TPS
                    'impact_area_pct': float,  # max area variation / mean area
                    'rank': int,               # 1 = most impactful
                    'is_zero_sensitivity': bool,  # < 2% impact → candidate to minimize
                    'optimal_value': any,      # value that gives best TPS/mm²
                    'values_tested': [...],
                }
            },
            'ranked': [param_names ordered by total impact],
            'zero_sensitivity_params': [param names with < 2% impact],
            'warnings': [human-readable findings],
        }
    """
    if metrics is None:
        metrics = ['tok_s', 'area_mm2']
    
    if len(results) < 10:
        return {'error': 'Need ≥10 results for meaningful sensitivity analysis'}
    
    # Parse all labels
    parsed = [_parse_label(r.config_label) for r in results]
    
    # Parameters to analyze (only those that vary across the result set)
    param_keys = ['engine', 'H', 'W', 'MACs', 'w_bits', 'freq_mhz', 'weight_cache', 'dram', 'sram_mb']
    varying_params = {}
    for key in param_keys:
        values = set()
        for p in parsed:
            if key in p:
                values.add(p[key])
        if len(values) > 1:
            varying_params[key] = sorted(values, key=str)
    
    sensitivity = {}
    warnings = []
    
    for param, values in varying_params.items():
        # Group results by this parameter's value
        groups = defaultdict(list)
        for i, r in enumerate(results):
            val = parsed[i].get(param)
            if val is not None:
                groups[val].append(r)
        
        # Compute mean metrics per group
        group_means = {}
        for val, group_results in groups.items():
            group_means[val] = {
                'tok_s': mean(r.tok_s for r in group_results),
                'area_mm2': mean(r.area_mm2 for r in group_results),
                'count': len(group_results),
            }
        
        # Impact = (max_mean - min_mean) / overall_mean
        overall_tps = mean(r.tok_s for r in results)
        overall_area = mean(r.area_mm2 for r in results)
        
        tps_vals = [m['tok_s'] for m in group_means.values()]
        area_vals = [m['area_mm2'] for m in group_means.values()]
        
        impact_tps = (max(tps_vals) - min(tps_vals)) / max(overall_tps, 0.01) * 100
        impact_area = (max(area_vals) - min(area_vals)) / max(overall_area, 0.01) * 100
        
        # Zero-sensitivity detection
        is_zero = (impact_tps < 2.0 and impact_area < 2.0)
        
        # Find optimal value (best TPS/mm² efficiency)
        best_val = None
        best_eff = -1
        for val, m in group_means.items():
            eff = m['tok_s'] / max(m['area_mm2'], 0.01)
            if eff > best_eff:
                best_eff = eff
                best_val = val
        
        sensitivity[param] = {
            'impact_tps_pct': round(impact_tps, 1),
            'impact_area_pct': round(impact_area, 1),
            'total_impact': round(impact_tps + impact_area, 1),
            'is_zero_sensitivity': is_zero,
            'optimal_value': str(best_val),
            'values_tested': [str(v) for v in values],
            'group_means': {str(k): {'tok_s': round(v['tok_s'], 1), 
                                      'area_mm2': round(v['area_mm2'], 1)}
                           for k, v in group_means.items()},
        }
        
        if is_zero:
            # Find the minimum-cost value
            min_cost_val = min(group_means.keys(), 
                              key=lambda v: group_means[v]['area_mm2'] 
                              if isinstance(v, (int, float)) else 0)
            warnings.append(
                f"⚠ {param}: zero sensitivity (TPS ±{impact_tps:.1f}%, area ±{impact_area:.1f}%). "
                f"Recommend {min_cost_val} to minimize cost."
            )
    
    # Rank by total impact
    ranked = sorted(sensitivity.keys(), 
                    key=lambda k: sensitivity[k]['total_impact'], 
                    reverse=True)
    
    zero_params = [k for k, v in sensitivity.items() if v['is_zero_sensitivity']]
    
    return {
        'parameters': sensitivity,
        'ranked': ranked,
        'zero_sensitivity_params': zero_params,
        'warnings': warnings,
        'varying_params_count': len(varying_params),
        'total_results_analyzed': len(results),
    }


def print_sensitivity_report(sa: Dict[str, Any]):
    """Print a human-readable sensitivity analysis report."""
    if 'error' in sa:
        print(f"  Sensitivity analysis skipped: {sa['error']}")
        return
    
    params = sa['parameters']
    print(f"\n{'='*80}")
    print(f"  Parameter Sensitivity Analysis ({sa['total_results_analyzed']} configs, "
          f"{sa['varying_params_count']} varying params)")
    print(f"{'='*80}")
    print(f"  {'Rank':<5} {'Parameter':<16} {'ΔTPS%':>8} {'ΔArea%':>8} {'Impact':>8} {'Flag'}")
    print(f"  {'-'*55}")
    
    for i, param in enumerate(sa['ranked'], 1):
        p = params[param]
        flag = "⚠ ZERO" if p['is_zero_sensitivity'] else ""
        print(f"  {i:<5} {param:<16} {p['impact_tps_pct']:>7.1f}% {p['impact_area_pct']:>7.1f}% "
              f"{p['total_impact']:>7.1f}  {flag}")
    
    if sa['warnings']:
        print(f"\n  ══ Optimization Opportunities ══")
        for w in sa['warnings']:
            print(f"  {w}")
    
    # Per-param detail
    print(f"\n  ══ Per-Parameter Detail ══")
    for param in sa['ranked']:
        p = params[param]
        print(f"\n  [{param}] — {'ZERO SENSITIVITY' if p['is_zero_sensitivity'] else 'ACTIVE DRIVER'}")
        print(f"    Values tested: {', '.join(p['values_tested'][:8])}")
        print(f"    Optimal (TPS/mm²): {p['optimal_value']}")
        print(f"    Impact: TPS {p['impact_tps_pct']:.1f}% | Area {p['impact_area_pct']:.1f}%")
        for val, metrics in sorted(p['group_means'].items()):
            print(f"      {val:<20s} → {metrics['tok_s']:6.1f} tok/s, {metrics['area_mm2']:6.1f} mm²")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--top", type=int, default=20,
                        help="Show top N results")
    parser.add_argument("--cv-model", choices=["mobilenetv3-small", "yolov8n", "vit-b16", "resnet18", "resnet50"],
                        default=None,
                        help="Run CV design-space exploration")
    parser.add_argument("--model-spec",
                        choices=[a for a in all_aliases() if get_spec(a).model_type == "llm"],
                        default=None,
                        help="LLM model spec alias for DSE")
    parser.add_argument("--batch-m", type=int, choices=[1, 2], default=None,
                        help="Batch M dimension for attention ops (1 or 2)")
    parser.add_argument("--scenario", default=None,
                        help="Scenario name from sim/config/scenarios.yaml")
    parser.add_argument("--requirements", default=None,
                        help="Path to a YAML application-requirements file")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Override scenario sequence length")
    args = parser.parse_args()

    if args.requirements:
        req_path = Path(args.requirements)
        with open(req_path, encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        custom = payload.get("scenario", payload)
        custom = copy.deepcopy(custom)
        name = str(custom.pop("name", req_path.stem))
        _CUSTOM_SCENARIOS[name] = custom
        args.scenario = name
    if args.seq_len is not None:
        if args.scenario:
            scenario_copy = _load_scenario(args.scenario) or {}
            scenario_copy["seq_len"] = args.seq_len
            _CUSTOM_SCENARIOS[args.scenario] = scenario_copy

    if args.cv_model and (args.model_spec is not None or args.batch_m is not None or args.scenario is not None):
        parser.error("--cv-model is mutually exclusive with --model-spec, --batch-m, and --scenario")

    scenario = _load_scenario(args.scenario)
    if args.scenario and args.model_spec is not None:
        parser.error("--scenario already defines the model; omit --model-spec")

    model_spec = args.model_spec if args.model_spec is not None else (
        scenario.get("model") if scenario else "qwen2.5-3b"
    )
    batch_m = args.batch_m if args.batch_m is not None else 1

    global _CV_MODEL, _CV_TRACE, _CV_ONNX_PATH, _LLM_TRACE, _NUM_LAYERS
    _CV_MODEL = args.cv_model or ""
    if _CV_MODEL:
        if args.cv_model == "mobilenetv3-small":
            from cv.cv_trace import generate_mobilenetv3_trace
            _CV_ONNX_PATH = str(Path(__file__).parent.parent / "assets" / "mobilenetv3_small.onnx")
            _CV_TRACE = generate_mobilenetv3_trace(_CV_ONNX_PATH)
        elif args.cv_model == "yolov8n":
            from cv.traces.yolov8n_trace import generate_yolov8n_trace
            _CV_TRACE = generate_yolov8n_trace()
        elif args.cv_model == "vit-b16":
            from cv.traces.vit_trace import generate_vit_trace
            _CV_TRACE = generate_vit_trace()
        elif args.cv_model == "resnet18":
            from cv.traces.resnet18_trace import generate_resnet18_trace
            _CV_TRACE = generate_resnet18_trace()
        elif args.cv_model == "resnet50":
            from cv.traces.resnet50_trace import generate_resnet50_trace
            _CV_TRACE = generate_resnet50_trace()
    else:
        _LLM_TRACE = generate_trace_from_spec(model_spec, batch_m)
        _NUM_LAYERS = get_spec(model_spec).layers

    base_cfg = _apply_scenario(_load_base_config(), args.scenario)
    base_cfg["_model_name"] = model_spec
    base_cfg["_seq_len"] = int(args.seq_len or (scenario or {}).get("seq_len", 128))

    area_model = AreaModel(base_cfg)
    power_model = PowerModel(base_cfg)

    if args.scenario and args.scenario not in _CUSTOM_SCENARIOS:
        from dse_scenario import check_requirements, preflight, print_preflight, print_requirements_check
        rc = check_requirements(args.scenario, base_cfg)
        print_requirements_check(rc)
        pf = preflight(args.scenario, base_cfg)
        print_preflight(pf)
    elif args.scenario:
        print(f"Application requirements: {args.scenario} (custom YAML, all fields explicit)")

    configs = generate_configs(quick=args.quick, scenario_name=args.scenario, base_config=base_cfg)
    print(f"Design space: {len(configs)} configurations")
    engine_set = sorted({c["mac_engine"]["type"] for c in configs})
    print(f"  Engine types: {', '.join(engine_set)}")
    dim_set = set((c['mac_engine']['array_height'],
                   c['mac_engine']['array_width']) for c in configs)
    print(f"  Array dims: {len(dim_set)}")
    print(f"  Sweeping...", end=" ", flush=True)

    results: List[PPA] = []
    invalid_configs: List[Dict[str, str]] = []
    for cfg in configs:
        try:
            ppa = evaluate_config(cfg, area_model, power_model)
            # Filter: unreasonable area
            if ppa.area_mm2 <= 1000:
                results.append(ppa)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            mac = cfg.get("mac_engine", {})
            invalid_configs.append({
                "engine": str(mac.get("type", "")),
                "array": f"{mac.get('array_height', '?')}x{mac.get('array_width', '?')}",
                "error": f"{type(e).__name__}: {e}",
            })

    print(f"{len(results)} valid")
    if invalid_configs:
        print(f"  {len(invalid_configs)} invalid configs")

    # Hard constraints are applied before Pareto/ranking.
    feasible = [r for r in results if r.constraints_passed]
    pareto = find_pareto(feasible)
    active_scenario = _load_scenario(args.scenario)
    reasonable = sorted(feasible, key=lambda p: ranking_key(p, active_scenario))
    rejected = [r for r in results if not r.constraints_passed]
    closest = sorted(rejected, key=lambda p: violation_score(p, active_scenario))
    print(f"  Constraints: {len(feasible)} passed, {len(rejected)} rejected")
    if not feasible and closest:
        best_effort = closest[0]
        print("  No feasible architecture. Closest candidate: "
              f"{best_effort.config_label} (violation score "
              f"{violation_score(best_effort, active_scenario):.3f})")
        for reason in best_effort.failed_reasons:
            print(f"    - {reason}")

    perf_label = "fps" if _CV_MODEL else "tok/s"
    eff_label = "fps/W" if _CV_MODEL else "tok/W"
    cv_extra_header = f" {'SRAM(MB)':>10} {'DW(%)':>8}" if _CV_MODEL else ""
    line_width = 100 if _CV_MODEL else 85

    # ── Output ──
    print(f"\n{'='*90}")
    print(f"  Pareto 前沿 (面积 vs 性能)")
    print(f"  {'Config':<45} {perf_label:>8} {'Area':>8} {'Power':>8} {eff_label:>8}{cv_extra_header}")
    print(f"  {'-'*line_width}")
    for p in pareto[:15]:
        arrow = "← Pareto" if p in pareto else ""
        extra = ""
        if _CV_MODEL:
            extra = f" {p.sram_spill_mb:>9.1f} {p.depthwise_util_pct:>7.3f}"
        print(f"  {p.config_label:<45} {p.tok_s:>7.0f} {p.area_mm2:>6.0f}mm² "
              f"{p.power_w:>6.1f}W {p.efficiency_tok_per_watt:>7.1f}{extra}")

    # ── Top by tok/s ──
    print(f"\n  Top {args.top} feasible candidates (scenario objective order):")
    print(f"  {'Config':<45} {perf_label:>8} {'Area':>8} {'Power':>8} {eff_label:>8}{cv_extra_header}")
    print(f"  {'-'*line_width}")
    for p in reasonable[:args.top]:
        pareto_flag = "←" if p in pareto else ""
        extra = ""
        if _CV_MODEL:
            extra = f" {p.sram_spill_mb:>9.1f} {p.depthwise_util_pct:>7.3f}"
        print(f"  {p.config_label:<45} {p.tok_s:>7.0f} {p.area_mm2:>6.0f}mm² "
              f"{p.power_w:>6.1f}W {p.efficiency_tok_per_watt:>7.1f}{extra} {pareto_flag}")

    # ── Best feasible point per engine type ──
    print(f"\n  Best feasible per engine type (scenario objective order):")
    for eng in ["systolic", "os_systolic", "block", "tensor_core", "wmma", "gmma", "fsa"]:
        eng_results = [r for r in feasible
                       if r.config.get("engine") == eng]
        if eng_results:
            best = min(eng_results, key=lambda x: ranking_key(x, active_scenario))
            print(f"    {eng}: {best.tok_s:.0f} {perf_label}, {best.area_mm2:.0f}mm², "
                  f"{best.power_w:.1f}W — {best.config_label}")

    # ── Sensitivity Analysis (always run after sweep) ──
    sa = analyze_sensitivity(results)
    print_sensitivity_report(sa)

    # ── Cross-Validation (compare best config against known products) ──
    if not _CV_MODEL and reasonable:
        from dse_scenario import cross_validate as cv_func, print_cross_validate as print_cv
        best = reasonable[0]
        # Auto-detect scenario: on-chip if any config has on_chip_memory
        has_onchip = best.config.get("memory_type") == "on_chip_3d_dram"
        scenario = 'onchip_7b' if has_onchip else 'lpddr5_3b'
        cv = cv_func({
            'process_nm': int(base_cfg.get('area_model', {}).get('process_node', 12)),
            'area_mm2': best.area_mm2,
            'tops_int8': _tops_int8_from_config({"mac_engine": {
                "array_height": best.config.get("array_height", 0),
                "array_width": best.config.get("array_width", 0),
                "frequency_mhz": best.config.get("frequency_mhz", 0),
                "ops_per_mac": 2,
            }}),
            'tok_s': best.tok_s,
        }, scenario)
        print_cv(cv)

    # ── Save ──
    if args.output:
        def _result_dict(p, on_pareto=False):
            d = {"label": p.config_label, "tok_s": p.tok_s,
                 "area_mm2": p.area_mm2, "power_w": p.power_w,
                 "config": p.config}
            if _CV_MODEL:
                d["sram_spill_mb"] = p.sram_spill_mb
                d["depthwise_util_pct"] = p.depthwise_util_pct
                prefix = (p.config_label or "").split()[0]
                engine_map = {
                    "syst": "systolic",
                    "os_s": "os_systolic",
                    "bloc": "block",
                    "tens": "tensor_core",
                    "wmma": "wmma",
                    "gmma": "gmma",
                    "inpu": "input_stationary",
                    "fsa ": "fsa",
                }
                d["engine_type"] = engine_map.get(prefix, prefix)
                d["pareto"] = on_pareto
            return d

        if _CV_MODEL:
            # CV mode: flat list of Pareto + top results so downstream tools
            # can verify engine diversity while keeping Pareto points primary.
            points = [_result_dict(p, True) for p in pareto]
            seen = {p.config_label for p in pareto}
            for p in reasonable[:args.top]:
                if p.config_label not in seen:
                    points.append(_result_dict(p, False))
            output = points
        else:
            output = {
                "arc_version": "v3.1-physics-baseline",
                "cv_model": _CV_MODEL,
                "scenario": args.scenario,
                "model_spec": model_spec,
                "batch_m": batch_m,
                "total_configs": len(configs),
                "valid_results": len(results),
                "invalid_configs": invalid_configs,
                "feasible": bool(reasonable),
                "rejected_results": [p.to_dict() for p in rejected],
                "closest_candidates": [p.to_dict() for p in closest[:args.top]],
                "pareto_frontier": [p.to_dict() for p in pareto],
                "top_results": [p.to_dict() for p in reasonable[:args.top]],
                "recommended": reasonable[0].to_dict() if reasonable else None,
            }
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = SIM_DIR / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
