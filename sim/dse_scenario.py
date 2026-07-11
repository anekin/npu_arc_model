"""DSE 场景建模 — Phase -1 需求澄清 + Phase 0 预分析 + Phase 2 交叉校验

Phase -1 (requirements gathering):
  1. 加载场景定义，检查关键字段是否显式提供
  2. 对信息缺口列出问题，要求用户明确

Phase 0 (pre-sweep):
  3. 瓶颈预测 — BW 天花板 vs 算力天花板 + TTFT 驱动 TOPS 需求
  4. 组件清单校验

Phase 2 (post-sweep):
  5. 交叉校验 — 最优配置 vs 已知产品对比
"""

import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

import yaml

SIM_DIR = Path(__file__).parent


class Confidence(Enum):
    EXPLICIT = "explicit"       # 用户显式提供
    INFERRED = "inferred"       # 从场景名/模型名推导
    DEFAULTED = "defaulted"     # 用了默认值
    MISSING = "missing"         # 完全缺失


# ═══════════════════════════════════════════════════════════
# Phase -1: Requirements Clarification
# ═══════════════════════════════════════════════════════════

# Critical fields that MUST be explicitly specified (no defaults accepted silently)
CRITICAL_FIELDS = {
    "seq_len": {
        "question": "What is the expected input sequence length?",
        "why": "TTFT/TOPS requirements scale linearly with seq_len. Chat ~128, RAG ~512, VLM/VLA ~1024-2048.",
        "impact": "High — changes minimum TOPS by 8× between seq_len=128 and 1024.",
        "default": 128,
        "validate": lambda v: isinstance(v, (int, float)) and v > 0,
    },
    "ttft_ms_max": {
        "question": "What is the maximum acceptable TTFT (time-to-first-token)?",
        "why": "Drives minimum TOPS requirement. 200ms is typical for interactive use.",
        "impact": "High — tighter TTFT = more TOPS = larger die.",
        "default": 200,
        "validate": lambda v: isinstance(v, (int, float)) and 0 < v < 5000,
    },
    "tps_min": {
        "question": "What is the minimum acceptable decode throughput (tok/s)?",
        "why": "Reading speed ~15-20 tok/s for humans. 100+ for real-time VLM.",
        "impact": "Medium — drives bandwidth requirements.",
        "default": 20,
        "validate": lambda v: isinstance(v, (int, float)) and v > 0,
    },
    "model": {
        "question": "Which model family and size? (e.g., qwen2.5-3b, qwen2.5-7b)",
        "why": "Determines parameter count, layer structure, KV dimensions.",
        "impact": "High — model size scales memory BW and TOPS linearly.",
        "default": None,
        "validate": lambda v: isinstance(v, str) and len(v) > 3,
    },
    "memory.type": {
        "question": "What memory architecture? (lpddr5 / on_chip_3d_dram / hbm2e / hbm3)",
        "why": "Determines component checklist (PHY/PCIe/TSV), bandwidth, and area model.",
        "impact": "High — different memory types have different component costs and BW ceilings.",
        "default": "lpddr5",
        "validate": lambda v: v in ("lpddr5", "lpddr5x", "on_chip_3d_dram", "hbm2e", "hbm3"),
    },
    "process_nm": {
        "question": "Target process node in nm? (e.g., 12, 7, 5, 3)",
        "why": "Area scales with (process/7)^2. Critically affects cost.",
        "impact": "High — 7nm is ~3× smaller than 12nm.",
        "default": 12,
        "validate": lambda v: isinstance(v, (int, float)) and 1 < v < 100,
    },
}


def check_requirements(scenario_name: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Phase -1: Check what information is available vs what needs clarification.

    Args:
        scenario_name: Name of the scenario in scenarios.yaml
        config: Optional DSE config dict (for checking process_node etc.)

    Returns:
        {
            'ready': bool,                # True if all critical fields resolved
            'fields': {                   # Per-field status
                'seq_len': {'confidence': Confidence, 'value': any, 'question': str, ...}
            },
            'questions': [str],           # Human-readable questions to ask user
            'warnings': [str],            # Non-blocking concerns
        }
    """
    scenario = load_scenario(scenario_name)
    if not scenario:
        return {
            "ready": False,
            "error": f"Scenario '{scenario_name}' not found. Available: {list_scenarios()}",
        }

    fields = {}
    questions = []
    warnings = []

    for field_name, spec in CRITICAL_FIELDS.items():
        # Resolve value from scenario or config
        value = None
        confidence = Confidence.MISSING
        source = ""

        # Check scenario first (explicit user input)
        if "." in field_name:
            # Nested field like "memory.type"
            parts = field_name.split(".")
            val = scenario
            for p in parts:
                val = val.get(p, {}) if isinstance(val, dict) else None
            if val is not None:
                value = val
                confidence = Confidence.EXPLICIT
                source = f"scenario.{field_name}"
        else:
            if field_name in scenario:
                value = scenario[field_name]
                confidence = Confidence.EXPLICIT
                source = f"scenario.{field_name}"

        # Check config for process_nm
        if field_name == "process_nm":
            # Prefer scenario value, fall back to config
            if "process_nm" in scenario:
                value = int(scenario["process_nm"])
                confidence = Confidence.EXPLICIT
                source = "scenario.process_nm"
            elif config:
                pn = config.get("area_model", {}).get("process_node")
                if pn is not None:
                    value = int(pn)
                    confidence = Confidence.EXPLICIT
                    source = "config.area_model.process_node"

        # Infer from model name
        if field_name == "model" and value and confidence == Confidence.EXPLICIT:
            # Model was explicitly set — infer params_b
            pass

        # Apply default as fallback
        if value is None and spec.get("default") is not None:
            value = spec["default"]
            confidence = Confidence.DEFAULTED
            source = "default"

        # Validate
        validator = spec.get("validate")
        valid = True
        if validator and value is not None:
            try:
                valid = validator(value)
            except Exception:
                valid = False

        fields[field_name] = {
            "value": value,
            "confidence": confidence,
            "source": source,
            "valid": valid,
            "question": spec["question"],
            "why": spec["why"],
            "impact": spec["impact"],
        }

        # Generate questions for DEFAULTED or MISSING fields
        if confidence in (Confidence.DEFAULTED, Confidence.MISSING):
            q = f"❓ {spec['question']}"
            q += f"\n   Why it matters: {spec['why']}"
            q += f"\n   Impact: {spec['impact']}"
            q += f"\n   Current: {value} (default)" if confidence == Confidence.DEFAULTED else ""
            questions.append(q)

        # Warnings for DEFAULTED high-impact fields
        if confidence == Confidence.DEFAULTED and spec.get("impact", "").startswith("High"):
            warnings.append(
                f"⚠ {field_name}={value} (DEFAULTED, may not match your use case). "
                f"{spec['why']}"
            )

        if not valid:
            warnings.append(f"⚠ {field_name}={value} (INVALID value — check scenario config)")

    ready = len(questions) == 0

    return {
        "ready": ready,
        "fields": fields,
        "questions": questions,
        "warnings": warnings,
        "scenario_name": scenario_name,
    }


def print_requirements_check(rc: Dict[str, Any]):
    """Print a human-readable Phase -1 requirements report."""
    if "error" in rc:
        print(f"  ✗ {rc['error']}")
        return

    print(f"\n{'='*65}")
    print(f"  Phase -1 — Requirements Clarification: {rc['scenario_name']}")
    print(f"{'='*65}")

    # Critical fields table
    print(f"  {'Field':<16s} {'Value':<12s} {'Confidence':<12s} {'Source'}")
    print(f"  {'-'*55}")
    for name, f in rc["fields"].items():
        val_str = str(f["value"]) if f["value"] is not None else "—"
        conf_icon = {
            Confidence.EXPLICIT: "✓ EXPLICIT",
            Confidence.INFERRED: "~ INFERRED",
            Confidence.DEFAULTED: "⚠ DEFAULTED",
            Confidence.MISSING: "✗ MISSING",
        }.get(f["confidence"], "?")
        print(f"  {name:<16s} {val_str:<12s} {conf_icon:<12s} {f['source']}")

    if rc["questions"]:
        print(f"\n  ══ Questions to Resolve ══")
        for i, q in enumerate(rc["questions"], 1):
            print(f"\n  [{i}] {q}")

    if rc["warnings"]:
        print(f"\n  ══ Warnings ══")
        for w in rc["warnings"]:
            print(f"  {w}")

    if rc["ready"]:
        print(f"\n  ✓ All critical fields resolved. Ready for Phase 0.")
    else:
        print(f"\n  ⚠ {len(rc['questions'])} question(s) need resolution before proceeding.")

def _extract_params_b(model_name: str) -> float:
    """Extract parameter count in billions from model name."""
    import re
    m = re.search(r'(\d+\.?\d*)\s*b', model_name.lower())
    if m:
        return float(m.group(1))
    return 3.0  # default

def _estimate_min_tops(params_b: float, seq_len: int, constraints: Dict) -> float:
    """Estimate minimum TOPS needed to meet TTFT constraint.
    
    Prefill FLOPs ≈ 2 × params × seq_len (per-token, per-layer is implicit).
    TTFT = FLOPs / TOPS → TOPS_min = FLOPs / TTFT_max
    """
    ttft_max = constraints.get('ttft_ms_max', 200)
    # params_b in billions, convert to trillion ops
    flops = 2 * params_b * seq_len / 1000  # TOPS-seconds
    min_tops = flops / (ttft_max / 1000)  # TTFT in seconds
    return min_tops

# ═══════════════════════════════════════════════════════════
# Phase 0: Pre-sweep analysis
# ═══════════════════════════════════════════════════════════

def _load_scenarios() -> Dict:
    path = SIM_DIR / "config" / "scenarios.yaml"
    with open(path) as f:
        return yaml.safe_load(f)

def list_scenarios() -> List[str]:
    """List available scenario names."""
    data = _load_scenarios()
    return list(data.get("scenarios", {}).keys())

def load_scenario(name: str) -> Optional[Dict]:
    """Load a named scenario definition."""
    data = _load_scenarios()
    return data.get("scenarios", {}).get(name)

def predict_bottleneck(scenario: Dict) -> Dict[str, Any]:
    """Pre-sweep: predict whether bandwidth or compute is the bottleneck.
    
    For decode (M=1):
      - BW-limited TPS = effective_BW_GBps / model_size_GB_per_token
        where model_size_GB_per_token = params_INT4_GB (all weights accessed)
      - Compute-limited TPS = TOPS / (2 × params_billion × 1e9 ops/token)
    """
    mem = scenario.get("memory", {})
    bw = mem.get("effective_bw_gbps", mem.get("bandwidth_gbps", 51.2))
    model_gb = scenario.get("model_params_gb", 1.5)
    
    # Bandwidth ceiling: all weights must stream per token in decode
    bw_tps_ceiling = bw / model_gb
    
    # Compute ceiling: 2 ops per parameter per token (M×K + accumulation)
    # This depends on TOPS, which is array-dependent. We estimate with a range.
    # A 128×128 @ 1GHz = 16.4 TOPS → ceil = 16.4e12 / (2 × params × 1e9)
    
    # Extract params from model name
    model_name = scenario.get("model", "")
    params_b = 3 if "3b" in model_name.lower() else 7 if "7b" in model_name.lower() else 1
    
    # Use a "reasonable TOPS range" for the prediction
    typical_tops = [6, 16, 49, 131]  # H=4,16,32,128 for 1536-wide array
    
    compute_ceilings = []
    for tops in typical_tops:
        # TOPS = Tera (10^12) ops/sec. params_b = billions.
        # ops_per_token = 2 × params_b × 10^9 (M×K multiply + accumulate)
        # compute_tps = tops × 10^12 / (2 × params_b × 10^9) = tops × 500 / params_b
        ops_per_token = 2 * params_b  # billion ops
        compute_tps = tops * 500 / ops_per_token  # 500 = 1000/2
        compute_ceilings.append(compute_tps)
    
    # Which is lower?
    conclusion = []
    min_compute = min(compute_ceilings)
    max_compute = max(compute_ceilings)
    # ── TTFT / TOPS prediction ──
    seq_len = scenario.get('seq_len', 128)
    params_b = _extract_params_b(scenario.get('model', ''))
    constraints = scenario.get('constraints', {})
    min_tops_needed = _estimate_min_tops(params_b, seq_len, constraints)
    ttft_max = constraints.get('ttft_ms_max', 200)
    
    # Which is lower: BW or compute?
    conclusion = []
    if bw_tps_ceiling < min(compute_ceilings):
        conclusion.append(f"BANDWIDTH BOTTLENECK: BW ceiling {bw_tps_ceiling:.0f} TPS "
                         f"< compute floor {min(compute_ceilings):.0f} TPS")
        conclusion.append("→ Engine type and TOPS are secondary; focus on maximizing BW utilization.")
        if bw < 200:
            conclusion.append("→ SRAM size matters (affects tile granularity and BW efficiency).")
    elif bw_tps_ceiling > max(compute_ceilings):
        conclusion.append(f"COMPUTE BOTTLENECK: BW ceiling {bw_tps_ceiling:.0f} TPS "
                         f"> compute ceiling {max(compute_ceilings):.0f} TPS")
        conclusion.append("→ Array dimensions (TOPS) are the primary lever; BW is not limiting.")
    else:
        conclusion.append(f"MIXED: BW ceiling {bw_tps_ceiling:.0f} TPS within compute range "
                         f"[{min(compute_ceilings):.0f}, {max(compute_ceilings):.0f}] TPS")
        conclusion.append("→ Both TOPS and BW utilization matter; sweep both dimensions.")
    
    return {
        "bw_ceiling_tps": round(bw_tps_ceiling, 1),
        "compute_ceiling_tps_range": [round(c, 1) for c in compute_ceilings],
        "ttft_constraint_ms": ttft_max,
        "ttft_min_tops_needed": round(min_tops_needed, 1),
        "seq_len": seq_len,
        "params_billion": params_b,
        "conclusion": conclusion,
        "effective_bw_gbps": bw,
        "model_size_gb": model_gb,
    }

def validate_components(scenario: Dict, config: Dict[str, Any]) -> List[str]:
    """Validate that the DSE config matches the scenario's component checklist.
    
    Returns a list of warnings (empty = all good).
    """
    warnings = []
    comp = scenario.get("components", {})
    required = comp.get("required", [])
    excluded = comp.get("excluded", [])
    
    oc = config.get("on_chip_memory", {})
    has_onchip = float(oc.get("capacity_gb", 0)) > 0
    
    mem = config.get("memory", {})
    has_ddr = float(mem.get("bandwidth_gbps", 0)) > 0
    
    # Check required components
    if "dram_phy" in required and not has_ddr:
        warnings.append("MISSING: scenario requires DRAM PHY but memory.bandwidth_gbps=0")
    if "pcie" in required:
        # PCIe is always present in AreaModel.estimate() — just note it's needed
        pass
    if "tsv" in required and not has_onchip:
        warnings.append("MISSING: scenario requires TSV but on_chip_memory.capacity_gb=0")
    
    # Check excluded components
    if "dram_phy" in excluded and has_ddr:
        warnings.append("EXCESS: scenario excludes DRAM PHY but memory.bandwidth_gbps > 0")
    if "tsv" in excluded and has_onchip:
        warnings.append("EXCESS: scenario excludes TSV but on_chip_memory is configured")
    
    # TSV overhead check
    if "tsv" in required:
        expected_tsv = comp.get("tsv_overhead_pct") or scenario.get("memory", {}).get("tsv_overhead_pct", 0.10)
        actual_tsv = float(config.get("area_model", {}).get("tsv_overhead_pct", 0))
        if abs(actual_tsv - expected_tsv) > 0.01:
            warnings.append(f"TSV MISMATCH: scenario expects {expected_tsv:.0%} but "
                          f"area_model.tsv_overhead_pct={actual_tsv:.0%}")
    
    return warnings

def preflight(scenario_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run full Phase 0 pre-sweep analysis on a scenario + config.
    
    Returns:
        {
            'scenario': scenario dict,
            'bottleneck': predict_bottleneck output,
            'component_warnings': list of issues found,
            'recommendations': list of pre-sweep recommendations,
        }
    """
    scenario = load_scenario(scenario_name)
    if not scenario:
        return {"error": f"Scenario '{scenario_name}' not found. "
                f"Available: {list_scenarios()}"}
    
    bottleneck = predict_bottleneck(scenario)
    component_warnings = validate_components(scenario, config)
    
    recommendations = []
    
    # SRAM recommendation based on bottleneck + bandwidth regime
    if "BANDWIDTH BOTTLENECK" in (bottleneck.get('conclusion', [''])[0]):
        bw_actual = bottleneck.get('effective_bw_gbps', 0)
        if bw_actual >= 200:
            recommendations.append(
                f"SRAM: sweep from 512KB. BW={bw_actual:.0f}GB/s is high — "
                "SRAM may have low impact (tile streaming cost negligible). "
                "Sensitivity analysis will confirm."
            )
        else:
            recommendations.append(
                f"SRAM: sweep L2 from 1MB to 8MB — larger SRAM reduces DRAM tile reads. "
                f"BW={bw_actual:.0f}GB/s is constrained."
            )
    else:
        recommendations.append("SRAM: 512KB-1MB likely sufficient. Start sweep from 512KB.")
    
    # Array dimension recommendation
    ttft_min = bottleneck.get('ttft_min_tops_needed', 0)
    sl = bottleneck.get('seq_len', 128)
    if ttft_min > 16:
        recommendations.append(
            f"Array: TTFT requires ≥{ttft_min:.0f} TOPS (seq_len={sl}). "
            f"Sweep from {max(4, int(ttft_min/1.5))} TOPS upward."
        )
    elif "BANDWIDTH BOTTLENECK" in (bottleneck.get('conclusion', [''])[0]):
        recommendations.append("Array: BW-limited → oversizing array wastes area. Prefer smaller array (≤16 TOPS).")
    elif "COMPUTE BOTTLENECK" in (bottleneck.get('conclusion', [''])[0]):
        recommendations.append("Array: compute-limited → increase array size or frequency.")
    
    return {
        "scenario": scenario,
        "bottleneck": bottleneck,
        "component_warnings": component_warnings,
        "recommendations": recommendations,
        "scenario_name": scenario_name,
    }

# ═══════════════════════════════════════════════════════════
# Phase 2: Post-sweep cross-validation
# ═══════════════════════════════════════════════════════════

def cross_validate(best_config: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
    """Compare the best DSE result against known products.
    
    best_config: dict with keys like 'tok_s', 'area_mm2', 'tops_int8', 'process_nm'
    """
    data = _load_scenarios()
    scenario = data.get("scenarios", {}).get(scenario_name, {})
    product_db = data.get("product_database", {})
    benchmarks = scenario.get("benchmarks", [])
    
    process_nm = best_config.get("process_nm", 12)
    area = best_config.get("area_mm2", 0)
    tops = best_config.get("tops_int8", 0)
    tps = best_config.get("tok_s", 0)
    
    comparisons = []
    warnings = []
    
    for bm in benchmarks:
        name = bm.get("name", "unknown")
        # Look up detailed product data if available
        product = product_db.get(name.lower().replace(" ", "_").replace("(", "").replace(")", ""), {})
        bm_process = bm.get("process_nm") or product.get("process_nm", 12)
        bm_area = bm.get("area_mm2") or product.get("area_mm2", 0)
        bm_tops = bm.get("tops_int8") or product.get("tops_int8", 0)
        
        if not bm_area or not bm_tops:
            continue
        
        # Scale to our process node for fair comparison
        scale = (process_nm / bm_process) ** 2
        bm_area_scaled = bm_area * scale
        bm_tops_scaled = bm_tops  # TOPS doesn't scale linearly with process
        
        our_mm2_per_tops = area / max(tops, 0.01)
        bm_mm2_per_tops = bm_area_scaled / max(bm_tops_scaled, 0.01)
        ratio_tops = our_mm2_per_tops / max(bm_mm2_per_tops, 0.01)
        
        # Also compare TPS efficiency (more relevant for BW-bottlenecked designs)
        tps_warning = ""
        bm_tps_7b = bm.get("tps_7b") or product.get("tps_7b_range")
        if bm_tps_7b and tps > 0:
            bm_tps_mid = sum(bm_tps_7b) / 2 if isinstance(bm_tps_7b, list) else bm_tps_7b
            our_mm2_per_tps = area / max(tps, 0.01)
            bm_mm2_per_tps = bm_area_scaled / max(bm_tps_mid, 0.01)
            ratio_tps = our_mm2_per_tps / max(bm_mm2_per_tps, 0.01)
            tps_compare = (f"mm²/TPS: ours={our_mm2_per_tps:.2f}, {name}={bm_mm2_per_tps:.2f} "
                          f"(×{1/ratio_tps:.1f} better/worse)")
        else:
            tps_compare = ""
            ratio_tps = None
        
        comp = {
            "benchmark": name,
            "bm_process_nm": bm_process,
            "bm_area_mm2": bm_area,
            "bm_area_scaled_mm2": round(bm_area_scaled, 1),
            "bm_tops": bm_tops,
            "our_mm2_per_tops": round(our_mm2_per_tops, 2),
            "bm_mm2_per_tops": round(bm_mm2_per_tops, 2),
            "ratio_tops": round(ratio_tops, 2),
            "tps_compare": tps_compare,
        }
        comparisons.append(comp)
        
        # Generate warnings — prefer TPS-based ratio when available
        primary_ratio = ratio_tps if ratio_tps is not None else ratio_tops
        metric_name = "mm²/TPS" if ratio_tps is not None else "mm²/TOPS"
        
        if primary_ratio > 2.0:
            warnings.append(
                f"⚠ AREA ANOMALY: our {metric_name}={primary_ratio:.1f}× worse than {name}. "
                f"Check PE area model or component overhead."
            )
        elif primary_ratio < 0.5:
            warnings.append(
                f"⚠ SUSPICIOUSLY EFFICIENT: our {metric_name} is {1/primary_ratio:.1f}× "
                f"better than {name}. Verify area model isn't omitting components."
            )
    
    return {
        "comparisons": comparisons,
        "warnings": warnings,
        "process_nm": process_nm,
    }

def print_preflight(preflight_result: Dict[str, Any]):
    """Print a human-readable Phase 0 preflight report."""
    if "error" in preflight_result:
        print(f"  ✗ {preflight_result['error']}")
        return
    
    print(f"\n{'='*70}")
    print(f"  Phase 0 — Scenario Preflight: {preflight_result['scenario_name']}")
    print(f"{'='*70}")
    
    sc = preflight_result['scenario']
    print(f"  Model: {sc.get('model')} ({sc.get('model_params_gb')} GB INT4)")
    
    # Bottleneck
    bn = preflight_result['bottleneck']
    print(f"\n  ── Bottleneck & TTFT Prediction ──")
    print(f"  Seq length:    {bn['seq_len']} tokens")
    print(f"  BW ceiling:    {bn['bw_ceiling_tps']:.0f} TPS "
          f"(={bn['effective_bw_gbps']} GB/s ÷ {bn['model_size_gb']} GB)")
    print(f"  Compute range: [{bn['compute_ceiling_tps_range'][0]:.0f}, "
          f"{bn['compute_ceiling_tps_range'][-1]:.0f}] TPS (6-131 TOPS)")
    print(f"  TTFT constraint: <{bn['ttft_constraint_ms']}ms")
    print(f"  → Min TOPS needed: ~{bn['ttft_min_tops_needed']:.1f} TOPS "
          f"(={2*bn['params_billion']:.0f}B × {bn['seq_len']}tok × 2ops ÷ {bn['ttft_constraint_ms']}ms)")
    for line in bn['conclusion']:
        print(f"  → {line}")
    
    # Components
    warnings = preflight_result.get('component_warnings', [])
    print(f"\n  ── Component Checklist ──")
    comp = sc.get('components', {})
    print(f"  Required: {comp.get('required', [])}")
    print(f"  Excluded: {comp.get('excluded', [])}")
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print(f"  ✓ All component constraints satisfied")
    
    # Recommendations
    recs = preflight_result.get('recommendations', [])
    if recs:
        print(f"\n  ── Pre-sweep Recommendations ──")
        for r in recs:
            print(f"  • {r}")

def print_cross_validate(cv_result: Dict[str, Any]):
    """Print human-readable cross-validation report."""
    comparisons = cv_result.get('comparisons', [])
    if not comparisons:
        print("  No benchmarks to compare against.")
        return
    
    print(f"\n  ── Cross-Validation (vs known products @{cv_result['process_nm']}nm) ──")
    print(f"  {'Product':<20s} {'Orig':>6s} {'Scaled':>7s} {'mm²/TOPS':>9s} {'Ours':>9s} {'Ratio':>7s}")
    print(f"  {'-'*65}")
    
    for c in comparisons:
        orig = f"{c['bm_area_mm2']:.0f}mm²@{c['bm_process_nm']}nm"
        print(f"  {c['benchmark']:<20s} {orig:>6s} {c['bm_area_scaled_mm2']:>6.0f}mm² "
              f"{c['bm_mm2_per_tops']:>8.2f} {c['our_mm2_per_tops']:>8.2f} {c['ratio_tops']:>6.1f}x (TOPS)")
        if c['tps_compare']:
            print(f"    {c['tps_compare']}")
    
    cv_warnings = cv_result.get('warnings', [])
    if cv_warnings:
        print(f"\n  ══ Cross-Validation Warnings ══")
        for w in cv_warnings:
            print(f"  {w}")
