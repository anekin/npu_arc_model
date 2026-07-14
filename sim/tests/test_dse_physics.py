import copy
import math

from dse.constraints import evaluate_constraints
from dse.evaluator import evaluate_candidate, estimate_layer
from dse.memory import bandwidth_bytes_per_cycle, estimate_memory_footprint
from dse.workload import load_workload
from engine.mac_engine import create_engine
from engine.ppa_model import AreaModel, PowerModel


def _config(engine="block", freq=1000, bw=51.2):
    return {
        "mac_engine": {"type": engine, "array_height": 64, "array_width": 128,
                       "frequency_mhz": freq, "weight_precision_bits": 4,
                       "activation_precision_bits": 8, "ops_per_mac": 2},
        "memory": {"type": "lpddr5", "bandwidth_gbps": bw,
                   "dram_efficiency": 0.85, "dram_width_bits": 64},
        "sram": {"l1_per_core_kb": 512, "l2_shared_kb": 2048},
        "optimizations": {"weight_cache": False, "dma_bw_multiplier": 1.0},
        "dma": {"channels": 2},
        "sfu": {"width": 128, "pipeline_cycles": {}},
        "vector": {"width": 128, "ops": {}},
        "area_model": {"process_node": 12},
    }


def test_physical_bandwidth_conversion_keeps_gbps_constant():
    assert bandwidth_bytes_per_cycle({"bandwidth_gbps": 51.2}, 800) == 64.0
    assert bandwidth_bytes_per_cycle({"bandwidth_gbps": 51.2}, 1000) == 51.2
    assert round(bandwidth_bytes_per_cycle({"bandwidth_gbps": 51.2}, 1200), 6) == 42.666667


def test_prefill_grows_with_sequence_length():
    cfg = _config()
    short = estimate_layer(cfg, load_workload("qwen2.5-3b", 64), "prefill")
    long = estimate_layer(cfg, load_workload("qwen2.5-3b", 128), "prefill")
    assert long.total_cycles > short.total_cycles
    assert long.attention_cycles > short.attention_cycles


def test_fsa_attention_scales_with_context_and_query_heads():
    engine = create_engine(_config(engine="fsa"))
    short = engine.estimate_attention(1, 128, 128, num_heads=16, num_kv_heads=2)
    long = engine.estimate_attention(1, 512, 128, num_heads=16, num_kv_heads=2)
    fewer_heads = engine.estimate_attention(1, 128, 128, num_heads=8, num_kv_heads=2)
    assert long.total_cycles > short.total_cycles
    assert short.compute_cycles > fewer_heads.compute_cycles


def test_constraints_reject_non_compliant_candidate():
    result = evaluate_constraints(
        {"tok_s": 19, "ttft_ms": 250, "area_mm2": 90, "power_w": 10},
        {"constraints": {"tps_min": 20, "ttft_ms_max": 200, "area_mm2_max": 80}},
    )
    assert not result.passed
    assert len(result.failed_reasons) == 3


def test_onchip_bandwidth_is_area_coupled():
    cfg = _config(bw=500)
    cfg["memory"].update({"type": "on_chip_3d_dram", "dram_efficiency": 1.0})
    cfg["on_chip_memory"] = {"capacity_gb": 5, "bandwidth_gbps": 500,
                              "bw_per_mm2_gbps": 7.5, "stack_area_mm2": 50}
    cfg["_model_name"] = "qwen2.5-3b"
    cfg["_seq_len"] = 64
    small = evaluate_candidate(copy.deepcopy(cfg), AreaModel(cfg), PowerModel(cfg), None)
    large_cfg = copy.deepcopy(cfg)
    large_cfg["mac_engine"]["array_width"] = 256
    large = evaluate_candidate(large_cfg, AreaModel(large_cfg), PowerModel(large_cfg), None)
    assert large.bandwidth_gbps > small.bandwidth_gbps


def test_all_searchable_engines_complete_candidate_evaluation():
    from design_space_explorer import generate_configs

    expected = {
        "block", "block_fused_attention", "fsa", "gmma",
        "input_stationary", "os_systolic",
        "os_systolic_fused_attention", "systolic",
        "tensor_core", "wmma",
    }
    first_by_engine = {}
    for cfg in generate_configs(quick=False):
        first_by_engine.setdefault(cfg["mac_engine"]["type"], cfg)
        if first_by_engine.keys() == expected:
            break

    assert first_by_engine.keys() == expected
    for engine, cfg in first_by_engine.items():
        point = evaluate_candidate(
            copy.deepcopy(cfg), AreaModel(cfg), PowerModel(cfg), None,
        )
        assert point.tok_s > 0, engine


def test_scenario_a_contract_is_low_cost_int4_only():
    from design_space_explorer import _load_scenario, generate_configs

    scenario = _load_scenario("lpddr5_3b")
    configs = generate_configs(quick=False, scenario_name="lpddr5_3b")
    assert configs
    assert scenario["constraints"]["ttft_ms_max"] == 1000
    assert scenario["targets"]["ttft_ms_max"] == 500
    assert scenario["memory"]["dram_efficiency"] == 0.85
    assert scenario["memory"]["efficiency_corners"] == [0.75, 0.85, 0.90]
    assert {cfg["mac_engine"]["weight_precision_bits"] for cfg in configs} == {4}
    assert any(
        cfg["mac_engine"]["array_height"] == 64
        and cfg["mac_engine"]["array_width"] == 64
        for cfg in configs
    )


def test_external_memory_capacity_does_not_create_onchip_memory():
    from design_space_explorer import (
        _CUSTOM_SCENARIOS, _apply_scenario, _load_base_config,
    )

    name = "test_external_lpddr_capacity"
    _CUSTOM_SCENARIOS[name] = {
        "model": "qwen2.5-3b",
        "memory": {
            "type": "lpddr5", "capacity_gb": 4,
            "bandwidth_gbps": 51.2, "dram_efficiency": 0.85,
        },
    }
    cfg = _apply_scenario(_load_base_config(), name)
    assert cfg["memory"]["capacity_gb"] == 4
    assert "on_chip_memory" not in cfg


def test_memory_footprint_scales_with_context_and_concurrency():
    cfg = _config()
    cfg["memory"]["capacity_gb"] = 8
    small = load_workload(
        "qwen2.5-3b", 128,
        {"output_tokens": 128, "concurrent_requests": 1, "kv_bits": 16},
    )
    large = load_workload(
        "qwen2.5-3b", 1024,
        {"output_tokens": 1024, "concurrent_requests": 8, "kv_bits": 16},
    )
    small_fp = estimate_memory_footprint(cfg, small)
    large_fp = estimate_memory_footprint(cfg, large)
    assert large_fp.kv_cache_gb > small_fp.kv_cache_gb * 10
    assert large_fp.required_gb > small_fp.required_gb
    assert small_fp.weights_gb == large_fp.weights_gb


def test_insufficient_memory_is_a_physical_constraint():
    cfg = _config()
    cfg["memory"].update({"capacity_gb": 1.0, "capacity_usable_fraction": 0.9})
    cfg["_model_name"] = "qwen2.5-3b"
    cfg["_seq_len"] = 128
    scenario = {
        "model": "qwen2.5-3b",
        "workload": {"prompt_tokens": 128, "output_tokens": 128},
        "constraints": {},
    }
    point = evaluate_candidate(cfg, AreaModel(cfg), PowerModel(cfg), scenario)
    assert not point.memory_fits
    assert not point.constraints_passed
    assert any("memory capacity" in reason for reason in point.failed_reasons)


def test_performance_contract_reports_batch_and_latency_metrics():
    cfg = _config(bw=102.4)
    cfg["memory"].update({"capacity_gb": 8.0, "capacity_usable_fraction": 0.9})
    cfg["_model_name"] = "qwen2.5-3b"
    cfg["_seq_len"] = 128
    cfg["_workload"] = {
        "prompt_tokens": 128, "output_tokens": 32,
        "concurrent_requests": 4, "decode_batch_size": 4,
    }
    scenario = {
        "model": "qwen2.5-3b", "workload": cfg["_workload"],
        "constraints": {"aggregate_tps_min": 1, "itl_ms_max": 10000},
    }
    point = evaluate_candidate(cfg, AreaModel(cfg), PowerModel(cfg), scenario)
    assert point.decode_tps > 0
    assert abs(point.aggregate_tps - point.decode_tps * 4) < 0.05
    assert point.prefill_tps > 0
    assert point.itl_ms == point.decode_ms
    assert point.e2e_latency_ms > point.ttft_ms
    assert point.breakdown["workload"]["decode_batch_size"] == 4


def test_new_performance_constraints_are_enforced():
    result = evaluate_constraints(
        {
            "decode_tps": 20, "aggregate_tps": 80, "prefill_tps": 900,
            "ttft_ms": 250, "itl_ms": 50, "e2e_latency_ms": 1000,
            "area_mm2": 50, "power_w": 10,
        },
        {"constraints": {
            "decode_tps_min": 25, "aggregate_tps_min": 100,
            "prefill_tps_min": 1000, "itl_ms_max": 40,
        }},
    )
    assert not result.passed
    assert len(result.failed_reasons) == 4


def test_os_and_block_have_distinct_dataflow_scaling():
    base = _config()
    base["mac_engine"].update({
        "array_height": 64,
        "array_width": 64,
        "frequency_mhz": 1000,
        "weight_precision_bits": 4,
    })

    results = {}
    for m_dim in (1, 128):
        results[m_dim] = {}
        for engine_type in ("block", "os_systolic"):
            cfg = copy.deepcopy(base)
            cfg["mac_engine"]["type"] = engine_type
            results[m_dim][engine_type] = create_engine(cfg).estimate(
                m_dim, 2048, 2048,
            )

    # Block maps H onto the K reduction and is efficient for M=1 decode.
    assert results[1]["block"].total_cycles < results[1]["os_systolic"].total_cycles
    # OS maps H onto M and amortizes K across a full prefill tile.
    assert results[128]["os_systolic"].total_cycles < results[128]["block"].total_cycles

    os_decode = results[1]["os_systolic"]
    assert os_decode.details["mapping"] == "M_by_N_spatial_K_temporal"
    assert os_decode.details["M_tiles"] == 1
    assert os_decode.compute_cycles >= math.ceil(1 * 2048 * 2048 / (64 * 64))


def test_os_ppa_uses_independent_area_baseline():
    block_cfg = _config(engine="block")
    block_cfg["area_model"].update({
        "block_pe_area_mm2": 4.0,
        "os_pe_area_mm2": 3.0,
    })
    os_cfg = copy.deepcopy(block_cfg)
    os_cfg["mac_engine"]["type"] = "os_systolic"

    block_area = AreaModel(block_cfg).estimate(block_cfg, "block")["total_mm2"]
    os_area = AreaModel(os_cfg).estimate(os_cfg, "os_systolic")["total_mm2"]
    assert os_area < block_area


def test_os_point_records_analytical_calibration():
    cfg = _config(engine="os_systolic")
    point = evaluate_candidate(
        cfg,
        AreaModel(cfg),
        PowerModel(cfg),
        {"model": "qwen2.5-3b", "workload": {"prompt_tokens": 128}},
    )
    assert point.provenance["calibration_tier"] == "dataflow_analytical"
    assert any("OS uses an analytical" in warning for warning in point.warnings)


def test_qwen25_3b_official_gqa_and_32k_kv_capacity():
    from model_specs import get_spec

    spec = get_spec("qwen2.5-3b")
    assert spec.num_heads == 16
    assert spec.kv_heads == 2
    workload = load_workload(
        "qwen2.5-3b", 875,
        {
            "output_tokens": 214,
            "max_context_tokens": 32768,
            "kv_bits": 16,
        },
    )
    cfg = _config()
    cfg["memory"].update({"capacity_gb": 4, "capacity_usable_fraction": 0.9})
    footprint = estimate_memory_footprint(cfg, workload)
    assert workload.max_context_tokens == 32768

    assert 1.20 < footprint.kv_cache_gb < 1.22
    assert footprint.fits


def test_agent_subscenario_uses_incremental_append_and_32k_context():
    from design_space_explorer import _load_scenario, generate_configs

    scenario = _load_scenario("lpddr5_3b_agent")
    assert scenario["workload"]["prompt_tokens"] == 875
    assert scenario["workload"]["output_tokens"] == 214
    assert scenario["workload"]["max_context_tokens"] == 32768
    assert scenario["workload"]["cached_prefix_tokens"] == 30000
    assert scenario["workload"]["attention_bits"] == 16
    assert scenario["agent_workload"]["prefix_cache_hit_rate"] == 0.90
    assert scenario["memory"]["capacity_gb"] == 4
    configs = generate_configs(quick=False, scenario_name="lpddr5_3b_agent")
    assert {cfg["mac_engine"]["weight_precision_bits"] for cfg in configs} == {4}


def test_decode_cannot_exceed_full_model_memory_ceiling():
    cfg = _config()
    cfg["_model_name"] = "qwen2.5-3b"
    cfg["_seq_len"] = 128
    point = evaluate_candidate(
        cfg,
        AreaModel(cfg),
        PowerModel(cfg),
        {"model": "qwen2.5-3b", "workload": {"prompt_tokens": 128}},
    )
    effective_bw_gbps = 51.2 * 0.85
    int4_weight_gb = 3.09 * 4 / 8
    assert point.decode_tps <= effective_bw_gbps / int4_weight_gb + 0.01

def test_cached_prefix_separates_prompt_from_attention_context():
    workload = load_workload(
        "qwen2.5-3b",
        875,
        {
            "output_tokens": 214,
            "cached_prefix_tokens": 30000,
            "max_context_tokens": 32768,
            "causal_attention": True,
        },
    )
    assert workload.prompt_tokens == 875
    assert workload.prefill_context_tokens == 30875
    assert workload.causal_prefill_compute_context_tokens == 30438
    assert workload.max_context_tokens == 32768

    short = load_workload(
        "qwen2.5-3b",
        128,
        {"causal_attention": True},
    )
    assert short.causal_prefill_compute_context_tokens == 65


def test_fsa_attention_uses_explicit_precision_and_rtl_schedule():
    cfg = _config(engine="fsa")
    cfg["mac_engine"].update({"array_height": 128, "array_width": 128})
    engine = create_engine(cfg)
    fp8 = engine.estimate_attention(128, 128, 128, attention_bits=8)
    fp16 = engine.estimate_attention(128, 128, 128, attention_bits=16)
    assert fp16.details["attention_bytes"] == 2 * fp8.details["attention_bytes"]
    assert fp16.details["schedule_source"] == "upstream_execution_plan"
    assert fp16.details["calibration_status"] == "paper_extrapolation"
    assert fp16.details["mapping_compatible"]


def test_fsa_gate_up_pair_counts_two_distinct_weights():
    engine = create_engine(_config(engine="fsa"))
    single = engine.estimate(32, 2048, 11008)
    pair = engine.estimate_weight_cache_pair(32, 2048, 11008)
    assert pair.total_cycles == 2 * single.total_cycles
    assert pair.weight_bytes == 2 * single.weight_bytes
    assert pair.ops == 2 * single.ops


def test_fsa_is_reported_but_not_automatically_recommended():
    cfg = _config(engine="fsa")
    cfg["mac_engine"].update({"array_height": 128, "array_width": 128})
    scenario = {
        "model": "qwen2.5-3b",
        "workload": {
            "prompt_tokens": 128,
            "output_tokens": 32,
            "attention_bits": 16,
            "causal_attention": True,
        },
        "constraints": {},
    }
    point = evaluate_candidate(cfg, AreaModel(cfg), PowerModel(cfg), scenario)
    assert point.constraints_passed
    assert not point.recommendation_eligible
    assert point.provenance["calibration_tier"] == "paper_extrapolation"
    assert any("research candidate" in warning for warning in point.warnings)


def test_fsa_search_respects_public_rtl_mapping():
    from design_space_explorer import generate_configs

    base = generate_configs(quick=False, scenario_name="lpddr5_3b")
    base_fsa = [cfg for cfg in base if cfg["mac_engine"]["type"] == "fsa"]
    assert base_fsa
    assert {
        (cfg["mac_engine"]["array_height"], cfg["mac_engine"]["array_width"])
        for cfg in base_fsa
    } == {(128, 128)}

    agent = generate_configs(quick=False, scenario_name="lpddr5_3b_agent")
    agent_fsa = [cfg for cfg in agent if cfg["mac_engine"]["type"] == "fsa"]
    assert agent_fsa
    assert {cfg["mac_engine"]["array_height"] for cfg in agent_fsa} == {128}

def test_fused_attention_projection_paths_match_their_baselines():
    pairs = (
        ("block", "block_fused_attention"),
        ("os_systolic", "os_systolic_fused_attention"),
    )
    for baseline_type, fused_type in pairs:
        baseline = create_engine(_config(engine=baseline_type))
        fused = create_engine(_config(engine=fused_type))
        for method in ("estimate", "estimate_weight_cache_pair"):
            base_result = getattr(baseline, method)(32, 2048, 11008)
            fused_result = getattr(fused, method)(32, 2048, 11008)
            assert fused_result.total_cycles == base_result.total_cycles
            assert fused_result.compute_cycles == base_result.compute_cycles
            assert fused_result.dma_cycles == base_result.dma_cycles
            assert fused_result.weight_bytes == base_result.weight_bytes


def test_fused_attention_reduces_agent_prefill_without_changing_context():
    workload = load_workload(
        "qwen2.5-3b",
        875,
        {
            "cached_prefix_tokens": 30000,
            "max_context_tokens": 32768,
            "attention_bits": 16,
            "causal_attention": True,
        },
    )
    baseline = estimate_layer(_config(engine="block"), workload, "prefill")
    fused = estimate_layer(
        _config(engine="block_fused_attention"), workload, "prefill",
    )
    assert fused.total_cycles < baseline.total_cycles
    assert fused.details["context_tokens"] == baseline.details["context_tokens"] == 30875
    attention = fused.details["attention"]
    assert attention["projection_engine"] == "block"
    assert attention["inline_softmax"]
    assert attention["external_softmax_cycles"] == 0
    assert attention["query_position_offset_required"]


def test_fused_os_and_block_preserve_distinct_dataflows():
    workload = load_workload(
        "qwen2.5-3b",
        875,
        {
            "cached_prefix_tokens": 30000,
            "max_context_tokens": 32768,
            "attention_bits": 16,
            "causal_attention": True,
        },
    )
    block_fused = estimate_layer(
        _config(engine="block_fused_attention"), workload, "prefill",
    )
    os_fused = estimate_layer(
        _config(engine="os_systolic_fused_attention"), workload, "prefill",
    )
    assert os_fused.total_cycles < block_fused.total_cycles
    assert (
        os_fused.details["attention"]["projection_engine"]
        == "os_systolic"
    )


def test_fused_attention_has_incremental_area_over_its_baseline():
    for baseline_type, fused_type in (
        ("block", "block_fused_attention"),
        ("os_systolic", "os_systolic_fused_attention"),
    ):
        baseline_cfg = _config(engine=baseline_type)
        fused_cfg = _config(engine=fused_type)
        baseline_area = AreaModel(baseline_cfg).estimate(
            baseline_cfg, baseline_type,
        )["total_mm2"]
        fused_area = AreaModel(fused_cfg).estimate(
            fused_cfg, fused_type,
        )["total_mm2"]
        assert fused_area > baseline_area
        assert fused_area - baseline_area < 1.0


def test_fused_attention_candidates_are_distinct_research_points():
    cfg = _config(engine="block_fused_attention")
    scenario = {
        "model": "qwen2.5-3b",
        "workload": {
            "prompt_tokens": 128,
            "attention_bits": 16,
            "causal_attention": True,
        },
        "constraints": {},
    }
    point = evaluate_candidate(cfg, AreaModel(cfg), PowerModel(cfg), scenario)
    assert point.config_label.startswith("bfsa ")
    assert point.config["engine"] == "block_fused_attention"
    assert point.config["projection_engine"] == "block"
    assert point.config["attention_mode"] == "fused_array"
    assert not point.recommendation_eligible
    assert point.provenance["calibration_tier"] == "fsa_inspired_analytical"


def test_fused_attention_search_is_not_limited_to_paper_fsa_mapping():
    from design_space_explorer import generate_configs

    configs = generate_configs(quick=True, scenario_name="lpddr5_3b")
    dims = {
        (cfg["mac_engine"]["array_height"], cfg["mac_engine"]["array_width"])
        for cfg in configs
        if cfg["mac_engine"]["type"] == "block_fused_attention"
    }
    assert (64, 64) in dims
    assert any(height != 128 for height, _ in dims)
