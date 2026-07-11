import copy

from dse.constraints import evaluate_constraints
from dse.evaluator import evaluate_candidate, estimate_layer
from dse.memory import bandwidth_bytes_per_cycle
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
        "block", "fsa", "gmma", "input_stationary", "os_systolic",
        "systolic", "tensor_core", "wmma",
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
