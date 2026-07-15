import copy

import pytest

from design_space_explorer import _load_base_config, generate_configs
from engine.mac_engine import ENGINE_TYPES, create_engine
from engine.manifest import (
    MATURITY_RANK,
    engine_names,
    get_engine_manifest,
    load_engine_manifests,
    validate_manifest_set,
)
from engine.ppa_model import AreaModel, PowerModel
from engine_audit import MICROBENCH_SHAPES, run_audit


def _config(engine, bandwidth_gbps=51.2):
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
        "bandwidth_gbps": bandwidth_gbps,
        "dram_efficiency": 0.85,
    })
    return config


def test_manifest_set_matches_factory_and_full_dse_search():
    validate_manifest_set(ENGINE_TYPES)
    manifests = load_engine_manifests()
    searched = {
        cfg["mac_engine"]["type"] for cfg in generate_configs(quick=False)
    }
    assert set(manifests) == set(ENGINE_TYPES) == searched
    assert set(engine_names("M2")) == {
        "systolic", "os_systolic", "block",
    }


@pytest.mark.parametrize("engine_name", ENGINE_TYPES)
def test_manifest_factory_identity_and_maturity_contract(engine_name):
    manifest = get_engine_manifest(engine_name)
    engine = create_engine(_config(engine_name))
    assert engine.engine_type == engine_name
    assert manifest.role == "dse_candidate"
    assert manifest.maturity in MATURITY_RANK
    assert manifest.raw_exploration_eligible
    assert bool(manifest.comparison_ready) == (manifest.maturity_rank >= 2)
    assert bool(manifest.product_qualified) == (manifest.maturity_rank >= 3)
    assert manifest.known_gaps
    assert manifest.sources


@pytest.mark.parametrize("engine_name", ENGINE_TYPES)
@pytest.mark.parametrize("shape", MICROBENCH_SHAPES.values())
def test_every_engine_microbench_is_finite_and_physically_bounded(
    engine_name, shape,
):
    engine = create_engine(_config(engine_name))
    result = engine.estimate(*shape)
    m_dim, k_dim, n_dim = shape
    assert result.total_cycles > 0
    assert result.compute_cycles >= 0
    assert result.dma_cycles >= 0
    assert result.total_cycles >= result.dma_cycles
    assert 0 <= result.utilization <= 1
    assert result.ops == m_dim * k_dim * n_dim * engine.ops_per_mac
    assert result.weight_bytes >= 0
    assert result.bottleneck in {"compute", "dma"}


@pytest.mark.parametrize("engine_name", ENGINE_TYPES)
def test_every_engine_bandwidth_preload_pair_and_ppa_properties(engine_name):
    shape = (32, 256, 256)
    config = _config(engine_name)
    engine = create_engine(config)
    baseline = engine.estimate(*shape)
    high_bw = create_engine(_config(engine_name, 102.4)).estimate(*shape)
    preloaded = engine.estimate(*shape, weight_preloaded=True)
    pair = engine.estimate_weight_cache_pair(*shape)
    assert high_bw.total_cycles <= baseline.total_cycles
    assert preloaded.total_cycles <= baseline.total_cycles
    assert pair.total_cycles <= 2 * baseline.total_cycles

    small_area = AreaModel(config).estimate(config, engine_name)["total_mm2"]
    large = copy.deepcopy(config)
    large["mac_engine"]["array_width"] = 128
    large_area = AreaModel(large).estimate(large, engine_name)["total_mm2"]
    power = PowerModel(config).estimate(AreaModel(config), config, engine_name)
    assert small_area > 0
    assert power > 0
    assert large_area >= small_area


def test_engine_specific_dataflow_directions():
    block = create_engine(_config("block"))
    os_engine = create_engine(_config("os_systolic"))
    input_stationary = create_engine(_config("input_stationary"))
    assert block.estimate(1, 2048, 2048).total_cycles < os_engine.estimate(
        1, 2048, 2048,
    ).total_cycles
    assert os_engine.estimate(128, 2048, 2048).total_cycles < block.estimate(
        128, 2048, 2048,
    ).total_cycles
    assert input_stationary.estimate(64, 256, 256).total_cycles < (
        input_stationary.estimate(1, 256, 256).total_cycles
    )


def test_fused_attention_ppa_is_not_free():
    for baseline, fused in (
        ("block", "block_fused_attention"),
        ("os_systolic", "os_systolic_fused_attention"),
    ):
        baseline_cfg = _config(baseline)
        fused_cfg = _config(fused)
        baseline_area = AreaModel(baseline_cfg).estimate(
            baseline_cfg, baseline,
        )["total_mm2"]
        fused_area = AreaModel(fused_cfg).estimate(fused_cfg, fused)["total_mm2"]
        baseline_power = PowerModel(baseline_cfg).estimate(
            AreaModel(baseline_cfg), baseline_cfg, baseline,
        )
        fused_power = PowerModel(fused_cfg).estimate(
            AreaModel(fused_cfg), fused_cfg, fused,
        )
        assert fused_area > baseline_area
        assert fused_power >= baseline_power


@pytest.mark.parametrize(
    "engine_name",
    ["systolic", "block", "block_fused_attention", "gmma"],
)
def test_weight_cache_hardware_variant_has_nonzero_ppa_cost(engine_name):
    off = _config(engine_name)
    off.setdefault("optimizations", {})["weight_cache"] = False
    on = copy.deepcopy(off)
    on["optimizations"]["weight_cache"] = True

    off_area_model = AreaModel(off)
    on_area_model = AreaModel(on)
    off_area = off_area_model.estimate(off, engine_name)
    on_area = on_area_model.estimate(on, engine_name)
    off_power = PowerModel(off).estimate(off_area_model, off, engine_name)
    on_power = PowerModel(on).estimate(on_area_model, on, engine_name)

    assert off_area["weight_cache_area_mm2"] == 0
    assert on_area["weight_cache_area_mm2"] > 0
    assert on_area["weight_cache_pe_overhead_pct"] > 0
    assert on_area["total_mm2"] > off_area["total_mm2"]
    assert on_power > off_power


def test_weight_cache_pair_is_monotonic_for_decode_and_agent_ffn_shapes():
    shapes = ((1, 2048, 11008), (875, 2048, 11008))
    for engine_name in (
        "systolic", "block", "block_fused_attention", "gmma",
    ):
        engine = create_engine(_config(engine_name))
        for shape in shapes:
            single = engine.estimate(*shape)
            pair = engine.estimate_weight_cache_pair(*shape)
            assert pair.total_cycles <= 2 * single.total_cycles


def test_repeatable_engine_audit_passes_all_admission_checks():
    first = run_audit()
    second = run_audit()
    assert first == second
    assert first["engine_count"] == len(ENGINE_TYPES)
    assert first["passed_engines"] == len(ENGINE_TYPES)
    assert first["all_passed"]
