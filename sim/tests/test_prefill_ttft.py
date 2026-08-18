"""Prefill/TTFT modeling: simulate_prefill and ttft_ms_from_prefill."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import design_space_explorer as dse
import pytest
import yaml
from dse.legacy_adapter import generate_configs
from engine.ppa_model import AreaModel, PowerModel

SIM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SIM_DIR.parent


def _run_dse(args, timeout=120):
    cmd = [sys.executable, str(SIM_DIR / "design_space_explorer.py"), "--quick", *args]
    env = {
        "PYTHONPATH": str(SIM_DIR),
        "PATH": str(Path(sys.executable).parent),
        **dict(__import__("os").environ.items()),
    }
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=timeout)
    return result


@pytest.fixture
def sample_config():
    for cfg in generate_configs(quick=True):
        if (
            cfg["mac_engine"]["type"] == "block"
            and cfg["mac_engine"]["array_height"] == 128
            and cfg["mac_engine"]["array_width"] == 128
            and cfg.get("_dram_label") == "LPDDR5-128b"
        ):
            return cfg
    raise RuntimeError("no matching sample config")


@pytest.fixture
def models():
    with open(SIM_DIR / "config" / "design_space.yaml") as f:
        base_cfg = yaml.safe_load(f)
    return AreaModel(base_cfg), PowerModel(base_cfg)


def _block_64x64_config(wc=False):
    for cfg in generate_configs(quick=True):
        if (
            cfg["mac_engine"]["type"] == "block"
            and cfg["mac_engine"]["array_height"] == 64
            and cfg["mac_engine"]["array_width"] == 64
            and cfg["mac_engine"]["frequency_mhz"] == 1000
            and cfg.get("_dram_label") == "LPDDR5-64b"
            and cfg["optimizations"]["weight_cache"] is wc
        ):
            return cfg
    raise RuntimeError("no block 64x64 config found")


def test_ttft_ms_formula():
    # 1M cycles/layer * 36 layers @ 1000 MHz -> 36 ms
    assert dse.ttft_ms_from_prefill(1_000_000, 36, 1000) == 36.0


def test_ttft_uses_units():
    calls = []

    def fake_cycles_to_us(cycles, freq_mhz):
        calls.append((cycles, freq_mhz))
        return cycles / freq_mhz

    with patch("contracts.units.cycles_to_microseconds", fake_cycles_to_us):
        result = dse.ttft_ms_from_prefill(1_000_000, 36, 1000)

    assert len(calls) == 1
    assert calls[0] == (36_000_000, 1000)
    assert result == 36.0


def test_prefill_linear_in_m(sample_config):
    """For compute-bound configs prefill cycles scale roughly linearly with batch_m."""
    small = dse.simulate_prefill(sample_config, 20)
    large = dse.simulate_prefill(sample_config, 200)
    ratio = large / small
    assert 8.0 <= ratio <= 12.0, f"expected ~10x linear scaling, got {ratio}"


def test_prefill_gt_decode(sample_config):
    decode_cycles, _ = dse.simulate_layer(sample_config)
    prefill_cycles = dse.simulate_prefill(sample_config, 2)
    assert prefill_cycles > decode_cycles


def test_kv_prefill_write_zero():
    spec = dse.get_spec(dse._DEFAULT_LLM_SPEC)
    cfg = list(generate_configs(quick=True))[0]

    decode_kv = dse._compute_kv_cycles(cfg, batch_m=1, kv_heads=spec.kv_heads, head_dim=spec.head_dim)
    prefill_kv = dse._compute_kv_cycles(cfg, batch_m=2, kv_heads=spec.kv_heads, head_dim=spec.head_dim)

    assert decode_kv > 0
    assert prefill_kv == 0


def test_prefill_uses_spec_kv_geometry(sample_config):
    cycles = dse.simulate_prefill(sample_config, 4, model_alias=dse._DEFAULT_LLM_SPEC)
    assert isinstance(cycles, int)
    assert cycles > 0

    # Doubling batch_m should increase cycles for a compute-bound config.
    cycles2 = dse.simulate_prefill(sample_config, 8, model_alias=dse._DEFAULT_LLM_SPEC)
    assert cycles2 > cycles


def test_batch_m_zero_rejected():
    result = _run_dse(["--batch-m", "0", "--output", "/dev/null"])
    assert result.returncode != 0
    assert "--batch-m must be >= 1" in result.stderr


def test_batch_m_non_integer_rejected():
    result = _run_dse(["--batch-m", "2.5", "--output", "/dev/null"])
    assert result.returncode != 0
    assert "invalid int value" in result.stderr


def test_batch_m_3_accepted(tmp_path):
    out = tmp_path / "m3.json"
    result = _run_dse(["--batch-m", "3", "--output", str(out)])
    assert result.returncode == 0, result.stderr
    with open(out) as f:
        data = json.load(f)
    assert data["batch_m"] == 3


def test_batch_m_128_same_tok_s_as_default(tmp_path):
    default_out = tmp_path / "default.json"
    batched_out = tmp_path / "m128.json"

    result_default = _run_dse(["--output", str(default_out)])
    assert result_default.returncode == 0, result_default.stderr
    result_batched = _run_dse(["--batch-m", "128", "--output", str(batched_out)])
    assert result_batched.returncode == 0, result_batched.stderr

    with open(default_out) as f:
        default = json.load(f)
    with open(batched_out) as f:
        batched = json.load(f)

    assert batched["batch_m"] == 128
    default_by_label = {r["label"]: r["tok_s"] for r in default["pareto_frontier"] + default["top_results"]}
    for r in batched["pareto_frontier"] + batched["top_results"]:
        assert r["label"] in default_by_label
        assert r["tok_s"] == default_by_label[r["label"]]


def test_ttft_ms_increases_with_batch_m(tmp_path):
    m1_out = tmp_path / "m1.json"
    m128_out = tmp_path / "m128.json"

    result_m1 = _run_dse(["--batch-m", "1", "--output", str(m1_out)])
    assert result_m1.returncode == 0, result_m1.stderr
    result_m128 = _run_dse(["--batch-m", "128", "--output", str(m128_out)])
    assert result_m128.returncode == 0, result_m128.stderr

    with open(m1_out) as f:
        m1 = json.load(f)
    with open(m128_out) as f:
        m128 = json.load(f)

    m1_by_label = {r["label"]: r["ttft_ms"] for r in m1["pareto_frontier"] + m1["top_results"]}
    for r in m128["pareto_frontier"] + m128["top_results"]:
        assert r["ttft_ms"] > 0
        assert r["ttft_ms"] > m1_by_label[r["label"]]


def test_ttft_monotonic_in_m(models):
    """Block 64x64 wc=True LPDDR5-64b 1GHz: TTFT strictly increases with batch_m."""
    area_model, power_model = models
    cfg = _block_64x64_config(wc=True)
    ttfts = []
    for m in (32, 64, 128):
        ppa = dse.evaluate_config(cfg, area_model, power_model, batch_m=m)
        ttfts.append(ppa.ttft_ms)
    assert ttfts[0] < ttfts[1] < ttfts[2], f"ttft_ms not monotonic: {ttfts}"


def test_ttft_bw_floor(models):
    """TTFT is above a simple weight-read bandwidth floor (wc=False, 10% tolerance)."""
    area_model, power_model = models
    cfg = _block_64x64_config(wc=False)
    ppa = dse.evaluate_config(cfg, area_model, power_model, batch_m=128)

    spec = dse.get_spec(dse._DEFAULT_LLM_SPEC)
    weights_bytes_total = spec.layers * (spec.hidden * spec.intermediate * 3 + spec.intermediate * spec.hidden) * 0.5
    bandwidth_gbps = cfg["memory"]["bandwidth_gbps"]
    floor_ms = weights_bytes_total / (bandwidth_gbps * 1e9 / 8) * 1000
    margin = ppa.ttft_ms / floor_ms
    print(f"ttft_bw_floor margin = {margin:.2f} (ttft={ppa.ttft_ms:.2f}ms, floor={floor_ms:.2f}ms)")
    assert ppa.ttft_ms >= 0.9 * floor_ms


def test_ttft_linearity_2000_128(models):
    """TTFT scales roughly linearly between M=128 and M=2000."""
    area_model, power_model = models
    cfg = _block_64x64_config(wc=True)
    ttft_128 = dse.evaluate_config(cfg, area_model, power_model, batch_m=128).ttft_ms
    ttft_2000 = dse.evaluate_config(cfg, area_model, power_model, batch_m=2000).ttft_ms
    ratio = ttft_2000 / ttft_128
    assert 14.0 <= ratio <= 17.5, f"ttft ratio {ratio} outside [14.0, 17.5]"


def test_all_quick_engines_prefill_finite(models):
    """Every quick engine configuration produces a finite positive TTFT at M=128."""
    area_model, power_model = models
    for cfg in generate_configs(quick=True):
        ppa = dse.evaluate_config(cfg, area_model, power_model, batch_m=128)
        assert ppa.ttft_ms > 0 and ppa.ttft_ms < float("inf")


def test_cv_ttft_zero(models):
    """CV mode evaluation produces ttft_ms == 0.0."""
    area_model, power_model = models
    cfg = next(iter(generate_configs(quick=True)))
    original_cv = dse._CV_MODEL
    dse._CV_MODEL = "mobilenetv3-small"
    try:
        with patch("cv.cv_sim.simulate_cv", return_value={"total_cycles": 1000, "sram_spill_mb": 0.0}):
            ppa = dse.evaluate_config(cfg, area_model, power_model)
    finally:
        dse._CV_MODEL = original_cv
    assert ppa.ttft_ms == 0.0


def test_prefill_output_deterministic(tmp_path):
    """Two identical CLI runs produce bit-identical v2 JSON outputs."""
    out1 = tmp_path / "run1.json"
    out2 = tmp_path / "run2.json"
    result1 = _run_dse(["--batch-m", "128", "--result-schema", "v2", "--output", str(out1)])
    assert result1.returncode == 0, result1.stderr
    result2 = _run_dse(["--batch-m", "128", "--result-schema", "v2", "--output", str(out2)])
    assert result2.returncode == 0, result2.stderr
    assert out1.read_bytes() == out2.read_bytes()


def test_evaluate_config_ttft_uses_spec_layers():
    """TTFT must be computed with the model-spec layer count, not the hard-coded 28."""
    with open(SIM_DIR / "config" / "design_space.yaml") as f:
        base_cfg = yaml.safe_load(f)
    area_model = AreaModel(base_cfg)
    power_model = PowerModel(base_cfg)

    cfg = next(
        c
        for c in generate_configs(quick=True)
        if c["mac_engine"]["type"] == "block"
        and c["mac_engine"]["array_height"] == 128
        and c["mac_engine"]["array_width"] == 128
    )

    original_alias = dse._MODEL_ALIAS
    dse._MODEL_ALIAS = "qwen2.5-3b"
    try:
        ppa = dse.evaluate_config(cfg, area_model, power_model, batch_m=128)
    finally:
        dse._MODEL_ALIAS = original_alias

    spec = dse.get_spec("qwen2.5-3b")
    prefill = dse.simulate_prefill(cfg, 128, "qwen2.5-3b")
    expected = dse.ttft_ms_from_prefill(prefill, spec.layers, cfg["mac_engine"]["frequency_mhz"])
    assert ppa.ttft_ms > 0
    assert round(ppa.ttft_ms, 2) == round(expected, 2)
