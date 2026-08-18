"""Prefill/TTFT modeling: simulate_prefill and ttft_ms_from_prefill."""

from pathlib import Path
from unittest.mock import patch

import design_space_explorer as dse
import pytest
from dse.legacy_adapter import generate_configs

SIM_DIR = Path(__file__).resolve().parents[1]


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
