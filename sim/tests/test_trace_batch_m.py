"""Trace batch_m parameterization and KV global side-effect removal."""

import math
from pathlib import Path

import design_space_explorer as dse
import pytest
import yaml
from dse.legacy_adapter import evaluate_config, generate_configs
from engine.ppa_model import AreaModel, PowerModel

SIM_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def models():
    with open(SIM_DIR / "config" / "design_space.yaml") as f:
        base_cfg = yaml.safe_load(f)
    return AreaModel(base_cfg), PowerModel(base_cfg)


@pytest.fixture
def sample_config():
    return list(generate_configs(quick=True))[0]


def test_module_has_no_kv_globals():
    assert not hasattr(dse, "_LLM_TRACE")
    assert not hasattr(dse, "_KV_HEADS")
    assert not hasattr(dse, "_HEAD_DIM")


def test_generate_trace_respects_batch_m():
    trace_decode = dse.generate_trace_from_spec("qwen2.5-3b", batch_m=1)
    trace_prefill = dse.generate_trace_from_spec("qwen2.5-3b", batch_m=2)

    # All attention projection rows should scale with batch_m.
    for name in ("Q_proj", "K_proj", "V_proj", "O_proj"):
        row_d = next(r for r in trace_decode if r[4] == name)
        row_p = next(r for r in trace_prefill if r[4] == name)
        assert row_p[0] == 2, f"{name} should have M=2 in prefill trace"
        assert row_d[0] == 1, f"{name} should have M=1 in decode trace"

    # FFN rows stay at M=1 for decode, M=batch_m for prefill only when >1.
    for name in ("FFN_gate", "FFN_up", "FFN_down"):
        row_d = next(r for r in trace_decode if r[4] == name)
        row_p = next(r for r in trace_prefill if r[4] == name)
        assert row_d[0] == 1, f"{name} should have M=1 in decode trace"
        assert row_p[0] == 2, f"{name} should have M=2 in prefill trace"


def test_simulate_layer_prefill_exceeds_decode_for_compute_bound_configs():
    # For compute-bound configs the prefill compute dominates the saved KV-read cost.
    found = False
    for cfg in generate_configs(quick=True):
        if cfg["mac_engine"]["type"] != "block":
            continue
        if cfg.get("_dram_label") != "LPDDR5-128b":
            continue
        decode_cycles, _ = dse.simulate_layer(cfg, batch_m=1)
        prefill_cycles, _ = dse.simulate_layer(cfg, batch_m=2)
        if prefill_cycles > decode_cycles:
            found = True
            break
    assert found, "expected at least one compute-bound config where prefill cycles exceed decode cycles"


def test_simulate_layer_default_matches_decode_and_baseline(models, sample_config):
    """Default simulate_layer call is decode and matches the FR-7 baseline."""
    default_cycles, default_weight = dse.simulate_layer(sample_config)
    decode_cycles, decode_weight = dse.simulate_layer(sample_config, batch_m=1)
    assert default_cycles == decode_cycles
    assert default_weight == decode_weight

    # Cross-check via evaluate_config (which uses the default decode path).
    area_model, power_model = models
    ppa = evaluate_config(sample_config, area_model, power_model)
    # Reconstruct tok_s from default cycles to confirm consistency.
    expected_tok_s = dse.tok_s_from_layer(default_cycles, dse._NUM_LAYERS, sample_config["mac_engine"]["frequency_mhz"])
    assert math.isclose(ppa.tok_s, expected_tok_s, rel_tol=0, abs_tol=1e-9)


def test_simulate_layer_accepts_kv_geometry(sample_config):
    spec = dse.get_spec(dse._DEFAULT_LLM_SPEC)
    cycles_default, _ = dse.simulate_layer(sample_config)
    cycles_explicit, _ = dse.simulate_layer(
        sample_config,
        kv_heads=spec.kv_heads,
        head_dim=spec.head_dim,
    )
    assert cycles_default == cycles_explicit
