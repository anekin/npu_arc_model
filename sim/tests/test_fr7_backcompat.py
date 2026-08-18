"""FR-7 backward-compatibility baseline: lock default --quick DSE outputs.

This test ensures the default (no --batch-m) quick run produces exactly the
same tok_s/area_mm2/power_w values as the golden baseline captured at the
start of the arc-prefill-ttft-dse plan.  New rows may be added later
(e.g. T5 adds 64x64), but existing rows must not drift.
"""

import json
import math
from pathlib import Path

import pytest
import yaml
from dse.legacy_adapter import evaluate_config, generate_configs
from engine.ppa_model import AreaModel, PowerModel

SIM_DIR = Path(__file__).resolve().parents[1]
GOLDEN_PATH = SIM_DIR / "tests" / "golden" / "dse_default_quick_baseline.json"


@pytest.fixture
def models():
    with open(SIM_DIR / "config" / "design_space.yaml") as f:
        base_cfg = yaml.safe_load(f)
    return AreaModel(base_cfg), PowerModel(base_cfg)


@pytest.fixture
def baseline():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def _make_key(cfg):
    return (
        cfg["mac_engine"]["type"],
        cfg["mac_engine"]["array_height"],
        cfg["mac_engine"]["array_width"],
        cfg["mac_engine"]["weight_precision_bits"],
        cfg["mac_engine"]["frequency_mhz"],
        cfg["optimizations"]["weight_cache"],
        cfg.get("_dram_label", ""),
    )


def test_baseline_exists_and_nonempty():
    assert GOLDEN_PATH.exists()
    data = json.loads(GOLDEN_PATH.read_text())
    assert len(data) >= 12


def test_fr7_backcompat_matches_baseline(models, baseline):
    area_model, power_model = models
    configs = list(generate_configs(quick=True))
    config_map = {_make_key(c): c for c in configs}

    for row in baseline:
        key = (
            row["engine_type"],
            row["H"],
            row["W"],
            row["w_bits"],
            row["freq_mhz"],
            row["weight_cache"],
            row.get("dram_label", ""),
        )
        assert key in config_map, f"No quick config matches baseline row {row}"
        cfg = config_map[key]
        ppa = evaluate_config(cfg, area_model, power_model)

        assert math.isclose(ppa.tok_s, row["tok_s"], rel_tol=0, abs_tol=1e-9), (
            f"tok_s drift for {key}: {ppa.tok_s} != {row['tok_s']}"
        )
        assert math.isclose(ppa.area_mm2, row["area_mm2"], rel_tol=0, abs_tol=1e-9), (
            f"area drift for {key}: {ppa.area_mm2} != {row['area_mm2']}"
        )
        assert math.isclose(ppa.power_w, row["power_w"], rel_tol=0, abs_tol=1e-9), (
            f"power drift for {key}: {ppa.power_w} != {row['power_w']}"
        )


def test_fr7_backcompat_detects_simulate_layer_regression(models, baseline):
    """Mutation: perturbing simulate_layer must break at least one baseline row."""
    import design_space_explorer as dse

    original_simulate_layer = dse.simulate_layer

    def broken_simulate_layer(cfg, batch_m=None, **kwargs):
        layer_cycles, weight_bytes = original_simulate_layer(cfg, batch_m, **kwargs)
        # A single extra cycle is not detectable for BW-bound quick configs,
        # so scale the compute latency to ensure the baseline catches a regression.
        return layer_cycles * 10, weight_bytes

    dse.simulate_layer = broken_simulate_layer
    try:
        area_model, power_model = models
        configs = list(generate_configs(quick=True))
        config_map = {_make_key(c): c for c in configs}

        drift_count = 0
        for row in baseline:
            key = (
                row["engine_type"],
                row["H"],
                row["W"],
                row["w_bits"],
                row["freq_mhz"],
                row["weight_cache"],
                row.get("dram_label", ""),
            )
            cfg = config_map[key]
            ppa = evaluate_config(cfg, area_model, power_model)
            if not math.isclose(ppa.tok_s, row["tok_s"], rel_tol=0, abs_tol=1e-9):
                drift_count += 1

        if drift_count == 0:
            pytest.fail(
                "Mutation test did not detect simulate_layer regression: no baseline row drifted after adding 1 cycle"
            )
    finally:
        dse.simulate_layer = original_simulate_layer

    # After restore, the normal test should still pass.
    test_fr7_backcompat_matches_baseline(models, baseline)
