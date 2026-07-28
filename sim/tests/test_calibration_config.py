"""Calibration config exposure tests.

Covers:
  - GMMA pipeline_scale read from config with class-constant fallback
  - TensorCore descriptor_overhead_cycles read from dma config
  - Validation and field-specific ValueError messages
"""

import pytest

from config.npu_config import load_config
from engine.mac_engine import create_engine


def _engine_config(engine_type: str) -> dict:
    """Return a minimal engine config with the requested engine type."""
    return {
        "mac_engine": {
            "type": engine_type,
            "array_height": 64,
            "array_width": 64,
            "frequency_mhz": 1000,
            "weight_precision_bits": 4,
            "activation_precision_bits": 8,
        },
        "memory": {
            "bandwidth_bytes_per_cycle": 51.2,
            "dram_efficiency": 0.85,
        },
        "dma": {
            "descriptor_overhead_cycles": 5,
        },
    }


# ── Default config contents ────────────────────────────────────────────────


def test_default_config_has_gmma_pipeline_scale():
    """npu_config.yaml exposes gmma.pipeline_scale = 0.05 at top level."""
    config = load_config()
    assert config.get("gmma", {}).get("pipeline_scale") == pytest.approx(0.05)


def test_default_config_has_dma_descriptor_overhead_cycles():
    """npu_config.yaml keeps dma.descriptor_overhead_cycles = 5."""
    config = load_config()
    assert config["dma"]["descriptor_overhead_cycles"] == 5


# ── GMMA pipeline_scale ────────────────────────────────────────────────────


def test_gmma_engine_uses_class_constant_fallback():
    """GMMAEngine falls back to GMMA_PIPELINE_SCALE when gmma block is absent."""
    engine = create_engine(_engine_config("gmma"))
    assert engine.pipeline_scale == pytest.approx(0.05)


def test_gmma_engine_uses_configured_pipeline_scale():
    """GMMAEngine reads pipeline_scale from config when present."""
    cfg = _engine_config("gmma")
    cfg["gmma"] = {"pipeline_scale": 0.1}
    engine = create_engine(cfg)
    assert engine.pipeline_scale == pytest.approx(0.1)


@pytest.mark.parametrize("bad_value", [0.0, -0.1, 1.01, "fast"])
def test_gmma_engine_rejects_invalid_pipeline_scale(bad_value):
    """pipeline_scale must satisfy 0 < scale <= 1."""
    cfg = _engine_config("gmma")
    cfg["gmma"] = {"pipeline_scale": bad_value}
    with pytest.raises(ValueError, match="gmma.pipeline_scale"):
        create_engine(cfg)


# ── TensorCore descriptor_overhead_cycles ──────────────────────────────────


def test_tensor_core_engine_uses_default_descriptor_overhead_cycles():
    """TensorCoreEngine defaults descriptor_overhead_cycles to 5."""
    engine = create_engine(_engine_config("tensor_core"))
    assert engine.descriptor_overhead_cycles == 5


def test_tensor_core_engine_uses_configured_descriptor_overhead_cycles():
    """TensorCoreEngine reads descriptor_overhead_cycles from dma config."""
    cfg = _engine_config("tensor_core")
    cfg["dma"]["descriptor_overhead_cycles"] = 10
    engine = create_engine(cfg)
    assert engine.descriptor_overhead_cycles == 10


@pytest.mark.parametrize("bad_value", [-1, 3.5, True])
def test_tensor_core_engine_rejects_invalid_descriptor_overhead_cycles(bad_value):
    """descriptor_overhead_cycles must be a non-negative integer."""
    cfg = _engine_config("tensor_core")
    cfg["dma"]["descriptor_overhead_cycles"] = bad_value
    with pytest.raises(ValueError, match="dma.descriptor_overhead_cycles"):
        create_engine(cfg)
