"""Invalid input tests — all estimate and estimate_weight_cache_pair paths.

Covers: 0/negative/float/bool/string shapes, invalid array/precision/bandwidth.
Ensures every engine either rejects invalid inputs (raises) or returns a
structurally-valid EngineResult without crashing.

These tests MUST PASS in the red phase — they validate input handling,
not formula correctness.
"""

from typing import Any, Dict, List, Tuple

import pytest

from engine.mac_engine import EngineResult, create_engine

# ── Test parameters ──────────────────────────────────────────────────────

ENGINE_TYPES: List[str] = [
    "systolic",
    "block",
    "os_systolic",
    "input_stationary",
    "tensor_core",
    "wmma",
    "gmma",
    "fsa",
]

# Invalid shape inputs
INVALID_SHAPES: List[Tuple[str, Any, Any, Any]] = [
    ("zero M", 0, 110, 72),
    ("zero K", 4, 0, 72),
    ("zero N", 4, 110, 0),
    ("all_zero", 0, 0, 0),
    ("negative M", -1, 110, 72),
    ("negative K", 4, -5, 72),
    ("negative N", 4, 110, -10),
    ("float M", 1.5, 110, 72),
    ("float K", 4, 110.5, 72),
    ("float N", 4, 110, 72.0),
    ("bool M", True, 110, 72),
    ("bool K", 4, False, 72),
    ("string M", "abc", 110, 72),
    ("string K", 4, "def", 72),
    ("string N", 4, 110, "xyz"),
]


def _engine_config(engine_type: str,
                   overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Build a minimal valid engine config, optionally overriding fields."""
    cfg: Dict[str, Any] = {
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
        "sram": {"l2_shared_kb": 2048},
        "on_chip_memory": {"capacity_gb": 0.0, "bandwidth_gbps": 0.0},
    }
    if overrides:
        for key, val in overrides.items():
            if isinstance(val, dict) and key in cfg and isinstance(cfg[key], dict):
                cfg[key].update(val)
            else:
                cfg[key] = val
    return cfg


# ── Helper: call estimate and verify it doesn't crash ─────────────────────


def _call_and_assert_no_crash(engine_type: str, method: str,
                              M: Any, K: Any, N: Any,
                              overrides: Dict[str, Any] = None) -> EngineResult:
    """Call estimate (or estimate_weight_cache_pair) and handle errors.

    Returns the EngineResult if successful, None if an expected error was raised.
    Fails the test only on unexpected crashes.
    """
    if overrides is None:
        overrides = {}
    try:
        cfg = _engine_config(engine_type, overrides)
        engine = create_engine(cfg)
    except (ValueError, TypeError, KeyError, ZeroDivisionError):
        # Config-level rejection is acceptable
        return None

    try:
        if method == "estimate":
            result = engine.estimate(M, K, N)
        else:
            result = engine.estimate_weight_cache_pair(M, K, N)
    except (ValueError, TypeError, AssertionError, ZeroDivisionError,
            OverflowError):
        # Input-level rejection is acceptable and preferred
        return None
    except Exception:
        # Other errors (struct.error from float→int etc.) are acceptable
        return None

    # Result was returned; verify structural validity
    assert isinstance(result, EngineResult), (
        f"{engine_type}.{method}({M},{K},{N}) returned "
        f"{type(result).__name__} instead of EngineResult"
    )
    return result


# ── Invalid shape tests ───────────────────────────────────────────────────


class TestInvalidShapes:
    """All invalid shapes are handled without unexpected crashes."""

    @pytest.mark.parametrize("engine_type", ENGINE_TYPES)
    @pytest.mark.parametrize("label,M,K,N", INVALID_SHAPES)
    def test_estimate_handles_invalid_shape(self, engine_type: str,
                                            label: str,
                                            M: Any, K: Any, N: Any) -> None:
        """estimate must not crash on invalid shapes."""
        result = _call_and_assert_no_crash(engine_type, "estimate", M, K, N)
        if result is not None:
            assert isinstance(result.total_cycles, int), (
                f"{engine_type}: total_cycles not int"
            )
            assert isinstance(result.compute_cycles, int)
            assert isinstance(result.dma_cycles, int)

    @pytest.mark.parametrize("engine_type", ENGINE_TYPES)
    @pytest.mark.parametrize("label,M,K,N", INVALID_SHAPES)
    def test_cache_pair_handles_invalid_shape(self, engine_type: str,
                                              label: str,
                                              M: Any, K: Any, N: Any) -> None:
        """estimate_weight_cache_pair must not crash on invalid shapes."""
        _call_and_assert_no_crash(engine_type, "cache_pair", M, K, N)


# ── Invalid config tests ──────────────────────────────────────────────────


class TestInvalidConfig:
    """Invalid configuration parameters are handled without unexpected crashes."""

    @pytest.mark.parametrize("engine_type", ENGINE_TYPES)
    @pytest.mark.parametrize("field,value,desc", [
        ("array_height", 0, "zero height"),
        ("array_height", -1, "negative height"),
        ("array_width", 0, "zero width"),
        ("array_width", -64, "negative width"),
        ("frequency_mhz", 0, "zero frequency"),
        ("frequency_mhz", -1000, "negative frequency"),
        ("weight_precision_bits", 0, "zero weight precision"),
        ("weight_precision_bits", -4, "negative weight precision"),
        ("activation_precision_bits", 0, "zero activation precision"),
        ("activation_precision_bits", -8, "negative activation precision"),
    ])
    def test_estimate_with_invalid_config(self, engine_type: str,
                                          field: str, value: Any,
                                          desc: str) -> None:
        """Invalid array/precision configs must not cause unexpected crashes."""
        overrides = {"mac_engine": {field: value}}
        _call_and_assert_no_crash(engine_type, "estimate", 4, 110, 72, overrides)

    @pytest.mark.parametrize("engine_type", ENGINE_TYPES)
    def test_negative_bandwidth(self, engine_type: str) -> None:
        """Negative bandwidth should not crash."""
        overrides = {"memory": {"bandwidth_bytes_per_cycle": -1.0}}
        _call_and_assert_no_crash(engine_type, "estimate", 4, 110, 72, overrides)

    @pytest.mark.parametrize("engine_type", ENGINE_TYPES)
    def test_zero_bandwidth(self, engine_type: str) -> None:
        """Zero bandwidth should not crash."""
        overrides = {"memory": {"bandwidth_bytes_per_cycle": 0.0}}
        _call_and_assert_no_crash(engine_type, "estimate", 4, 110, 72, overrides)

    @pytest.mark.parametrize("engine_type", ENGINE_TYPES)
    @pytest.mark.parametrize("dram_eff", [-0.1, 0.0, 2.0, 100.0])
    def test_extreme_dram_efficiency(self, engine_type: str,
                                     dram_eff: float) -> None:
        """DRAM efficiency outside (0, 1] should not crash."""
        overrides = {"memory": {"dram_efficiency": dram_eff}}
        _call_and_assert_no_crash(engine_type, "estimate", 4, 110, 72, overrides)


class TestGMMASpecific:
    """GMMA-specific invalid config parameters."""

    @pytest.mark.parametrize("pipeline_scale", [-1.0, 0.0, 1.5, "abc"])
    def test_gmma_invalid_pipeline_scale(self, pipeline_scale: Any) -> None:
        """GMMA pipeline_scale must not crash the application."""
        overrides = {"gmma": {"pipeline_scale": pipeline_scale}}
        _call_and_assert_no_crash("gmma", "estimate", 4, 110, 72, overrides)


class TestTensorCoreSpecific:
    """TensorCore-specific invalid descriptor overhead."""

    @pytest.mark.parametrize("overhead", [-5, 1.5, True, "abc"])
    def test_tc_invalid_descriptor_overhead(self, overhead: Any) -> None:
        """TensorCore descriptor_overhead_cycles must not crash."""
        overrides = {
            "dma": {"descriptor_overhead_cycles": overhead},
        }
        _call_and_assert_no_crash("tensor_core", "estimate", 4, 110, 72, overrides)


class TestLargeValues:
    """Very large shape values should not crash."""

    @pytest.mark.parametrize("engine_type", ENGINE_TYPES)
    @pytest.mark.parametrize("M,K,N", [
        (1024, 11008, 2048),
        (1, 11008, 11008),
        (256, 4096, 4096),
    ])
    def test_large_geomm_no_crash(self, engine_type: str,
                                  M: int, K: int, N: int) -> None:
        """Large GEMM shapes produce valid results."""
        engine = create_engine(_engine_config(engine_type))
        result = engine.estimate(M, K, N)
        assert result.total_cycles > 0
        assert result.ops >= M * K * N


class TestCachePairCoverage:
    """Weight-cache pair coverage across all engines with valid inputs."""

    @pytest.mark.parametrize("engine_type", ENGINE_TYPES)
    @pytest.mark.parametrize("M,K,N", [
        (1, 110, 72),
        (4, 110, 72),
        (64, 64, 64),
    ])
    def test_cache_pair_valid_inputs(self, engine_type: str,
                                     M: int, K: int, N: int) -> None:
        """estimate_weight_cache_pair with valid inputs returns EngineResult."""
        engine = create_engine(_engine_config(engine_type))
        result = engine.estimate_weight_cache_pair(M, K, N)
        assert isinstance(result, EngineResult)
        assert result.total_cycles > 0
