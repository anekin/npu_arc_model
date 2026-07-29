"""Cross-engine result contract tests.

Every engine produced by ``create_engine`` must return an ``EngineResult``
that satisfies the shared base contract.  Diagnostics requirements are
derived from the unified engine registry oracle and enforced as hard
failures — no skip permitted.
"""

import copy
from typing import Any, Dict

import pytest

from engine.mac_engine import EngineResult, create_engine
from engine.registry import canonical_engine_ids, is_valid_engine
from tests.oracles.physics import required_diagnostics


ENGINE_TYPES = list(canonical_engine_ids())


def _typed_config(engine_config: Dict[str, Any], engine_type: str) -> Dict[str, Any]:
    """Return a deep-copied config with the requested engine type."""
    cfg = copy.deepcopy(engine_config)
    cfg["mxu"]["type"] = engine_type
    return cfg


def _validate_base(result: EngineResult, M: int, K: int, N: int) -> None:
    """Shared base contract for every EngineResult."""
    assert result.total_cycles > 0, "total_cycles must be positive"
    assert result.compute_cycles >= 0, "compute_cycles must be non-negative"
    assert result.dma_cycles >= 0, "dma_cycles must be non-negative"
    assert 0 < result.utilization <= 1.0, (
        f"utilization must be in (0, 1], got {result.utilization}"
    )
    assert result.mac_count >= M * K * N, (
        f"mac_count {result.mac_count} must cover the full GEMM volume {M * K * N}"
    )
    assert result.op_count == result.mac_count * 2, (
        f"op_count {result.op_count} must equal 2 × mac_count {result.mac_count}"
    )
    assert result.bottleneck in {"compute", "dma"}, (
        f"bottleneck must be 'compute' or 'dma', got {result.bottleneck!r}"
    )
    assert result.weight_bytes >= 0, "weight_bytes must be non-negative"
    assert isinstance(result.details, dict) and result.details, (
        "details must be a non-empty dict"
    )
    assert result.ideal_compute_cycles >= 0, "ideal_compute_cycles must be non-negative"
    assert result.raw_dma_cycles >= 0, "raw_dma_cycles must be non-negative"


def _validate_engine_specific(engine_type: str, result: EngineResult) -> None:
    """Validate that all required diagnostics are present (hard failure if missing)."""
    required = required_diagnostics(engine_type)
    is_cache_pair = bool(result.details.get("weight_cache"))
    if is_cache_pair:
        # Cache-pair paths don't include per_fragment_dma
        check_required = required - {"per_fragment_dma"}
    else:
        check_required = required

    missing = check_required - set(result.details.keys())
    assert not missing, (
        f"Missing required diagnostics for {engine_type}: {missing}"
    )


@pytest.mark.parametrize("engine_type", ENGINE_TYPES)
def test_engine_estimate_contract(engine_config, engine_type):
    """Every engine satisfies the base contract plus engine-specific diagnostics.

    Uses a representative decode GEMM (M=1, K=11008, N=2048) so that the
    weight tensor is larger than the SRAM weight buffer; this guarantees
    ``weight_bytes > 0`` for every engine, including FSA.
    """
    M, K, N = 1, 11008, 2048
    engine = create_engine(_typed_config(engine_config, engine_type))
    result = engine.estimate(M, K, N)

    _validate_base(result, M, K, N)
    _validate_engine_specific(engine_type, result)


def test_valid_engine_result_passes_base_validation():
    """A syntactically valid result passes the base validator."""
    M, K, N = 64, 64, 64
    result = EngineResult(
        compute_cycles=10,
        dma_cycles=5,
        total_cycles=15,
        utilization=0.5,
        mac_count=M * K * N,
        op_count=2 * M * K * N,
        num_tiles=1,
        weight_bytes=1024,
        bottleneck="compute",
        details={"reason": "ok"},
        ideal_compute_cycles=1,
        raw_dma_cycles=0,
    )
    _validate_base(result, M, K, N)


def test_invalid_engine_results_are_rejected():
    """Base validator rejects negative cycles, empty details, and illegal labels."""
    M, K, N = 64, 64, 64
    base = {
        "compute_cycles": 10,
        "dma_cycles": 5,
        "total_cycles": 15,
        "utilization": 0.5,
        "mac_count": M * K * N,
        "op_count": 2 * M * K * N,
        "num_tiles": 1,
        "weight_bytes": 1024,
        "bottleneck": "compute",
        "details": {"reason": "ok"},
        "ideal_compute_cycles": 1,
        "raw_dma_cycles": 0,
    }

    invalid_cases = [
        ("zero total_cycles", {**base, "total_cycles": 0}),
        ("zero utilization", {**base, "utilization": 0.0}),
        ("utilization above one", {**base, "utilization": 1.01}),
        ("mac_count below gemm volume", {**base, "mac_count": M * K * N - 1}),
        ("op_count mismatch", {**base, "op_count": 3 * M * K * N}),
        ("illegal bottleneck", {**base, "bottleneck": "memory"}),
        ("empty details", {**base, "details": {}}),
    ]

    for name, kwargs in invalid_cases:
        bad = EngineResult(**kwargs)
        with pytest.raises(AssertionError):
            _validate_base(bad, M, K, N)


def test_engine_result_rejects_non_finite_values():
    """EngineResult construction rejects NaN/Inf values."""
    M, K, N = 64, 64, 64
    import math
    with pytest.raises(ValueError):
        EngineResult(
            compute_cycles=float("nan"),
            dma_cycles=5,
            total_cycles=15,
            utilization=0.5,
            mac_count=M * K * N,
            op_count=2 * M * K * N,
            weight_bytes=1024,
            bottleneck="compute",
            details={"reason": "ok"},
        )


def test_engine_result_rejects_negative_mac_count():
    """EngineResult construction rejects non-positive mac_count."""
    with pytest.raises(ValueError):
        EngineResult(
            compute_cycles=10,
            dma_cycles=5,
            total_cycles=15,
            utilization=0.5,
            mac_count=0,
            op_count=0,
            weight_bytes=1024,
            bottleneck="compute",
            details={"reason": "ok"},
        )
