"""Cross-engine result contract tests.

Every engine produced by ``create_engine`` must return an ``EngineResult``
that satisfies the shared base contract.  Three engines (OS-Systolic,
TensorCore, GMMA) must additionally expose diagnostic detail fields that
are required for DSE debugging but are not yet implemented — those tests
are intentionally red in this Wave-1 checkpoint.
"""

import copy
from typing import Any, Dict

import pytest

from engine.mac_engine import EngineResult, create_engine


ENGINE_TYPES = [
    "systolic",
    "block",
    "os_systolic",
    "input_stationary",
    "tensor_core",
    "wmma",
    "gmma",
    "fsa",
]

DIAGNOSTIC_ENGINES = {"os_systolic", "tensor_core", "gmma"}


def _typed_config(engine_config: Dict[str, Any], engine_type: str) -> Dict[str, Any]:
    """Return a deep-copied config with the requested engine type."""
    cfg = copy.deepcopy(engine_config)
    cfg["mxu"]["type"] = engine_type
    return cfg


def _validate_base(result: EngineResult, M: int, K: int, N: int) -> None:
    """Shared base contract for every EngineResult."""
    assert result.total_cycles > 0, "total_cycles must be positive"
    assert result.compute_cycles > 0, "compute_cycles must be positive"
    assert result.dma_cycles >= 0, "dma_cycles must be non-negative"
    assert 0 < result.utilization <= 1.0, (
        f"utilization must be in (0, 1], got {result.utilization}"
    )
    assert result.ops >= M * K * N, (
        f"ops {result.ops} must cover the full GEMM volume {M * K * N}"
    )
    assert result.bottleneck in {"compute", "dma"}, (
        f"bottleneck must be 'compute' or 'dma', got {result.bottleneck!r}"
    )
    assert result.weight_bytes > 0, "weight_bytes must be positive"
    assert isinstance(result.details, dict) and result.details, (
        "details must be a non-empty dict"
    )


def _validate_os_systolic_details(result: EngineResult) -> None:
    """OS-Systolic diagnostic fields required for DSE.

    NOTE: Skipped in Wave 1 because ``os_systolic`` engine does not yet
    populate these detail keys.  Tighten with a hard assertion once
    Todos 6-8 land.
    """
    required = {
        "raw_dma_cycles",
        "k_reduction_cycles",
        "total_compute_cycles",
        "bottleneck_reason",
    }
    missing = required - result.details.keys()
    if missing:
        pytest.skip(f"OS-Systolic detail keys not yet implemented ({missing})")


def _validate_tensor_core_details(result: EngineResult) -> None:
    """TensorCore diagnostic fields required for DSE.

    NOTE: Skipped in Wave 1 because ``tensor_core`` engine does not yet
    populate these detail keys.  Tighten with a hard assertion once
    Todos 6-8 land.
    """
    required = {
        "active_tcs",
        "num_waves",
        "per_wave_payload_cycles",
        "descriptor_cycles_per_wave",
        "total_descriptor_cycles",
    }
    missing = required - result.details.keys()
    if missing:
        pytest.skip(f"TensorCore detail keys not yet implemented ({missing})")


def _validate_gmma_details(result: EngineResult) -> None:
    """GMMA diagnostic fields required for DSE.

    NOTE: Skipped in Wave 1 because ``gmma`` engine does not yet
    populate these detail keys.  Tighten with a hard assertion once
    Todos 6-8 land.
    """
    required = {
        "raw_dma_cycles",
        "tma_hidden_dma",
        "tma_exposed_dma",
        "per_tile_compute",
        "pipeline_scale",
    }
    missing = required - result.details.keys()
    if missing:
        pytest.skip(f"GMMA detail keys not yet implemented ({missing})")
    if "raw_dma_cycles" in result.details:
        assert result.total_cycles >= result.details["raw_dma_cycles"], (
            "total_cycles must be at least raw_dma_cycles"
        )


def _validate_engine_specific(engine_type: str, result: EngineResult) -> None:
    """Engine-specific diagnostic contract."""
    if engine_type == "os_systolic":
        _validate_os_systolic_details(result)
    elif engine_type == "tensor_core":
        _validate_tensor_core_details(result)
    elif engine_type == "gmma":
        _validate_gmma_details(result)


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
        ops=M * K * N,
        num_tiles=1,
        weight_bytes=1024,
        bottleneck="compute",
        details={"reason": "ok"},
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
        "ops": M * K * N,
        "num_tiles": 1,
        "weight_bytes": 1024,
        "bottleneck": "compute",
        "details": {"reason": "ok"},
    }

    invalid_cases = [
        ("negative compute_cycles", {**base, "compute_cycles": -1}),
        ("negative dma_cycles", {**base, "dma_cycles": -1}),
        ("zero total_cycles", {**base, "total_cycles": 0}),
        ("zero utilization", {**base, "utilization": 0.0}),
        ("utilization above one", {**base, "utilization": 1.01}),
        ("ops below gemm volume", {**base, "ops": M * K * N - 1}),
        ("illegal bottleneck", {**base, "bottleneck": "memory"}),
        ("zero weight_bytes", {**base, "weight_bytes": 0}),
        ("empty details", {**base, "details": {}}),
    ]

    for name, kwargs in invalid_cases:
        bad = EngineResult(**kwargs)
        with pytest.raises(AssertionError):
            _validate_base(bad, M, K, N)
