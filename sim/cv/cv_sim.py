"""CV trace simulator — wraps MACEngine.estimate() for per-layer and aggregate metrics.

Feeds each trace entry through the appropriate execution model:
  - Conv layers         -> MACEngine.estimate(M, K, N)
  - SFU layers          -> sfu_cycles from trace entry
  - Element-wise        -> 1 cycle per ~128 elements (vector width)
  - Metadata ops        -> 0 cycles (fused/in-place)
  - GEMM                -> MACEngine.estimate(M, K, N)

Aggregates total cycles, MACs, DMA cycles, SRAM spill, and per-layer breakdown.
"""

import math
from typing import Any, Dict, List

from engine.mac_engine import create_engine


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default vector/SIMD width (elements/cycle) - overridden by config if present
_DEFAULT_VECTOR_WIDTH = 128

# Layer types that produce no intermediate tensor worth spilling
_METADATA_TYPES = frozenset({"reshape", "shape", "concat", "reduce_mean"})

# Layer types that execute on the SFU (no MXU)
_SFU_TYPES = frozenset({"hard_swish", "hard_sigmoid", "relu", "global_avg_pool"})

# Layer types that run on the vector/SIMD unit (element-wise)
_ELEMENTWISE_TYPES = frozenset({"add", "mul"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_sram_config(config: dict) -> tuple:
    """Extract SRAM sizes from config, returning (l1_kb, l2_kb)."""
    sram = config.get("sram", {})
    l1 = int(sram.get("l1_per_core_kb", 512))
    l2 = int(sram.get("l2_shared_kb", 2048))
    return l1, l2


def _get_bytes_per_element(config: dict) -> float:
    """Return element size in bytes from activation precision in config."""
    mac = config.get("mac_engine", config.get("mxu", {}))
    a_bits = int(mac.get("activation_precision_bits", 8))
    return a_bits / 8.0


def _get_vector_width(config: dict) -> int:
    """Return SIMD/vector width from config (elements/cycle)."""
    vec = config.get("vector", {})
    return int(vec.get("width", _DEFAULT_VECTOR_WIDTH))


def _estimate_activation_bytes(entry: dict, bytes_per_element: float) -> float:
    """Estimate intermediate tensor byte volume for a trace entry.

    Uses ``in_shape`` when present in the entry (for precise sizes);
    falls back to ``M x N`` as a conservative approximation.
    """
    if "in_shape" in entry:
        vol = 1
        for d in entry["in_shape"]:
            vol *= d
        return vol * bytes_per_element
    # Fallback: output activation volume ~ M x N elements
    m = entry.get("M", 0)
    n = entry.get("N", 0)
    return m * n * bytes_per_element


# ---------------------------------------------------------------------------
# Per-layer simulation dispatch
# ---------------------------------------------------------------------------

def _simulate_conv(entry: dict, engine) -> dict:
    """Simulate a convolution or GEMM layer through the MAC engine."""
    M = entry.get("M", 0)
    K = entry.get("K", 0)
    N = entry.get("N", 0)
    im2col = entry.get("im2col_overhead_cycles", 0.0)

    result = engine.estimate(M, K, N)
    macs = M * K * N

    compute_cycles = result.compute_cycles
    dma_cycles = result.dma_cycles + im2col
    cycles = result.total_cycles + im2col
    mxu_util = result.utilization * 100.0   # convert fraction -> percent

    return {
        "cycles": int(cycles),
        "compute_cycles": int(compute_cycles),
        "dma_cycles": dma_cycles,
        "macs": macs,
        "mxu_util_pct": mxu_util,
    }


def _simulate_sfu(entry: dict) -> dict:
    """Simulate an SFU-mapped layer using precomputed sfu_cycles."""
    sfu = entry.get("sfu_cycles", 0)
    return {
        "cycles": sfu,
        "compute_cycles": sfu,
        "dma_cycles": 0.0,
        "macs": 0,
        "mxu_util_pct": 0.0,
    }


def _simulate_elementwise(entry: dict, vector_width: int) -> dict:
    """Simulate an element-wise operation at 1 cycle / ~vector_width elements."""
    M = entry.get("M", 0)
    N = entry.get("N", 0)
    num_elements = M * N if N > 0 else M
    cycles = max(1, math.ceil(num_elements / vector_width))
    return {
        "cycles": cycles,
        "compute_cycles": cycles,
        "dma_cycles": 0.0,
        "macs": 0,
        "mxu_util_pct": 0.0,
    }


def _simulate_metadata(entry: dict) -> dict:
    """Metadata ops (reshape, shape, concat, reduce_mean) - fused, 0 cycles."""
    return {
        "cycles": 0,
        "compute_cycles": 0,
        "dma_cycles": 0.0,
        "macs": 0,
        "mxu_util_pct": 0.0,
    }


# ---------------------------------------------------------------------------
# Main simulation entry point
# ---------------------------------------------------------------------------

def simulate_cv(trace: list, config: dict) -> dict:
    """Simulate a CV trace through the NPU, returning cycle/MAC/layer breakdown.

    Args:
        trace: List of trace entries from ``cv_trace.generate_mobilenetv3_trace()``.
               Each entry follows the schema:
                 {type, name, M, K, N, im2col_overhead_cycles, sfu_cycles}
        config: ``npu_config.yaml`` loaded as a dict via ``yaml.safe_load``.

    Returns:
        {
            "total_cycles": int,
            "total_macs": int,
            "total_dma_cycles": float,
            "sram_spill_mb": float,
            "layers": [ {name, type, cycles, compute_cycles, dma_cycles, macs, mxu_util_pct}, ... ]
        }
    """
    engine = create_engine(config)
    bytes_per_element = _get_bytes_per_element(config)
    vector_width = _get_vector_width(config)
    l1_kb, l2_kb = _get_sram_config(config)
    total_sram_kb = l1_kb + l2_kb

    total_cycles = 0
    total_macs = 0
    total_dma_cycles = 0.0
    total_activation_bytes = 0.0
    layers: List[Dict[str, Any]] = []

    for entry in trace:
        entry_type = entry.get("type", "")
        name = entry.get("name", "")

        # --- Dispatch to layer-specific simulation -------------------------
        if entry_type in ("pointwise_conv", "depthwise_conv", "gemm"):
            result = _simulate_conv(entry, engine)
        elif entry_type in _SFU_TYPES:
            result = _simulate_sfu(entry)
        elif entry_type in _ELEMENTWISE_TYPES:
            result = _simulate_elementwise(entry, vector_width)
        else:
            # reshape, shape, concat, reduce_mean, or unknown
            result = _simulate_metadata(entry)

        # --- Accumulate activation volume for SRAM spill -------------------
        if entry_type not in _METADATA_TYPES:
            total_activation_bytes += _estimate_activation_bytes(
                entry, bytes_per_element
            )

        # --- Aggregate totals ----------------------------------------------
        cycles = result["cycles"]
        macs = result["macs"]
        dma_cycles = result["dma_cycles"]

        total_cycles += cycles
        total_macs += macs
        total_dma_cycles += dma_cycles

        layers.append({
            "name": name,
            "type": entry_type,
            "cycles": cycles,
            "compute_cycles": result["compute_cycles"],
            "dma_cycles": dma_cycles,
            "macs": macs,
            "mxu_util_pct": result["mxu_util_pct"],
        })

    # --- SRAM spill calculation -------------------------------------------
    total_activation_kb = total_activation_bytes / 1024.0
    sram_spill_kb = total_activation_kb - total_sram_kb
    sram_spill_mb = max(0.0, sram_spill_kb / 1024.0)

    return {
        "total_cycles": total_cycles,
        "total_macs": total_macs,
        "total_dma_cycles": total_dma_cycles,
        "sram_spill_mb": sram_spill_mb,
        "layers": layers,
    }
