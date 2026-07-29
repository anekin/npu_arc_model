"""Independent closed-form physical oracle.

Computes conservation-law lower bounds for MAC count, byte transfers,
and cycle floors using first-principles formulas ONLY.

Does NOT import from ``sim.engine.*`` — this is the key independence guarantee.
Any engine whose result violates these bounds has a formula bug,
not a calibration issue.

Bounds enforced:
  - mac_count = M × K × N  (op_count = 2 × mac_count)
  - total_cycles >= ceil(mac_count / peak_macs_per_cycle)   [compute floor]
  - total_cycles >= ceil(raw_transfer_bytes / eff_bytes_per_cycle)  [DMA floor]
  - 0 < utilization <= 1
  - M increase → total_cycles must not decrease
  - bandwidth ↑ → total_cycles monotonically non-increase, saturating at compute floor
"""

import math
from typing import Dict, Set, Tuple


# ── Atomic conservation laws ──────────────────────────────────────────────


def mac_count(M: int, K: int, N: int) -> int:
    """M × K × N multiply-accumulates for a single GEMM."""
    return M * K * N


def op_count(M: int, K: int, N: int) -> int:
    """Total operations: multiply + accumulate = 2 × mac_count."""
    return 2 * mac_count(M, K, N)


def peak_macs_per_cycle(array_height: int, array_width: int,
                        ops_per_mac: int = 2) -> float:
    """Theoretical peak MACs per cycle from array dimensions."""
    return array_height * array_width * ops_per_mac


def bytes_per_cycle(bandwidth_gbps: float, frequency_mhz: int) -> float:
    """Convert GB/s to bytes/cycle at the given clock frequency.

    Derivation:
      bandwidth_bytes_per_second = bandwidth_gbps × 1e9
      cycles_per_second = frequency_mhz × 1e6
      bytes_per_cycle = bandwidth_bytes_per_second / cycles_per_second
                      = bandwidth_gbps × 1e9 / (frequency_mhz × 1e6)
                      = bandwidth_gbps × 1000 / frequency_mhz
    """
    if frequency_mhz <= 0:
        raise ValueError(f"frequency_mhz must be positive, got {frequency_mhz}")
    return bandwidth_gbps * 1000.0 / frequency_mhz


# ── Byte-transfer accounting ──────────────────────────────────────────────


def weight_bytes(K: int, N: int, weight_precision_bits: int) -> int:
    """K × N weight matrix bytes (INT4 → 4 bits/element)."""
    return K * N * weight_precision_bits // 8


def activation_bytes(M: int, K: int,
                     activation_precision_bits: int) -> int:
    """M × K activation matrix bytes."""
    return M * K * activation_precision_bits // 8


def raw_transfer_bytes(M: int, K: int, N: int,
                       weight_precision_bits: int,
                       activation_precision_bits: int) -> int:
    """Total raw DMA bytes: weights + activations (no caching assumed)."""
    return (weight_bytes(K, N, weight_precision_bits) +
            activation_bytes(M, K, activation_precision_bits))


def weight_cache_pair_bytes(M: int, K: int, N: int,
                            weight_precision_bits: int,
                            activation_precision_bits: int) -> int:
    """Gate+up pair: 2 × weight bytes + 1 × activation bytes."""
    return (2 * weight_bytes(K, N, weight_precision_bits) +
            activation_bytes(M, K, activation_precision_bits))


# ── Lower bounds (ceil) ───────────────────────────────────────────────────


def compute_lower_bound(macs: int, peak_macs: float) -> int:
    """Cycles cannot be less than ceil(mac_count / peak_macs_per_cycle)."""
    if peak_macs <= 0:
        raise ValueError(f"peak_macs must be positive, got {peak_macs}")
    return math.ceil(macs / peak_macs)


def dma_lower_bound(total_bytes: int, eff_bytes_per_cycle: float) -> int:
    """Cycles cannot be less than ceil(bytes / effective_bytes_per_cycle)."""
    if eff_bytes_per_cycle <= 0:
        raise ValueError(
            f"eff_bytes_per_cycle must be positive, got {eff_bytes_per_cycle}"
        )
    if total_bytes < 0:
        raise ValueError(f"total_bytes must be non-negative, got {total_bytes}")
    if total_bytes == 0:
        return 0
    return math.ceil(total_bytes / eff_bytes_per_cycle)


def combined_lower_bound(macs: int, peak_macs: float,
                         total_bytes: int,
                         eff_bytes_per_cycle: float) -> int:
    """Max of compute floor and DMA floor."""
    return max(
        compute_lower_bound(macs, peak_macs),
        dma_lower_bound(total_bytes, eff_bytes_per_cycle),
    )


# ── Utilization bounds ────────────────────────────────────────────────────


def utilization_lower(macs: int, peak_macs: float,
                      total_cycles: int) -> float:
    """Ideal utilization = mac_count / (peak_macs × total_cycles)."""
    if total_cycles <= 0 or peak_macs <= 0:
        return 0.0
    return macs / (peak_macs * total_cycles)


def validate_utilization(macs: int, peak_macs: float,
                         total_cycles: int) -> Tuple[bool, str]:
    """Return (is_valid, message) for utilization in (0, 1]."""
    if total_cycles <= 0:
        return False, "total_cycles must be positive for utilization"
    util = macs / (peak_macs * total_cycles)
    if util <= 0:
        return False, f"utilization {util:.6f} is not > 0"
    if util > 1.0:
        return False, f"utilization {util:.6f} exceeds 1.0"
    return True, f"utilization {util:.4f} in (0, 1]"


# ── Monotonicity helpers ──────────────────────────────────────────────────


def validate_m_monotonic(results: Dict[int, int]) -> Tuple[bool, str]:
    """M increase must not decrease total_cycles: tot_cycles non-decreasing."""
    sorted_ms = sorted(results.keys())
    for i in range(len(sorted_ms) - 1):
        m_lo, m_hi = sorted_ms[i], sorted_ms[i + 1]
        if results[m_hi] < results[m_lo]:
            return False, (
                f"M={m_lo}→{m_hi}: total_cycles decreased "
                f"({results[m_lo]} → {results[m_hi]})"
            )
    return True, "M-monotonic (non-decreasing)"


def validate_bandwidth_monotonic(
    bw_results: Dict[float, int],
    compute_floor: int,
    tolerance_pct: float = 5.0,
) -> Tuple[bool, str]:
    """Bandwidth increase must not increase total_cycles.

    As bandwidth grows, DMA shrinks. Total cycles should approach
    the compute floor and then saturate there (within tolerance).
    """
    sorted_bws = sorted(bw_results.keys())
    for i in range(len(sorted_bws) - 1):
        bw_lo, bw_hi = sorted_bws[i], sorted_bws[i + 1]
        cycles_lo, cycles_hi = bw_results[bw_lo], bw_results[bw_hi]
        if cycles_lo == 0:
            continue
        if cycles_hi > cycles_lo * 1.01:  # >1% increase is wrong
            return False, (
                f"BW {bw_lo}→{bw_hi} GB/s: total_cycles increased "
                f"({cycles_lo} → {cycles_hi}) — violates monotonicity"
            )
    # Check saturation near compute floor
    highest_bw = sorted_bws[-1]
    if bw_results[highest_bw] > 0 and compute_floor > 0:
        ratio = bw_results[highest_bw] / compute_floor
        if ratio > 1.0 + tolerance_pct / 100.0:
            return False, (
                f"At {highest_bw} GB/s, total_cycles={bw_results[highest_bw]} "
                f"is {ratio - 1:.1%} above compute floor {compute_floor} — "
                f"should saturate within {tolerance_pct}%"
            )
    return True, (
        f"BW-monotonic, highest-BW {bw_results[highest_bw]:,}c "
        f"vs floor {compute_floor:,}c"
    )


# ── Required diagnostics ──────────────────────────────────────────────────


_REQUIRED_DIAGNOSTICS: Dict[str, Set[str]] = {
    "systolic": {"K_tiles", "N_tiles", "per_tile_compute",
                 "per_tile_dma", "pipeline_fill", "pipeline_drain"},
    "block": {"M_tiles", "K_tiles", "N_tiles", "per_tile_compute",
              "weight_dram_eff", "token_multiplex"},
    "os_systolic": {"K_tiles", "N_tiles", "per_tile_compute",
                    "broadcast_sync", "k_reduction_cycles",
                    "raw_dma_cycles", "total_compute_cycles",
                    "bottleneck_reason", "dataflow"},
    "input_stationary": {"per_tile_compute", "per_tile_dma",
                         "K_tiles"},
    "tensor_core": {"sub_K", "sub_M", "sub_N",
                    "total_invocations", "num_tcs", "waves",
                    "per_wave_payload_cycles", "per_wave_dma",
                    "per_wave_compute", "descriptor_cycles_per_wave",
                    "total_descriptor_cycles", "subtile_size"},
    "wmma": {"total_fragments", "fragments_per_tile",
             "per_fragment_compute", "per_fragment_dma"},
    "gmma": {"K_tiles", "N_tiles", "per_tile_compute",
             "per_tile_dma", "raw_dma_cycles",
             "tma_hidden_dma", "tma_exposed_dma",
             "pipeline_scale", "tma_overlap", "weight_dram_eff"},
    "fsa": {"tiles_k", "tiles_m", "tiles_n",
            "pipe_depth", "engine", "inline_softmax", "dram_eff"},
}


def required_diagnostics(engine_type: str) -> Set[str]:
    """Return the set of required detail keys for the given engine type.

    Raises ValueError for unknown engine types.
    """
    keys = _REQUIRED_DIAGNOSTICS.get(engine_type)
    if keys is None:
        raise ValueError(f"Unknown engine type: {engine_type!r}")
    return keys


def validate_diagnostics(engine_type: str,
                         details: dict) -> Tuple[bool, str]:
    """Return (is_valid, message) for diagnostics completeness check."""
    required = required_diagnostics(engine_type)
    missing = required - set(details.keys())
    if missing:
        return False, f"Missing diagnostics for {engine_type}: {missing}"
    return True, f"All {len(required)} diagnostics present for {engine_type}"


# ── Wall-time conversion (for cross-frequency validation) ─────────────────


def cycles_to_seconds(cycles: int, frequency_mhz: int) -> float:
    """Convert cycles to seconds: cycles / (freq_mhz × 1e6)."""
    if frequency_mhz <= 0:
        raise ValueError(f"frequency_mhz must be positive, got {frequency_mhz}")
    return cycles / (frequency_mhz * 1e6)


def cycles_to_us(cycles: int, frequency_mhz: int) -> float:
    """Convert cycles to microseconds."""
    return cycles_to_seconds(cycles, frequency_mhz) * 1e6
