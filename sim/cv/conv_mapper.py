"""Convolution-to-GEMM mapper with im2col transformation.

Maps convolution layers to GEMM dimensions suitable for systolic array
execution. Supports pointwise (1x1) and depthwise (groups=C_in) convolutions
plus channel-tiling analysis for depthwise utilization improvement.

Formulas follow the Caduceus NPU architecture:
  - BW: 51.2 GB/s (LPDDR5-6400 @ 1 GHz => 51.2 bytes/cycle)
  - DRAM efficiency: 0.85
  - Systolic array: 128 x 128 (default)
"""

import math

# ---------------------------------------------------------------------------
# Constants (aligned with sim/config/npu_config.yaml)
# ---------------------------------------------------------------------------
BW_BYTES_PER_CYCLE = 51.2   # bytes/cycle @ 1 GHz
DRAM_EFFICIENCY = 0.85       # effective BW fraction
EFF_BW = BW_BYTES_PER_CYCLE * DRAM_EFFICIENCY  # ~43.52 bytes/cycle
BYTES_PER_ELEMENT = 4        # FP32 / int32 element size


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _compute_output_size(H: int, K: int, stride: int, pad: int) -> int:
    """Compute spatial output dimension after convolution."""
    return (H + 2 * pad - K) // stride + 1


# ---------------------------------------------------------------------------
# Primary mapping
# ---------------------------------------------------------------------------

def map_conv_to_gemm(
    C_in: int,
    C_out: int,
    H: int,
    W: int,
    K: int,
    stride: int = 1,
    pad: int = 0,
    groups: int = 1,
) -> dict:
    """Map a convolution layer to GEMM dimensions via im2col.

    Args:
        C_in:   Number of input channels.
        C_out:  Number of output channels.
        H:      Input spatial height.
        W:      Input spatial width.
        K:      Kernel spatial size (assumed square).
        stride: Convolution stride.
        pad:    Spatial padding (applied equally to all sides).
        groups: Number of channel groups.

    Returns:
        dict with keys:
          M, K, N                  -- GEMM dimensions
          H_out, W_out             -- output spatial shape
          im2col_overhead_cycles   -- estimated DMA cycles for im2col data
          is_depthwise             -- True if groups == C_in and groups > 1
    """
    H_out = _compute_output_size(H, K, stride, pad)
    W_out = _compute_output_size(W, K, stride, pad)

    # --- Classify convolution type ----------------------------------------
    is_pointwise = K == 1 and groups == 1
    is_depthwise = groups > 1 and groups == C_in

    if is_pointwise:
        # (1x1 conv)  =>  no spatial tiling needed
        M = H_out * W_out
        K_dim = C_in
        N = C_out

    elif is_depthwise:
        # Spatial positions x input channels  =>  one row per (channel, pixel)
        M = H_out * W_out * C_in
        K_dim = K * K
        N = 1

    else:
        # General grouped or regular convolution
        M = H_out * W_out
        K_dim = (C_in * K * K) // groups
        N = C_out // groups

    # --- im2col DMA overhead estimate -------------------------------------
    # Data to move: M rows x K_dim columns x 4 bytes
    # Effective bandwidth: 51.2 * 0.85 bytes/cycle
    im2col_overhead_cycles = M * K_dim * BYTES_PER_ELEMENT / EFF_BW

    return {
        "M": M,
        "K": K_dim,
        "N": N,
        "H_out": H_out,
        "W_out": W_out,
        "im2col_overhead_cycles": im2col_overhead_cycles,
        "is_depthwise": is_depthwise,
    }


# ---------------------------------------------------------------------------
# Depthwise channel-tiling analysis
# ---------------------------------------------------------------------------

def map_depthwise_with_tiling(
    C_in: int,
    H: int,
    W: int,
    K: int,
    array_W: int = 128,
) -> dict:
    """Analyze depthwise convolution under channel tiling.

    Depthwise convolutions have N=1 per group, underutilising wide systolic
    arrays.  Channel tiling groups multiple input channels together so that
    the array's N dimension is better utilised.

    Args:
        C_in:    Number of input channels (equals output channels for DW).
        H:       Input spatial height.
        W:       Input spatial width.
        K:       Kernel spatial size.
        array_W: Systolic array width  (height assumed = width = 128).

    Returns:
        dict mapping N_tile -> {
            mxu_util_pct:   array utilisation percentage
            compute_cycles: estimated MXU compute cycles
            dma_cycles:     estimated im2col DMA cycles for all tiles
            total_cycles:   compute_cycles + dma_cycles
        }
    """
    base = map_conv_to_gemm(C_in, C_in, H, W, K, stride=1, pad=0, groups=C_in)
    M_total = base["M"]          # H_out * W_out * C_in
    K_dim = base["K"]            # K * K
    H_out = base["H_out"]
    W_out = base["W_out"]

    array_H = array_W   # square systolic array (128 x 128)

    results = {}

    for N_tile in (1, 4, 8, 16):
        # --- MXU utilisation ----------------------------------------------
        mxu_util_pct = min(N_tile / array_W, 1.0) * 100.0

        # --- Compute cycles -----------------------------------------------
        # Total MAC operations are spread across the array.  Estimate cycles
        # as the number of array "fills" in the M, K, and N dimensions.
        #   ceil(M / (array_H * array_W))  -- how many passes over M
        #   ceil(K / array_H)              -- weight-tile trips
        #   ceil(N_tile / array_W)         -- output-channel tile trips
        compute_cycles = (
            math.ceil(M_total / (array_H * array_W))
            * math.ceil(K_dim / array_H)
            * math.ceil(N_tile / array_W)
        )

        # --- DMA cycles ---------------------------------------------------
        # Each tile moves N_tile channels of im2col data:
        #   rows per tile = H_out * W_out * N_tile
        #   cols per tile = K_dim
        #   bytes = rows * cols * 4
        num_tiles = math.ceil(C_in / N_tile)
        rows_per_tile = H_out * W_out * N_tile
        dma_per_tile = rows_per_tile * K_dim * BYTES_PER_ELEMENT / EFF_BW
        dma_cycles_total = dma_per_tile * num_tiles

        results[N_tile] = {
            "mxu_util_pct": mxu_util_pct,
            "compute_cycles": compute_cycles,
            "dma_cycles": dma_cycles_total,
            "total_cycles": compute_cycles + dma_cycles_total,
        }

    return results
