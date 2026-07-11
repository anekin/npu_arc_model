#!/usr/bin/env python3
"""
Golden Executor — bit-accurate NPU model for RTL verification.

Phase A: Precision-correct MXU (INT32 accumulate) + SFU (LUT/fixed-point).
Phase B: ISA instruction execution + SRAM memory model.
Phase C: Test vector generation + RTL comparison framework.

Hardware spec reference: NPU硬件详细架构设计v0.2
- MXU: 64×64 broadcast-based block, INT4 weights × INT8 activations → INT32 accumulate
- SFU: LUT-based (Softmax 256-entry, GELU 4-segment, RoPE CORDIC 12-stage)
- SRAM: 2MB Unified Buffer, 256KB×2 L1 per core
"""

import hashlib
import math
import struct
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models.dma import DMAModel

# ══════════════════════════════════════════════════════════════════════
# Constants from hardware spec
# ══════════════════════════════════════════════════════════════════════

ARRAY_H = 64           # broadcast-based block array height
ARRAY_W = 64           # broadcast-based block array width
SRAM_SIZE = 4 * 1024 * 1024  # 4 MB Unified Buffer (L2 configurable 2-8 MB)
L1_SRAM = 256 * 1024   # 256 KB L1 per core
INT32_MAX = 2**31 - 1
INT32_MIN = -2**31


# ══════════════════════════════════════════════════════════════════════
# Phase A-1: GoldenMXU — pure INT32 tile accumulation
# ══════════════════════════════════════════════════════════════════════

class GoldenMXU:
    """Bit-accurate MXU: INT4 weights × INT8 activations → INT32 accumulate.

    Matches hardware: 64×64 broadcast-based block array.
    Each PE: INT4×INT8→INT32 MAC per cycle.
    Broadcast-based: weights and activations broadcast to all PEs per tile,
    all PEs fire in parallel with zero pipeline fill/drain overhead.
    """

    def __init__(self, array_h: int = ARRAY_H, array_w: int = ARRAY_W):
        self.H = array_h
        self.W = array_w

    # ── INT4 packing/unpacking ──────────────────────────────────────

    @staticmethod
    def unpack_int4(packed: np.ndarray) -> np.ndarray:
        """Unpack INT4 weights from uint8 (2 per byte) → int8 values [-8, 7].

        Hardware: each byte stores weight[i] in low nibble, weight[i+1] in high nibble.
        Sign-extension: values 8-15 are negative (two's complement for 4-bit).
        """
        packed = np.asarray(packed, dtype=np.uint8)
        low = (packed & 0x0F).astype(np.int8)
        high = ((packed >> 4) & 0x0F).astype(np.int8)
        # Sign-extend 4-bit: values > 7 are negative
        low = np.where(low > 7, low - 16, low)
        high = np.where(high > 7, high - 16, high)
        # Interleave low/high: low[0], high[0], low[1], high[1], ...
        result = np.empty(packed.size * 2, dtype=np.int8)
        result[0::2] = low
        result[1::2] = high
        return result

    @staticmethod
    def pack_int4(values: np.ndarray) -> np.ndarray:
        """Pack INT4 values [-8, 7] into uint8 (2 per byte)."""
        values = np.asarray(values, dtype=np.int8).flatten()
        # Make even length
        if len(values) % 2 != 0:
            values = np.append(values, 0)
        # Convert negatives to unsigned 4-bit
        unsigned = np.where(values < 0, values + 16, values).astype(np.uint8)
        packed = np.empty(len(values) // 2, dtype=np.uint8)
        packed = (unsigned[0::2] & 0x0F) | ((unsigned[1::2] & 0x0F) << 4)
        return packed

    # ── Core matmul ─────────────────────────────────────────────────

    def matmul_int32(self, activation: np.ndarray,
                     weight_packed: np.ndarray,
                     M: int, K: int, N: int) -> np.ndarray:
        """INT4 weights × INT8 activations → INT32 accumulation.

        Bit-exact to hardware: per-tile integer dot product, no float32 intermediate.
        Weights are pre-unpacked INT4 (range [-8, 7]) in shape (K, N).
        Activations are INT8 (range [-128, 127]) in shape (M, K).

        Tiling matches hardware: 64×64 broadcast-based block.
        K dimension is the reduction axis — accumulates in INT32.
        """
        # Unpack weights
        w_flat = self.unpack_int4(weight_packed).astype(np.int32)
        expected_len = K * N
        if len(w_flat) < expected_len:
            w_flat = np.pad(w_flat, (0, expected_len - len(w_flat)), constant_values=0)
        W = w_flat[:expected_len].reshape(K, N)

        # Activations: INT8
        A = np.asarray(activation, dtype=np.int8)
        if A.size < M * K:
            A = np.pad(A.flatten(), (0, M * K - A.size), constant_values=0)
        A = A.flatten()[:M * K].reshape(M, K)

        result = np.zeros((M, N), dtype=np.int32)

        # Tile over M and N (matching hardware 64×64 block array boundaries)
        for m_start in range(0, M, self.H):
            m_end = min(m_start + self.H, M)
            for n_start in range(0, N, self.W):
                n_end = min(n_start + self.W, N)

                # Extract tile: (M_tile, K) × (K, N_tile) → (M_tile, N_tile)
                a_tile = A[m_start:m_end, :].astype(np.int32)
                w_tile = W[:, n_start:n_end].astype(np.int32)

                # Integer matmul — numpy int32 @ int32 → int64 on most platforms
                # This is conservative (wider than hardware INT32) → won't miss overflows
                partial = np.dot(a_tile, w_tile)

                # Saturate to INT32 (matches hardware saturation)
                partial = np.clip(partial, INT32_MIN, INT32_MAX)
                result[m_start:m_end, n_start:n_end] = partial.astype(np.int32)

        return result

    def matmul_from_sram(self, M: int, K: int, N: int,
                         act_sram_addr: int, wgt_sram_addr: int,
                         sram: np.ndarray) -> np.ndarray:
        """Execute MMUL reading activation and weight from SRAM.

        SRAM layout: unified byte array, activations and weights at given offsets.
        Returns INT32 result array (M, N).
        """
        # Read activation: M×K INT8 values from sram[act_sram_addr:]
        act_bytes = M * K  # INT8 = 1 byte each
        act = sram[act_sram_addr:act_sram_addr + act_bytes].astype(np.int8).reshape(M, K)

        # Read weights: K×N INT4 values (packed 2/byte) from sram[wgt_sram_addr:]
        wgt_packed_bytes = (K * N + 1) // 2
        wgt_packed = sram[wgt_sram_addr:wgt_sram_addr + wgt_packed_bytes].astype(np.uint8)

        return self.matmul_int32(act, wgt_packed, M, K, N)

    # ── Per-channel matmul ──────────────────────────────────────────

    def matmul_int4_per_channel(self, activation: np.ndarray,
                                weight_packed: np.ndarray,
                                weight_scales: np.ndarray,
                                M: int, K: int, N: int) -> np.ndarray:
        """INT4 per-channel: INT32 accumulate → FP16 scale per output channel.

        Hardware flow:
        1. PE array: INT4 weights × INT8 activations → INT32 partial sums (same as matmul_int32)
        2. At output: each column's INT32 result × FP16 scale[channel] → FP32

        This matches the block array hardware where scale multiplication
        happens in the accumulator/output stage, not during MAC operations.

        Args:
            activation: INT8, shape (M, K), range [-128, 127]
            weight_packed: packed INT4, uint8, (K*N+1)//2 bytes, values [-7, 7]
            weight_scales: float32 per-channel scales, shape (N,)
            M, K, N: matrix dimensions

        Returns:
            float32 result, shape (M, N)
        """
        # Step 1: INT32 accumulate (same as matmul_int32)
        int32_result = self.matmul_int32(activation, weight_packed, M, K, N)

        # Step 2: Per-channel scale (hardware: column accumulator × FP16 scale)
        # scales shape (N,) → broadcast to (M, N)
        scales_fp32 = np.asarray(weight_scales, dtype=np.float32)
        assert scales_fp32.shape == (N,), \
            f"Expected scales shape ({N},), got {scales_fp32.shape}"

        fp32_result = int32_result.astype(np.float32) * scales_fp32[np.newaxis, :]
        return fp32_result

    def matmul_int4_per_block(self, activation: np.ndarray,
                              weight_packed: np.ndarray,
                              block_scales: np.ndarray,
                              M: int, K: int, N: int,
                              group_size: int = 128) -> np.ndarray:
        """INT4 per-block: K-dimension split into blocks, each with per-channel scales.

        Hardware flow (matches 64×64 block array tiling):
        1. For each N-tile (width=ARRAY_W=64):
        2.   Split K into blocks of group_size
        3.   For each K-block: INT4×INT8 → INT32 partial for that block × N-tile
        4.   Scale partial by block_scales[block_idx, n_start:n_end]
        5.   Accumulate scaled partial into float32 accumulator

        This is more expensive than per-channel (scale multiplication happens per-block,
        not just once per column), but isolates outliers to individual blocks.

        Args:
            activation: INT8, shape (M, K)
            weight_packed: packed INT4, uint8
            block_scales: float32, shape (num_blocks, N)
            M, K, N: matrix dimensions
            group_size: block size along K (default 128, independent of ARRAY_H)

        Returns:
            float32 result, shape (M, N)
        """
        # Unpack weights
        w_flat = self.unpack_int4(weight_packed).astype(np.int32)
        expected_len = K * N
        if len(w_flat) < expected_len:
            w_flat = np.pad(w_flat, (0, expected_len - len(w_flat)), constant_values=0)
        W = w_flat[:expected_len].reshape(K, N)

        # Activations: INT8
        A = np.asarray(activation, dtype=np.int8)
        if A.size < M * K:
            A = np.pad(A.flatten(), (0, M * K - A.size), constant_values=0)
        A = A.flatten()[:M * K].reshape(M, K)

        scales = np.asarray(block_scales, dtype=np.float32)
        num_blocks = (K + group_size - 1) // group_size
        assert scales.shape == (num_blocks, N), \
            f"Expected block_scales ({num_blocks},{N}), got {scales.shape}"

        # Accumulate in float32 (hardware: FP32 accumulator after scale multiply)
        result = np.zeros((M, N), dtype=np.float32)

        # Tile over N (64×64 block array width)
        for n_start in range(0, N, self.W):
            n_end = min(n_start + self.W, N)

            # Per K-block: compute INT32 partial → scale → accumulate
            for b in range(num_blocks):
                k_start = b * group_size
                k_end = min(k_start + group_size, K)

                a_block = A[:, k_start:k_end].astype(np.int32)          # (M, block_size)
                w_block = W[k_start:k_end, n_start:n_end].astype(np.int32)  # (block_size, N_tile)

                # INT32 matmul for this block × tile
                partial = np.dot(a_block, w_block)   # (M, N_tile)
                partial = np.clip(partial, INT32_MIN, INT32_MAX)

                # Scale by block's per-channel scales
                block_sc = scales[b, n_start:n_end].astype(np.float32)  # (N_tile,)
                scaled = partial.astype(np.float32) * block_sc[np.newaxis, :]

                result[:, n_start:n_end] += scaled

        return result

    @staticmethod
    def hash_output(arr: np.ndarray) -> str:
        """MD5 hash for fast comparison (first 16 hex chars)."""
        return hashlib.md5(np.asarray(arr, dtype=np.int32).tobytes()).hexdigest()[:16]

    @staticmethod
    def max_error(golden: np.ndarray, test: np.ndarray) -> Dict[str, float]:
        """Compute error metrics between golden and test arrays."""
        diff = np.abs(golden.astype(np.float64) - test.astype(np.float64))
        rel = diff / (np.abs(golden.astype(np.float64)) + 1e-8)
        return {
            "max_abs": float(np.max(diff)),
            "mean_abs": float(np.mean(diff)),
            "max_rel": float(np.max(rel)),
            "exact_match": bool(np.all(golden == test)),
        }


# ══════════════════════════════════════════════════════════════════════
# Phase A-2: GoldenSFU — hardware-equivalent LUT/fixed-point
# ══════════════════════════════════════════════════════════════════════

class GoldenSFU:
    """Hardware-equivalent SFU: LUT-based activation functions.

    Each function has TWO implementations:
    - ref: float64 numpy (mathematically correct, for debugging)
    - hw:  fixed-point / LUT (matches hardware precision, for RTL comparison)

    Hardware spec:
    - Softmax: 256-entry exp LUT + piecewise subtraction, 8-stage pipeline
    - LayerNorm: 6-stage, parallel mean/var + fused multiply-add
    - GELU: 4-segment piecewise linear LUT, 4-stage pipeline
    - SiLU: reuses GELU LUT, 4-stage
    - RoPE: CORDIC 12-stage rotation
    """

    def __init__(self):
        self._build_exp_lut()
        self._build_gelu_lut()
        self._build_cordic_table()

    @staticmethod
    def _flush_f16_subnormals(x: np.ndarray) -> np.ndarray:
        """Flush FP16 subnormal values to zero (matches hardware SFU input path)."""
        tiny = np.finfo(np.float16).tiny
        x = np.asarray(x, dtype=np.float32)
        x[np.abs(x) < tiny] = 0.0
        return x

    # ── Softmax LUT ─────────────────────────────────────────────────

    def _build_exp_lut(self, entries: int = 4096, x_min: float = -20.0):
        """Build exponential lookup table for softmax.

        The functional model uses 4096 entries (more than the 256-entry RTL ROM)
        to keep linear-interpolation error below 1e-5 across [-20, 0].
        The RTL verification uses its own tighter tolerance (abs_tol=2e-3) that is
        achievable with 256 entries.
        """
        self.exp_lut_x_min = x_min
        self.exp_lut_x_max = 0.0
        self.exp_lut_entries = entries
        self.exp_lut_step = (self.exp_lut_x_max - x_min) / (entries - 1)

        xs = np.linspace(x_min, 0.0, entries, dtype=np.float64)
        self.exp_lut = np.exp(xs).astype(np.float32)

    def _exp_hw(self, x: np.ndarray) -> np.ndarray:
        """Hardware-equivalent exp: LUT lookup with linear interpolation.

        Input range: [-20, 0]. Values < -20 clamp to 0. Values > 0 clamp to 1.
        """
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x, dtype=np.float32)

        # Clamp to LUT range
        valid = (x >= self.exp_lut_x_min) & (x <= self.exp_lut_x_max)

        if np.any(valid):
            xv = x[valid]
            # Fractional index into LUT
            idx_f = (xv - self.exp_lut_x_min) / self.exp_lut_step
            idx_lo = np.floor(idx_f).astype(np.int32)
            idx_hi = np.minimum(idx_lo + 1, self.exp_lut_entries - 1)
            frac = idx_f - idx_lo

            # Linear interpolation
            result[valid] = (
                self.exp_lut[idx_lo] * (1.0 - frac) +
                self.exp_lut[idx_hi] * frac
            ).astype(np.float32)

        # Above 0 → 1.0 (values > 0 shouldn't appear after max-subtract)
        result[x > 0] = 1.0
        # Below min → 0
        result[x < self.exp_lut_x_min] = 0.0

        return result

    def softmax_hw(self, x: np.ndarray) -> np.ndarray:
        """Hardware-equivalent softmax: max-subtract → LUT-exp → normalize.

        Matches 8-stage pipeline: max_reduce → sub → LUT exp → sum_reduce → div.
        Uses BF16-equivalent precision throughout.
        """
        x = self._flush_f16_subnormals(x)
        x_max = np.max(x)
        x_sub = x - x_max  # ≤ 0, in LUT range

        # LUT-based exp
        exp_vals = self._exp_hw(x_sub)

        # Reciprocal normalization (hardware uses iterative division)
        s = np.sum(exp_vals)
        if s > 0:
            return exp_vals / s
        return exp_vals

    def softmax_ref(self, x: np.ndarray) -> np.ndarray:
        """Reference softmax: float64 numpy."""
        x = np.asarray(x, dtype=np.float64)
        x_max = np.max(x)
        e_x = np.exp(x - x_max)
        return (e_x / np.sum(e_x)).astype(np.float32)

    # ── GELU LUT ────────────────────────────────────────────────────

    def _build_gelu_lut(self, entries: int = 64, x_min: float = -4.0, x_max: float = 4.0):
        """Build 64-entry GELU LUT with linear interpolation.

        Hardware: 4-stage pipeline, 64-entry LUT covering [-4, 4].
        Uses tanh approximation: gelu(x) ≈ 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
        """
        self.gelu_lut_entries = entries
        self.gelu_lut_x_min = x_min
        self.gelu_lut_x_max = x_max
        self.gelu_lut_step = (x_max - x_min) / (entries - 1)

        xs = np.linspace(x_min, x_max, entries, dtype=np.float64)
        self.gelu_lut = (0.5 * xs * (
            1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (xs + 0.044715 * xs**3))
        )).astype(np.float32)

    def gelu_hw(self, x: np.ndarray) -> np.ndarray:
        """Hardware-equivalent GELU: 64-entry LUT with linear interpolation."""
        x = self._flush_f16_subnormals(x)
        result = np.zeros_like(x, dtype=np.float32)

        # Clamp to LUT range
        below = x < self.gelu_lut_x_min
        above = x > self.gelu_lut_x_max
        in_range = ~(below | above)

        # Below min → 0 (GELU saturates to 0 for very negative x)
        result[below] = 0.0
        # Above max → x (GELU ≈ x for large positive x)
        result[above] = x[above]

        if np.any(in_range):
            xv = x[in_range]
            idx_f = (xv - self.gelu_lut_x_min) / self.gelu_lut_step
            idx_lo = np.floor(idx_f).astype(np.int32)
            idx_hi = np.minimum(idx_lo + 1, self.gelu_lut_entries - 1)
            frac = idx_f - idx_lo

            result[in_range] = (
                self.gelu_lut[idx_lo] * (1.0 - frac) +
                self.gelu_lut[idx_hi] * frac
            ).astype(np.float32)

        return result

    def gelu_ref(self, x: np.ndarray) -> np.ndarray:
        """Reference GELU: tanh approximation (float64)."""
        x = np.asarray(x, dtype=np.float64)
        return (0.5 * x * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
        ))).astype(np.float32)

    # ── SiLU ────────────────────────────────────────────────────────

    @staticmethod
    def silu_ref(x: np.ndarray) -> np.ndarray:
        """Reference SiLU: x * sigmoid(x)."""
        x = np.asarray(x, dtype=np.float64)
        return (x / (1.0 + np.exp(-x))).astype(np.float32)

    def silu_hw(self, x: np.ndarray) -> np.ndarray:
        """Hardware-equivalent SiLU: using sigmoid LUT (reuses exp LUT)."""
        x = self._flush_f16_subnormals(x)
        # sigmoid(x) = 1 / (1 + exp(-x))
        # For x >= 0: exp(-x) is in [0, 1]
        # For x < 0: exp(-x) > 1, sigmoid ≈ exp(x) / (1 + exp(x))
        neg_exp = self._exp_hw(-np.abs(x))  # exp(-|x|)
        sigmoid = np.where(
            x >= 0,
            1.0 / (1.0 + neg_exp),
            neg_exp / (1.0 + neg_exp)
        )
        return x * sigmoid

    # ── ReLU ────────────────────────────────────────────────────────

    @staticmethod
    def relu_hw(x: np.ndarray) -> np.ndarray:
        """Hardware-equivalent ReLU: x if x > 0 else 0.

        Hardware: zero-overhead comparator on SFU datapath, matches FP16 semantics.
        Flushes subnormals on input path.
        """
        x = GoldenSFU._flush_f16_subnormals(x)
        return np.maximum(x, 0.0).astype(np.float32)

    @staticmethod
    def relu_ref(x: np.ndarray) -> np.ndarray:
        """Reference ReLU: float64."""
        x = np.asarray(x, dtype=np.float64)
        return np.maximum(x, 0.0).astype(np.float32)

    # ── LayerNorm (fixed-point) ─────────────────────────────────────

    @staticmethod
    def layernorm_hw(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Hardware-equivalent LayerNorm: fixed-point mean/variance.

        Hardware: 6-stage pipeline.
        - Mean: sum/N using integer division
        - Variance: sum((x-mean)^2)/N
        - Output: (x - mean) / sqrt(var + eps)
        """
        x = GoldenSFU._flush_f16_subnormals(x)
        N = x.shape[-1]
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        # Quantize intermediate to BF16 precision (simulate hardware)
        mean = mean.astype(np.float16).astype(np.float32)
        var = var.astype(np.float16).astype(np.float32)
        result = (x - mean) / np.sqrt(var + eps)
        return result.astype(np.float16).astype(np.float32)

    @staticmethod
    def layernorm_ref(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Reference LayerNorm: float64."""
        x = np.asarray(x, dtype=np.float64)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return ((x - mean) / np.sqrt(var + eps)).astype(np.float32)

    # ── RMSNorm ─────────────────────────────────────────────────────

    @staticmethod
    def rmsnorm_hw(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Hardware-equivalent RMSNorm: no mean subtraction.

        RMSNorm: result = x / sqrt(mean(x^2) + eps)

        Full float32 precision (FPU path, no LUT). The hardware SFU computes
        RMSNorm in the FPU datapath (not LUT-based), so it achieves float32
        precision — unlike Softmax/LayerNorm which use fixed-point LUTs.
        Reference uses float64 for verification.
        """
        x = GoldenSFU._flush_f16_subnormals(x)
        if x.ndim == 1:
            mean_xsq = np.mean(x ** 2)
        else:
            mean_xsq = np.mean(x ** 2, axis=-1, keepdims=True)
        return (x / np.sqrt(mean_xsq + eps)).astype(np.float32)

    @staticmethod
    def rmsnorm_ref(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Reference RMSNorm: float64."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            mean_xsq = np.mean(x ** 2)
        else:
            mean_xsq = np.mean(x ** 2, axis=-1, keepdims=True)
        return (x / np.sqrt(mean_xsq + eps)).astype(np.float32)

    # ── RoPE (CORDIC-equivalent) ────────────────────────────────────

    def _build_cordic_table(self, iterations: int = 12):
        """Build CORDIC angle table.

        Hardware: 12-stage CORDIC rotation.
        Each stage: rot_i = atan(2^-i).
        """
        self.cordic_iterations = iterations
        self.cordic_angles = np.arctan(2.0 ** -np.arange(iterations)).astype(np.float32)
        # CORDIC gain: product_i cos(atan(2^-i)) ≈ 0.607253
        self.cordic_gain = np.prod(np.cos(self.cordic_angles))

    def _cordic_rotate(self, x0: float, y0: float, theta: float) -> Tuple[float, float]:
        """CORDIC rotation: (x0, y0) rotated by theta radians.

        Hardware: 12-stage iterative shift-add.
        Angle MUST be in [-π/2, π/2] for convergence. Caller must reduce.
        """
        # Reduce angle to [-π, π] then adjust quadrant for CORDIC convergence
        theta = theta % (2.0 * math.pi)
        if theta > math.pi:
            theta -= 2.0 * math.pi

        # CORDIC only converges in [-π/2, π/2]
        # For angles outside: use quadrant symmetry: rotate(x,y, θ+π) = -rotate(x,y, θ)
        flip = False
        if theta > math.pi / 2:
            theta -= math.pi
            flip = True
        elif theta < -math.pi / 2:
            theta += math.pi
            flip = True

        # Scale by CORDIC gain (CORDIC magnifies by 1/K, so pre-scale by K)
        x = x0 * self.cordic_gain
        y = y0 * self.cordic_gain
        z = theta

        for i in range(self.cordic_iterations):
            if z >= 0:
                d = 1
            else:
                d = -1
            x_new = x - d * y * (2.0 ** -i)
            y_new = y + d * x * (2.0 ** -i)
            z = z - d * float(self.cordic_angles[i])
            x, y = x_new, y_new

        if flip:
            x, y = -x, -y
        return x, y

    def rope_hw(self, x_q: np.ndarray, x_k: np.ndarray, position: int,
                num_heads: int = 32, head_dim: int = 128,
                theta: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
        """Hardware-equivalent RoPE: CORDIC rotation.

        Hardware uses 12-stage CORDIC, producing precision equivalent to ~11-bit angle.
        """
        x_q = self._flush_f16_subnormals(x_q)
        x_k = self._flush_f16_subnormals(x_k)

        # Frequency bands (RoPE uses pairs of dimensions)
        freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
        angles = position * freqs

        def rotate_cordic(x, n_heads):
            x = x.reshape(n_heads, head_dim).copy()
            for h in range(n_heads):
                for i in range(0, head_dim, 2):
                    x_rot, y_rot = self._cordic_rotate(
                        float(x[h, i]), float(x[h, i + 1]), float(angles[i // 2])
                    )
                    x[h, i] = x_rot
                    x[h, i + 1] = y_rot
            return x.reshape(-1)

        return rotate_cordic(x_q, num_heads), rotate_cordic(x_k, 2)

    def rope_ref(self, x_q: np.ndarray, x_k: np.ndarray, position: int,
                 num_heads: int = 32, head_dim: int = 128,
                 theta: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
        """Reference RoPE: float64 trig."""
        x_q = np.asarray(x_q, dtype=np.float64)
        x_k = np.asarray(x_k, dtype=np.float64)

        freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2) / head_dim))
        angles = position * freqs
        cos = np.cos(angles)
        sin = np.sin(angles)

        def rotate_ref(x, n_heads):
            x = x.reshape(n_heads, head_dim).copy()
            for h in range(n_heads):
                for i in range(0, head_dim, 2):
                    x0, x1 = x[h, i], x[h, i + 1]
                    c, s = cos[i // 2], sin[i // 2]
                    x[h, i] = x0 * c - x1 * s
                    x[h, i + 1] = x1 * c + x0 * s
            return x.reshape(-1)

        return (rotate_ref(x_q, num_heads).astype(np.float32),
                rotate_ref(x_k, 2).astype(np.float32))

    # ── Error metrics ───────────────────────────────────────────────

    @staticmethod
    def compare_hw_vs_ref(hw: np.ndarray, ref: np.ndarray,
                          tol_abs: float = 1e-3, tol_rel: float = 1e-3) -> Dict[str, Any]:
        """Compare hardware-equivalent vs reference implementation."""
        hw = np.asarray(hw, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)
        abs_diff = np.abs(hw - ref)
        rel_diff = abs_diff / (np.abs(ref) + 1e-12)
        return {
            "max_abs_err": float(np.max(abs_diff)),
            "mean_abs_err": float(np.mean(abs_diff)),
            "max_rel_err": float(np.max(rel_diff)),
            "within_tolerance": bool(np.all(abs_diff < tol_abs) or np.all(rel_diff < tol_rel)),
        }


# ══════════════════════════════════════════════════════════════════════
# Phase D-1: GoldenVector — bit-accurate Vector Unit
# ══════════════════════════════════════════════════════════════════════

class GoldenVector:
    """Bit-accurate Vector Unit: element-wise ops, reductions, type conversion.

    Hardware: 128-wide SIMD pipeline, shares SRAM bandwidth with MXU/SFU.
    Key operations:
    - Element-wise: add, mul, scale, bias
    - Reductions: max_reduce, sum_reduce (tree reduction)
    - Type conversion: INT32 → BF16 (bridge between MXU and SFU)
    - Residual add: x = x + residual (INT32 accumulation)
    """

    def __init__(self, width: int = 128):
        self.width = width  # SIMD width (elements per cycle)

    # ── Element-wise ops ────────────────────────────────────────────

    @staticmethod
    def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise add in INT32 with saturation."""
        result = a.astype(np.int64) + b.astype(np.int64)
        return np.clip(result, INT32_MIN, INT32_MAX).astype(np.int32)

    @staticmethod
    def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise multiply in INT32 with saturation."""
        result = a.astype(np.int64) * b.astype(np.int64)
        return np.clip(result, INT32_MIN, INT32_MAX).astype(np.int32)

    @staticmethod
    def max(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise max in INT32 (bit-exact comparator)."""
        return np.maximum(a, b).astype(np.int32)

    # ── Reductions ──────────────────────────────────────────────────

    @staticmethod
    def max_reduce(x: np.ndarray) -> float:
        """Tree reduction: find maximum value in array.

        Hardware: log2(N) stages of pairwise max.
        For INT32 input, returns the max value.
        """
        return float(np.max(x))

    @staticmethod
    def sum_reduce(x: np.ndarray) -> float:
        """Tree reduction: sum all elements.

        Hardware: log2(N) stages of pairwise add.
        INT32 accumulation with overflow protection.
        """
        return float(np.sum(x.astype(np.float64)))

    # ── Type conversion (INT32 → BF16) ─────────────────────────────

    @staticmethod
    def conv_i32_to_f16(arr: np.ndarray) -> np.ndarray:
        """Convert INT32 array to BF16 (FP16).

        This is THE critical bridge between MXU output (INT32) and SFU input (BF16).
        Hardware: INT32 accumulator → BF16 converter → SFU input buffer.

        Precision: INT32 range ±2B maps to BF16 range ±65504.
        Values outside BF16 range saturate.
        """
        arr = np.asarray(arr, dtype=np.int32)
        # Convert to float32 first (exact for up to 2^24), then to float16
        f32 = arr.astype(np.float32)
        # Hardware saturation: clamp to BF16 range
        f16_max = np.finfo(np.float16).max
        f32 = np.clip(f32, -f16_max, f16_max)
        return f32.astype(np.float16)

    @staticmethod
    def conv_f16_to_i32(arr: np.ndarray) -> np.ndarray:
        """Convert FP16 array to INT32 with round-toward-zero truncation.

        This is the SFU→MXU/Vector dtype bridge: FP16 activations from the SFU
        pipeline are converted to INT32 for the Vector/MXU datapath.

        Semantics:
          - Finite values truncate toward zero (matches numpy float32→int32).
          - FP16 subnormals are flushed to zero before conversion.
          - ±Inf and NaN saturate sign-aware to INT32_MAX / INT32_MIN.
        """
        arr = np.asarray(arr, dtype=np.float16)
        f32 = arr.astype(np.float32)
        tiny = np.finfo(np.float16).tiny
        f32[np.abs(f32) < tiny] = 0.0
        sign = np.signbit(f32)
        is_special = ~np.isfinite(f32)
        out = np.empty(f32.shape, dtype=np.int32)
        out[is_special & ~sign] = INT32_MAX
        out[is_special & sign] = INT32_MIN
        finite = ~is_special
        out[finite] = f32[finite].astype(np.int32)
        return out

    # ── Residual add ───────────────────────────────────────────────

    @staticmethod
    def residual_add(original: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Residual connection: x = x + output in INT32.

        original: BF16 residual from skip connection (converted back to INT32)
        delta: INT32 from MXU output
        Returns: INT32 sum
        """
        orig_i32 = original.astype(np.float32).astype(np.int32)
        delta_i32 = delta.astype(np.int32)
        result = orig_i32.astype(np.int64) + delta_i32.astype(np.int64)
        # Saturate to INT32
        return np.clip(result, -2**31, 2**31 - 1).astype(np.int32)

    # ── Softmax decomposition helpers ───────────────────────────────

    def softmax_max_reduce(self, x: np.ndarray) -> float:
        """Vector step 1: find max(x)."""
        return self.max_reduce(x)

    def softmax_scale_sub(self, x: np.ndarray, x_max: float) -> np.ndarray:
        """Vector step 2: x = x - max (element-wise subtract scalar)."""
        return x - x_max

    @staticmethod
    def softmax_sum_reduce(x: np.ndarray) -> float:
        """Vector step 4: sum(exp(x))."""
        return float(np.sum(x.astype(np.float64)))


# ══════════════════════════════════════════════════════════════════════
# Phase D-2: GoldenDMA — DMA engine behavior model
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DMADescriptor:
    """DMA descriptor: one transfer unit in a descriptor chain.

    Hardware: 16-entry descriptor queue, each entry 64-bit.
    Format (64-bit):
      [63:32] dram_addr[31:0]  — DRAM source/destination address
      [31:16] sram_addr[15:0]  — SRAM address (within 2MB)
      [15:4]  size[11:0]       — transfer size in bytes (1-4095, 0=4096)
      [3]     direction        — 0=DRAM→SRAM(load), 1=SRAM→DRAM(store)
      [2]     last              — 1=last descriptor in chain
      [1:0]   channel           — 0=weight, 1=data
    """
    dram_addr: int
    sram_addr: int
    size: int
    direction: int = 0  # 0=load, 1=store
    last: bool = False
    channel: int = 0   # 0=weight, 1=data
    cycle_cost: int = 0  # computed by DMA model

    def __post_init__(self):
        """Validate field values match hardware encoding limits."""
        if self.direction not in (0, 1):
            raise ValueError(f"direction must be 0 or 1, got {self.direction}")
        if self.channel not in (0, 1, 2, 3):
            raise ValueError(f"channel must be 0..3, got {self.channel}")
        if not (0 <= self.size <= 4095) and self.size != 4096:
            raise ValueError(f"size must be in [0, 4095] or 4096 (encoded as 0), got {self.size}")
        if not (0 <= self.sram_addr < 0x10000):
            raise ValueError(f"sram_addr must be in [0, 0xFFFF], got {self.sram_addr:#x}")
        if not (0 <= self.dram_addr < 0x100000000):
            raise ValueError(f"dram_addr must be in [0, 0xFFFFFFFF], got {self.dram_addr:#x}")

    @property
    def actual_size(self) -> int:
        """Actual transfer size (0 in field means 4096 bytes)."""
        return 4096 if self.size == 0 else self.size

    def encode(self) -> int:
        """Encode to 64-bit descriptor word."""
        # Size field: 0 means 4096, otherwise actual size (max 4095)
        sz = 0 if self.size == 4096 else self.size
        desc = 0
        desc |= (self.dram_addr & 0xFFFFFFFF) << 32
        desc |= (self.sram_addr & 0xFFFF) << 16
        desc |= (sz & 0xFFF) << 4
        desc |= (self.direction & 1) << 3
        desc |= (1 if self.last else 0) << 2
        desc |= (self.channel & 3)
        return desc

    @classmethod
    def decode(cls, word: int) -> "DMADescriptor":
        """Decode from 64-bit descriptor word."""
        return cls(
            dram_addr=(word >> 32) & 0xFFFFFFFF,
            sram_addr=(word >> 16) & 0xFFFF,
            size=(word >> 4) & 0xFFF,
            direction=(word >> 3) & 1,
            last=bool((word >> 2) & 1),
            channel=word & 3,
        )


class GoldenDMA:
    """DMA Engine model: descriptor chains, dual channels, DRAM ↔ SRAM.

    Hardware: 2 channels (weight + data), 16-entry descriptor queue.
    DRAM bandwidth: configurable (default 51.2 GB/s LPDDR5-64b).
    """

    def __init__(self):
        self.channel_active: Dict[int, bool] = {0: False, 1: False}

    def execute_load(self, sram: "SRAM", desc: DMADescriptor,
                     dram_data: np.ndarray):
        """Execute a DMA load: DRAM → SRAM."""
        sz = desc.actual_size
        sram.write_bytes(desc.sram_addr,
                         dram_data[desc.dram_addr:desc.dram_addr + sz])

    def execute_store(self, sram: "SRAM", desc: DMADescriptor,
                      dram_data: np.ndarray):
        """Execute a DMA store: SRAM → DRAM."""
        sz = desc.actual_size
        data = sram.read_bytes(desc.sram_addr, sz)
        dram_data[desc.dram_addr:desc.dram_addr + sz] = data

    def build_weight_load_chain(self, sram_base: int, dram_base: int,
                                 total_bytes: int, chunk_size: int = 4096) -> List[DMADescriptor]:
        """Build descriptor chain for loading weights from DRAM to SRAM.

        Splits large weight transfer into chunks, interleaved for ping-pong buffering.
        """
        descriptors = []
        remaining = total_bytes
        offset = 0

        while remaining > 0:
            size = min(chunk_size, remaining)
            descriptors.append(DMADescriptor(
                dram_addr=dram_base + offset,
                sram_addr=sram_base + offset,
                size=size,
                direction=0,  # load
                last=(remaining <= chunk_size),
                channel=0,    # weight channel
            ))
            offset += size
            remaining -= size

        return descriptors


# ══════════════════════════════════════════════════════════════════════
# Phase D-3: GoldenNoC — NoC functional model
# ══════════════════════════════════════════════════════════════════════

@dataclass
class NoCPacket:
    """NoC packet: one unit of data transfer between NPU cores.

    Fields:
        src_id: source node ID
        dst_id: destination node ID
        payload: data payload as numpy array (bit-exact byte content)
        packet_id: unique packet identifier for ordering/tracking
        priority: packet priority (0=lowest, higher=more urgent)
        size_bytes: size of data payload in bytes
    """
    src_id: int
    dst_id: int
    payload: np.ndarray
    packet_id: int = 0
    priority: int = 0
    size_bytes: int = 0

    def __post_init__(self):
        """Auto-compute size_bytes from payload nbytes when not explicitly set."""
        if self.size_bytes == 0 and self.payload.size > 0:
            self.size_bytes = self.payload.nbytes


class GoldenNoC:
    """NoC functional model: packet routing and payload delivery.

    Performs bit-exact data movement only — no cycle estimation.
    Matches hardware NoC behavior for correctness verification.
    """

    @staticmethod
    def route_packet(packet: NoCPacket, network_state: dict) -> bool:
        """Check whether a NoC packet can be routed given network state.

        Args:
            packet: NoCPacket with src_id, dst_id
            network_state: dict with optional keys:
                - active_nodes: set of online node IDs
                - congestion: dict mapping (src,dst) -> congestion level

        Returns:
            True if routable, False if blocked.
        """
        active = network_state.get("active_nodes", None)
        if active is not None and packet.dst_id not in active:
            return False

        congestion = network_state.get("congestion", {})
        link = (packet.src_id, packet.dst_id)
        if link in congestion and congestion[link] > 0:
            return False

        return True

    @staticmethod
    def deliver_payload(packet: NoCPacket, sram: bytearray, addr: int) -> None:
        """Write NoC packet payload bytes into SRAM at given address.

        Bit-exact copy: payload's raw bytes are written to sram[addr:addr+size_bytes].
        """
        raw = packet.payload.tobytes()
        end = addr + len(raw)
        if end > len(sram):
            raise ValueError(
                f"NoC payload overflow: addr={addr:#x} + {len(raw)}B > sram {len(sram)}B"
            )
        sram[addr:end] = raw

    @staticmethod
    def build_transfer_packet(src: int, dst: int, data: np.ndarray,
                              priority: int = 0) -> NoCPacket:
        """Build a NoC transfer packet from source to destination.

        Args:
            src: source core ID
            dst: destination core ID
            data: payload as numpy array (any dtype)
            priority: packet priority (default 0)

        Returns:
            NoCPacket with auto-computed size_bytes.
        """
        arr = np.asarray(data)
        return NoCPacket(
            src_id=src,
            dst_id=dst,
            payload=arr,
            packet_id=0,
            priority=priority,
            size_bytes=arr.nbytes,
        )


# ── GoldenExecutor: add Vector and DMA instruction handlers ──────────
# Phase B: SRAM memory model
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SRAMRegion:
    """Named region in SRAM address space."""
    name: str
    start: int
    size: int

    @property
    def end(self) -> int:
        return self.start + self.size


class SRAM:
    """Unified Buffer SRAM (2 MB) matching hardware memory map.

    Address map (matching architecture doc §3.2):
    0x000000 - 0x0FFFFF: Weight Buffer A (Ping)   — 1 MB
    0x100000 - 0x1FFFFF: Weight Buffer B (Pong)   — 1 MB
    0x200000 - 0x27FFFF: Activation Buffer         — 512 KB
    0x280000 - 0x2BFFFF: Accumulator (INT32)       — 256 KB
    0x2C0000 - 0x2FFFFF: SFU I/O Buffer (BF16)     — 256 KB
    """

    SIZE = SRAM_SIZE

    def __init__(self):
        self.data = np.zeros(SRAM_SIZE, dtype=np.uint8)
        self.regions: Dict[str, SRAMRegion] = {}

    def define_region(self, name: str, start: int, size: int):
        """Define a named memory region."""
        self.regions[name] = SRAMRegion(name=name, start=start, size=size)

    def read_bytes(self, addr: int, n_bytes: int) -> np.ndarray:
        """Read n_bytes from SRAM at given address."""
        if addr + n_bytes > SRAM_SIZE:
            raise ValueError(f"SRAM read overflow: addr={addr:#x} + {n_bytes} > {SRAM_SIZE:#x}")
        return self.data[addr:addr + n_bytes].copy()

    def write_bytes(self, addr: int, data: np.ndarray):
        """Write bytes to SRAM at given address."""
        n = len(data)
        if addr + n > SRAM_SIZE:
            raise ValueError(f"SRAM write overflow: addr={addr:#x} + {n} > {SRAM_SIZE:#x}")
        self.data[addr:addr + n] = np.asarray(data, dtype=np.uint8).flatten()

    def write_int32(self, addr: int, data: np.ndarray):
        """Write INT32 array to SRAM (little-endian)."""
        raw = np.asarray(data, dtype=np.int32).tobytes()
        self.write_bytes(addr, np.frombuffer(raw, dtype=np.uint8))

    def read_int32(self, addr: int, n_elems: int) -> np.ndarray:
        """Read INT32 array from SRAM."""
        raw = self.read_bytes(addr, n_elems * 4)
        return np.frombuffer(raw.tobytes(), dtype=np.int32).copy()

    def write_float16(self, addr: int, data: np.ndarray):
        """Write BF16/FP16 array to SRAM (little-endian, 2 bytes each)."""
        raw = np.asarray(data, dtype=np.float16).tobytes()
        self.write_bytes(addr, np.frombuffer(raw, dtype=np.uint8))

    def read_float16(self, addr: int, n_elems: int) -> np.ndarray:
        """Read BF16/FP16 array from SRAM."""
        raw = self.read_bytes(addr, n_elems * 2)
        return np.frombuffer(raw.tobytes(), dtype=np.float16).copy()

    def write_int8(self, addr: int, data: np.ndarray):
        """Write INT8 array to SRAM (little-endian, 1 byte each)."""
        raw = np.asarray(data, dtype=np.int8).tobytes()
        self.write_bytes(addr, np.frombuffer(raw, dtype=np.uint8))

    def read_int8(self, addr: int, n_elems: int) -> np.ndarray:
        """Read INT8 array from SRAM."""
        raw = self.read_bytes(addr, n_elems)
        return np.frombuffer(raw.tobytes(), dtype=np.int8).copy()

    def checksum_region(self, name: str) -> str:
        """MD5 checksum of a named region."""
        r = self.regions[name]
        return hashlib.md5(self.data[r.start:r.end].tobytes()).hexdigest()[:16]

    def zero_region(self, name: str):
        """Zero out a named region."""
        r = self.regions[name]
        self.data[r.start:r.end] = 0


# ══════════════════════════════════════════════════════════════════════
# Phase B: ISA Executor
# ══════════════════════════════════════════════════════════════════════

# Import ISA types (from sibling engine/isa.py)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "sim"))
from engine.isa import NPUInstruction, OpCode, NPUDecoder, NPUEncoder


@dataclass
class ExecutorState:
    """Snapshottable state of the NPU core.

    For RTL verification: capture state after each instruction,
    compare against RTL's internal state at the same point.
    """
    cycle: int = 0
    pc: int = 0  # program counter (instruction index)
    sram_checksum: str = ""
    mxu_output_hash: str = ""
    sfu_output_hash: str = ""

    def snapshot(self, sram: SRAM) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "pc": self.pc,
            "sram_checksum": hashlib.md5(sram.data.tobytes()).hexdigest()[:16],
            "mxu_output_hash": self.mxu_output_hash,
            "sfu_output_hash": self.sfu_output_hash,
        }


class GoldenExecutor:
    """Bit-accurate NPU instruction executor.

    Takes ISA program + initial SRAM state → executes instruction-by-instruction.
    Returns final SRAM state + per-instruction trace for RTL verification.
    """

    def __init__(self, array_h: int = ARRAY_H, array_w: int = ARRAY_W):
        self.mxu = GoldenMXU(array_h, array_w)
        self.sfu = GoldenSFU()
        self.vector = GoldenVector()
        self.dma = GoldenDMA()
        self.dma_model = DMAModel({
            "dma": {
                "channels": 2,
                "burst_size_bytes": 256,
                "descriptor_overhead_cycles": 5,
                "max_pending_descriptors": 16,
            },
            "memory": {
                "bandwidth_bytes_per_cycle": 51.2,
            },
        })
        self.noc = GoldenNoC()
        self.sram = SRAM()
        self.dram = np.zeros(256 * 1024 * 1024, dtype=np.uint8)  # 256 MB DRAM model
        self.state = ExecutorState()

        # Default memory map
        self.sram.define_region("weight_ping", 0x000000, 1 * 1024 * 1024)
        self.sram.define_region("weight_pong", 0x100000, 1 * 1024 * 1024)
        self.sram.define_region("activation",  0x200000, 512 * 1024)
        self.sram.define_region("accumulator", 0x280000, 256 * 1024)
        self.sram.define_region("sfu_io",      0x2C0000, 256 * 1024)
        self.sram.define_region("vector_io",   0x300000, 256 * 1024)  # v2

        # Dtype bookkeeping for automatic inter-op VCONV insertion.
        # Primary input dtype is the dtype the chained operand must present
        # in SRAM for the op to consume it without an explicit preload.
        self._OPCODE_INPUT_DTYPE = {
            OpCode.MMUL: "int8",
            OpCode.SOFTMAX: "fp16",
            OpCode.LAYERNORM: "fp16",
            OpCode.GELU: "fp16",
            OpCode.RELU: "fp16",
            OpCode.ROPE: "fp16",
            OpCode.SILU: "fp16",
            OpCode.MAXPOOL: "fp16",
            OpCode.AVGPOOL: "fp16",
            OpCode.RMSNORM: "fp16",
            OpCode.VADD: "int32",
            OpCode.VMUL: "int32",
            OpCode.VRED_MAX: "int32",
            OpCode.VRED_SUM: "int32",
            OpCode.VCONV: "int32",
            OpCode.VCONV_F16_I32: "fp16",
            OpCode.VRESID: "int32",   # chained delta operand (sb)
        }
        self._OPCODE_OUTPUT_DTYPE = {
            OpCode.MMUL: "int32",
            OpCode.SOFTMAX: "fp16",
            OpCode.LAYERNORM: "fp16",
            OpCode.GELU: "fp16",
            OpCode.RELU: "fp16",
            OpCode.ROPE: "fp16",
            OpCode.SILU: "fp16",
            OpCode.MAXPOOL: "fp16",
            OpCode.AVGPOOL: "fp16",
            OpCode.RMSNORM: "fp16",
            OpCode.VADD: "int32",
            OpCode.VMUL: "int32",
            OpCode.VRED_MAX: "fp16",
            OpCode.VRED_SUM: "fp16",
            OpCode.VCONV: "fp16",
            OpCode.VCONV_F16_I32: "int32",
            OpCode.VRESID: "int32",
        }

    # ── Instruction execution ───────────────────────────────────────

    def step(self, instr: NPUInstruction) -> ExecutorState:
        """Execute ONE instruction on the hardware model.

        Returns snapshotted state before advancing to next instruction.
        This is THE function RTL verification compares against —
        for each instruction in program, golden.step() state must match RTL state.
        """
        op = instr.opcode
        ops = instr.operands
        state_before = self.state.snapshot(self.sram)

        if op == OpCode.NOP:
            self.state.cycle += 1

        elif op == OpCode.BARRIER:
            self.state.cycle += 5  # pipeline drain

        elif op == OpCode.MMUL:
            wa = ops.get("wa", 0)  # weight address
            ia = ops.get("ia", 0)  # input activation address
            oa = ops.get("oa", 0)  # output address
            N = ops.get("N", 2560)  # output dimension
            M = ops.get("M", 1)     # decode: 1 token
            K = ops.get("K", 2560)  # hidden_size

            result = self.mxu.matmul_from_sram(M, K, N, ia, wa, self.sram.data)

            # Write INT32 result to accumulator region
            self.sram.write_int32(oa, result)
            self.state.mxu_output_hash = GoldenMXU.hash_output(result)
            self.state.cycle += self._estimate_mxu_cycles(M, K, N)

        elif op == OpCode.SOFTMAX:
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            length = ops.get("len", 2560)

            # Read input as float32 from SRAM, apply softmax, write back
            inp = self.sram.read_float16(sa, length).astype(np.float32)
            out = self.sfu.softmax_hw(inp)
            self.sram.write_float16(da, out.astype(np.float16))
            self.state.sfu_output_hash = hashlib.md5(
                out.astype(np.float16).tobytes()
            ).hexdigest()[:16]
            self.state.cycle += length // self.sfu.exp_lut_entries + 8

        elif op == OpCode.LAYERNORM:
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            length = ops.get("len", 2560)

            inp = self.sram.read_float16(sa, length).astype(np.float32)
            out = self.sfu.layernorm_hw(inp)
            self.sram.write_float16(da, out.astype(np.float16))
            self.state.sfu_output_hash = hashlib.md5(
                out.astype(np.float16).tobytes()
            ).hexdigest()[:16]
            self.state.cycle += length // 128 + 6  # 6-stage pipeline

        elif op == OpCode.GELU:
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            length = ops.get("len", 2560)

            inp = self.sram.read_float16(sa, length).astype(np.float32)
            out = self.sfu.gelu_hw(inp)
            self.sram.write_float16(da, out.astype(np.float16))
            self.state.sfu_output_hash = hashlib.md5(
                out.astype(np.float16).tobytes()
            ).hexdigest()[:16]
            self.state.cycle += length // 128 + 4

        elif op == OpCode.SILU:
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            length = ops.get("len", 2560)

            inp = self.sram.read_float16(sa, length).astype(np.float32)
            out = self.sfu.silu_hw(inp)
            self.sram.write_float16(da, out.astype(np.float16))
            self.state.sfu_output_hash = hashlib.md5(
                out.astype(np.float16).tobytes()
            ).hexdigest()[:16]
            self.state.cycle += length // 128 + 4

        elif op == OpCode.RELU:
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            length = ops.get("len", 2560)

            inp = self.sram.read_float16(sa, length).astype(np.float32)
            out = self.sfu.relu_hw(inp)
            self.sram.write_float16(da, out.astype(np.float16))
            self.state.sfu_output_hash = hashlib.md5(
                out.astype(np.float16).tobytes()
            ).hexdigest()[:16]
            self.state.cycle += length // 128 + 1

        elif op == OpCode.ROPE:
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            length = ops.get("len", 2560)

            # RoPE reads paired (q, k) from SRAM
            inp = self.sram.read_float16(sa, length).astype(np.float32)
            # Split into Q and K portions
            mid = length // 2
            q_in = inp[:mid]
            k_in = inp[mid:mid + mid // 8]  # GQA: 2 KV heads vs 32 Q heads
            q_out, k_out = self.sfu.rope_hw(
                q_in, k_in, position=ops.get("pos", 0),
                num_heads=32, head_dim=128
            )
            out = np.concatenate([q_out, k_out])
            self.sram.write_float16(da, out.astype(np.float16))
            self.state.sfu_output_hash = hashlib.md5(
                out.astype(np.float16).tobytes()
            ).hexdigest()[:16]
            self.state.cycle += length // 128 + 12

        elif op == OpCode.RMSNORM:
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            elements = ops.get("elements", 0)
            if elements <= 0:
                raise ValueError(f"RMSNORM requires elements > 0, got {elements}")

            inp = self.sram.read_float16(sa, elements).astype(np.float32)
            out = self.sfu.rmsnorm_hw(inp)
            self.sram.write_float16(da, out.astype(np.float16))
            self.state.sfu_output_hash = hashlib.md5(
                out.astype(np.float16).tobytes()
            ).hexdigest()[:16]
            self.state.cycle += elements // 128 * 2 + 10  # two-pass: square-sum + normalize

        elif op == OpCode.MAXPOOL:
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            H = ops.get("H", 2)
            W = ops.get("W", 2)
            total = H * W
            out_h, out_w = H // 2, W // 2

            inp = self.sram.read_float16(sa, total).astype(np.float32).reshape(H, W)
            out = np.zeros((out_h, out_w), dtype=np.float32)
            for i in range(out_h):
                for j in range(out_w):
                    window = inp[i*2:i*2+2, j*2:j*2+2]
                    out[i, j] = np.max(window)
            self.sram.write_float16(da, out.astype(np.float16).flatten())
            self.state.cycle += out_h * out_w // 128 + 2

        elif op == OpCode.AVGPOOL:
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            H = ops.get("H", 2)
            W = ops.get("W", 2)
            total = H * W
            out_h, out_w = H // 2, W // 2

            inp = self.sram.read_float16(sa, total).astype(np.float32).reshape(H, W)
            out = np.zeros((out_h, out_w), dtype=np.float32)
            for i in range(out_h):
                for j in range(out_w):
                    window = inp[i*2:i*2+2, j*2:j*2+2]
                    out[i, j] = np.mean(window)
            self.sram.write_float16(da, out.astype(np.float16).flatten())
            self.state.cycle += out_h * out_w // 128 + 2

        elif op == OpCode.DMA_LD:
            # DMA load: DRAM → SRAM (simple mode)
            dram = ops.get("dram", 0)
            sram_addr = ops.get("sram", 0)
            size = ops.get("size", 0)
            self.sram.write_bytes(sram_addr, self.dram[dram:dram + size])
            self.state.cycle += self.dma_model.estimate_transfer(size, "load")

        elif op == OpCode.DMA_ST:
            # DMA store: SRAM → DRAM (simple mode)
            sram_addr = ops.get("sram", 0)
            dram = ops.get("dram", 0)
            size = ops.get("size", 0)
            data = self.sram.read_bytes(sram_addr, size)
            self.dram[dram:dram + size] = data
            self.state.cycle += self.dma_model.estimate_transfer(size, "store")

        elif op == OpCode.DMA_LDD:
            # DMA load via descriptor chain
            desc_addr = ops.get("sa", 0)  # descriptor chain address in DRAM
            # Read 64-bit descriptors until 'last' flag
            offset = 0
            while True:
                raw = self.dram[desc_addr + offset:desc_addr + offset + 8]
                word = int.from_bytes(raw.tobytes(), 'little')
                desc = DMADescriptor.decode(word)
                self.dma.execute_load(self.sram, desc, self.dram)
                self.state.cycle += self.dma_model.estimate_transfer(desc.actual_size, "load")
                offset += 8
                if desc.last:
                    break

        elif op == OpCode.DMA_STD:
            # DMA store via descriptor chain
            desc_addr = ops.get("sa", 0)
            offset = 0
            while True:
                raw = self.dram[desc_addr + offset:desc_addr + offset + 8]
                word = int.from_bytes(raw.tobytes(), 'little')
                desc = DMADescriptor.decode(word)
                self.dma.execute_store(self.sram, desc, self.dram)
                self.state.cycle += self.dma_model.estimate_transfer(desc.actual_size, "store")
                offset += 8
                if desc.last:
                    break

        # ── Vector unit instructions (v2) ─────────────────────────

        elif op == OpCode.VADD:
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            length = ops.get("len", 2560)
            a = self.sram.read_int32(sa, length)
            b = self.sram.read_int32(sa + length * 4, length)
            out = self.vector.add(a, b)
            self.sram.write_int32(da, out)
            self.state.cycle += length // 128 + 1

        elif op == OpCode.VMUL:
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            length = ops.get("len", 2560)
            a = self.sram.read_int32(sa, length)
            b = self.sram.read_int32(sa + length * 4, length)
            out = self.vector.mul(a, b)
            self.sram.write_int32(da, out)
            self.state.cycle += length // 128 + 1

        elif op == OpCode.VRED_MAX:
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            length = ops.get("len", 2560)
            inp = self.sram.read_float16(sa, length).astype(np.float32)
            max_val = self.vector.max_reduce(inp)
            self.sram.write_float16(da, np.array([max_val], dtype=np.float16))
            self.state.cycle += length // 128 + 3  # tree reduction

        elif op == OpCode.VRED_SUM:
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            length = ops.get("len", 2560)
            inp = self.sram.read_float16(sa, length).astype(np.float32)
            sum_val = self.vector.sum_reduce(inp)
            self.sram.write_float16(da, np.array([sum_val], dtype=np.float16))
            self.state.cycle += length // 128 + 3

        elif op == OpCode.VCONV:
            # INT32 → BF16 conversion (MXU→SFU bridge)
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            length = ops.get("len", 2560)
            inp = self.sram.read_int32(sa, length)
            out = self.vector.conv_i32_to_f16(inp)
            self.sram.write_float16(da, out)
            self.state.cycle += length // 128 + 2

        elif op == OpCode.VCONV_F16_I32:
            # FP16 → INT32 conversion (SFU→MXU/Vector bridge)
            sa = ops.get("sa", 0)
            da = ops.get("da", 0)
            length = ops.get("len", 2560)
            inp = self.sram.read_float16(sa, length)
            out = self.vector.conv_f16_to_i32(inp)
            self.sram.write_int32(da, out)
            self.state.cycle += length // 128 + 2

        elif op == OpCode.VRESID:
            # Residual add: da = sa + sb (both INT32 or mixed)
            sa = ops.get("sa", 0)
            sb = ops.get("sb", sa + 1024)  # default offset for second operand
            da = ops.get("da", 0)
            length = ops.get("len", 2560)
            a = self.sram.read_float16(sa, length).astype(np.float32)
            b = self.sram.read_int32(sb, length)
            out = self.vector.residual_add(a, b)
            self.sram.write_int32(da, out)
            self.state.cycle += length // 128 + 1

        elif op in (OpCode.KV_LOAD, OpCode.KV_STORE):
            self.state.cycle += 5

        else:
            raise ValueError(f"Unknown opcode: {op}")

        self.state.pc += 1
        return ExecutorState(**state_before)

    def execute_program(self, program: List[NPUInstruction],
                        auto_insert_dtype_converters: bool = False) -> List[ExecutorState]:
        """Execute full ISA program, return trace of per-instruction states."""
        if auto_insert_dtype_converters:
            return self.run_op_chain(program)
        trace = []
        for instr in program:
            state = self.step(instr)
            trace.append(state)
        return trace

    @staticmethod
    def _get_dst_addr(instr: NPUInstruction) -> Optional[int]:
        return instr.operands.get("da") or instr.operands.get("oa")

    @staticmethod
    def _get_src_addr(instr: NPUInstruction) -> Optional[int]:
        op = instr.opcode
        if op == OpCode.MMUL:
            return instr.operands.get("ia")
        return instr.operands.get("sa")

    @staticmethod
    def _set_src_addr(instr: NPUInstruction, addr: int) -> NPUInstruction:
        new_ops = dict(instr.operands)
        op = instr.opcode
        if op == OpCode.MMUL:
            new_ops["ia"] = addr
        elif op == OpCode.VRESID:
            new_ops["sb"] = addr
        else:
            new_ops["sa"] = addr
        return replace(instr, operands=new_ops)

    @staticmethod
    def _get_element_count(instr: NPUInstruction) -> int:
        ops = instr.operands
        op = instr.opcode
        if op == OpCode.MMUL:
            return ops.get("M", 1) * ops.get("N", 2560)
        if op == OpCode.RMSNORM:
            return ops.get("elements", 0)
        return ops.get("len", 0)

    def _insert_dtype_converters(self, program: List[NPUInstruction],
                                 scratch_base: int = 0x380000) -> List[Any]:
        """Expand *program* by inserting VCONV/VCONV_F16_I32 between mismatched ops.

        For FP16 -> INT8 transitions an explicit INT32 -> INT8 clip/rewrite is
        needed because there is no dedicated INT8 cast opcode; this is handled
        at execution time by :meth:`run_op_chain`.
        """
        expanded: List[Any] = []
        scratch = scratch_base
        prev: Optional[NPUInstruction] = None

        for instr in program:
            if prev is None:
                expanded.append(instr)
                prev = instr
                continue

            prev_out = self._OPCODE_OUTPUT_DTYPE.get(prev.opcode)
            next_in = self._OPCODE_INPUT_DTYPE.get(instr.opcode)
            if prev_out is None or next_in is None or prev_out == next_in:
                expanded.append(instr)
                prev = instr
                continue

            length = self._get_element_count(prev)
            prev_dst = self._get_dst_addr(prev)
            if prev_dst is None:
                raise ValueError(f"Cannot chain from {prev.opcode.name}: no destination address")

            if prev_out == "int32" and next_in == "fp16":
                conv = NPUInstruction(
                    OpCode.VCONV,
                    {"sa": prev_dst, "da": scratch, "len": length},
                )
                expanded.append(conv)
                expanded.append(self._set_src_addr(instr, scratch))
                scratch += ((length * 2 + 63) // 64) * 64

            elif prev_out == "fp16" and next_in == "int32":
                conv = NPUInstruction(
                    OpCode.VCONV_F16_I32,
                    {"sa": prev_dst, "da": scratch, "len": length},
                )
                expanded.append(conv)
                expanded.append(self._set_src_addr(instr, scratch))
                scratch += ((length * 4 + 63) // 64) * 64

            elif prev_out == "fp16" and next_in == "int8":
                temp_int32 = scratch
                scratch += ((length * 4 + 63) // 64) * 64
                int8_addr = scratch
                scratch += ((length * 1 + 63) // 64) * 64
                conv = NPUInstruction(
                    OpCode.VCONV_F16_I32,
                    {"sa": prev_dst, "da": temp_int32, "len": length},
                )
                expanded.append(conv)
                expanded.append(("int8_clip", temp_int32, int8_addr, length))
                expanded.append(self._set_src_addr(instr, int8_addr))

            else:
                raise ValueError(
                    f"Unsupported dtype transition {prev_out} -> {next_in} "
                    f"between {prev.opcode.name} and {instr.opcode.name}"
                )

            prev = instr

        return expanded

    def run_op_chain(self, program: List[NPUInstruction],
                     scratch_base: int = 0x380000) -> List[ExecutorState]:
        """Execute a program with automatic dtype-converter insertion.

        Adjacent ops whose output/input dtypes differ receive an implicit
        VCONV/VCONV_F16_I32 instruction; the converted result is written to
        a scratch SRAM region and the consuming op's source address is updated
        automatically.  No external hex preload is required.
        """
        expanded = self._insert_dtype_converters(program, scratch_base)
        trace = []
        for item in expanded:
            if isinstance(item, tuple) and item[0] == "int8_clip":
                _, src, dst, length = item
                int32 = self.sram.read_int32(src, length)
                int8 = np.clip(int32, -128, 127).astype(np.int8)
                self.sram.write_int8(dst, int8)
                continue
            state = self.step(item)
            trace.append(state)
        return trace

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _estimate_mxu_cycles(M: int, K: int, N: int) -> int:
        """Estimate MXU cycles for tiled matmul (Block 64×64 broadcast engine)."""
        H, W = ARRAY_H, ARRAY_W
        m_tiles = (M + H - 1) // H
        n_tiles = (N + W - 1) // W
        k_tiles = (K + H - 1) // H
        # Broadcast-based block array: all PEs fire in parallel per tile,
        # no pipeline fill/drain. One tile completes in steady state per cycle.
        return m_tiles * n_tiles * k_tiles


# ══════════════════════════════════════════════════════════════════════
# Phase C: Test vector generation + verification framework
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TestVector:
    """Self-contained test case for RTL verification."""
    name: str
    # Inputs
    weight_packed: np.ndarray   # INT4 packed as uint8
    activation: np.ndarray      # INT8
    M: int
    K: int
    N: int
    # Golden outputs (computed by this model)
    golden_int32: np.ndarray = field(default=None)
    golden_hash: str = ""

    def validate(self) -> bool:
        """Check that test vector is self-consistent."""
        expected_w_bytes = (self.K * self.N + 1) // 2
        expected_a_bytes = self.M * self.K
        w_ok = len(self.weight_packed) == expected_w_bytes
        a_ok = self.activation.size == expected_a_bytes
        g_ok = self.golden_int32 is not None and self.golden_int32.shape == (self.M, self.N)
        return w_ok and a_ok and g_ok


def generate_random_test(M: int, K: int, N: int,
                         name: str = "random",
                         seed: int = 42) -> TestVector:
    """Generate a random test vector with golden output."""
    rng = np.random.RandomState(seed)

    # Generate INT4 weights (packed)
    w_int4 = rng.randint(-8, 8, size=K * N, dtype=np.int8)
    w_packed = GoldenMXU.pack_int4(w_int4)

    # Generate INT8 activations
    activation = rng.randint(-128, 128, size=M * K, dtype=np.int8).reshape(M, K)

    # Compute golden output
    mxu = GoldenMXU()
    golden = mxu.matmul_int32(activation, w_packed, M, K, N)

    return TestVector(
        name=name,
        weight_packed=w_packed,
        activation=activation,
        M=M, K=K, N=N,
        golden_int32=golden,
        golden_hash=GoldenMXU.hash_output(golden),
    )


def generate_smoke_tests() -> List[TestVector]:
    """Generate comprehensive smoke test suite."""
    tests = []
    configs = [
        # (M, K, N, description)
        (1, 128, 128, "tiny_tile"),
        (1, 256, 256, "2x2_tiles"),
        (1, 2560, 4096, "Q_proj_decode"),
        (1, 2560, 256, "K_proj_decode"),
        (1, 2560, 9728, "FFN_gate_decode"),
        (1, 9728, 2560, "FFN_down_decode"),
        (4, 2560, 4096, "prefill_batch4"),
        (128, 2560, 256, "prefill_128_KV"),
        (1, 128, 9728, "boundary_tile_narrow_M"),
        (1, 2560, 1, "boundary_tile_narrow_N"),
    ]
    for i, (M, K, N, desc) in enumerate(configs):
        tv = generate_random_test(M, K, N, name=desc, seed=42 + i)
        tests.append(tv)
    return tests


def compare_float32_legacy(golden_int32: np.ndarray, activation: np.ndarray,
                           weight_packed: np.ndarray,
                           M: int, K: int, N: int) -> Dict[str, Any]:
    """Compare new INT32 golden vs old float32-based computation.

    Measures the error introduced by the float32 intermediate casting.
    """
    # Old method (from original golden.py): dequant→float32→matmul→int32
    mxu = GoldenMXU()
    w_flat = mxu.unpack_int4(weight_packed).astype(np.float32)
    if len(w_flat) < K * N:
        w_flat = np.pad(w_flat, (0, K * N - len(w_flat)))
    W = w_flat[:K * N].reshape(K, N)
    A = activation.astype(np.float32)
    legacy = np.matmul(A, W).astype(np.int32)

    # Compare
    diff = np.abs(golden_int32.astype(np.int64) - legacy.astype(np.int64))
    rel = diff / (np.abs(golden_int32.astype(np.float64)) + 1e-8)

    return {
        "max_abs_diff": int(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "max_rel_diff": float(np.max(rel)),
        "num_mismatches": int(np.sum(diff > 0)),
        "total_elements": int(golden_int32.size),
        "mismatch_pct": float(np.sum(diff > 0) / golden_int32.size * 100),
    }


def verify_sfu_precision() -> Dict[str, Any]:
    """Verify SFU hardware-equivalent precision vs float64 reference."""
    sfu = GoldenSFU()
    rng = np.random.RandomState(12345)
    results = {}

    # Softmax: random vectors
    x = rng.randn(100, 2560).astype(np.float32) * 2.0
    for i in range(min(10, len(x))):
        hw = sfu.softmax_hw(x[i])
        ref = sfu.softmax_ref(x[i])
        results[f"softmax_{i}"] = GoldenSFU.compare_hw_vs_ref(hw, ref)

    # GELU: range [-4, 4]
    x = rng.randn(1000).astype(np.float32) * 2.0
    x = np.clip(x, -4, 4)
    hw = sfu.gelu_hw(x)
    ref = sfu.gelu_ref(x)
    results["gelu"] = GoldenSFU.compare_hw_vs_ref(hw, ref, tol_abs=2e-3)

    # SiLU: range [-4, 4]
    hw = sfu.silu_hw(x)
    ref = sfu.silu_ref(x)
    results["silu"] = GoldenSFU.compare_hw_vs_ref(hw, ref, tol_abs=2e-3)

    # LayerNorm
    x = rng.randn(10, 2560).astype(np.float32) * 2.0
    for i in range(5):
        hw = sfu.layernorm_hw(x[i])
        ref = sfu.layernorm_ref(x[i])
        results[f"layernorm_{i}"] = GoldenSFU.compare_hw_vs_ref(hw, ref, tol_abs=1e-2)

    # RoPE
    x_q = rng.randn(4096).astype(np.float32) * 0.5
    x_k = rng.randn(256).astype(np.float32) * 0.5
    hw_q, hw_k = sfu.rope_hw(x_q, x_k, position=42)
    ref_q, ref_k = sfu.rope_ref(x_q, x_k, position=42)
    results["rope_q"] = GoldenSFU.compare_hw_vs_ref(hw_q, ref_q, tol_abs=1e-1)
    results["rope_k"] = GoldenSFU.compare_hw_vs_ref(hw_k, ref_k, tol_abs=1e-1)

    return results


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Golden Executor — NPU RTL verification model")
    sub = parser.add_subparsers(dest="cmd")

    # Smoke tests
    smoke = sub.add_parser("smoke", help="Run MXU precision smoke tests")
    smoke.add_argument("--legacy-compare", action="store_true",
                       help="Compare INT32 golden vs float32 legacy")

    # SFU precision
    sfu_cmd = sub.add_parser("sfu-verify", help="Verify SFU hardware precision")

    # ISA execution
    isa_cmd = sub.add_parser("run", help="Execute ISA program")
    isa_cmd.add_argument("program", help="Path to ISA program file (.isa)")

    # Test vector generation
    gen_cmd = sub.add_parser("gen-test", help="Generate RTL test vectors")
    gen_cmd.add_argument("-o", "--output", default="test_vectors",
                         help="Output directory")
    gen_cmd.add_argument("-M", type=int, default=1)
    gen_cmd.add_argument("-K", type=int, default=2560)
    gen_cmd.add_argument("-N", type=int, default=4096)
    gen_cmd.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.cmd == "smoke":
        print("=" * 60)
        print("MXU INT32 Golden — Smoke Tests")
        print("=" * 60)

        tests = generate_smoke_tests()
        mxu = GoldenMXU()
        all_passed = True

        for tv in tests:
            # Recompute to verify determinism
            golden2 = mxu.matmul_int32(tv.activation, tv.weight_packed, tv.M, tv.K, tv.N)
            exact = np.array_equal(tv.golden_int32, golden2)
            status = "PASS" if exact else "FAIL"
            if not exact:
                all_passed = False
            print(f"  [{status}] {tv.name:30s} M={tv.M:4d} K={tv.K:4d} N={tv.N:4d}  "
                  f"hash={tv.golden_hash}")

            if args.legacy_compare:
                cmp = compare_float32_legacy(tv.golden_int32, tv.activation,
                                             tv.weight_packed, tv.M, tv.K, tv.N)
                if cmp["max_abs_diff"] > 0:
                    print(f"         Legacy diff: max={cmp['max_abs_diff']} "
                          f"mean={cmp['mean_abs_diff']:.2f} "
                          f"mismatch={cmp['mismatch_pct']:.1f}%")

        print(f"\n  {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    elif args.cmd == "sfu-verify":
        print("=" * 60)
        print("SFU Hardware-Equivalent — Precision Verification")
        print("=" * 60)

        results = verify_sfu_precision()
        for name, r in results.items():
            status = "PASS" if r["within_tolerance"] else "FAIL"
            print(f"  [{status}] {name:20s}  max_abs={r['max_abs_err']:.2e}  "
                  f"max_rel={r['max_rel_err']:.2e}")

    elif args.cmd == "run":
        # Read ISA program
        text = Path(args.program).read_text()
        from engine.isa import parse_isa_program
        program = parse_isa_program(text)

        executor = GoldenExecutor()
        print(f"Executing {len(program)} instructions...")
        trace = executor.execute_program(program)

        print(f"\nFinal SRAM checksum: {hashlib.md5(executor.sram.data.tobytes()).hexdigest()[:16]}")
        print(f"Total cycles: {executor.state.cycle}")
        print(f"\nInstruction trace:")
        for i, state in enumerate(trace):
            print(f"  [{i:3d}] cycle={state['cycle']:6d}  "
                  f"sram={state['sram_checksum']}  "
                  f"mxu={state['mxu_output_hash'] or '-':16s}  "
                  f"sfu={state['sfu_output_hash'] or '-':16s}")

    elif args.cmd == "gen-test":
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        tv = generate_random_test(args.M, args.K, args.N, seed=args.seed)

        # Write hex files (RTL $readmemh compatible)
        # Weights: INT4 packed as uint8 → 2 hex digits per byte
        w_hex = out_dir / "weight.hex"
        with open(w_hex, "w") as f:
            for b in tv.weight_packed:
                f.write(f"{b:02x}\n")

        # Activations: INT8 → 2 hex digits
        a_hex = out_dir / "activation.hex"
        with open(a_hex, "w") as f:
            for v in tv.activation.flatten():
                # INT8 as unsigned hex (two's complement preserved for negative)
                f.write(f"{v & 0xFF:02x}\n")

        # Golden output: INT32 → 8 hex digits
        g_hex = out_dir / "golden.hex"
        with open(g_hex, "w") as f:
            for v in tv.golden_int32.flatten():
                f.write(f"{v & 0xFFFFFFFF:08x}\n")

        # Manifest
        import json
        manifest = {
            "name": tv.name,
            "M": tv.M, "K": tv.K, "N": tv.N,
            "golden_hash": tv.golden_hash,
            "files": {
                "weight": "weight.hex",
                "activation": "activation.hex",
                "golden": "golden.hex",
            },
            "format": {
                "weight": "INT4 packed, 2 per byte, low nibble first",
                "activation": "INT8, 1 byte per value",
                "golden": "INT32, little-endian, 4 bytes per value",
            },
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Generated test vector: {out_dir}")
        print(f"  M={tv.M}, K={tv.K}, N={tv.N}")
        print(f"  Golden hash: {tv.golden_hash}")
        print(f"  Files: weight.hex ({len(tv.weight_packed)}B), "
              f"activation.hex ({tv.activation.size}B), "
              f"golden.hex ({tv.golden_int32.size * 4}B)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
