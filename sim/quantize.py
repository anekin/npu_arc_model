#!/usr/bin/env python3
"""Per-channel INT4 quantization — hardware-matched scheme.

Produces packed INT4 weights + per-channel FP16 scales.
Matches hardware: each output channel gets its own scale.
"""

import numpy as np


def quantize_int4_per_channel(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize float32 weight matrix to INT4 per-channel.

    Args:
        W: float32 weight matrix, shape (K, N) where
           K = input features, N = output features (channels)

    Returns:
        packed: uint8 packed INT4 array, shape (K * N // 2,)
        scales: float32 per-channel scales, shape (N,)
        dequant_ref: float32 reconstruction (K, N) — for error measurement
    """
    K, N = W.shape
    W_f32 = W.astype(np.float32)

    # Per-channel: one scale per output channel (column)
    # scale[c] = max(abs(W[:, c])) / 7.0
    # quantized = round(W[:, c] / scale[c]), clamped to [-7, 7]
    scales = np.empty(N, dtype=np.float32)
    quantized = np.empty((K, N), dtype=np.int8)

    for c in range(N):
        col = W_f32[:, c]
        max_abs = np.max(np.abs(col))
        if max_abs < 1e-12:
            scales[c] = 1.0
            quantized[:, c] = 0
        else:
            scales[c] = max_abs / 7.0
            q = np.clip(np.round(col / scales[c]), -7, 7).astype(np.int8)
            quantized[:, c] = q

    # Pack: 2 INT4 values per uint8 (low nibble first)
    flat = quantized.flatten()
    if len(flat) % 2 != 0:
        flat = np.append(flat, 0)
    unsigned = np.where(flat < 0, flat + 16, flat).astype(np.uint8)
    packed = np.empty(len(flat) // 2, dtype=np.uint8)
    packed = (unsigned[0::2] & 0x0F) | ((unsigned[1::2] & 0x0F) << 4)

    # Dequant for reference
    dequant = (quantized.astype(np.float32) * scales[np.newaxis, :]).astype(np.float32)

    return packed, scales, dequant


def dequantize_int4_per_channel(packed: np.ndarray, scales: np.ndarray, K: int, N: int) -> np.ndarray:
    """Dequantize per-channel INT4 back to float32."""
    # Unpack INT4
    flat_unsigned = np.empty(len(packed) * 2, dtype=np.uint8)
    flat_unsigned[0::2] = packed & 0x0F
    flat_unsigned[1::2] = (packed >> 4) & 0x0F
    # Sign-extend
    flat_signed = np.where(flat_unsigned > 7, flat_unsigned.astype(np.int16) - 16,
                           flat_unsigned.astype(np.int16)).astype(np.int8)
    flat_signed = flat_signed[:K * N]

    quantized = flat_signed.reshape(K, N)
    dequant = (quantized.astype(np.float32) * scales[np.newaxis, :]).astype(np.float32)
    return dequant


# ══════════════════════════════════════════════════════════════════════
# Per-block INT4 quantization (group_size along K dimension)
# ══════════════════════════════════════════════════════════════════════

def quantize_int4_per_block(W: np.ndarray, group_size: int = 128
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize float32 weight matrix to INT4 per-block.

    Weights are divided into blocks of `group_size` along the K dimension.
    Each block has N per-channel scales (one per output column).
    This isolates outlier damage: a weight outlier only degrades its block,
    not the entire channel.

    Industry standard: group_size=128 (TensorRT, GPTQ, AWQ).

    Args:
        W: float32 weight matrix, shape (K, N)
        group_size: block size along K dimension (default 128)

    Returns:
        packed: uint8 packed INT4 array, shape (K * N // 2,)
        scales: float32 block scales, shape (num_blocks, N)
        dequant_ref: float32 reconstruction (K, N) — for error measurement
    """
    K, N = W.shape
    W_f32 = W.astype(np.float32)
    num_blocks = (K + group_size - 1) // group_size

    scales = np.empty((num_blocks, N), dtype=np.float32)
    quantized = np.empty((K, N), dtype=np.int8)

    for b in range(num_blocks):
        k_start = b * group_size
        k_end = min(k_start + group_size, K)
        block = W_f32[k_start:k_end, :]  # (block_size, N)

        for c in range(N):
            col = block[:, c]
            max_abs = np.max(np.abs(col))
            if max_abs < 1e-12:
                scales[b, c] = 1.0
                quantized[k_start:k_end, c] = 0
            else:
                scales[b, c] = max_abs / 7.0
                q = np.clip(np.round(col / scales[b, c]), -7, 7).astype(np.int8)
                quantized[k_start:k_end, c] = q

    # Pack
    flat = quantized.flatten()
    if len(flat) % 2 != 0:
        flat = np.append(flat, 0)
    unsigned = np.where(flat < 0, flat + 16, flat).astype(np.uint8)
    packed = np.empty(len(flat) // 2, dtype=np.uint8)
    packed = (unsigned[0::2] & 0x0F) | ((unsigned[1::2] & 0x0F) << 4)

    # Dequant for reference: apply block scales
    dequant = np.empty((K, N), dtype=np.float32)
    for b in range(num_blocks):
        k_start = b * group_size
        k_end = min(k_start + group_size, K)
        dequant[k_start:k_end, :] = (
            quantized[k_start:k_end, :].astype(np.float32)
            * scales[b, np.newaxis, :]
        )

    return packed, scales, dequant
