"""Stable Diffusion 1.5 UNet trace — single denoising step.

Models the UNet architecture used in SD 1.5:
  - Input: 64×64 latent (4096 tokens), 4 input channels
  - Cross-attention with CLIP text embeddings (77 tokens)
  - Down blocks (ResNet + self-attention) → middle block → up blocks
  - Total: ~850 GMACs per denoising step (~1.7 TFLOPs)
  - Typical generation: 20-50 steps

Reference: Rombach et al., "High-Resolution Image Synthesis with
Latent Diffusion Models", CVPR 2022.
"""

from __future__ import annotations

import math
from typing import Any

_SFU_WIDTH = 128

# ── SD 1.5 UNet topology ──
_LATENT_H = 64
_LATENT_W = 64
_LATENT_TOKENS = _LATENT_H * _LATENT_W  # 4096
_TEXT_TOKENS = 77  # CLIP text encoder output

# UNet channel dimensions (progressively downsampled)
_CHANNELS = [320, 640, 1280, 1280]  # per resolution level
_HIDDEN_DIMS = [320, 640, 1280, 1280]
_ATTN_HEAD_DIM = 64
_NUM_HEADS = [5, 10, 20, 20]  # channels / head_dim

# Cross-attention dimension
_CROSS_ATTN_DIM = 768  # CLIP text embedding dim

# Block counts per resolution level
_RESNET_BLOCKS = [2, 2, 2, 2]  # per down/up level
_ATTN_RESOLUTIONS = [2, 3]  # which levels have attention (0-indexed: 640, 1280)


def _gemm(name: str, M: int, K: int, N: int) -> dict[str, Any]:
    return {"type": "gemm", "name": name, "M": M, "K": K, "N": N,
            "im2col_overhead_cycles": 0, "sfu_cycles": 0}


def _conv_im2col(name: str, M: int, K: int, N: int,
                 overhead: float = 0) -> dict[str, Any]:
    """3×3 convolution mapped to GEMM via im2col."""
    return {"type": "conv", "name": name, "M": M, "K": K, "N": N,
            "im2col_overhead_cycles": overhead if overhead else M * K * 0.05,
            "sfu_cycles": 0}


def _sfu(name: str, element_count: int, op_type: str = "sfu") -> dict[str, Any]:
    return {"type": op_type, "name": name, "M": 0, "K": 0, "N": 0,
            "im2col_overhead_cycles": 0,
            "sfu_cycles": math.ceil(element_count / _SFU_WIDTH)}


def _resnet_block(prefix: str, tokens: int, in_ch: int, out_ch: int,
                  time_emb_dim: int = 1280) -> list[dict[str, Any]]:
    """ResNet block with time embedding."""
    ops = []
    # GroupNorm + SiLU
    ops.append(_sfu(f"{prefix}_gn1", tokens * in_ch, "group_norm"))
    ops.append(_sfu(f"{prefix}_silu1", tokens * in_ch, "silu"))
    # 3×3 conv: im2col maps spatial to GEMM
    k = 3 * 3 * in_ch  # im2col kernel dimension
    ops.append(_conv_im2col(f"{prefix}_conv1", tokens, k, out_ch))
    # Time embedding projection → add
    ops.append(_gemm(f"{prefix}_time_proj", tokens, time_emb_dim, out_ch))
    # GroupNorm + SiLU
    ops.append(_sfu(f"{prefix}_gn2", tokens * out_ch, "group_norm"))
    ops.append(_sfu(f"{prefix}_silu2", tokens * out_ch, "silu"))
    # 3×3 conv (zero init → skip connection handled by caller)
    k2 = 3 * 3 * out_ch
    ops.append(_conv_im2col(f"{prefix}_conv2", tokens, k2, out_ch))
    # Skip connection 1×1 conv if in_ch != out_ch
    if in_ch != out_ch:
        ops.append(_gemm(f"{prefix}_skip_conv", tokens, in_ch, out_ch))
    return ops


def _self_attn_block(prefix: str, tokens: int, channels: int,
                     num_heads: int) -> list[dict[str, Any]]:
    """Self-attention block (used at 640 and 1280 channel levels)."""
    ops = []
    head_dim = channels // num_heads
    # GroupNorm
    ops.append(_sfu(f"{prefix}_gn", tokens * channels, "group_norm"))
    # QKV projection (combined)
    ops.append(_gemm(f"{prefix}_qkv", tokens, channels, 3 * channels))
    # Attention softmax
    ops.append(_sfu(f"{prefix}_softmax",
                   num_heads * tokens * tokens, "softmax"))
    # Output projection
    ops.append(_gemm(f"{prefix}_o_proj", tokens, channels, channels))
    return ops


def _cross_attn_block(prefix: str, latent_tokens: int, text_tokens: int,
                      channels: int, cross_dim: int,
                      num_heads: int) -> list[dict[str, Any]]:
    """Cross-attention: latent queries attend to text embeddings."""
    ops = []
    head_dim = channels // num_heads
    # GroupNorm
    ops.append(_sfu(f"{prefix}_gn", latent_tokens * channels, "group_norm"))
    # Q from latent, KV from text
    ops.append(_gemm(f"{prefix}_q", latent_tokens, channels, channels))
    ops.append(_gemm(f"{prefix}_kv_text", text_tokens, cross_dim, 2 * channels))
    # Attention softmax (latent × text)
    ops.append(_sfu(f"{prefix}_softmax",
                   num_heads * latent_tokens * text_tokens, "softmax"))
    # Output projection
    ops.append(_gemm(f"{prefix}_o_proj", latent_tokens, channels, channels))
    return ops


def generate_sd_unet_step() -> list[dict[str, Any]]:
    """Generate a single SD 1.5 UNet denoising step trace.

    Returns ~850 GMACs trace suitable for the CV simulator.
    """
    trace: list[dict[str, Any]] = []

    # ── Input projection: 4ch → 320ch ──
    trace.append(_gemm("input_proj", _LATENT_TOKENS, 4, 320))

    # Time embedding (simplified: one GEMM)
    trace.append(_gemm("time_embed", 1, 1280, _LATENT_TOKENS))

    # ── Down blocks ──
    spatial_tokens = _LATENT_TOKENS  # 4096
    for level in range(4):
        in_ch = _CHANNELS[level - 1] if level > 0 else 320
        out_ch = _CHANNELS[level]

        # ResNet blocks at this level
        for b in range(_RESNET_BLOCKS[level]):
            ch_in = in_ch if b == 0 else out_ch
            trace.extend(_resnet_block(
                f"down_l{level}_b{b}", spatial_tokens, ch_in, out_ch))

        # Self-attention + cross-attention at specific levels
        if level in _ATTN_RESOLUTIONS:
            trace.extend(_self_attn_block(
                f"down_l{level}_selfattn", spatial_tokens, out_ch,
                _NUM_HEADS[level]))
            trace.extend(_cross_attn_block(
                f"down_l{level}_crossattn", spatial_tokens, _TEXT_TOKENS,
                out_ch, _CROSS_ATTN_DIM, _NUM_HEADS[level]))

        # Downsample (except last level): 3×3 stride-2 conv
        # Reduces spatial tokens by 4×
        if level < 3:
            trace.append(_sfu(f"down_l{level}_downsample",
                            spatial_tokens * out_ch, "silu"))
            new_tokens = spatial_tokens // 4
            k = 3 * 3 * out_ch
            trace.append(_conv_im2col(
                f"down_l{level}_down_conv", new_tokens, k, out_ch))
            spatial_tokens = new_tokens

    # ── Middle block ──
    mid_ch = _CHANNELS[-1]  # 1280
    trace.extend(_resnet_block("mid_b0", spatial_tokens, mid_ch, mid_ch))
    trace.extend(_self_attn_block("mid_selfattn", spatial_tokens, mid_ch,
                                  _NUM_HEADS[-1]))
    trace.extend(_cross_attn_block("mid_crossattn", spatial_tokens,
                                   _TEXT_TOKENS, mid_ch, _CROSS_ATTN_DIM,
                                   _NUM_HEADS[-1]))
    trace.extend(_resnet_block("mid_b1", spatial_tokens, mid_ch, mid_ch))

    # ── Up blocks ──
    for level in range(3, -1, -1):
        in_ch = _CHANNELS[level + 1] if level < 3 else _CHANNELS[3]
        out_ch = _CHANNELS[level]

        # Upsample (except last level)
        if level < 3:
            new_tokens = spatial_tokens * 4
            trace.append(_sfu(f"up_l{level}_upsample",
                            spatial_tokens * in_ch, "silu"))
            k = 3 * 3 * in_ch
            trace.append(_conv_im2col(
                f"up_l{level}_up_conv", new_tokens, k, in_ch))
            spatial_tokens = new_tokens

        # Concatenated skip → double channels for first ResNet
        skip_ch = out_ch  # skip connection channels
        cat_ch = in_ch + skip_ch

        # ResNet blocks
        for b in range(_RESNET_BLOCKS[level] + 1):
            ch_in = cat_ch if b == 0 else out_ch
            trace.extend(_resnet_block(
                f"up_l{level}_b{b}", spatial_tokens, ch_in, out_ch))

        # Attention at specific levels
        if level in _ATTN_RESOLUTIONS:
            trace.extend(_self_attn_block(
                f"up_l{level}_selfattn", spatial_tokens, out_ch,
                _NUM_HEADS[level]))
            trace.extend(_cross_attn_block(
                f"up_l{level}_crossattn", spatial_tokens, _TEXT_TOKENS,
                out_ch, _CROSS_ATTN_DIM, _NUM_HEADS[level]))

    # ── Output: GroupNorm + SiLU + 3×3 conv → 4ch ──
    trace.append(_sfu("output_gn", _LATENT_TOKENS * 320, "group_norm"))
    trace.append(_sfu("output_silu", _LATENT_TOKENS * 320, "silu"))
    k_out = 3 * 3 * 320
    trace.append(_conv_im2col("output_conv", _LATENT_TOKENS, k_out, 4))

    # ── Validation ──
    total_macs = sum(e["M"] * e["K"] * e["N"] for e in trace)
    assert 150_000_000_000 <= total_macs <= 500_000_000_000, (
        f"SD UNet step MACs {total_macs:,} outside expected [150G, 500G]"
    )

    return trace


if __name__ == "__main__":
    t = generate_sd_unet_step()
    total_macs = sum(e["M"] * e["K"] * e["N"] for e in t)
    gemm_entries = [e for e in t if e["type"] in ("gemm", "conv")]
    sfu_entries = [e for e in t if e["type"] not in ("gemm", "conv")]
    conv_entries = [e for e in t if e["type"] == "conv"]
    print(f"SD 1.5 UNet (1 denoising step): {len(t)} entries")
    print(f"  GEMM: {len(gemm_entries) - len(conv_entries)}, "
          f"Conv: {len(conv_entries)}, SFU: {len(sfu_entries)}")
    print(f"  MACs: {total_macs:,} ({total_macs/1e9:.1f} G) "
          f"≈ {total_macs*2/1e12:.1f} TFLOPs")
    print(f"  @ 49 TOPS: {49000/(total_macs*2/1e12):.1f} steps/s")
    print(f"  20 steps: {20/(49000/(total_macs*2/1e12)):.1f}s")
    print(f"  50 steps: {50/(49000/(total_macs*2/1e12)):.1f}s")
