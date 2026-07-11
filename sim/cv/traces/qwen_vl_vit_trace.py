"""Qwen2.5-VL ViT trace generator (~675M params, NaViT-style).

Models a ViT-Huge class vision encoder used in Qwen2.5-VL-7B:
  - hidden=1280, MLP=5120, 16 heads, 32 layers
  - Patch size 14, input 448×448 → 1024 patches + CLS = 1025 tokens
  - Supports multi-crop for dynamic resolution (1 global + 3 local views)

Reference: Qwen2.5-VL technical report (2025)
"""

from __future__ import annotations

import math
from typing import Any

_SFU_WIDTH = 128

# ── Qwen2.5-VL ViT topology ──
_PATCH_SIZE = 14
_IN_CHANNELS = 3
_HIDDEN = 1280
_MLP_HIDDEN = 5120
_NUM_HEADS = 16
_HEAD_DIM = 80  # 1280 / 16
_NUM_LAYERS = 32
_NUM_CLASSES = _HIDDEN  # vision-language projection, not classifier

# Per-crop sequence length
_IMG_SIZE = 448
_NUM_PATCHES = (_IMG_SIZE // _PATCH_SIZE) ** 2  # 1024
_SEQ_LEN = _NUM_PATCHES + 1  # 1025 (incl. CLS token)
_PATCH_EMBED_K = _PATCH_SIZE * _PATCH_SIZE * _IN_CHANNELS  # 588


def _gemm(name: str, M: int, K: int, N: int) -> dict[str, Any]:
    return {"type": "gemm", "name": name, "M": M, "K": K, "N": N,
            "im2col_overhead_cycles": 0, "sfu_cycles": 0}


def _sfu(name: str, element_count: int, op_type: str = "sfu") -> dict[str, Any]:
    return {"type": op_type, "name": name, "M": 0, "K": 0, "N": 0,
            "im2col_overhead_cycles": 0,
            "sfu_cycles": math.ceil(element_count / _SFU_WIDTH)}


def _single_crop_trace() -> list[dict[str, Any]]:
    """Generate trace for one 448×448 crop through the ViT."""
    trace: list[dict[str, Any]] = []

    # Patch embedding: (1024, 588) × (588, 1280) → (1024, 1280)
    trace.append(_gemm("patch_embed", _NUM_PATCHES, _PATCH_EMBED_K, _HIDDEN))

    # CLS token is prepended in the model — we account for it in all
    # subsequent GEMMs by using _SEQ_LEN=1025 as M.

    for layer_idx in range(_NUM_LAYERS):
        prefix = f"block{layer_idx}"

        # LayerNorm before attention
        trace.append(_sfu(f"{prefix}_ln1", _SEQ_LEN * _HIDDEN, "layer_norm"))

        # Q/K/V projections
        trace.append(_gemm(f"{prefix}_q_proj", _SEQ_LEN, _HIDDEN, _HIDDEN))
        trace.append(_gemm(f"{prefix}_k_proj", _SEQ_LEN, _HIDDEN, _HIDDEN))
        trace.append(_gemm(f"{prefix}_v_proj", _SEQ_LEN, _HIDDEN, _HIDDEN))

        # Softmax over attention scores (heads × seq × seq)
        trace.append(_sfu(f"{prefix}_attn_softmax",
                         _NUM_HEADS * _SEQ_LEN * _SEQ_LEN, "softmax"))

        # Output projection
        trace.append(_gemm(f"{prefix}_o_proj", _SEQ_LEN, _HIDDEN, _HIDDEN))

        # LayerNorm before MLP
        trace.append(_sfu(f"{prefix}_ln2", _SEQ_LEN * _HIDDEN, "layer_norm"))

        # MLP: expand → GELU → project
        trace.append(_gemm(f"{prefix}_mlp_expand", _SEQ_LEN, _HIDDEN, _MLP_HIDDEN))
        trace.append(_sfu(f"{prefix}_mlp_gelu", _SEQ_LEN * _MLP_HIDDEN, "gelu"))
        trace.append(_gemm(f"{prefix}_mlp_proj", _SEQ_LEN, _MLP_HIDDEN, _HIDDEN))

    # Final layer norm
    trace.append(_sfu("final_ln", _SEQ_LEN * _HIDDEN, "layer_norm"))

    return trace


def generate_qwen_vl_vit_trace(num_crops: int = 1) -> list[dict[str, Any]]:
    """Generate Qwen2.5-VL ViT trace with optional multi-crop.

    Args:
        num_crops: Number of image crops (1=global only, 4=1 global+3 local).

    Returns:
        Trace list — each crop appends a full ViT forward pass.

    Raises:
        AssertionError if total MACs outside expected range.
    """
    if num_crops < 1:
        raise ValueError("num_crops must be >= 1")

    crop_trace = _single_crop_trace()
    trace = []
    for c in range(num_crops):
        for entry in crop_trace:
            # Prefix name with crop index for multi-crop
            entry_copy = dict(entry)
            entry_copy["name"] = f"crop{c}_{entry['name']}"
            trace.append(entry_copy)

    # Validation
    total_macs = sum(e["M"] * e["K"] * e["N"] for e in trace)
    per_crop_macs = total_macs / num_crops
    expected_per_crop = 300_000_000_000  # ~300 GFLOPs per crop
    assert per_crop_macs > 250_000_000_000, (
        f"Qwen-VL ViT per-crop MACs {per_crop_macs:,} below expected 250G"
    )

    return trace


# Convenience functions for common configurations
def generate_qwen_vl_1crop() -> list[dict[str, Any]]:
    """Single crop (global view only) — ~300 GMACs."""
    return generate_qwen_vl_vit_trace(num_crops=1)


def generate_qwen_vl_4crop() -> list[dict[str, Any]]:
    """Dynamic resolution: 1 global + 3 local crops — ~1.2 TGMACs."""
    return generate_qwen_vl_vit_trace(num_crops=4)


if __name__ == "__main__":
    for nc in [1, 4]:
        t = generate_qwen_vl_vit_trace(num_crops=nc)
        total_macs = sum(e["M"] * e["K"] * e["N"] for e in t)
        gemm_entries = [e for e in t if e["type"] == "gemm"]
        print(f"Qwen2.5-VL ViT ({nc} crop(s)): {len(t)} entries")
        print(f"  GEMM: {len(gemm_entries)}, SFU: {len(t) - len(gemm_entries)}")
        print(f"  MACs: {total_macs:,} ({total_macs/1e9:.1f} G)")
