"""
ViT-B/16 CV trace generator.

Produces a GEMM-only trace for the Vision Transformer (ViT-B/16) architecture:
  - Patch embedding is a single dense GEMM (no im2col).
  - Each transformer block exposes Q/K/V/O projections and the MLP pair as GEMMs.
  - LayerNorm, Softmax and GELU are represented as SFU-only entries with zero
    GEMM dimensions.

Trace entries follow the schema defined in ``sim.cv.cv_trace``:

    {
        "type": str,
        "name": str,
        "M": int,
        "K": int,
        "N": int,
        "im2col_overhead_cycles": 0,
        "sfu_cycles": int,
    }
"""

from __future__ import annotations

import math
from typing import Any

# SFU width (elements per cycle), aligned with sim/config/npu_config.yaml.
_SFU_WIDTH = 128

# ViT-B/16 topology constants.
_IMAGE_SIZE = 224
_PATCH_SIZE = 16
_IN_CHANNELS = 3
_HIDDEN = 768
_MLP_HIDDEN = 3072
_NUM_HEADS = 12
_NUM_LAYERS = 12
_NUM_CLASSES = 1000

# Derived sequence / shape values.
_NUM_PATCHES = (_IMAGE_SIZE // _PATCH_SIZE) ** 2  # 196
_SEQ_LEN = _NUM_PATCHES + 1  # 197, includes the learned class token
_PATCH_EMBED_K = _PATCH_SIZE * _PATCH_SIZE * _IN_CHANNELS  # 768


def _gemm(name: str, M: int, K: int, N: int) -> dict[str, Any]:
    """Build a ``type="gemm"`` trace entry."""
    return {
        "type": "gemm",
        "name": name,
        "M": M,
        "K": K,
        "N": N,
        "im2col_overhead_cycles": 0,
        "sfu_cycles": 0,
    }


def _sfu(name: str, element_count: int, op_type: str = "sfu") -> dict[str, Any]:
    """Build an SFU-only trace entry with zero GEMM dimensions."""
    return {
        "type": op_type,
        "name": name,
        "M": 0,
        "K": 0,
        "N": 0,
        "im2col_overhead_cycles": 0,
        "sfu_cycles": math.ceil(element_count / _SFU_WIDTH),
    }


def generate_vit_trace() -> list[dict[str, Any]]:
    """Generate a ViT-B/16 trace as a list of accelerator entries.

    Returns
    -------
    list[dict[str, Any]]
        Ordered trace entries.  All compute-heavy layers are ``type="gemm"``;
        normalization / activation / softmax layers are SFU-only entries with
        ``M=K=N=0``.

    Raises
    ------
    AssertionError
        If the total GEMM MAC count is outside the expected ViT-B/16 range
        [16 G, 19 G].
    """
    trace: list[dict[str, Any]] = []

    # ---- Patch embedding ----------------------------------------------------
    # Flattened patches: (196, 768) x (768, 768) -> (196, 768)
    trace.append(_gemm("patch_embed", _NUM_PATCHES, _PATCH_EMBED_K, _HIDDEN))

    # ---- Transformer blocks -------------------------------------------------
    for layer_idx in range(_NUM_LAYERS):
        prefix = f"block{layer_idx}"

        # LayerNorm before self-attention.
        trace.append(_sfu(f"{prefix}_ln1", _SEQ_LEN * _HIDDEN, "layer_norm"))

        # Q/K/V/O projections all operate on the full sequence incl. class token.
        trace.append(_gemm(f"{prefix}_q_proj", _SEQ_LEN, _HIDDEN, _HIDDEN))
        trace.append(_gemm(f"{prefix}_k_proj", _SEQ_LEN, _HIDDEN, _HIDDEN))
        trace.append(_gemm(f"{prefix}_v_proj", _SEQ_LEN, _HIDDEN, _HIDDEN))

        # Attention softmax over the key dimension (heads x seq x seq).
        trace.append(
            _sfu(
                f"{prefix}_attn_softmax",
                _NUM_HEADS * _SEQ_LEN * _SEQ_LEN,
                "softmax",
            )
        )

        trace.append(_gemm(f"{prefix}_o_proj", _SEQ_LEN, _HIDDEN, _HIDDEN))

        # LayerNorm before MLP.
        trace.append(_sfu(f"{prefix}_ln2", _SEQ_LEN * _HIDDEN, "layer_norm"))

        # MLP expansion + GELU + projection.
        trace.append(_gemm(f"{prefix}_mlp_expand", _SEQ_LEN, _HIDDEN, _MLP_HIDDEN))
        trace.append(_sfu(f"{prefix}_mlp_gelu", _SEQ_LEN * _MLP_HIDDEN, "gelu"))
        trace.append(_gemm(f"{prefix}_mlp_proj", _SEQ_LEN, _MLP_HIDDEN, _HIDDEN))

    # ---- Final layer norm + classifier head ---------------------------------
    trace.append(_sfu("final_ln", _SEQ_LEN * _HIDDEN, "layer_norm"))
    trace.append(_gemm("classifier", 1, _HIDDEN, _NUM_CLASSES))

    # ---- Validation ---------------------------------------------------------
    total_macs = sum(entry["M"] * entry["K"] * entry["N"] for entry in trace)
    assert 16_000_000_000 <= total_macs <= 19_000_000_000, (
        f"ViT-B/16 total MACs {total_macs:,} outside expected range [16G, 19G]"
    )

    return trace


if __name__ == "__main__":
    t = generate_vit_trace()
    total_macs = sum(e["M"] * e["K"] * e["N"] for e in t)
    gemm_entries = [e for e in t if e["type"] == "gemm"]
    sfu_entries = [e for e in t if e["type"] != "gemm"]
    print(f"Generated ViT-B/16 trace with {len(t)} entries")
    print(f"  GEMM entries: {len(gemm_entries)}")
    print(f"  SFU entries:  {len(sfu_entries)}")
    print(f"  Total MACs:   {total_macs:,} ({total_macs / 1e9:.2f} G)")
