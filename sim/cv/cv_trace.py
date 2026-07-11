"""
CV Trace Generator — MobileNetV3-Small trace JSON using ONNX importer and conv mapper.

Generates a structured trace that the Caduceus NPU simulator (Task 8) can consume,
mapping each layer to its GEMM dimensions (via ``map_conv_to_gemm``) and SFU cycles.

Trace entries follow a strict schema:

.. code-block:: python

    {
        "type": "pointwise_conv" | "depthwise_conv" | "hard_swish"
                | "hard_sigmoid" | "add" | "mul" | "global_avg_pool"
                | "gemm" | "relu" | "reduce_mean" | "reshape"
                | "shape" | "concat",
        "name": str,
        "M": int,           # GEMM M dimension (0 for non-conv / non-gemm)
        "K": int,           # GEMM K dimension
        "N": int,           # GEMM N dimension
        "im2col_overhead_cycles": float,  # DMA cycles for im2col (0 if not conv)
        "sfu_cycles": int,  # SFU cycles (0 if not SFU op)
    }
"""

from __future__ import annotations

import json
import math
from typing import Any
import os as _os
import sys as _sys

# ---- Path setup: ensure CaduceusCore/sim is importable when run directly ---
_sim_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _sim_dir not in _sys.path:
    _sys.path.insert(0, _sim_dir)
# --------------------------------------------------------------------------------

from cv.conv_mapper import map_conv_to_gemm
from cv.onnx_importer import import_mobilenetv3

# ---------------------------------------------------------------------------
# Constants (aligned with sim/config/npu_config.yaml)
# ---------------------------------------------------------------------------

# SFU processes this many elements per cycle (npu_config.yaml: sfu.width)
SFU_WIDTH = 128

# ---------------------------------------------------------------------------
# Type mapping: ONNX op type -> trace entry type
# ---------------------------------------------------------------------------

TYPE_MAP: dict[str, str] = {
    "HardSwish": "hard_swish",
    "HardSigmoid": "hard_sigmoid",
    "Relu": "relu",
    "Add": "add",
    "Mul": "mul",
    "GlobalAveragePool": "global_avg_pool",
    "ReduceMean": "global_avg_pool",
    "Reshape": "reshape",
    "Squeeze": "reshape",
    "Shape": "shape",
    "Concat": "concat",
}

# Set of trace types that are executed on the SFU.
SFU_TYPES: frozenset[str] = frozenset({"hard_swish", "hard_sigmoid", "relu"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _element_count(shape: list[int] | None) -> int:
    """Compute total element count from a tensor shape.

    Skips symbolic dimensions (value == 0) by treating them as 1, since
    the batch axis is always 1 at inference time.
    """
    if shape is None:
        return 0
    n = 1
    for d in shape:
        n *= max(d, 1)
    return n


def _compute_sfu_cycles(shape: list[int] | None) -> int:
    """Estimate SFU cycles from the output tensor element count.

    The SFU processes ``SFU_WIDTH`` elements per cycle (fully pipelined),
    so the number of cycles is simply the number of 128-element batches.
    """
    return math.ceil(_element_count(shape) / SFU_WIDTH)


def _conv_from_layer(layer: dict[str, Any]) -> dict[str, Any]:
    """Build a trace entry for a Conv (pointwise or depthwise) layer.

    Extracts all spatial / channel parameters from the importer's layer dict
    and delegates to ``map_conv_to_gemm`` for GEMM dimensions and im2col
    overhead.
    """
    in_shape = layer["in_shape"]
    out_shape = layer["out_shape"]

    C_in = in_shape[1] if in_shape and len(in_shape) >= 2 else 1
    C_out = out_shape[1] if out_shape and len(out_shape) >= 2 else 1
    H = in_shape[2] if in_shape and len(in_shape) >= 3 else 1
    W = in_shape[3] if in_shape and len(in_shape) >= 4 else 1

    kernel = layer.get("kernel")
    K = kernel[0] if kernel else 1

    stride_arr = layer.get("stride", [1, 1])
    stride = stride_arr[0] if stride_arr else 1

    pad_arr = layer.get("padding", [0, 0, 0, 0])
    pad = pad_arr[0] if pad_arr else 0

    groups = layer.get("groups", 1)

    result = map_conv_to_gemm(
        C_in, C_out, H, W, K, stride=stride, pad=pad, groups=groups,
    )

    # Determine trace type. The importer already marks depthwise convs
    # with type "depthwise_conv"; everything else -> "pointwise_conv".
    trace_type = "depthwise_conv" if layer.get("type") == "depthwise_conv" else "pointwise_conv"

    return {
        "type": trace_type,
        "name": layer["name"],
        "M": result["M"],
        "K": result["K"],
        "N": result["N"],
        "im2col_overhead_cycles": result["im2col_overhead_cycles"],
        "sfu_cycles": 0,
    }


def _gemm_from_layer(layer: dict[str, Any]) -> dict[str, Any]:
    """Build a trace entry for a Gemm (fully-connected / classifier) layer.

    Gemm computes ``Y = alpha * A * B + beta * C`` where:
    - ``A`` has shape ``[M, K]``  (input activation)
    - ``B`` has shape ``[K, N]``  (weight)
    - ``Y`` has shape ``[M, N]``  (output)

    For single-image inference ``M = 1``.
    """
    in_shape = layer["in_shape"]
    out_shape = layer["out_shape"]

    M = 1  # single-image batch
    K = in_shape[1] if in_shape and len(in_shape) >= 2 else 0
    N = out_shape[1] if out_shape and len(out_shape) >= 2 else 0

    return {
        "type": "gemm",
        "name": layer["name"],
        "M": M,
        "K": K,
        "N": N,
        "im2col_overhead_cycles": 0,
        "sfu_cycles": 0,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_mobilenetv3_trace(onnx_path: str) -> list[dict[str, Any]]:
    """Generate a CV accelerator trace from a MobileNetV3-Small ONNX file.

    The pipeline is:

    1. **Import** topology via :func:`~cv.onnx_importer.import_mobilenetv3`
       --- returns per-layer dicts with shapes, conv parameters, and types.
    2. **Map convs** via :func:`~cv.conv_mapper.map_conv_to_gemm` --- converts
       each convolution to its GEMM ``(M, K, N)`` dimensions plus im2col
       DMA overhead.
    3. **Classify** every remaining layer: SFU activations (HardSwish,
       HardSigmoid, Relu), element-wise (Add, Mul), pool / reduce,
       reshape / shape, concat, and Gemm classifier heads.

    Parameters
    ----------
    onnx_path : str
        Path to the ``mobilenetv3_small.onnx`` file.

    Returns
    -------
    list[dict[str, Any]]
        Ordered list of trace entries consumable by the Caduceus NPU
        simulator.  Each entry follows the schema documented at the top
        of this module.

    Raises
    ------
    AssertionError
        If the total number of MACs falls outside the expected range
        50 M - 62 M.
    """
    layers = import_mobilenetv3(onnx_path)
    trace: list[dict[str, Any]] = []

    for layer in layers:
        op_type = layer["type"]
        out_shape = layer["out_shape"]
        name = layer["name"]

        # ---- Conv layers (pointwise / depthwise) -------------------------
        if op_type in ("Conv", "depthwise_conv"):
            entry = _conv_from_layer(layer)
            trace.append(entry)

        # ---- Gemm (fully-connected classifier heads) ---------------------
        elif op_type == "Gemm":
            entry = _gemm_from_layer(layer)
            trace.append(entry)

        # ---- SFU activation layers ---------------------------------------
        elif op_type in ("HardSwish", "HardSigmoid", "Relu"):
            trace_type = TYPE_MAP.get(op_type, op_type.lower())
            trace.append({
                "type": trace_type,
                "name": name,
                "M": 0,
                "K": 0,
                "N": 0,
                "im2col_overhead_cycles": 0,
                "sfu_cycles": _compute_sfu_cycles(out_shape),
            })

        # ---- Element-wise / shape / misc ops -----------------------------
        elif op_type in TYPE_MAP:
            trace_type = TYPE_MAP[op_type]
            trace.append({
                "type": trace_type,
                "name": name,
                "M": 0,
                "K": 0,
                "N": 0,
                "im2col_overhead_cycles": 0,
                "sfu_cycles": 0,
            })

        else:
            # Fallback: lowercased type (should not be reached for
            # MobileNetV3-Small, but keeps the generator robust).
            trace.append({
                "type": op_type.lower(),
                "name": name,
                "M": 0,
                "K": 0,
                "N": 0,
                "im2col_overhead_cycles": 0,
                "sfu_cycles": 0,
            })

    # ---- Validation -------------------------------------------------------
    total_macs = sum(entry["M"] * entry["K"] * entry["N"] for entry in trace)
    assert 50_000_000 <= total_macs <= 62_000_000, (
        f"Total MACs {total_macs:,} is outside expected range [50M, 62M]"
    )

    return trace


def save_trace(trace: list[dict[str, Any]], json_path: str) -> None:
    """Write a trace to a JSON file (one entry per line, pretty-printed)."""
    with open(json_path, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"Trace saved to {json_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    # Ensure the sim package is on the path when running as a script
    _sim_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _sim_dir not in sys.path:
        sys.path.insert(0, _sim_dir)

    # Repo root (one level up from .../sim)
    repo_root = os.path.dirname(_sim_dir)
    default_onnx = os.path.join(repo_root, "assets", "mobilenetv3_small.onnx")

    trace = generate_mobilenetv3_trace(default_onnx)

    total_macs = sum(e["M"] * e["K"] * e["N"] for e in trace)
    print(f"Generated trace with {len(trace)} entries")
    print(f"Total MACs: {total_macs:,}  (expected 50M-62M)")

    # Save trace JSON for downstream tools
    trace_out = os.path.join(repo_root, "results", "cv", "mobilenetv3_small", "trace.json")
    os.makedirs(os.path.dirname(trace_out), exist_ok=True)
    save_trace(trace, trace_out)

    # Save evidence
    evidence_dir = os.path.join(repo_root, ".omo", "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    evidence_path = os.path.join(evidence_dir, "cv-task-6-trace.txt")
    with open(evidence_path, "w") as f:
        f.write(f"cv_trace.py - MobileNetV3-Small trace generation\n")
        f.write(f"===============================================\n\n")
        f.write(f"ONNX file: {default_onnx}\n")
        f.write(f"Trace entries: {len(trace)}\n")
        f.write(f"Total MACs: {total_macs:,}\n")
        f.write(f"MACs in range [50M, 62M]: {50_000_000 <= total_macs <= 62_000_000}\n\n")
        f.write("Layer-by-layer breakdown:\n")
        f.write(f"{'#':>4} {'Type':20s} {'Name':40s} {'M':>8} {'K':>8} {'N':>8} {'MACs':>12} {'im2col':>10} {'SFU':>6}\n")
        f.write("-" * 120 + "\n")
        for i, entry in enumerate(trace):
            macs = entry["M"] * entry["K"] * entry["N"]
            f.write(
                f"{i:4d} {entry['type']:20s} {entry['name']:40s} "
                f"{entry['M']:8d} {entry['K']:8d} {entry['N']:8d} "
                f"{macs:12,} {entry['im2col_overhead_cycles']:10.1f} {entry['sfu_cycles']:6d}\n"
            )
        f.write("\n")
        f.write(f"Total MACs: {total_macs:,}\n")

    print(f"Evidence written to {evidence_path}")
