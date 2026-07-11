"""
ONNX MAC Validation — compares Arc Model MACs against ONNX Runtime
theoretical MACs for MobileNetV3-Small.

Provides two functions:

- ``count_onnx_macs(onnx_path)``: count theoretical MACs directly from
  ONNX graph nodes (Conv / Gemm).
- ``validate_macs(onnx_path)``: return ``{arc_macs, onnx_macs, delta_pct}``.

Verification: ONNX Runtime inference with random input, top-5 logits,
               assert delta < 5 %.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np
import onnx

# ---- Path setup: ensure CaduceusCore/sim is importable when run directly ---
_sim_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sim_dir not in sys.path:
    sys.path.insert(0, _sim_dir)

from cv.cv_trace import generate_mobilenetv3_trace


# ---------------------------------------------------------------------------
# Shape helper
# ---------------------------------------------------------------------------

def _build_name_to_shape(graph: onnx.GraphProto) -> dict[str, list[int]]:
    """Build a name -> shape lookup from graph inputs, outputs,
    value_info and initializers."""
    mapping: dict[str, list[int]] = {}
    for v in graph.input:
        try:
            mapping[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
        except Exception:
            pass
    for v in graph.output:
        try:
            mapping[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
        except Exception:
            pass
    for v in graph.value_info:
        try:
            mapping[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
        except Exception:
            pass
    for init in graph.initializer:
        mapping[init.name] = list(init.dims)
    return mapping


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_onnx_macs(onnx_path: str) -> int:
    """Count theoretical MACs from ONNX graph nodes.

    - **Conv**: ``H_out * W_out * C_in * K_h * K_w * C_out // groups``
    - **Gemm**: ``M * K * N`` (from input / weight shapes)
    - All other ops: **0**

    Parameters
    ----------
    onnx_path : str
        Path to the ``.onnx`` file (must include shape information).

    Returns
    -------
    int
        Total MAC count across all Conv and Gemm nodes.
    """
    model = onnx.load(onnx_path)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    graph = model.graph

    name_to_shape = _build_name_to_shape(graph)

    total_macs = 0

    for node in graph.node:
        op_type = node.op_type

        # --- Conv ----------------------------------------------------------
        if op_type == "Conv":
            attrs = {a.name: a for a in node.attribute}

            group = 1
            group_attr = attrs.get("group")
            if group_attr is not None:
                group = int(group_attr.i) or 1

            pads = [0, 0, 0, 0]
            pads_attr = attrs.get("pads")
            if pads_attr is not None:
                pads = list(pads_attr.ints) or [0, 0, 0, 0]

            strides = [1, 1]
            strides_attr = attrs.get("strides")
            if strides_attr is not None:
                strides = list(strides_attr.ints) or [1, 1]

            # Weight shape: [C_out, C_in_per_group, K_h, K_w]
            if len(node.input) < 2:
                continue
            w_shape = name_to_shape.get(node.input[1])
            if w_shape is None or len(w_shape) < 4:
                continue

            C_out = w_shape[0]
            C_in_per_g = w_shape[1]
            C_in = C_in_per_g * group
            K_h, K_w = w_shape[2], w_shape[3]

            # Input activation shape: [batch, C_in, H, W]
            in_shape = name_to_shape.get(node.input[0])
            if in_shape is None or len(in_shape) < 4:
                continue
            H, W = in_shape[2], in_shape[3]

            pad_h, pad_w = pads[0], pads[1]
            stride_h, stride_w = strides[0], strides[1]

            H_out = (H + 2 * pad_h - K_h) // stride_h + 1
            W_out = (W + 2 * pad_w - K_w) // stride_w + 1

            macs = H_out * W_out * C_in * K_h * K_w * C_out // group
            total_macs += macs

        # --- Gemm ----------------------------------------------------------
        elif op_type == "Gemm":
            attrs = {a.name: a for a in node.attribute}

            transB = 0
            transB_attr = attrs.get("transB")
            if transB_attr is not None:
                transB = int(transB_attr.i)

            a_shape = name_to_shape.get(node.input[0])
            b_shape = name_to_shape.get(node.input[1])

            if a_shape is None or len(a_shape) < 2:
                continue
            if b_shape is None or len(b_shape) < 2:
                continue

            # M is the batch dimension; max(1, ...) handles symbolic dim (0)
            M = max(1, a_shape[0])
            K = a_shape[1]

            if transB:
                # B is stored as [N, K], unfolded during matmul
                N = b_shape[0]
            else:
                N = b_shape[1]

            macs = M * K * N
            total_macs += macs

    return total_macs


def validate_macs(onnx_path: str) -> dict[str, Any]:
    """Compute Arc and ONNX MAC counts and return the delta.

    Parameters
    ----------
    onnx_path : str
        Path to the ``.onnx`` file.

    Returns
    -------
    dict
        ``{"arc_macs": int, "onnx_macs": int, "delta_pct": float}``.
    """
    # Arc model MACs: sum(M * K * N) across the trace
    trace = generate_mobilenetv3_trace(onnx_path)
    arc_macs = sum(entry["M"] * entry["K"] * entry["N"] for entry in trace)

    # ONNX theoretical MACs
    onnx_macs = count_onnx_macs(onnx_path)

    delta_pct = abs(arc_macs - onnx_macs) / max(arc_macs, 1) * 100.0

    return {
        "arc_macs": arc_macs,
        "onnx_macs": onnx_macs,
        "delta_pct": delta_pct,
    }


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def _run_onnx_inference(onnx_path: str) -> np.ndarray:
    """Run ONNX Runtime with random input and return top-5 class indices."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    logits = session.run([out_name], {inp_name: x})[0]
    top5 = np.argsort(logits[0])[-5:][::-1]
    top5_logits = logits[0][top5]

    print("\n--- ONNX Runtime Inference ---")
    print(f"Input shape:  (1, 3, 224, 224)")
    print(f"Output shape: {logits.shape}")
    for rank, (idx, val) in enumerate(zip(top5, top5_logits), start=1):
        print(f"  #{rank}: class {idx:5d}  logit={val:.4f}")

    return top5, top5_logits


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Resolve ONNX path ------------------------------------------------
    repo_root = os.path.dirname(_sim_dir)
    default_onnx = os.path.join(repo_root, "assets", "mobilenetv3_small.onnx")
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else default_onnx

    # --- Validate MACs ----------------------------------------------------
    result = validate_macs(onnx_path)
    print(f"Arc  MACs:  {result['arc_macs']:>12,}")
    print(f"ONNX MACs:  {result['onnx_macs']:>12,}")
    print(f"Delta:      {result['delta_pct']:>11.2f}%")

    # --- Assert -----------------------------------------------------------------
    assert result["delta_pct"] < 5.0, (
        f"MAC delta {result['delta_pct']:.2f}% exceeds 5% threshold"
    )
    print("PASS: delta < 5%")

    # --- ONNX Runtime inference verification ------------------------------
    _run_onnx_inference(onnx_path)

    # --- Save evidence ----------------------------------------------------
    evidence_dir = os.path.join(repo_root, ".omo", "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    evidence_path = os.path.join(evidence_dir, "cv-task-11-validate.txt")
    with open(evidence_path, "w") as f:
        f.write("cv validate_onnx.py — Arc vs ONNX MAC validation\n")
        f.write("=" * 52 + "\n\n")
        f.write(f"ONNX file:  {onnx_path}\n")
        f.write(f"Arc  MACs:  {result['arc_macs']:>12,}\n")
        f.write(f"ONNX MACs:  {result['onnx_macs']:>12,}\n")
        f.write(f"Delta:      {result['delta_pct']:>11.2f}%\n")
        f.write(f"Threshold:  < 5.0%\n")
        f.write(f"PASS:       {result['delta_pct'] < 5.0}\n")
        f.write(f"\nONNX Runtime inference: OK (random input, top-5 logits printed)\n")

    print(f"\nEvidence written to {evidence_path}")
