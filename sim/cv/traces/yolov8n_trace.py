"""YOLOv8n CV trace generator.

Generates a structured trace of the YOLOv8n object-detection network
(backbone + neck + head) for cycle-level NPU simulation.  Every convolution
is mapped to im2col-equivalent GEMM dimensions via ``map_conv_to_gemm``;
SiLU activations, MaxPool, Upsample and Concat layers are emitted as
non-GEMM entries.

Trace entries follow the schema defined in ``sim.cv.cv_trace``:

    {
        "type": str,          # pointwise_conv | depthwise_conv | relu |
                              # max_pool | upsample | concat | gemm
        "name": str,
        "M": int,
        "K": int,
        "N": int,
        "im2col_overhead_cycles": float,
        "sfu_cycles": int,
    }
"""

from __future__ import annotations

import math
import os
from typing import Any

from cv.conv_mapper import map_conv_to_gemm

# SFU width (elements per cycle), aligned with sim/config/npu_config.yaml.
_SFU_WIDTH = 128

# YOLOv8n topology constants.
_IMAGE_SIZE = 640
_NUM_CLASSES = 80  # COCO


def _conv_entry(
    name: str,
    C_in: int,
    C_out: int,
    H: int,
    W: int,
    K: int,
    stride: int = 1,
    pad: int = 0,
    groups: int = 1,
) -> dict[str, Any]:
    """Build a pointwise/standard convolution trace entry via map_conv_to_gemm."""
    result = map_conv_to_gemm(
        C_in, C_out, H, W, K, stride=stride, pad=pad, groups=groups
    )
    trace_type = "depthwise_conv" if groups > 1 else "pointwise_conv"
    return {
        "type": trace_type,
        "name": name,
        "M": result["M"],
        "K": result["K"],
        "N": result["N"],
        "im2col_overhead_cycles": result["im2col_overhead_cycles"],
        "sfu_cycles": 0,
    }


def _silu_entry(name: str, element_count: int) -> dict[str, Any]:
    """Build a SiLU activation SFU-only trace entry (typed as relu per schema)."""
    return {
        "type": "relu",
        "name": name,
        "M": 0,
        "K": 0,
        "N": 0,
        "im2col_overhead_cycles": 0,
        "sfu_cycles": math.ceil(element_count / _SFU_WIDTH),
    }


def _misc_entry(name: str, op_type: str) -> dict[str, Any]:
    """Build a MaxPool/Upsample/Concat trace entry with zero GEMM dims."""
    return {
        "type": op_type,
        "name": name,
        "M": 0,
        "K": 0,
        "N": 0,
        "im2col_overhead_cycles": 0,
        "sfu_cycles": 0,
    }


def _add_conv_with_silu(
    trace: list[dict[str, Any]],
    name: str,
    C_in: int,
    C_out: int,
    H: int,
    W: int,
    K: int,
    stride: int = 1,
    pad: int = 0,
) -> tuple[int, int]:
    """Append a Conv + SiLU pair and return the output spatial size."""
    trace.append(_conv_entry(name, C_in, C_out, H, W, K, stride=stride, pad=pad))
    H_out = (H + 2 * pad - K) // stride + 1
    W_out = (W + 2 * pad - K) // stride + 1
    trace.append(_silu_entry(f"{name}.act", C_out * H_out * W_out))
    return H_out, W_out


def _c2f_block(
    trace: list[dict[str, Any]],
    prefix: str,
    C_in: int,
    C_out: int,
    n: int,
    H: int,
    W: int,
) -> None:
    """Append a YOLOv8 C2f block (cv1 + n bottlenecks + cv2) with SiLUs."""
    C_ = C_out // 2

    # cv1: 1x1 conv C_in -> C_out
    trace.append(_conv_entry(f"{prefix}.cv1", C_in, C_out, H, W, 1))
    trace.append(_silu_entry(f"{prefix}.cv1.act", C_out * H * W))

    # n Bottleneck blocks: 3x3 conv -> SiLU -> 3x3 conv -> SiLU
    for i in range(n):
        trace.append(_conv_entry(f"{prefix}.m.{i}.cv1", C_, C_, H, W, 3, pad=1))
        trace.append(_silu_entry(f"{prefix}.m.{i}.cv1.act", C_ * H * W))
        trace.append(_conv_entry(f"{prefix}.m.{i}.cv2", C_, C_, H, W, 3, pad=1))
        trace.append(_silu_entry(f"{prefix}.m.{i}.cv2.act", C_ * H * W))

    # cv2: 1x1 conv (n+2)*C_ -> C_out
    trace.append(_conv_entry(f"{prefix}.cv2", (n + 2) * C_, C_out, H, W, 1))
    trace.append(_silu_entry(f"{prefix}.cv2.act", C_out * H * W))


def _sppf_block(
    trace: list[dict[str, Any]],
    prefix: str,
    C: int,
    H: int,
    W: int,
) -> None:
    """Append a YOLOv8 SPPF block (cv1 + 3 maxpools + cv2)."""
    # cv1: 1x1 conv C -> C//2
    trace.append(_conv_entry(f"{prefix}.cv1", C, C // 2, H, W, 1))
    trace.append(_silu_entry(f"{prefix}.cv1.act", (C // 2) * H * W))

    # Three sequential 5x5 max-pools (stride 1, padding 2 => same spatial size)
    trace.append(_misc_entry(f"{prefix}.m.0", "max_pool"))
    trace.append(_misc_entry(f"{prefix}.m.1", "max_pool"))
    trace.append(_misc_entry(f"{prefix}.m.2", "max_pool"))

    # cv2: 1x1 conv 2*C -> C (concatenation of 4 x C//2)
    trace.append(_conv_entry(f"{prefix}.cv2", 2 * C, C, H, W, 1))
    trace.append(_silu_entry(f"{prefix}.cv2.act", C * H * W))


def _detect_branch(
    trace: list[dict[str, Any]],
    prefix: str,
    C_in: int,
    C_hidden: int,
    C_out: int,
    H: int,
    W: int,
) -> None:
    """Append one YOLOv8 Detect branch: two 3x3 SiLU convs + one 1x1 conv."""
    trace.append(_conv_entry(f"{prefix}.0", C_in, C_hidden, H, W, 3, pad=1))
    trace.append(_silu_entry(f"{prefix}.0.act", C_hidden * H * W))

    trace.append(_conv_entry(f"{prefix}.1", C_hidden, C_hidden, H, W, 3, pad=1))
    trace.append(_silu_entry(f"{prefix}.1.act", C_hidden * H * W))

    # Final 1x1 projection has no activation in YOLOv8 Detect.
    trace.append(_conv_entry(f"{prefix}.2", C_hidden, C_out, H, W, 1))


def _count_params(trace: list[dict[str, Any]]) -> int:
    """Estimate parameter count from pointwise_conv trace entries.

    For a standard conv, ``K`` already encodes the kernel spatial size
    (``C_in`` for 1x1, ``C_in * 9`` for 3x3), so weights + bias = ``N*K + N``.
    YOLOv8n has no depthwise convolutions.  The result is for reporting only.
    """
    total = 0
    for entry in trace:
        if entry["type"] == "pointwise_conv":
            total += entry["N"] * entry["K"] + entry["N"]
    return total


def generate_yolov8n_trace() -> list[dict[str, Any]]:
    """Generate a YOLOv8n detection trace as a list of accelerator entries.

    Architecture:
      - Input: 640x640x3
      - Backbone: Conv downsampling + C2f blocks + SPPF
      - Neck: FPN-PAN upsampling + concat + C2f blocks
      - Head: detection conv branches for P3/P4/P5 scales

    Returns
    -------
    list[dict[str, Any]]
        Ordered trace entries.  Convolutions map to ``pointwise_conv``;
        activations are ``relu`` SFU entries; MaxPool/Upsample/Concat have
        ``M=K=N=0``.

    Raises
    ------
    AssertionError
        If the total MAC count is outside the expected YOLOv8n range
        [8 G, 9.5 G].  YOLOv8n is conventionally quoted at ~8.7 GFLOPs;
        we treat that figure as the MAC target (multiply + add counted
        once), which equals ``2 * sum(M*K*N)`` over the GEMM entries.
    """
    trace: list[dict[str, Any]] = []

    H = W = _IMAGE_SIZE

    # ---- Backbone -----------------------------------------------------------
    H, W = _add_conv_with_silu(
        trace, "backbone.stem", 3, 16, H, W, 3, stride=2, pad=1
    )
    H, W = _add_conv_with_silu(
        trace, "backbone.stage1", 16, 32, H, W, 3, stride=2, pad=1
    )
    _c2f_block(trace, "backbone.stage2", 32, 32, 1, H, W)
    H, W = _add_conv_with_silu(
        trace, "backbone.stage3", 32, 64, H, W, 3, stride=2, pad=1
    )
    _c2f_block(trace, "backbone.stage4", 64, 64, 2, H, W)
    H, W = _add_conv_with_silu(
        trace, "backbone.stage5", 64, 128, H, W, 3, stride=2, pad=1
    )
    _c2f_block(trace, "backbone.stage6", 128, 128, 2, H, W)
    H, W = _add_conv_with_silu(
        trace, "backbone.stage7", 128, 256, H, W, 3, stride=2, pad=1
    )
    _c2f_block(trace, "backbone.stage8", 256, 256, 1, H, W)
    _sppf_block(trace, "backbone.sppf", 256, H, W)

    # Save backbone output shape for neck concat.
    H_sppf, W_sppf = H, W  # 20x20
    C_sppf = 256

    # ---- Neck (FPN-PAN) -----------------------------------------------------
    # P5 -> P4 upsample + concat + C2f
    trace.append(_misc_entry("neck.upsample0", "upsample"))
    H, W = H_sppf * 2, W_sppf * 2  # 40x40
    trace.append(_misc_entry("neck.concat0", "concat"))
    _c2f_block(trace, "neck.c2f_p4", 384, 128, 1, H, W)
    C_p4 = 128
    H_p4, W_p4 = H, W

    # P4 -> P3 upsample + concat + C2f
    trace.append(_misc_entry("neck.upsample1", "upsample"))
    H, W = H * 2, W * 2  # 80x80
    trace.append(_misc_entry("neck.concat1", "concat"))
    _c2f_block(trace, "neck.c2f_p3", 192, 64, 1, H, W)
    C_p3 = 64
    H_p3, W_p3 = H, W

    # P3 -> P4 downsample + concat + C2f
    H, W = _add_conv_with_silu(
        trace, "neck.conv_p4", C_p3, C_p3, H_p3, W_p3, 3, stride=2, pad=1
    )
    trace.append(_misc_entry("neck.concat2", "concat"))
    _c2f_block(trace, "neck.c2f_p4_2", C_p3 + C_p4, 128, 1, H, W)

    # P4 -> P5 downsample + concat + C2f
    H, W = _add_conv_with_silu(
        trace, "neck.conv_p5", 128, 128, H, W, 3, stride=2, pad=1
    )
    trace.append(_misc_entry("neck.concat3", "concat"))
    _c2f_block(trace, "neck.c2f_p5", 128 + C_sppf, 256, 1, H, W)

    # ---- Head (Detect) ------------------------------------------------------
    # P3 branch: 80x80x64 -> box(64) + cls(80)
    _detect_branch(trace, "head.p3.box", 64, 64, 64, H_p3, W_p3)
    _detect_branch(trace, "head.p3.cls", 64, 80, 80, H_p3, W_p3)

    # P4 branch: 40x40x128 -> box(64) + cls(80)
    _detect_branch(trace, "head.p4.box", 128, 64, 64, H_p4, W_p4)
    _detect_branch(trace, "head.p4.cls", 128, 80, 80, H_p4, W_p4)

    # P5 branch: 20x20x256 -> box(64) + cls(80)
    _detect_branch(trace, "head.p5.box", 256, 64, 64, H_sppf, W_sppf)
    _detect_branch(trace, "head.p5.cls", 256, 80, 80, H_sppf, W_sppf)

    # ---- Validation ----------------------------------------------------------
    gemm_ops = sum(entry["M"] * entry["K"] * entry["N"] for entry in trace)
    # YOLOv8n is quoted at ~8.7 GFLOPs; in this trace convention that maps to
    # 2 * gemm_ops because literature counts multiply and add separately.
    total_macs = 2 * gemm_ops
    assert 8_000_000_000 <= total_macs <= 9_500_000_000, (
        f"YOLOv8n total MACs {total_macs:,} outside expected range [8G, 9.5G] "
        f"(raw GEMM ops = {gemm_ops:,})"
    )

    return trace


if __name__ == "__main__":
    trace = generate_yolov8n_trace()
    gemm_ops = sum(e["M"] * e["K"] * e["N"] for e in trace)
    total_macs = 2 * gemm_ops
    gemm_entries = [e for e in trace if e["M"] != 0]
    sfu_entries = [e for e in trace if e["M"] == 0]

    # Parameter sanity check (approximate from trace entries).
    approx_params = _count_params(trace)

    print(f"Generated YOLOv8n trace with {len(trace)} entries")
    print(f"  GEMM entries: {len(gemm_entries)}")
    print(f"  SFU entries:  {len(sfu_entries)}")
    print(f"  Raw GEMM ops: {gemm_ops:,} ({gemm_ops / 1e9:.2f} G)")
    print(f"  Total MACs:   {total_macs:,} ({total_macs / 1e9:.2f} G)")
    print(f"  Approx params: {approx_params:,} ({approx_params / 1e6:.2f} M)")
    print(f"  MACs in [8G, 9.5G]: {8_000_000_000 <= total_macs <= 9_500_000_000}")

    # Save evidence to repository root .omo/evidence/.
    # File path: repo_root/CaduceusCore/sim/cv/traces/yolov8n_trace.py
    repo_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        )
    )
    evidence_dir = os.path.join(repo_root, ".omo", "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    evidence_path = os.path.join(evidence_dir, "timing-task-10-yolo.txt")
    with open(evidence_path, "w") as f:
        f.write("yolov8n_trace.py - YOLOv8n trace generation\n")
        f.write("==========================================\n\n")
        f.write(f"Trace entries: {len(trace)}\n")
        f.write(f"  GEMM entries: {len(gemm_entries)}\n")
        f.write(f"  SFU entries:  {len(sfu_entries)}\n")
        f.write(f"Raw GEMM ops (sum M*K*N): {gemm_ops:,} ({gemm_ops / 1e9:.3f} G)\n")
        f.write(f"Total MACs (2 * raw GEMM ops): {total_macs:,} ({total_macs / 1e9:.3f} G)\n")
        f.write(f"MACs in [8G, 9.5G]: {8_000_000_000 <= total_macs <= 9_500_000_000}\n")
        f.write(f"Approx params: {approx_params:,} ({approx_params / 1e6:.3f} M)\n\n")
        f.write("Layer-by-layer breakdown:\n")
        f.write(
            f"{'#':>4} {'Type':20s} {'Name':40s} "
            f"{'M':>8} {'K':>8} {'N':>8} {'GEMM_ops':>14} {'im2col':>10} {'SFU':>6}\n"
        )
        f.write("-" * 130 + "\n")
        for i, entry in enumerate(trace):
            ops = entry["M"] * entry["K"] * entry["N"]
            f.write(
                f"{i:4d} {entry['type']:20s} {entry['name']:40s} "
                f"{entry['M']:8d} {entry['K']:8d} {entry['N']:8d} "
                f"{ops:14,} {entry['im2col_overhead_cycles']:10.1f} "
                f"{entry['sfu_cycles']:6d}\n"
            )
        f.write("\n")
        f.write(f"Total MACs: {total_macs:,}\n")

    print(f"Evidence written to {evidence_path}")
