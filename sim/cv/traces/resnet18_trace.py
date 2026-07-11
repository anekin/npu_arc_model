"""
ResNet-18 CV trace generator.

Produces a CNN trace for the torchvision ResNet-18 architecture.  Every
convolution is mapped to GEMM dimensions via ``map_conv_to_gemm``; ReLU,
MaxPool and global average-pool layers are emitted as SFU-only entries.

Trace entries follow the schema defined in ``sim.cv.cv_trace``:

    {
        "type": str,
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
from typing import Any

from cv.conv_mapper import map_conv_to_gemm

# SFU width (elements per cycle), aligned with sim/config/npu_config.yaml.
_SFU_WIDTH = 128

# ResNet-18 topology constants (ImageNet-1k).
_IMAGE_SIZE = 224
_NUM_CLASSES = 1000


def _conv(
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
    trace_type = "pointwise_conv" if K == 1 and groups == 1 else "gemm"
    return {
        "type": trace_type,
        "name": name,
        "M": result["M"],
        "K": result["K"],
        "N": result["N"],
        "im2col_overhead_cycles": result["im2col_overhead_cycles"],
        "sfu_cycles": 0,
    }


def _relu(name: str, element_count: int) -> dict[str, Any]:
    """Build a ReLU SFU-only trace entry."""
    return {
        "type": "relu",
        "name": name,
        "M": 0,
        "K": 0,
        "N": 0,
        "im2col_overhead_cycles": 0,
        "sfu_cycles": math.ceil(element_count / _SFU_WIDTH),
    }


def _pool(name: str, element_count: int, pool_type: str) -> dict[str, Any]:
    """Build a MaxPool/AvgPool SFU-only trace entry."""
    return {
        "type": pool_type,
        "name": name,
        "M": 0,
        "K": 0,
        "N": 0,
        "im2col_overhead_cycles": 0,
        "sfu_cycles": math.ceil(element_count / _SFU_WIDTH),
    }


def _basic_block(
    trace: list[dict[str, Any]],
    prefix: str,
    C_in: int,
    C_out: int,
    H: int,
    W: int,
    stride: int,
    downsample: bool,
) -> tuple[int, int]:
    """Append one ResNet-18 BasicBlock and return output spatial size."""
    # First 3x3 conv; stride may downsample.
    trace.append(_conv(f"{prefix}.conv1", C_in, C_out, H, W, 3, stride=stride, pad=1))
    H_out = (H + 2 * 1 - 3) // stride + 1
    W_out = (W + 2 * 1 - 3) // stride + 1
    trace.append(_relu(f"{prefix}.relu1", C_out * H_out * W_out))

    # Second 3x3 conv.
    trace.append(_conv(f"{prefix}.conv2", C_out, C_out, H_out, W_out, 3, stride=1, pad=1))
    trace.append(_relu(f"{prefix}.relu2", C_out * H_out * W_out))

    # Projection shortcut when dimensions change.
    if downsample:
        trace.append(
            _conv(
                f"{prefix}.downsample.0",
                C_in,
                C_out,
                H,
                W,
                1,
                stride=stride,
                pad=0,
            )
        )

    return H_out, W_out


def generate_resnet18_trace() -> list[dict[str, Any]]:
    """Generate a ResNet-18 trace as a list of accelerator entries.

    Returns
    -------
    list[dict[str, Any]]
        Ordered trace entries.  Convolutions map to ``pointwise_conv`` or
        ``gemm``; activation / pool layers are SFU-only entries with
        ``M=K=N=0``.

    Raises
    ------
    AssertionError
        If the total GEMM MAC count is outside the expected ResNet-18 range
        [1.7 G, 2.0 G].
    """
    trace: list[dict[str, Any]] = []

    H = W = _IMAGE_SIZE

    # ---- conv1 + maxpool ----------------------------------------------------
    trace.append(_conv("conv1", 3, 64, H, W, 7, stride=2, pad=3))
    H = (H + 2 * 3 - 7) // 2 + 1  # 112
    W = (W + 2 * 3 - 7) // 2 + 1
    trace.append(_relu("relu", 64 * H * W))

    trace.append(_pool("maxpool", 64 * (H // 2) * (W // 2), "max_pool"))
    H = H // 2  # 56
    W = W // 2

    # ---- conv2_x ------------------------------------------------------------
    H, W = _basic_block(trace, "layer1.0", 64, 64, H, W, stride=1, downsample=False)
    H, W = _basic_block(trace, "layer1.1", 64, 64, H, W, stride=1, downsample=False)

    # ---- conv3_x ------------------------------------------------------------
    H, W = _basic_block(trace, "layer2.0", 64, 128, H, W, stride=2, downsample=True)
    H, W = _basic_block(trace, "layer2.1", 128, 128, H, W, stride=1, downsample=False)

    # ---- conv4_x ------------------------------------------------------------
    H, W = _basic_block(trace, "layer3.0", 128, 256, H, W, stride=2, downsample=True)
    H, W = _basic_block(trace, "layer3.1", 256, 256, H, W, stride=1, downsample=False)

    # ---- conv5_x ------------------------------------------------------------
    H, W = _basic_block(trace, "layer4.0", 256, 512, H, W, stride=2, downsample=True)
    H, W = _basic_block(trace, "layer4.1", 512, 512, H, W, stride=1, downsample=False)

    # ---- global average pool + classifier -----------------------------------
    trace.append(_pool("avgpool", 512 * H * W, "avg_pool"))
    # Output is 512-D vector.
    trace.append(
        {
            "type": "gemm",
            "name": "fc",
            "M": 1,
            "K": 512,
            "N": _NUM_CLASSES,
            "im2col_overhead_cycles": 0,
            "sfu_cycles": 0,
        }
    )

    # ---- Validation ---------------------------------------------------------
    total_macs = sum(entry["M"] * entry["K"] * entry["N"] for entry in trace)
    assert 1_700_000_000 <= total_macs <= 2_000_000_000, (
        f"ResNet-18 total MACs {total_macs:,} outside expected range [1.7G, 2.0G]"
    )

    return trace


if __name__ == "__main__":
    t = generate_resnet18_trace()
    total_macs = sum(e["M"] * e["K"] * e["N"] for e in t)
    gemm_entries = [e for e in t if e["M"] != 0]
    sfu_entries = [e for e in t if e["M"] == 0]
    print(f"Generated ResNet-18 trace with {len(t)} entries")
    print(f"  GEMM entries: {len(gemm_entries)}")
    print(f"  SFU entries:  {len(sfu_entries)}")
    print(f"  Total MACs:   {total_macs:,} ({total_macs / 1e9:.2f} G)")
