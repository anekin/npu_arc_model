"""
Lightweight ONNX importer for MobileNetV3-Small topology extraction.

Extracts graph topology (layer types, shapes, conv parameters) without
loading weight values.  White-lists only ops known to appear in the
MobileNetV3-Small export.
"""

from __future__ import annotations

from typing import Any

import onnx

# ---------------------------------------------------------------------------
# Whitelist
# ---------------------------------------------------------------------------

WHITELIST_OPS: frozenset[str] = frozenset({
    "Conv",
    "Add",
    "GlobalAveragePool",
    "MaxPool",
    "HardSwish",
    "HardSigmoid",
    "Mul",
    "Gemm",
    "Reshape",
    "Squeeze",
    "Relu",
    "ReduceMean",
    "Concat",
    "Shape",
})

# SE block: ReduceMean -> Conv -> Relu -> Conv -> HardSigmoid -> Mul
SE_PATTERN: tuple[str, ...] = (
    "ReduceMean",
    "Conv",
    "Relu",
    "Conv",
    "HardSigmoid",
    "Mul",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_shape(
    value_info: onnx.ValueInfoProto,
) -> list[int] | None:
    """Extract a list of dimension sizes from a ValueInfoProto.

    Symbolic dimensions (e.g. the batch axis) are encoded as 0 so that
    downstream code can still inspect spatial/channel axes.
    """
    if value_info is None:
        return None
    try:
        tensor_type = value_info.type.tensor_type
        shape = tensor_type.shape
        dims: list[int] = []
        for d in shape.dim:
            if d.dim_value != 0:
                dims.append(d.dim_value)
            else:
                # symbolic dimension – store as 0
                dims.append(0)
        return dims if dims else None
    except Exception:  # noqa: BLE001
        return None


def _build_name_to_shape(
    graph: onnx.GraphProto,
) -> dict[str, list[int] | None]:
    """Build a name -> shape lookup from inputs, outputs, value_info and initializers."""
    mapping: dict[str, list[int] | None] = {}
    for v in graph.input:
        mapping[v.name] = _get_shape(v)
    for v in graph.output:
        mapping[v.name] = _get_shape(v)
    for v in graph.value_info:
        mapping[v.name] = _get_shape(v)
    for init in graph.initializer:
        mapping[init.name] = list(init.dims) if init.dims else None
    return mapping


def _parse_attrs(node: onnx.NodeProto) -> dict[str, Any]:
    """Convert ONNX node attributes to a plain dict."""
    attrs: dict[str, Any] = {}
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.INT:
            attrs[attr.name] = attr.i
        elif attr.type == onnx.AttributeProto.INTS:
            attrs[attr.name] = list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOAT:
            attrs[attr.name] = attr.f
        elif attr.type == onnx.AttributeProto.STRING:
            raw = attr.s
            attrs[attr.name] = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    return attrs


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def _get_conv_kernel(weight_shape: list[int] | None) -> list[int] | None:
    """Extract [H, W] from a conv weight shape [C_out, C_in, H, W].

    Returns None when the shape is not a 4-D tensor.
    """
    if weight_shape is None or len(weight_shape) != 4:
        return None
    return list(weight_shape[2:])


# ---------------------------------------------------------------------------
# SE-block detection (post-processing)
# ---------------------------------------------------------------------------

def _tag_se_blocks(layers: list[dict[str, Any]]) -> None:
    """Walk *layers* and tag every SE-block layer with ``se_block=True``.

    SE pattern: ReduceMean -> Conv -> Relu -> Conv -> HardSigmoid -> Mul.

    Layers that form part of a recognised SE block get an additional
    ``"se_block"`` key set to ``True``.  The final ``Mul`` of the block
    also gets ``"se_output": True``.
    """
    pattern_len = len(SE_PATTERN)
    for i in range(len(layers) - pattern_len + 1):
        window = layers[i : i + pattern_len]
        if all(w["type"] == exp for w, exp in zip(window, SE_PATTERN)):
            for w in window:
                w["se_block"] = True
            window[-1]["se_output"] = True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def import_mobilenetv3(onnx_path: str) -> list[dict[str, Any]]:
    """Load a MobileNetV3-Small ONNX file and extract layer topology.

    Parameters
    ----------
    onnx_path: str
        Path to the ``.onnx`` file.

    Returns
    -------
    list[dict[str, Any]]
        One entry per graph node with keys:

        - ``type``       - operator type (or ``"depthwise_conv"``)
        - ``name``       - node name (auto-generated if empty)
        - ``in_shape``   - input tensor shape or ``None``
        - ``out_shape``  - output tensor shape or ``None``
        - ``kernel``     - ``[H, W]`` for Conv, else ``None``
        - ``stride``     - ``[stride_h, stride_w]`` for Conv
        - ``padding``    - ``[top, left, bottom, right]`` for Conv
        - ``groups``     - group count for Conv
        - ``se_block``   - ``True`` if part of an SE block (post-processed)

    Raises
    ------
    NotImplementedError
        For any ONNX operator **not** in the whitelist.
    """
    model = onnx.load(onnx_path)
    model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph

    name_to_shape = _build_name_to_shape(graph)

    layers: list[dict[str, Any]] = []

    for node in graph.node:
        op_type = node.op_type

        if op_type not in WHITELIST_OPS:
            raise NotImplementedError(
                f"Operator '{op_type}' is not in the whitelist. "
                f"Supported ops: {sorted(WHITELIST_OPS)}"
            )

        # -- Basic info ----------------------------------------------------
        layer: dict[str, Any] = {
            "type": op_type,
            "name": node.name if node.name else f"{op_type}_{len(layers)}",
            "in_shape": None,
            "out_shape": None,
            "kernel": None,
            "stride": None,
            "padding": None,
            "groups": None,
        }

        if node.input:
            layer["in_shape"] = name_to_shape.get(node.input[0])
        if node.output:
            layer["out_shape"] = name_to_shape.get(node.output[0])

        attrs = _parse_attrs(node)

        # -- Conv-specific -------------------------------------------------
        if op_type == "Conv":
            groups = int(attrs.get("group", 1))
            layer["groups"] = groups
            layer["stride"] = attrs.get("strides", [1, 1])
            layer["padding"] = attrs.get("pads", [0, 0, 0, 0])

            # Kernel size from weight tensor shape (index 1 of inputs)
            if len(node.input) > 1:
                w_shape = name_to_shape.get(node.input[1])
                layer["kernel"] = _get_conv_kernel(w_shape)

            # Depthwise detection: groups == in_channels
            in_shape = layer["in_shape"]
            if groups > 1 and in_shape is not None and len(in_shape) >= 2:
                in_channels = in_shape[1]
                if groups == in_channels:
                    layer["type"] = "depthwise_conv"

        layers.append(layer)

    # -- Post-process: tag SE blocks ---------------------------------------
    _tag_se_blocks(layers)

    return layers
