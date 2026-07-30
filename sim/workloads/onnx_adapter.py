"""ONNX → WorkloadGraphV1 lowering adapter.

This adapter translates an ONNX model into the canonical declarative workload
graph schema.  It is fail-closed:

* Symbolic dimension names are preserved (never silently converted to 0).
* Only operators registered as ``modeled`` or ``explicitly_free_or_fused``
  are lowered.
* Unknown or explicitly unsupported operators raise
  ``UnsupportedOperatorError`` with node name, op type, opset version, and
  model path.

The adapter produces the graph only; callers must bind all symbolic
dimensions via ``DimensionBindings`` and run ``validate_all`` before cost
estimation.
"""

from __future__ import annotations

import contextlib
from typing import Any

import onnx
from contracts.errors import UnsupportedOperatorError
from workloads.operators import DEFAULT_REGISTRY, OperatorRegistry
from workloads.schema import (
    Layout,
    NodeSpec,
    Precision,
    SymbolicDim,
    TensorSpec,
    WorkloadGraphV1,
    WorkloadProvenance,
)

# ONNX op type -> workload op type key (must exist in the operator registry)
ONNX_TO_WORKLOAD_OP: dict[str, str] = {
    "Conv": "conv",
    "Gemm": "gemm",
    "MatMul": "matmul",
    "Relu": "relu",
    "Add": "add",
    "Mul": "mul",
    "MaxPool": "max_pool",
    "GlobalAveragePool": "global_avg_pool",
    "ReduceMean": "reduce_mean",
    "Softmax": "softmax",
    "LayerNormalization": "layernorm",
    "RMSNorm": "rms_norm",
    "Gelu": "gelu",
    "HardSwish": "hard_swish",
    "HardSigmoid": "hard_sigmoid",
    "Reshape": "reshape",
    "Concat": "concat",
    "Shape": "shape",
    "Transpose": "transpose",
    "Squeeze": "reshape",
    "Unsqueeze": "reshape",
}


def _onnx_opset_version(model: onnx.ModelProto) -> int:
    """Return the ONNX opset version from the model, or 0 if unknown."""
    if not model.opset_import:
        return 0
    return int(model.opset_import[0].version)


def _get_shape(value_info: onnx.ValueInfoProto | None) -> list[int | str] | None:
    """Extract tensor shape from a ValueInfoProto, preserving symbolic names.

    Symbolic dimensions are encoded via ``dim_param`` in ONNX.  Those names
    are kept verbatim so that callers can bind them with ``DimensionBindings``.
    Fixed dimensions use ``dim_value``.
    """
    if value_info is None:
        return None
    try:
        shape = value_info.type.tensor_type.shape
        dims: list[int | str] = []
        for d in shape.dim:
            if d.dim_param:
                dims.append(d.dim_param)
            elif d.dim_value:
                dims.append(int(d.dim_value))
            else:
                # Unspecified scalar-like dimension; represent as 1 so the
                # graph remains valid while still allowing shape inference.
                dims.append(1)
        return dims if dims else None
    except Exception:  # noqa: BLE001
        return None


def _build_value_info_map(
    graph: onnx.GraphProto,
) -> dict[str, list[int | str] | None]:
    """Build name -> shape lookup from inputs, outputs, value_info and initializers."""
    mapping: dict[str, list[int | str] | None] = {}
    for v in graph.input:
        mapping[v.name] = _get_shape(v)
    for v in graph.output:
        mapping[v.name] = _get_shape(v)
    for v in graph.value_info:
        mapping[v.name] = _get_shape(v)
    for init in graph.initializer:
        mapping[init.name] = [int(x) for x in init.dims] if init.dims else None
    return mapping


def _parse_attrs(node: onnx.NodeProto) -> dict[str, Any]:
    """Convert ONNX node attributes to a plain Python dict."""
    attrs: dict[str, Any] = {}
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.INT:
            attrs[attr.name] = int(attr.i)
        elif attr.type == onnx.AttributeProto.INTS:
            attrs[attr.name] = [int(x) for x in attr.ints]
        elif attr.type == onnx.AttributeProto.FLOAT:
            attrs[attr.name] = float(attr.f)
        elif attr.type == onnx.AttributeProto.FLOATS:
            attrs[attr.name] = [float(x) for x in attr.floats]
        elif attr.type == onnx.AttributeProto.STRING:
            raw = attr.s
            attrs[attr.name] = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    return attrs


def _map_onnx_op_type(
    node: onnx.NodeProto,
    model: onnx.ModelProto,
    path: str,
) -> str:
    """Map an ONNX op type to a workload op type and verify registry support.

    Raises:
        UnsupportedOperatorError: if the op is unknown or explicitly unsupported.
    """
    onnx_op = node.op_type
    workload_op = ONNX_TO_WORKLOAD_OP.get(onnx_op)

    if workload_op is None:
        raise UnsupportedOperatorError(
            f"ONNX node {node.name!r} uses unsupported operator {onnx_op!r} "
            f"(opset {model.opset_import[0].version}, path {path!r})",
            op_type=onnx_op,
            node_id=node.name or f"<{onnx_op}>",
        )

    # Fail-closed registry check: explicitly unsupported entries still raise.
    if DEFAULT_REGISTRY.is_unsupported(workload_op):
        raise UnsupportedOperatorError(
            f"ONNX node {node.name!r} maps to unsupported operator {workload_op!r} "
            f"(opset {model.opset_import[0].version}, path {path!r})",
            op_type=workload_op,
            node_id=node.name or f"<{onnx_op}>",
        )

    return workload_op


def lower_onnx_to_graph(
    onnx_path: str,
    *,
    graph_name: str = "onnx",
    registry: OperatorRegistry = DEFAULT_REGISTRY,
) -> WorkloadGraphV1:
    """Load an ONNX model from *onnx_path* and lower it to ``WorkloadGraphV1``.

    Args:
        onnx_path: Path to the ``.onnx`` file.
        graph_name: Human-readable name for the resulting graph.
        registry: Operator registry used to reject unsupported ops.

    Raises:
        UnsupportedOperatorError: when an operator is unknown or unsupported.
    """
    model = onnx.load(onnx_path)
    return lower_onnx_model_to_graph(
        model,
        graph_name=graph_name,
        registry=registry,
        path=onnx_path,
    )


def lower_onnx_model_to_graph(
    model: onnx.ModelProto,
    *,
    graph_name: str = "onnx",
    registry: OperatorRegistry = DEFAULT_REGISTRY,
    path: str = "<buffer>",
) -> WorkloadGraphV1:
    """Lower an in-memory ONNX model to ``WorkloadGraphV1``.

    Args:
        model: The ONNX model protobuf.
        graph_name: Human-readable name for the resulting graph.
        registry: Operator registry used to reject unsupported ops.
        path: Source path for error messages.

    Raises:
        UnsupportedOperatorError: when an operator is unknown or unsupported.
    """
    with contextlib.suppress(Exception):
        model = onnx.shape_inference.infer_shapes(model)

    onnx_graph = model.graph
    shape_map = _build_value_info_map(onnx_graph)
    _onnx_opset_version(model)

    provenance = WorkloadProvenance(
        source=f"onnx:{graph_name}",
        reference_uri=path if path != "<buffer>" else None,
    )

    # Tensor ID -> TensorSpec
    tensors: dict[str, TensorSpec] = {}

    def _ensure_tensor(name: str, is_initializer: bool = False) -> TensorSpec:
        if name in tensors:
            return tensors[name]
        shape = shape_map.get(name)
        if shape is None:
            shape = [1]
        tensor = TensorSpec(
            tensor_id=name,
            shape=shape,
            precision=Precision.FP16,
            layout=Layout.NCHW,
            producer_node="input" if is_initializer else "",
            bytes=0,
            provenance=provenance,
        )
        tensors[name] = tensor
        return tensor

    # Graph inputs and initializers are graph inputs.
    for inp in onnx_graph.input:
        _ensure_tensor(inp.name, is_initializer=False)
    for init in onnx_graph.initializer:
        _ensure_tensor(init.name, is_initializer=True)

    nodes: list[NodeSpec] = []
    symbol_names: set[str] = set()

    for idx, node in enumerate(onnx_graph.node):
        node_name = node.name or f"{node.op_type}_{idx}"
        workload_op = _map_onnx_op_type(node, model, path)

        # Verify the mapped op is supported by the caller-supplied registry.
        entry = registry.lookup(workload_op)
        if entry.disposition.value == "unsupported":
            raise UnsupportedOperatorError(
                f"node {node_name!r} operator {workload_op!r} is unsupported",
                op_type=workload_op,
                node_id=node_name,
            )

        inputs = [inp for inp in node.input if inp]
        outputs = [out for out in node.output if out]

        # Ensure input/output tensors exist.
        for inp in inputs:
            _ensure_tensor(inp)
        for out in outputs:
            _ensure_tensor(out)

        attrs = _parse_attrs(node)
        node_spec = NodeSpec(
            node_id=node_name,
            op_type=workload_op,
            op_label=f"{node.op_type}:{node_name}",
            inputs=inputs,
            outputs=outputs,
            attributes=attrs,
            provenance=provenance,
        )
        nodes.append(node_spec)

    # Resolve producers and consumers for every tensor.
    for node in nodes:
        for out in node.outputs:
            tensor = tensors[out]
            tensor.producer_node = node.node_id
        for inp in node.inputs:
            tensor = tensors[inp]
            if node.node_id not in tensor.consumed_by:
                tensor.consumed_by.append(node.node_id)

    # Collect symbolic dimension names.
    for tensor in tensors.values():
        for dim in tensor.shape:
            if isinstance(dim, str):
                symbol_names.add(dim)

    symbols = [SymbolicDim(name=name, description="ONNX symbolic dimension") for name in sorted(symbol_names)]

    return WorkloadGraphV1(
        version="1",
        graph_name=graph_name,
        nodes=nodes,
        tensors=list(tensors.values()),
        symbols=symbols,
        provenance=provenance,
    )
