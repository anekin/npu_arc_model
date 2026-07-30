"""Legacy trace → WorkloadGraphV1 lowering adapter.

Converts existing LLM tuple traces and CV dict traces into the canonical
``WorkloadGraphV1`` schema.  Handles the legacy ``--batch-m`` CLI semantics:

* ``--batch-m 1`` maps to decode mode with ``active_sequences=1``.
* ``--batch-m 2`` maps to a two-token prefill with ``token_block=2``.

Conflicts between legacy batch flags and the new explicit dimension bindings
raise ``ConfigError``.
"""

from __future__ import annotations

from typing import Any

from contracts.errors import ConfigError, UnsupportedOperatorError
from workloads.dimensions import AXIS_SEQUENCES, AXIS_TOKEN_BLOCK, DimensionBindings
from workloads.operators import DEFAULT_REGISTRY, OperatorRegistry
from workloads.schema import (
    Layout,
    NodeSpec,
    Precision,
    TensorSpec,
    WorkloadGraphV1,
    WorkloadProvenance,
)

# Map legacy CV trace entry "type" strings to workload op types.
CV_TRACE_TYPE_MAP: dict[str, str] = {
    "pointwise_conv": "pointwise_conv",
    "depthwise_conv": "depthwise_conv",
    "gemm": "gemm",
    "hard_swish": "hard_swish",
    "hard_sigmoid": "hard_sigmoid",
    "relu": "relu",
    "global_avg_pool": "global_avg_pool",
    "add": "add",
    "mul": "mul",
    "reshape": "reshape",
    "shape": "shape",
    "concat": "concat",
    "reduce_mean": "reduce_mean",
}


def _provenance(source: str) -> WorkloadProvenance:
    return WorkloadProvenance(source=source)


def apply_legacy_batch_m(
    batch_m: int | None,
    bindings: DimensionBindings,
) -> DimensionBindings:
    """Merge legacy ``--batch-m`` semantics into explicit dimension bindings.

    Args:
        batch_m: Legacy batch flag value (1 or 2), or ``None``.
        bindings: Existing explicit dimension bindings.

    Raises:
        ConfigError: if the legacy flag conflicts with an already-set dimension.

    Returns:
        A new ``DimensionBindings`` with the legacy dimension applied.
    """
    if batch_m is None:
        return bindings

    if batch_m not in (1, 2):
        raise ConfigError(
            f"legacy --batch-m must be 1 or 2, got {batch_m}",
            field_path="batch_m",
            value=batch_m,
        )

    current = bindings.to_dict()

    if batch_m == 1:
        if AXIS_SEQUENCES in current and current[AXIS_SEQUENCES] != 1:
            raise ConfigError(
                f"legacy --batch-m 1 conflicts with active_sequences={current[AXIS_SEQUENCES]}",
                field_path="active_sequences",
                value=current[AXIS_SEQUENCES],
            )
        return DimensionBindings(
            request_batch=bindings.request_batch,
            active_sequences=1,
            token_block=bindings.token_block,
            image_count=bindings.image_count,
            action_horizon=bindings.action_horizon,
            flow_steps=bindings.flow_steps,
            resident_models=bindings.resident_models,
            inflight_jobs=bindings.inflight_jobs,
            extra=bindings.extra,
        )

    # batch_m == 2
    if AXIS_TOKEN_BLOCK in current and current[AXIS_TOKEN_BLOCK] != 2:
        raise ConfigError(
            f"legacy --batch-m 2 conflicts with token_block={current[AXIS_TOKEN_BLOCK]}",
            field_path="token_block",
            value=current[AXIS_TOKEN_BLOCK],
        )
    return DimensionBindings(
        request_batch=bindings.request_batch,
        active_sequences=bindings.active_sequences,
        token_block=2,
        image_count=bindings.image_count,
        action_horizon=bindings.action_horizon,
        flow_steps=bindings.flow_steps,
        resident_models=bindings.resident_models,
        inflight_jobs=bindings.inflight_jobs,
        extra=bindings.extra,
    )


def lower_llm_tuple_trace(
    trace: list[tuple[int, int, int, int, str]],
    *,
    graph_name: str = "legacy_llm",
    batch_m: int | None = None,
    bindings: DimensionBindings | None = None,
) -> tuple[WorkloadGraphV1, DimensionBindings]:
    """Convert a legacy LLM tuple trace into a workload graph.

    Each trace tuple is ``(M, K, N, layer, op_name)``.  Every entry becomes a
    ``gemm`` node in a linear dependency chain.  Weight tensors are modelled
    as separate graph inputs.

    Args:
        trace: Legacy LLM tuple trace.
        graph_name: Human-readable graph name.
        batch_m: Optional legacy ``--batch-m`` value (1 or 2).
        bindings: Optional explicit dimension bindings.

    Returns:
        ``(graph, effective_bindings)`` where ``effective_bindings`` includes
        the legacy batch mapping.
    """
    bindings = bindings or DimensionBindings()
    effective_bindings = apply_legacy_batch_m(batch_m, bindings)
    provenance = _provenance(f"legacy:llm:{graph_name}")

    nodes: list[NodeSpec] = []
    tensors: list[TensorSpec] = []

    prev_output: str | None = None
    for idx, (m, k, n, _layer, op_name) in enumerate(trace):
        node_id = f"n_{op_name}_{idx}"

        # Activation input
        act_id = f"t_act_{idx}"
        act_shape: list[int | str] = [m, k]
        act_tensor = TensorSpec(
            tensor_id=act_id,
            shape=act_shape,
            precision=Precision.INT8,
            layout=Layout.ROW_MAJOR,
            producer_node="input" if prev_output is None else f"n_{trace[idx - 1][4]}_{idx - 1}",
            consumed_by=[node_id],
            bytes=0,
            provenance=provenance,
        )
        tensors.append(act_tensor)

        # Weight input
        weight_id = f"t_weight_{idx}"
        weight_tensor = TensorSpec(
            tensor_id=weight_id,
            shape=[k, n],
            precision=Precision.INT4,
            layout=Layout.ROW_MAJOR,
            producer_node="input",
            consumed_by=[node_id],
            bytes=0,
            provenance=provenance,
        )
        tensors.append(weight_tensor)

        # Output activation
        out_id = f"t_out_{idx}"
        out_tensor = TensorSpec(
            tensor_id=out_id,
            shape=[m, n],
            precision=Precision.INT8,
            layout=Layout.ROW_MAJOR,
            producer_node=node_id,
            consumed_by=[],
            bytes=0,
            provenance=provenance,
        )
        tensors.append(out_tensor)

        # If there is a previous output, it already points to this node via consumed_by
        # but we created a new activation tensor above.  For a linear chain the previous
        # output should be this node's input instead.
        if prev_output is not None:
            # Replace the placeholder activation tensor with the real previous output.
            tensors.pop(-3)  # remove placeholder act_tensor
            act_id = prev_output
            # Ensure consumed_by includes this node
            for t in tensors:
                if t.tensor_id == act_id and node_id not in t.consumed_by:
                    t.consumed_by.append(node_id)

        node = NodeSpec(
            node_id=node_id,
            op_type="gemm",
            op_label=op_name,
            inputs=[act_id, weight_id],
            outputs=[out_id],
            dependencies=[f"n_{trace[idx - 1][4]}_{idx - 1}"] if idx > 0 else [],
            provenance=provenance,
        )
        nodes.append(node)
        prev_output = out_id

    graph = WorkloadGraphV1(
        version="1",
        graph_name=graph_name,
        nodes=nodes,
        tensors=tensors,
        provenance=provenance,
    )
    return graph, effective_bindings


def lower_cv_dict_trace(
    trace: list[dict[str, Any]],
    *,
    graph_name: str = "legacy_cv",
    image_count: int | None = None,
    bindings: DimensionBindings | None = None,
    registry: OperatorRegistry = DEFAULT_REGISTRY,
) -> tuple[WorkloadGraphV1, DimensionBindings]:
    """Convert a legacy CV dict trace into a workload graph.

    Each trace entry must contain ``type`` and ``name`` keys.  Supported types
    are mapped through the operator registry; unknown types raise
    ``UnsupportedOperatorError``.

    Args:
        trace: Legacy CV trace (list of dicts).
        graph_name: Human-readable graph name.
        image_count: Optional explicit image count binding.
        bindings: Optional explicit dimension bindings.
        registry: Operator registry used to reject unsupported ops.

    Returns:
        ``(graph, effective_bindings)``.
    """
    bindings = bindings or DimensionBindings()
    if image_count is not None:
        if bindings.image_count is not None and bindings.image_count != image_count:
            raise ConfigError(
                f"image_count={image_count} conflicts with bindings.image_count={bindings.image_count}",
                field_path="image_count",
                value=image_count,
            )
        bindings = DimensionBindings(
            request_batch=bindings.request_batch,
            active_sequences=bindings.active_sequences,
            token_block=bindings.token_block,
            image_count=image_count,
            action_horizon=bindings.action_horizon,
            flow_steps=bindings.flow_steps,
            resident_models=bindings.resident_models,
            inflight_jobs=bindings.inflight_jobs,
            extra=bindings.extra,
        )

    provenance = _provenance(f"legacy:cv:{graph_name}")
    nodes: list[NodeSpec] = []
    tensors: list[TensorSpec] = []

    prev_output: str | None = None
    for idx, entry in enumerate(trace):
        entry_type = entry.get("type", "")
        name = entry.get("name", f"{entry_type}_{idx}")
        workload_op = CV_TRACE_TYPE_MAP.get(entry_type)

        if workload_op is None:
            raise UnsupportedOperatorError(
                f"legacy CV trace entry {name!r} has unsupported type {entry_type!r}",
                op_type=entry_type,
                node_id=name,
            )

        # Fail-closed registry check.
        registry.lookup(workload_op)

        node_id = f"n_{name}_{idx}"

        # Activation input (reuse previous output when available)
        if prev_output is not None:
            act_id = prev_output
            for t in tensors:
                if t.tensor_id == act_id and node_id not in t.consumed_by:
                    t.consumed_by.append(node_id)
            inputs = [act_id]
        else:
            act_id = "t_input"
            act_tensor = TensorSpec(
                tensor_id=act_id,
                shape=[1, 3, 224, 224],
                precision=Precision.INT8,
                layout=Layout.NCHW,
                producer_node="input",
                consumed_by=[node_id],
                bytes=0,
                provenance=provenance,
            )
            tensors.append(act_tensor)
            inputs = [act_id]

        # Weight tensor for conv/gemm layers
        weight_id: str | None = None
        if workload_op in {"pointwise_conv", "depthwise_conv", "conv", "gemm"}:
            weight_id = f"t_weight_{idx}"
            weight_tensor = TensorSpec(
                tensor_id=weight_id,
                shape=[1],
                precision=Precision.INT4,
                layout=Layout.ROW_MAJOR,
                producer_node="input",
                consumed_by=[node_id],
                bytes=0,
                provenance=provenance,
            )
            tensors.append(weight_tensor)
            inputs.append(weight_id)

        out_id = f"t_out_{idx}"
        out_tensor = TensorSpec(
            tensor_id=out_id,
            shape=[1],
            precision=Precision.INT8,
            layout=Layout.NCHW,
            producer_node=node_id,
            consumed_by=[],
            bytes=0,
            provenance=provenance,
        )
        tensors.append(out_tensor)

        node = NodeSpec(
            node_id=node_id,
            op_type=workload_op,
            op_label=name,
            inputs=inputs,
            outputs=[out_id],
            dependencies=[f"n_{trace[idx - 1].get('name', f'op_{idx - 1}')}_{idx - 1}"] if idx > 0 else [],
            attributes={
                "M": entry.get("M", 0),
                "K": entry.get("K", 0),
                "N": entry.get("N", 0),
                "im2col_overhead_cycles": entry.get("im2col_overhead_cycles", 0.0),
                "sfu_cycles": entry.get("sfu_cycles", 0),
            },
            provenance=provenance,
        )
        nodes.append(node)
        prev_output = out_id

    graph = WorkloadGraphV1(
        version="1",
        graph_name=graph_name,
        nodes=nodes,
        tensors=tensors,
        provenance=provenance,
    )
    return graph, bindings
