"""Pre-lowering validation for workload graphs.

Validators that run BEFORE cost estimation / execution:

1. ``validate_graph_dag``: cycle detection and structural integrity
2. ``validate_dimensions``: all symbolic axes bound, no implicit substitution
3. ``validate_operators``: every op type in the registry, fail-closed on unsupported
4. ``validate_tensor_lifetime``: producer/consumer consistency, no dangling refs

Each validator returns ``None`` on success and raises a typed error on failure.
"""

from __future__ import annotations

from contracts.errors import (
    ConfigError,
    DimensionBindingError,
    UnsupportedOperatorError,
)
from workloads.dimensions import DimensionBindings
from workloads.operators import OperatorRegistry
from workloads.schema import WorkloadGraphV1


def validate_graph_dag(graph: WorkloadGraphV1) -> None:
    """Verify the graph is a valid DAG with unique IDs and consistent references.

    The graph's own ``model_validator`` already checks this at construction,
    but this function provides a standalone entry point for explicit validation
    in test and adapter code.

    Raises:
        ConfigError: if duplicate IDs, cycles, or dangling references are found.
    """
    node_ids: set[str] = set()
    for node in graph.nodes:
        if node.node_id in node_ids:
            raise ConfigError(f"duplicate node_id: {node.node_id!r}", field_path="nodes")
        node_ids.add(node.node_id)

    tensor_ids: set[str] = set()
    for tensor in graph.tensors:
        if tensor.tensor_id in tensor_ids:
            raise ConfigError(f"duplicate tensor_id: {tensor.tensor_id!r}", field_path="tensors")
        tensor_ids.add(tensor.tensor_id)

    # Cycle detection via topological sort
    graph._check_dag_cycles()

    # Tensor reference integrity
    graph._check_tensor_references()
    graph._check_alias_validity()


def validate_dimensions(
    graph: WorkloadGraphV1,
    bindings: DimensionBindings,
) -> None:
    """Verify all symbolic dimensions referenced by the graph are bound.

    This enforces that no symbolic dimension is implicitly substituted:
    every named symbol in tensor shapes must have an explicit binding.

    Args:
        graph: The workload graph to validate.
        bindings: Concrete dimension bindings.

    Raises:
        DimensionBindingError: if any symbolic dimension is unbound.
    """
    # Collect all symbolic names used in tensor shapes
    used_symbols: set[str] = set()
    for tensor in graph.tensors:
        for dim in tensor.shape:
            if isinstance(dim, str):
                used_symbols.add(dim)

    if not used_symbols:
        return  # no symbolic dimensions — trivially bound

    bound_symbols = set(bindings.to_dict().keys())

    unbound = used_symbols - bound_symbols
    if unbound:
        raise DimensionBindingError(
            f"unbound symbolic dimensions: {sorted(unbound)!r}",
            dimension=sorted(unbound)[0],
        )


def validate_operators(
    graph: WorkloadGraphV1,
    registry: OperatorRegistry,
) -> None:
    """Verify every node's op_type is registered and supported.

    Unknown or unsupported operators cause fail-closed errors.

    Args:
        graph: The workload graph to validate.
        registry: The operator registry to check against.

    Raises:
        UnsupportedOperatorError: if any node has an unsupported or unregistered op_type.
    """
    for node in graph.nodes:
        disposition = registry.check(node.op_type)
        if disposition.value == "unsupported":
            raise UnsupportedOperatorError(
                f"node {node.node_id!r} uses unsupported operator {node.op_type!r}",
                op_type=node.op_type,
                node_id=node.node_id,
            )


def validate_tensor_lifetime(graph: WorkloadGraphV1) -> None:
    """Verify tensor lifetime declarations are internally consistent.

    Checks:
    - Every tensor's producer_node exists in the graph or is 'input'.
    - Every tensor's consumed_by entries reference existing nodes.
    - Output tensors of a node appear in the node's outputs list.
    - Input tensors of a node appear in the node's inputs list.
    - No tensor is both input and output of the same node (self-loop data).

    Raises:
        ConfigError: if any lifetime inconsistency is found.
    """
    node_ids = {n.node_id for n in graph.nodes}
    node_ids.add("input")
    tensor_ids = {t.tensor_id for t in graph.tensors}

    for tensor in graph.tensors:
        # Producer must exist
        if tensor.producer_node and tensor.producer_node not in node_ids:
            raise ConfigError(
                f"tensor {tensor.tensor_id!r} has unknown producer {tensor.producer_node!r}",
                field_path=f"tensors.{tensor.tensor_id}",
            )

        # Consumers must exist
        for consumer in tensor.consumed_by:
            if consumer not in node_ids:
                raise ConfigError(
                    f"tensor {tensor.tensor_id!r} has unknown consumer {consumer!r}",
                    field_path=f"tensors.{tensor.tensor_id}",
                )

    # Build producer map for cross-checking
    producer_map = {}
    for tensor in graph.tensors:
        if tensor.producer_node:
            producer_map[tensor.tensor_id] = tensor.producer_node

    for node in graph.nodes:
        # Inputs: should reference tensors produced by other nodes or inputs
        for inp_id in node.inputs:
            if inp_id not in tensor_ids:
                raise ConfigError(
                    f"node {node.node_id!r} input {inp_id!r} is not a known tensor",
                    field_path=f"nodes.{node.node_id}",
                )

        # Outputs: should reference tensors produced by this node
        for out_id in node.outputs:
            if out_id not in tensor_ids:
                raise ConfigError(
                    f"node {node.node_id!r} output {out_id!r} is not a known tensor",
                    field_path=f"nodes.{node.node_id}",
                )
            prod = producer_map.get(out_id)
            if prod and prod != node.node_id:
                raise ConfigError(
                    f"tensor {out_id!r} is listed as output of node {node.node_id!r} but producer_node is {prod!r}",
                    field_path=f"nodes.{node.node_id}",
                )

    # Detect self-loop data edges: tensor produced AND consumed by same node
    for node in graph.nodes:
        for out_id in node.outputs:
            if out_id in node.inputs:
                raise ConfigError(
                    f"node {node.node_id!r} has self-loop on tensor {out_id!r} "
                    f"(tensor appears in both inputs and outputs)",
                    field_path=f"nodes.{node.node_id}",
                )


def validate_all(
    graph: WorkloadGraphV1,
    bindings: DimensionBindings,
    registry: OperatorRegistry,
) -> None:
    """Run all validation gates on a workload graph before execution.

    This is the canonical pre-execution validation entry point. It runs
    graph DAG, dimension, operator, and tensor lifetime validation in order,
    failing fast on the first error.

    Args:
        graph: The workload graph to validate.
        bindings: Concrete dimension bindings.
        registry: The operator registry to check against.

    Raises:
        ConfigError: on structural graph errors.
        DimensionBindingError: on unbound symbolic dimensions.
        UnsupportedOperatorError: on unsupported or unregistered operators.
    """
    validate_graph_dag(graph)
    validate_dimensions(graph, bindings)
    validate_operators(graph, registry)
    validate_tensor_lifetime(graph)
