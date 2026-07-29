"""Versioned declarative workload graph schema (v1).

Defines the canonical representation of NPU workloads as typed DAGs:
- ``WorkloadGraphV1``: root container with version-gated validation
- ``NodeSpec``: compute/data-movement nodes with stable IDs and typed dependencies
- ``TensorSpec``: typed tensor descriptors with shape, layout, precision, lifetime, alias
- ``SymbolicDim``: named symbolic axis for dimension-binding before cost
- ``WorkloadProvenance``: audit trail for graph, node, and tensor origins

Rules enforced at construction:
1. Graph **must** be a DAG — cycle detection returns typed ``SchemaVersionError`` or ``ConfigError``.
2. Unknown fields and unsupported versions fail-closed (``extra='forbid'``)
3. Tensor shapes accept ``int`` (fixed) or ``str`` (named symbolic dim).
4. Alias can only reference existing tensor IDs.
5. Lifetimes (producer node, consumers) must be internally consistent.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Annotated, Dict, List, Optional, Set, Tuple, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from contracts.errors import ConfigError, SchemaVersionError


# ── Shape element: int | str ─────────────────────────────────────────────────

ShapeElement = Union[int, str]
"""A single dimension value: either a positive integer (fixed shape) or a
named symbolic dimension (e.g. ``"batch"``, ``"seq_len"``)."""

Shape = List[ShapeElement]
"""A list of shape elements representing a full tensor shape tuple."""


def _validate_shape_element(v: Any) -> ShapeElement:
    """Validate a single shape element."""
    if isinstance(v, bool):
        raise ValueError("bool value is not allowed as shape element")
    if isinstance(v, int):
        if v <= 0:
            raise ValueError(f"shape dimension must be positive, got {v}")
        return v
    if isinstance(v, str):
        if not v.strip():
            raise ValueError("symbolic dimension name must not be empty")
        return v
    raise ValueError(f"shape element must be int or str, got {type(v).__name__}")


def _validate_shape(v: Any) -> Shape:
    """Validate a full shape list."""
    if not isinstance(v, list):
        raise ValueError("shape must be a list")
    return [_validate_shape_element(e) for e in v]


ValidatedShape = Annotated[Shape, AfterValidator(_validate_shape)]


# ── Provenance ───────────────────────────────────────────────────────────────


class WorkloadProvenance(BaseModel):
    """Audit trail for workload graph elements.

    Distinct from ``contracts.hardware.Provenance`` — this carries workload-level
    provenance (e.g. ONNX model origin, hand-crafted trace, paper specification).
    """

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="Human-readable source description (e.g. 'onnx:vit-b16', 'hand-crafted:qwen2.5-3b')")
    reference_uri: Optional[str] = Field(default=None, description="Optional URI/DOI/path to source specification")


# ── Symbolic dimension ───────────────────────────────────────────────────────


class SymbolicDim(BaseModel):
    """A named symbolic axis for dimension-binding.

    Example: ``SymbolicDim(name="batch", description="Request batch size")``
    is resolved to a concrete integer value before cost estimation.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Unique symbolic axis name (e.g. 'batch', 'seq_len', 'image_count')")
    description: str = Field(default="", description="Human-readable description of what this dimension represents")


# ── Tensor spec ──────────────────────────────────────────────────────────────


class Layout(str, Enum):
    """Tensor data layout."""
    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"
    NHWC = "nhwc"
    NCHW = "nchw"
    BLOCKED = "blocked"


class Precision(str, Enum):
    """Numeric precision."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"
    MIXED_INT8_INT4 = "int8_int4"
    MIXED_INT8_INT8 = "int8_int8"


class TensorSpec(BaseModel):
    """Typed tensor descriptor with stable ID, shape, and lifetime information.

    Lifetime fields:
    - ``producer_node``: the node that writes this tensor (may be ``"input"`` for graph inputs)
    - ``consumed_by``: list of node IDs that read this tensor (empty for graph outputs)
    """

    model_config = ConfigDict(extra="forbid")

    tensor_id: str = Field(..., description="Stable, unique tensor identifier (e.g. 't_3f1a')")
    shape: ValidatedShape = Field(..., description="Tensor shape: fixed int dims or named symbolic dims")
    precision: Precision = Field(default=Precision.FP16, description="Element precision")
    layout: Layout = Field(default=Layout.ROW_MAJOR, description="Data layout in memory")
    bytes: int = Field(default=0, description="Total tensor size in bytes (can be 0 when not yet computed from shape/precision)")

    # Lifetime
    producer_node: str = Field(default="", description="Node that produces this tensor; 'input' for graph inputs")
    consumed_by: List[str] = Field(default_factory=list, description="Node IDs that consume this tensor")

    @field_validator("shape", mode="before")
    @classmethod
    def _reject_bool_in_shape(cls, v: Any) -> Any:
        """Reject boolean values in shape before Pydantic coerces bool→int."""
        if isinstance(v, bool):
            raise ValueError("bool value is not allowed as shape element")
        if isinstance(v, list):
            for item in v:
                if isinstance(item, bool):
                    raise ValueError("bool value is not allowed as shape element")
        return v

    # Alias support
    alias_of: Optional[str] = Field(default=None, description="If this tensor is an alias of another tensor, reference its tensor_id")

    provenance: Optional[WorkloadProvenance] = None

    @field_validator("tensor_id")
    @classmethod
    def _tensor_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("tensor_id must not be empty")
        return v

    @field_validator("bytes")
    @classmethod
    def _bytes_nonnegative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"bytes must be non-negative, got {v}")
        return v


# ── Node spec ────────────────────────────────────────────────────────────────


class NodeSpec(BaseModel):
    """A single operation node in the workload graph.

    Dependency model:
    - ``inputs``: tensor IDs consumed by this node.
    - ``outputs``: tensor IDs produced by this node.
    - ``dependencies``: node IDs that must complete before this node can start
      (control dependencies, not data-flow).
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(..., description="Stable, unique node identifier (e.g. 'n_7b2c')")
    op_type: str = Field(..., description="Operator type key (e.g. 'gemm', 'layernorm', 'softmax', 'reshape', 'reduce_mean')")
    op_label: str = Field(default="", description="Human-readable label (e.g. 'Q_proj', 'FFN_gate')")

    # Data-flow edges as tensor IDs
    inputs: List[str] = Field(default_factory=list, description="Input tensor IDs consumed by this node")
    outputs: List[str] = Field(default_factory=list, description="Output tensor IDs produced by this node")

    # Control dependencies as node IDs
    dependencies: List[str] = Field(default_factory=list, description="Node IDs that must complete before this node starts")

    # Attributes (op-type-specific)
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Op-type-specific key-value attributes")

    provenance: Optional[WorkloadProvenance] = None

    @field_validator("node_id")
    @classmethod
    def _node_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("node_id must not be empty")
        return v

    @field_validator("op_type")
    @classmethod
    def _op_type_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("op_type must not be empty")
        return v


# ── Workload graph ────────────────────────────────────────────────────────────


class WorkloadGraphV1(BaseModel):
    """Versioned workload graph (v1).

    This is the **authoritative** representation of a workload.  All lowering
    adapters (JSON, ONNX, legacy trace) produce this schema, and all executors
    consume it.

    Validation gates (run at construction and available via ``validate()``):
    1. Version must be ``"1"`` — unsupported versions fail-closed.
    2. Graph must be a DAG (cycle detection returns typed error).
    3. All node/tensor IDs must be unique.
    4. All tensor references (inputs, outputs, alias) must resolve.
    5. All producer/consumer edges must reference existing nodes.
    """

    model_config = ConfigDict(extra="forbid")

    version: str = Field(default="1", description="Schema version — must be '1'")
    graph_name: str = Field(default="unnamed", description="Human-readable graph name")

    nodes: List[NodeSpec] = Field(default_factory=list, description="Operation nodes in the graph")
    tensors: List[TensorSpec] = Field(default_factory=list, description="Tensors flowing between nodes")
    symbols: List[SymbolicDim] = Field(default_factory=list, description="Declared symbolic dimensions")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Optional global constraints (e.g. memory budget, schedule hints)")

    provenance: Optional[WorkloadProvenance] = None

    @field_validator("version")
    @classmethod
    def _version_must_be_one(cls, v: str) -> str:
        if v != "1":
            raise ValueError(f"unsupported schema version: {v!r} (expected '1')")
        return v

    @model_validator(mode="after")
    def _validate_graph_integrity(self) -> "WorkloadGraphV1":
        """Run full DAG and referential integrity validation."""
        self._check_duplicate_ids()
        self._check_dag_cycles()
        self._check_tensor_references()
        self._check_alias_validity()
        return self

    # ── internal validation helpers ──────────────────────────────────────────

    def _check_duplicate_ids(self) -> None:
        """Verify all node and tensor IDs are unique."""
        node_ids: Set[str] = set()
        for node in self.nodes:
            if node.node_id in node_ids:
                raise ConfigError(
                    f"duplicate node_id: {node.node_id!r}",
                    field_path="nodes",
                )
            node_ids.add(node.node_id)

        tensor_ids: Set[str] = set()
        for tensor in self.tensors:
            if tensor.tensor_id in tensor_ids:
                raise ConfigError(
                    f"duplicate tensor_id: {tensor.tensor_id!r}",
                    field_path="tensors",
                )
            tensor_ids.add(tensor.tensor_id)

    def _check_dag_cycles(self) -> None:
        """Detect cycles in the dependency graph using topological sort (Kahn's algorithm)."""
        node_ids = {n.node_id for n in self.nodes}
        if not node_ids:
            return  # empty graph is a valid DAG

        # Build adjacency from explicit node dependencies
        in_degree: Dict[str, int] = {nid: 0 for nid in node_ids}
        adj: Dict[str, Set[str]] = {nid: set() for nid in node_ids}

        # Data-flow dependencies: if node A produces tensor T and node B consumes T,
        # then A → B
        producer_map: Dict[str, str] = {}  # tensor_id → node_id
        for tensor in self.tensors:
            if tensor.producer_node and tensor.producer_node != "input":
                producer_map[tensor.tensor_id] = tensor.producer_node

        for node in self.nodes:
            for inp_id in node.inputs:
                prod_node = producer_map.get(inp_id)
                if prod_node and prod_node in node_ids:
                    if prod_node == node.node_id:
                        raise ConfigError(
                            f"graph contains a cycle: node {node.node_id!r} produces "
                            f"and consumes tensor {inp_id!r} (self-loop)",
                            field_path="nodes",
                        )
                    if node.node_id not in adj[prod_node]:
                        adj[prod_node].add(node.node_id)
                        in_degree[node.node_id] += 1

        # Explicit node dependencies
        for node in self.nodes:
            for dep_id in node.dependencies:
                if dep_id in node_ids and dep_id != node.node_id:
                    if node.node_id not in adj[dep_id]:
                        adj[dep_id].add(node.node_id)
                        in_degree[node.node_id] += 1

        # Kahn topological sort
        queue: List[str] = [nid for nid, deg in in_degree.items() if deg == 0]
        visited_count = 0

        while queue:
            current = queue.pop(0)
            visited_count += 1
            for neighbor in adj[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited_count != len(node_ids):
            # Find the cycle participants
            cycle_nodes = [nid for nid, deg in in_degree.items() if deg > 0]
            raise ConfigError(
                f"graph contains a cycle involving {len(cycle_nodes)} nodes "
                f"(e.g. {sorted(cycle_nodes)[:5]!r})",
                field_path="nodes",
            )

    def _check_tensor_references(self) -> None:
        """Verify all tensor references in nodes and lifetimes are valid."""
        tensor_ids = {t.tensor_id for t in self.tensors}
        node_ids = {n.node_id for n in self.nodes}
        node_ids.add("input")  # special: graph input nodes

        for node in self.nodes:
            # Check inputs reference existing tensors
            for inp_id in node.inputs:
                if inp_id not in tensor_ids:
                    raise ConfigError(
                        f"node {node.node_id!r} references non-existent input tensor {inp_id!r}",
                        field_path=f"nodes.{node.node_id}.inputs",
                    )
            # Check outputs reference existing tensors
            for out_id in node.outputs:
                if out_id not in tensor_ids:
                    raise ConfigError(
                        f"node {node.node_id!r} references non-existent output tensor {out_id!r}",
                        field_path=f"nodes.{node.node_id}.outputs",
                    )

        for tensor in self.tensors:
            # Check producer_node references an existing node
            if tensor.producer_node and tensor.producer_node not in node_ids:
                raise ConfigError(
                    f"tensor {tensor.tensor_id!r} has non-existent producer_node {tensor.producer_node!r}",
                    field_path=f"tensors.{tensor.tensor_id}.producer_node",
                )
            # Check consumed_by references existing nodes
            for consumer in tensor.consumed_by:
                if consumer not in node_ids:
                    raise ConfigError(
                        f"tensor {tensor.tensor_id!r} has non-existent consumer {consumer!r}",
                        field_path=f"tensors.{tensor.tensor_id}.consumed_by",
                    )

    def _check_alias_validity(self) -> None:
        """Verify alias_of references an existing tensor and is not self-referential."""
        tensor_ids = {t.tensor_id for t in self.tensors}

        for tensor in self.tensors:
            if tensor.alias_of:
                if tensor.alias_of not in tensor_ids:
                    raise ConfigError(
                        f"tensor {tensor.tensor_id!r} aliases non-existent tensor {tensor.alias_of!r}",
                        field_path=f"tensors.{tensor.tensor_id}.alias_of",
                    )
                if tensor.alias_of == tensor.tensor_id:
                    raise ConfigError(
                        f"tensor {tensor.tensor_id!r} cannot alias itself",
                        field_path=f"tensors.{tensor.tensor_id}.alias_of",
                    )

    # ── public API ────────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> NodeSpec:
        """Look up a node by ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(f"node not found: {node_id!r}")

    def get_tensor(self, tensor_id: str) -> TensorSpec:
        """Look up a tensor by ID."""
        for tensor in self.tensors:
            if tensor.tensor_id == tensor_id:
                return tensor
        raise KeyError(f"tensor not found: {tensor_id!r}")

    def topological_order(self) -> List[str]:
        """Return node IDs in topological order.

        Returns an empty list for an empty graph. Raises ``ConfigError`` if a
        cycle is present (should not happen after construction, but is checked).
        """
        self._check_dag_cycles()
        node_ids = {n.node_id for n in self.nodes}
        if not node_ids:
            return []

        # Build adjacency
        in_degree: Dict[str, int] = {nid: 0 for nid in node_ids}
        adj: Dict[str, Set[str]] = {nid: set() for nid in node_ids}

        producer_map: Dict[str, str] = {}
        for tensor in self.tensors:
            if tensor.producer_node and tensor.producer_node != "input":
                producer_map[tensor.tensor_id] = tensor.producer_node

        for node in self.nodes:
            for inp_id in node.inputs:
                prod_node = producer_map.get(inp_id)
                if prod_node and prod_node in node_ids and prod_node != node.node_id:
                    if node.node_id not in adj[prod_node]:
                        adj[prod_node].add(node.node_id)
                        in_degree[node.node_id] += 1
            for dep_id in node.dependencies:
                if dep_id in node_ids and dep_id != node.node_id:
                    if node.node_id not in adj[dep_id]:
                        adj[dep_id].add(node.node_id)
                        in_degree[node.node_id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: List[str] = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for neighbor in adj[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    def has_symbolic_shapes(self) -> bool:
        """Return True if any tensor in the graph has the name of a symbolic dimension in its shape."""
        symbol_names = {s.name for s in self.symbols}
        for tensor in self.tensors:
            for dim in tensor.shape:
                if isinstance(dim, str):
                    return True
                if isinstance(dim, str) and dim in symbol_names:
                    return True
        return False

    def unbound_symbols(self, bindings: Optional[Dict[str, int]] = None) -> Set[str]:
        """Return the set of symbolic dimension names referenced in shapes but not bound.

        Args:
            bindings: Current dimension bindings (symbol_name → concrete int).
                      If None, returns *all* symbolic names found.
        """
        symbol_names: Set[str] = set()
        for tensor in self.tensors:
            for dim in tensor.shape:
                if isinstance(dim, str):
                    symbol_names.add(dim)

        if bindings is None:
            return symbol_names

        return symbol_names - set(bindings.keys())

    def node_ids(self) -> Set[str]:
        """Return the set of all node IDs."""
        return {n.node_id for n in self.nodes}

    def tensor_ids(self) -> Set[str]:
        """Return the set of all tensor IDs."""
        return {t.tensor_id for t in self.tensors}


