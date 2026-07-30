"""Tests for workload graph schema — round-trip, DAG validation, and failure modes.

Covers:
- Happy path: JSON round-trip normalized graph equality
- DAG: cycle detection, dangling tensor references
- ID uniqueness: duplicate node/tensor IDs
- Alias validity: self-alias, non-existent alias target
- Symbolic dimensions: unbound symbols
- Version: unsupported versions fail-closed
"""

from __future__ import annotations

import pytest

# Pydantic model_validators wrap ValueError/ValueError into ValidationError,
# so tests use ValueError (the common ancestor) or Exception for robustness.
_WRAPPED_ERROR = Exception
from contracts.errors import DimensionBindingError
from workloads.dimensions import DimensionBindings
from workloads.operators import DEFAULT_REGISTRY
from workloads.schema import (
    Layout,
    NodeSpec,
    Precision,
    SymbolicDim,
    TensorSpec,
    WorkloadGraphV1,
    WorkloadProvenance,
)
from workloads.validate import validate_all, validate_dimensions

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_tensor(tid: str, shape: list, producer: str = "", consumers: list | None = None, **kwargs) -> TensorSpec:
    return TensorSpec(
        tensor_id=tid,
        shape=shape,
        producer_node=producer,
        consumed_by=consumers or [],
        **kwargs,
    )


def _make_node(
    nid: str, op: str, inputs: list | None = None, outputs: list | None = None, deps: list | None = None
) -> NodeSpec:
    return NodeSpec(
        node_id=nid,
        op_type=op,
        inputs=inputs or [],
        outputs=outputs or [],
        dependencies=deps or [],
    )


def _make_minimal_graph(name: str = "test_graph") -> WorkloadGraphV1:
    """Create a minimal valid 2-node DAG: gemm → relu."""
    t_input = _make_tensor("t_in", [1, 64, 64], producer="input", consumers=["n0"])
    t_mid = _make_tensor("t_mid", [1, 64, 64], producer="n0", consumers=["n1"])
    t_out = _make_tensor("t_out", [1, 64, 64], producer="n1")

    n0 = _make_node("n0", "gemm", inputs=["t_in"], outputs=["t_mid"])
    n1 = _make_node("n1", "relu", inputs=["t_mid"], outputs=["t_out"], deps=["n0"])

    return WorkloadGraphV1(
        graph_name=name,
        nodes=[n0, n1],
        tensors=[t_input, t_mid, t_out],
    )


# ── Happy path tests ─────────────────────────────────────────────────────────


class TestWorkloadGraphConstruction:
    """Test that valid graphs can be constructed without errors."""

    def test_minimal_dag_constructs(self):
        """A minimal gemm→relu DAG should construct without errors."""
        g = _make_minimal_graph()
        assert len(g.nodes) == 2
        assert len(g.tensors) == 3
        assert g.version == "1"

    def test_larger_dag_constructs(self):
        """A 4-node dag with skip connection should construct."""
        t_in = _make_tensor("t_in", [8, 256, 256], producer="input", consumers=["n0"])
        t_a = _make_tensor("t_a", [8, 256, 256], producer="n0", consumers=["n1"])
        t_b = _make_tensor("t_b", [8, 256, 256], producer="n1", consumers=["n2"])
        t_c = _make_tensor("t_c", [8, 256, 256], producer="n2", consumers=["n3"])
        t_d = _make_tensor("t_d", [8, 256, 256], producer="n3")

        g = WorkloadGraphV1(
            nodes=[
                _make_node("n0", "gemm", inputs=["t_in"], outputs=["t_a"]),
                _make_node("n1", "layernorm", inputs=["t_a"], outputs=["t_b"], deps=["n0"]),
                _make_node("n2", "gemm", inputs=["t_b"], outputs=["t_c"], deps=["n1"]),
                _make_node("n3", "add", inputs=["t_b", "t_c"], outputs=["t_d"], deps=["n2"]),
            ],
            tensors=[t_in, t_a, t_b, t_c, t_d],
        )
        assert g.topological_order() == ["n0", "n1", "n2", "n3"]

    def test_empty_graph_constructs(self):
        """An empty graph is a valid DAG."""
        g = WorkloadGraphV1()
        assert g.topological_order() == []

    def test_single_node_graph(self):
        """A single-node graph with no dependencies is valid."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n0"])
        t_out = _make_tensor("t_out", [1, 64], producer="n0")
        g = WorkloadGraphV1(
            nodes=[_make_node("n0", "relu", inputs=["t_in"], outputs=["t_out"])],
            tensors=[t_in, t_out],
        )
        assert g.topological_order() == ["n0"]

    def test_parallel_nodes(self):
        """Two independent nodes with no dependencies are valid."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n0", "n1"])
        t_a = _make_tensor("t_a", [1, 64], producer="n0")
        t_b = _make_tensor("t_b", [1, 64], producer="n1")
        g = WorkloadGraphV1(
            nodes=[
                _make_node("n0", "relu", inputs=["t_in"], outputs=["t_a"]),
                _make_node("n1", "gelu", inputs=["t_in"], outputs=["t_b"]),
            ],
            tensors=[t_in, t_a, t_b],
        )
        order = g.topological_order()
        assert set(order) == {"n0", "n1"}


class TestJsonRoundTrip:
    """Test JSON round-trip preserves normalized graph equality."""

    def test_round_trip_minimal(self):
        """JSON round-trip of a minimal graph should yield equal model."""
        g = _make_minimal_graph()
        j = g.model_dump_json(indent=2)
        g2 = WorkloadGraphV1.model_validate_json(j)
        assert g2.version == g.version
        assert g2.graph_name == g.graph_name
        assert len(g2.nodes) == len(g.nodes)
        assert len(g2.tensors) == len(g.tensors)
        # Stable IDs preserved
        assert g2.nodes[0].node_id == g.nodes[0].node_id
        assert g2.tensors[0].tensor_id == g.tensors[0].tensor_id

    def test_round_trip_with_symbolic_shapes(self):
        """Round-trip preserves symbolic dimension names."""
        t_in = _make_tensor("t_in", ["batch", 64, 64], producer="input", consumers=["n0"])
        t_out = _make_tensor("t_out", ["batch", 64, 64], producer="n0")
        g = WorkloadGraphV1(
            nodes=[_make_node("n0", "gemm", inputs=["t_in"], outputs=["t_out"])],
            tensors=[t_in, t_out],
            symbols=[SymbolicDim(name="batch", description="Request batch size")],
        )
        j = g.model_dump_json()
        g2 = WorkloadGraphV1.model_validate_json(j)
        assert g2.tensors[0].shape == ["batch", 64, 64]
        assert g2.symbols[0].name == "batch"

    def test_round_trip_with_enum_fields(self):
        """Round-trip preserves enum fields (layout, precision)."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n0"])
        t_out = _make_tensor("t_out", [1, 64], producer="n0", layout=Layout.NHWC, precision=Precision.INT8)
        g = WorkloadGraphV1(
            nodes=[_make_node("n0", "gemm", inputs=["t_in"], outputs=["t_out"])],
            tensors=[t_in, t_out],
        )
        j = g.model_dump_json()
        g2 = WorkloadGraphV1.model_validate_json(j)
        assert g2.tensors[1].layout == Layout.NHWC
        assert g2.tensors[1].precision == Precision.INT8

    def test_round_trip_with_provenance(self):
        """Round-trip preserves provenance information."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n0"])
        t_out = _make_tensor("t_out", [1, 64], producer="n0")
        g = WorkloadGraphV1(
            nodes=[_make_node("n0", "gemm", inputs=["t_in"], outputs=["t_out"])],
            tensors=[t_in, t_out],
            provenance=WorkloadProvenance(source="test:round-trip"),
        )
        j = g.model_dump_json()
        g2 = WorkloadGraphV1.model_validate_json(j)
        assert g2.provenance is not None
        assert g2.provenance.source == "test:round-trip"

    def test_normalized_graph_equal_after_double_roundtrip(self):
        """Two round-trips from the same graph produce identical normalized JSON."""
        g = _make_minimal_graph("normalized-test")
        j1 = g.model_dump_json(indent=2)
        g2 = WorkloadGraphV1.model_validate_json(j1)
        j2 = g2.model_dump_json(indent=2)
        assert j1 == j2

    def test_round_trip_complex_graph(self):
        """Full round-trip of a Gemm+GELU+Softmax+Reshape graph."""
        g, bindings = _make_gemm_gelu_softmax_reshape_graph()
        j = g.model_dump_json(indent=2)
        g2 = WorkloadGraphV1.model_validate_json(j)
        assert g2.graph_name == g.graph_name
        assert len(g2.nodes) == len(g.nodes)
        assert len(g2.tensors) == len(g.tensors)
        assert g2.symbols[0].name == "batch"


# ── DAG failure tests ────────────────────────────────────────────────────────


class TestDagCycleDetection:
    """Test that cycles in the graph are detected and fail with typed errors."""

    def test_simple_2_node_cycle(self):
        """A → B → A cycle must fail."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n0"])
        t_a = _make_tensor("t_a", [1, 64], producer="n0", consumers=["n1"])
        t_b = _make_tensor("t_b", [1, 64], producer="n1", consumers=["n0"])  # feeds back
        with pytest.raises(ValueError, match="cycle"):
            WorkloadGraphV1(
                nodes=[
                    _make_node("n0", "gemm", inputs=["t_in", "t_b"], outputs=["t_a"]),
                    _make_node("n1", "relu", inputs=["t_a"], outputs=["t_b"]),
                ],
                tensors=[t_in, t_a, t_b],
            )

    def test_self_loop_via_producer(self):
        """A node that produces a tensor it also consumes creates a self-loop."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n0"])
        t_out = _make_tensor("t_out", [1, 64], producer="n0", consumers=["n0"])  # self-loop
        with pytest.raises(ValueError, match="cycle"):
            WorkloadGraphV1(
                nodes=[_make_node("n0", "gemm", inputs=["t_in", "t_out"], outputs=["t_out"])],
                tensors=[t_in, t_out],
            )

    def test_explicit_dep_cycle(self):
        """Explicit node dependency cycle (n0→n1→n0 via deps field)."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n0", "n1"])
        t_a = _make_tensor("t_a", [1, 64], producer="n0")
        t_b = _make_tensor("t_b", [1, 64], producer="n1")
        with pytest.raises(ValueError, match="cycle"):
            WorkloadGraphV1(
                nodes=[
                    _make_node("n0", "gemm", inputs=["t_in"], outputs=["t_a"], deps=["n1"]),
                    _make_node("n1", "relu", inputs=["t_in"], outputs=["t_b"], deps=["n0"]),
                ],
                tensors=[t_in, t_a, t_b],
            )

    def test_three_node_cycle(self):
        """A→B→C→A triangle cycle must fail."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n0"])
        t_a = _make_tensor("t_a", [1, 64], producer="n0", consumers=["n1"])
        t_b = _make_tensor("t_b", [1, 64], producer="n1", consumers=["n2"])
        t_c = _make_tensor("t_c", [1, 64], producer="n2", consumers=["n0"])  # cycle
        with pytest.raises(ValueError, match="cycle"):
            WorkloadGraphV1(
                nodes=[
                    _make_node("n0", "gemm", inputs=["t_in", "t_c"], outputs=["t_a"]),
                    _make_node("n1", "relu", inputs=["t_a"], outputs=["t_b"]),
                    _make_node("n2", "gelu", inputs=["t_b"], outputs=["t_c"]),
                ],
                tensors=[t_in, t_a, t_b, t_c],
            )


class TestDanglingTensorReferences:
    """Test that references to non-existent tensors are caught."""

    def test_input_refs_missing_tensor(self):
        """A node input that references a non-existent tensor must fail."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n0"])
        with pytest.raises(ValueError, match="non-existent input tensor"):
            WorkloadGraphV1(
                nodes=[_make_node("n0", "gemm", inputs=["t_in", "t_missing"], outputs=["t_out"])],
                tensors=[t_in, _make_tensor("t_out", [1, 64], producer="n0")],
            )

    def test_output_refs_missing_tensor(self):
        """A node output that references a non-existent tensor must fail."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n0"])
        with pytest.raises(ValueError, match="non-existent output tensor"):
            WorkloadGraphV1(
                nodes=[_make_node("n0", "gemm", inputs=["t_in"], outputs=["t_missing"])],
                tensors=[t_in],
            )

    def test_producer_refs_missing_node(self):
        """A tensor whose producer_node references a non-existent node must fail."""
        t_in = _make_tensor("t_in", [1, 64], producer="n_missing", consumers=[])
        with pytest.raises(ValueError, match="non-existent producer_node"):
            WorkloadGraphV1(nodes=[], tensors=[t_in])

    def test_consumer_refs_missing_node(self):
        """A tensor whose consumed_by references a non-existent node must fail."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n_missing"])
        with pytest.raises(ValueError, match="non-existent consumer"):
            WorkloadGraphV1(nodes=[], tensors=[t_in])


# ── ID uniqueness tests ──────────────────────────────────────────────────────


class TestDuplicateIds:
    """Test that duplicate node/tensor IDs are rejected."""

    def test_duplicate_node_id(self):
        """Two nodes with the same ID must fail."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n0"])
        t_a = _make_tensor("t_a", [1, 64], producer="n0")
        with pytest.raises(ValueError, match="duplicate node_id"):
            WorkloadGraphV1(
                nodes=[
                    _make_node("n0", "gemm", inputs=["t_in"], outputs=["t_a"]),
                    _make_node("n0", "relu", inputs=["t_a"], outputs=["t_b"]),
                ],
                tensors=[t_in, t_a, _make_tensor("t_b", [1, 64], producer="n0")],
            )

    def test_duplicate_tensor_id(self):
        """Two tensors with the same ID must fail."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n0"])
        t_a1 = _make_tensor("t_a", [1, 64], producer="n0", consumers=["n1"])
        t_a2 = _make_tensor("t_a", [1, 32], producer="n1")  # duplicate ID
        with pytest.raises(ValueError, match="duplicate tensor_id"):
            WorkloadGraphV1(
                nodes=[
                    _make_node("n0", "gemm", inputs=["t_in"], outputs=["t_a"]),
                    _make_node("n1", "relu", inputs=["t_a"], outputs=["t_out"]),
                ],
                tensors=[t_in, t_a1, t_a2, _make_tensor("t_out", [1, 32], producer="n1")],
            )


# ── Alias validity tests ─────────────────────────────────────────────────────


class TestIllegalAlias:
    """Test that illegal tensor aliases are rejected."""

    def test_alias_of_self(self):
        """A tensor that aliases itself must fail."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n0"])
        t_out = _make_tensor("t_out", [1, 64], producer="n0", alias_of="t_out")
        with pytest.raises(ValueError, match="cannot alias itself"):
            WorkloadGraphV1(
                nodes=[_make_node("n0", "gemm", inputs=["t_in"], outputs=["t_out"])],
                tensors=[t_in, t_out],
            )

    def test_alias_of_non_existent(self):
        """A tensor that aliases a non-existent tensor must fail."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n0"])
        t_out = _make_tensor("t_out", [1, 64], producer="n0", alias_of="t_missing")
        with pytest.raises(ValueError, match="aliases non-existent tensor"):
            WorkloadGraphV1(
                nodes=[_make_node("n0", "gemm", inputs=["t_in"], outputs=["t_out"])],
                tensors=[t_in, t_out],
            )

    def test_valid_alias(self):
        """A valid alias (referencing another existing tensor) should succeed."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n0"])
        t_out = _make_tensor("t_out", [1, 64], producer="n0")
        t_alias = _make_tensor("t_alias", [1, 64], producer="n0", alias_of="t_out")
        g = WorkloadGraphV1(
            nodes=[_make_node("n0", "gemm", inputs=["t_in"], outputs=["t_out"])],
            tensors=[t_in, t_out, t_alias],
        )
        assert g.tensors[2].alias_of == "t_out"


# ── Version fail tests ───────────────────────────────────────────────────────


class TestVersionFail:
    """Test that unsupported versions fail-closed."""

    def test_version_2_fails(self):
        """Version '2' should fail (only '1' is supported)."""
        t_in = _make_tensor("t_in", [1, 64], producer="input")
        with pytest.raises(ValueError, match="unsupported schema version"):
            WorkloadGraphV1(
                version="2",
                nodes=[],
                tensors=[t_in],
            )

    def test_version_empty_fails(self):
        """Empty version string should fail."""
        t_in = _make_tensor("t_in", [1, 64], producer="input")
        with pytest.raises(ValueError):
            WorkloadGraphV1(version="", nodes=[], tensors=[t_in])

    def test_unknown_field_fails(self):
        """Unknown field in the root model must be rejected."""
        t_in = _make_tensor("t_in", [1, 64], producer="input")
        with pytest.raises(ValueError):
            WorkloadGraphV1.model_validate(
                {
                    "version": "1",
                    "nodes": [],
                    "tensors": [t_in.model_dump()],
                    "extra_unknown_field": "should_fail",
                }
            )


# ── Dimension binding tests ─────────────────────────────────────────────────


class TestSymbolicDimensions:
    """Test symbolic dimension binding and validation."""

    def test_unbound_symbol_detected(self):
        """Graph with symbolic dims and no bindings should report unbound symbols."""
        t_in = _make_tensor("t_in", ["batch", 64, 64], producer="input", consumers=["n0"])
        t_out = _make_tensor("t_out", ["batch", 64, 64], producer="n0")
        g = WorkloadGraphV1(
            nodes=[_make_node("n0", "gemm", inputs=["t_in"], outputs=["t_out"])],
            tensors=[t_in, t_out],
            symbols=[SymbolicDim(name="batch")],
        )
        assert g.unbound_symbols() == {"batch"}
        # Validate with no bindings should raise
        with pytest.raises(DimensionBindingError):
            validate_dimensions(g, DimensionBindings())

    def test_bound_symbol_passes(self):
        """Graph with symbolic dims and matching bindings should pass."""
        t_in = _make_tensor("t_in", ["batch", 64, 64], producer="input", consumers=["n0"])
        t_out = _make_tensor("t_out", ["batch", 64, 64], producer="n0")
        g = WorkloadGraphV1(
            nodes=[_make_node("n0", "gemm", inputs=["t_in"], outputs=["t_out"])],
            tensors=[t_in, t_out],
            symbols=[SymbolicDim(name="batch")],
        )
        bindings = DimensionBindings(request_batch=8)
        # "batch" axis maps to request_batch
        validate_dimensions(g, bindings)  # should not raise

    def test_mixed_fixed_and_symbolic_shape(self):
        """Tensors with mixed fixed ints and symbolic names are valid."""
        t_in = _make_tensor("t_in", ["batch", "seq_len", 64], producer="input", consumers=["n0"])
        t_out = _make_tensor("t_out", ["batch", "seq_len", 64], producer="n0")
        g = WorkloadGraphV1(
            nodes=[_make_node("n0", "gemm", inputs=["t_in"], outputs=["t_out"])],
            tensors=[t_in, t_out],
            symbols=[
                SymbolicDim(name="batch"),
                SymbolicDim(name="seq_len"),
            ],
        )
        assert g.unbound_symbols() == {"batch", "seq_len"}
        bindings = DimensionBindings(
            request_batch=4,
            extra={"seq_len": 128},
        )
        unbound = g.unbound_symbols(bindings.to_dict())
        assert unbound == set()


# ── Shape validation tests ───────────────────────────────────────────────────


class TestShapeValidation:
    """Test shape element validation."""

    def test_zero_dimension_fails(self):
        """A shape with 0 should be rejected."""
        with pytest.raises(ValueError, match="positive"):
            _make_tensor("t_in", [0, 64], producer="input")

    def test_negative_dimension_fails(self):
        """A shape with a negative number should be rejected."""
        with pytest.raises(ValueError, match="positive"):
            _make_tensor("t_in", [-1, 64], producer="input")

    def test_empty_string_symbolic_fails(self):
        """An empty string symbolic dim should be rejected."""
        with pytest.raises(ValueError, match="not be empty"):
            _make_tensor("t_in", ["", 64], producer="input")

    def test_bool_not_allowed(self):
        """Bool values should not be accepted as shape elements."""
        with pytest.raises(ValueError):
            _make_tensor("t_in", [True, 64], producer="input")


# ── Full validation (validate_all) ───────────────────────────────────────────


class TestValidateAll:
    """Test the comprehensive validate_all pre-execution gate."""

    def test_valid_graph_passes_all(self):
        """A valid graph with bindings and registry should pass all validators."""
        g = _make_minimal_graph()
        bindings = DimensionBindings()
        validate_all(g, bindings, DEFAULT_REGISTRY)

    def test_unsupported_op_fails_validate_all(self):
        """An unsupported op should fail validate_all."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n0"])
        t_out = _make_tensor("t_out", [1, 64], producer="n0")
        g = WorkloadGraphV1(
            nodes=[_make_node("n0", "scatter", inputs=["t_in"], outputs=["t_out"])],
            tensors=[t_in, t_out],
        )
        with pytest.raises(Exception, match="unsupported"):
            validate_all(g, DimensionBindings(), DEFAULT_REGISTRY)

    def test_unregistered_op_fails_validate_all(self):
        """An unregistered op should fail (treated as unsupported)."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n0"])
        t_out = _make_tensor("t_out", [1, 64], producer="n0")
        g = WorkloadGraphV1(
            nodes=[_make_node("n0", "custom_fancy_op", inputs=["t_in"], outputs=["t_out"])],
            tensors=[t_in, t_out],
        )
        with pytest.raises(Exception, match="unsupported"):
            validate_all(g, DimensionBindings(), DEFAULT_REGISTRY)


# ── Complex graph fixture for round-trip testing ─────────────────────────────


def _make_gemm_gelu_softmax_reshape_graph():
    """Build a Gemm→GELU→Softmax→Reshape graph with symbolic batch dim."""
    t_in = _make_tensor("t_in", ["batch", "seq_len", 256], producer="input", consumers=["n_gemm"])
    t_w = _make_tensor("t_w", [256, 256], producer="input", consumers=["n_gemm"])
    t_gemm_out = _make_tensor("t_gemm_out", ["batch", "seq_len", 256], producer="n_gemm", consumers=["n_gelu"])
    t_gelu_out = _make_tensor("t_gelu_out", ["batch", "seq_len", 256], producer="n_gelu", consumers=["n_softmax"])
    t_softmax_out = _make_tensor(
        "t_softmax_out", ["batch", "seq_len", 256], producer="n_softmax", consumers=["n_reshape"]
    )
    t_reshape_out = _make_tensor("t_reshape_out", ["batch", "seq_len", 256], producer="n_reshape")

    g = WorkloadGraphV1(
        graph_name="gemm-gelu-softmax-reshape",
        nodes=[
            _make_node("n_gemm", "gemm", inputs=["t_in", "t_w"], outputs=["t_gemm_out"]),
            _make_node("n_gelu", "gelu", inputs=["t_gemm_out"], outputs=["t_gelu_out"], deps=["n_gemm"]),
            _make_node("n_softmax", "softmax", inputs=["t_gelu_out"], outputs=["t_softmax_out"], deps=["n_gelu"]),
            _make_node("n_reshape", "reshape", inputs=["t_softmax_out"], outputs=["t_reshape_out"], deps=["n_softmax"]),
        ],
        tensors=[t_in, t_w, t_gemm_out, t_gelu_out, t_softmax_out, t_reshape_out],
        symbols=[
            SymbolicDim(name="batch", description="Request batch size"),
            SymbolicDim(name="seq_len", description="Sequence length"),
        ],
    )
    bindings = DimensionBindings(request_batch=4, extra={"seq_len": 128})
    return g, bindings
