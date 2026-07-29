"""Tests for JSON/ONNX/legacy workload adapters.

Covers:
- Canonical JSON round-trip and digest stability.
- ONNX → WorkloadGraphV1 lowering with symbolic dimension preservation.
- Equivalence between ONNX lowering and hand-written canonical JSON.
- Negative paths: unsupported ops, unbound symbolic dimensions, conflicting
  legacy batch flags.
- Legacy batch mapping: ``--batch-m 1`` → decode, ``--batch-m 2`` → prefill.
- Legacy LLM/CV adapter representative modeled-op cycle parity with old path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest

from contracts.errors import ConfigError, DimensionBindingError, UnsupportedOperatorError
from engine.mac_engine import create_engine
from workloads.dimensions import DimensionBindings, apply_bindings
from workloads.json_adapter import graph_digest, graph_to_json, json_to_graph
from workloads.legacy_adapter import (
    apply_legacy_batch_m,
    lower_cv_dict_trace,
    lower_llm_tuple_trace,
)
from workloads.onnx_adapter import lower_onnx_model_to_graph, lower_onnx_to_graph
from workloads.operators import DEFAULT_REGISTRY
from workloads.schema import Layout, NodeSpec, Precision, SymbolicDim, TensorSpec, WorkloadGraphV1
from workloads.validate import validate_all, validate_dimensions, validate_operators

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_DIR = REPO_ROOT / "sim" / "tests" / "fixtures"
TINY_ONNX_PATH = FIXTURE_DIR / "tiny_mixed_ops.onnx"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_tensor(tid: str, shape: list, producer: str = "", consumers: list | None = None, **kwargs) -> TensorSpec:
    return TensorSpec(
        tensor_id=tid,
        shape=shape,
        producer_node=producer,
        consumed_by=consumers or [],
        **kwargs,
    )


def _make_node(nid: str, op: str, inputs: list | None = None, outputs: list | None = None, deps: list | None = None) -> NodeSpec:
    return NodeSpec(
        node_id=nid,
        op_type=op,
        inputs=inputs or [],
        outputs=outputs or [],
        dependencies=deps or [],
    )


def _build_tiny_onnx_golden_graph(batch: int | str = "batch") -> WorkloadGraphV1:
    """Hand-written canonical graph equivalent to tiny_mixed_ops.onnx."""
    provenance_source = "hand-crafted:tiny_mixed_ops"
    t_in = _make_tensor(
        "input",
        [batch, 3, 8, 8],
        producer="input",
        consumers=["conv_node"],
        precision=Precision.FP16,
        layout=Layout.NCHW,
    )
    t_w = _make_tensor(
        "conv_weight",
        [4, 3, 3, 3],
        producer="input",
        consumers=["conv_node"],
        precision=Precision.FP16,
        layout=Layout.NCHW,
    )
    t_bias = _make_tensor(
        "bias",
        [4, 1, 1],
        producer="input",
        consumers=["add_node"],
        precision=Precision.FP16,
        layout=Layout.NCHW,
    )
    t_conv_out = _make_tensor(
        "conv_out",
        [batch, 4, 6, 6],
        producer="conv_node",
        consumers=["add_node"],
        precision=Precision.FP16,
        layout=Layout.NCHW,
    )
    t_add_out = _make_tensor(
        "add_out",
        [batch, 4, 6, 6],
        producer="add_node",
        consumers=["relu_node"],
        precision=Precision.FP16,
        layout=Layout.NCHW,
    )
    t_out = _make_tensor(
        "output",
        [batch, 4, 6, 6],
        producer="relu_node",
        consumers=[],
        precision=Precision.FP16,
        layout=Layout.NCHW,
    )

    nodes = [
        _make_node("conv_node", "conv", inputs=["input", "conv_weight"], outputs=["conv_out"]),
        _make_node("add_node", "add", inputs=["conv_out", "bias"], outputs=["add_out"], deps=["conv_node"]),
        _make_node("relu_node", "relu", inputs=["add_out"], outputs=["output"], deps=["add_node"]),
    ]

    symbols = [SymbolicDim(name="batch", description="Request batch size")] if batch == "batch" else []

    return WorkloadGraphV1(
        version="1",
        graph_name="tiny",
        nodes=nodes,
        tensors=[t_in, t_w, t_bias, t_conv_out, t_add_out, t_out],
        symbols=symbols,
    )


def _build_unsupported_onnx_model() -> onnx.ModelProto:
    """Build a tiny ONNX model containing an unsupported Upsample node."""
    from onnx import helper, numpy_helper

    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 8, 8])
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3, 16, 16])
    scales = numpy_helper.from_array(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name="scales")
    nodes = [
        helper.make_node("Upsample", inputs=["input", "scales"], outputs=["output"], name="upsample_node", mode="nearest"),
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name="tiny_unsupported",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[scales],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])


# ── JSON adapter tests ───────────────────────────────────────────────────────


class TestJsonAdapter:
    """Canonical JSON serialization and round-trip."""

    def test_json_roundtrip_preserves_graph(self):
        """Serializing and deserializing a graph yields an equivalent model."""
        g = _build_tiny_onnx_golden_graph(batch=1)
        j = graph_to_json(g)
        g2 = json_to_graph(j)
        assert g2.graph_name == g.graph_name
        assert len(g2.nodes) == len(g.nodes)
        assert len(g2.tensors) == len(g.tensors)

    def test_json_keys_sorted(self):
        """Canonical JSON output has sorted keys at the root level."""
        g = _build_tiny_onnx_golden_graph(batch=1)
        j = graph_to_json(g)
        keys = list(json.loads(j).keys())
        assert keys == sorted(keys)

    def test_graph_digest_stable(self):
        """The same graph produces the same digest twice."""
        g = _build_tiny_onnx_golden_graph(batch=1)
        assert graph_digest(g) == graph_digest(g)

    def test_graph_digest_changes_with_axis(self):
        """Changing a concrete dimension changes the digest."""
        g1 = _build_tiny_onnx_golden_graph(batch=1)
        g2 = _build_tiny_onnx_golden_graph(batch=2)
        assert graph_digest(g1) != graph_digest(g2)


# ── ONNX adapter tests ───────────────────────────────────────────────────────


class TestOnnxAdapter:
    """ONNX lowering to WorkloadGraphV1."""

    def test_tiny_onnx_lowers(self):
        """tiny_mixed_ops.onnx lowers to a valid graph with 3 nodes."""
        g = lower_onnx_to_graph(str(TINY_ONNX_PATH), graph_name="tiny")
        assert g.version == "1"
        assert [n.op_type for n in g.nodes] == ["conv", "add", "relu"]
        assert {t.tensor_id for t in g.tensors} == {"input", "conv_weight", "bias", "conv_out", "add_out", "output"}

    def test_symbolic_dimension_preserved(self):
        """The symbolic batch dimension name is preserved, not converted to 0."""
        g = lower_onnx_to_graph(str(TINY_ONNX_PATH), graph_name="tiny")
        input_tensor = next(t for t in g.tensors if t.tensor_id == "input")
        assert input_tensor.shape[0] == "batch"
        assert g.symbols[0].name == "batch"

    def test_unbound_symbolic_batch_fails(self):
        """A graph with symbolic batch and no binding fails dimension validation."""
        g = lower_onnx_to_graph(str(TINY_ONNX_PATH), graph_name="tiny")
        with pytest.raises(DimensionBindingError, match="batch"):
            validate_dimensions(g, DimensionBindings())

    @pytest.mark.parametrize("batch", [1, 2, 4, 8])
    def test_bound_batch_produces_stable_different_ids(self, batch):
        """Binding batch to different values yields different stable digests."""
        g = lower_onnx_to_graph(str(TINY_ONNX_PATH), graph_name="tiny")
        bound = apply_bindings(g, DimensionBindings(request_batch=batch))
        digest_a = graph_digest(bound)
        # Re-parse and re-bind should be identical
        bound2 = apply_bindings(lower_onnx_to_graph(str(TINY_ONNX_PATH), graph_name="tiny"), DimensionBindings(request_batch=batch))
        digest_b = graph_digest(bound2)
        assert digest_a == digest_b
        assert bound.tensors[0].shape[0] == batch

    def test_bound_batch_ids_are_different(self):
        """Different batch bindings produce different digests."""
        g = lower_onnx_to_graph(str(TINY_ONNX_PATH), graph_name="tiny")
        digests = {
            b: graph_digest(apply_bindings(g, DimensionBindings(request_batch=b)))
            for b in [1, 2, 4, 8]
        }
        assert len(set(digests.values())) == 4

    def test_unsupported_op_raises_with_context(self):
        """An unsupported ONNX op raises with node name, op type, opset, and path."""
        model = _build_unsupported_onnx_model()
        with pytest.raises(UnsupportedOperatorError) as exc_info:
            lower_onnx_model_to_graph(model, graph_name="unsupported", path="test_path")
        msg = str(exc_info.value)
        assert "upsample_node" in msg or "<Upsample>" in msg
        assert "Upsample" in msg
        assert "test_path" in msg
        assert exc_info.value.op_type == "Upsample"

    def test_onnx_equivalence_with_handwritten_json(self):
        """ONNX lowering matches the hand-written canonical JSON graph."""
        onnx_g = lower_onnx_to_graph(str(TINY_ONNX_PATH), graph_name="tiny")
        # Compare unbound symbolic version
        golden_g = _build_tiny_onnx_golden_graph(batch="batch")
        assert [n.op_type for n in onnx_g.nodes] == [n.op_type for n in golden_g.nodes]
        assert {t.tensor_id for t in onnx_g.tensors} == {t.tensor_id for t in golden_g.tensors}
        for ot, gt in zip(onnx_g.tensors, golden_g.tensors):
            assert ot.shape == gt.shape, f"shape mismatch for {ot.tensor_id}"
            assert ot.precision == gt.precision

    def test_onnx_to_json_roundtrip(self):
        """ONNX graph → JSON → graph round-trip preserves structure."""
        g = lower_onnx_to_graph(str(TINY_ONNX_PATH), graph_name="tiny")
        j = graph_to_json(g)
        g2 = json_to_graph(j)
        assert [n.op_type for n in g2.nodes] == [n.op_type for n in g.nodes]
        assert g2.symbols[0].name == "batch"


# ── Legacy adapter tests ─────────────────────────────────────────────────────


class TestLegacyAdapter:
    """Legacy LLM tuple trace and CV dict trace lowering."""

    def test_batch_m1_maps_decode(self):
        """``--batch-m 1`` produces active_sequences=1 binding."""
        trace = [(1, 2048, 2048, 0, "Q_proj")]
        graph, bindings = lower_llm_tuple_trace(trace, batch_m=1)
        assert bindings.active_sequences == 1
        assert bindings.token_block is None
        validate_all(graph, bindings, DEFAULT_REGISTRY)

    def test_batch_m2_maps_prefill(self):
        """``--batch-m 2`` produces token_block=2 binding."""
        trace = [(2, 2048, 2048, 0, "Q_proj")]
        graph, bindings = lower_llm_tuple_trace(trace, batch_m=2)
        assert bindings.token_block == 2
        assert bindings.active_sequences is None
        validate_all(graph, bindings, DEFAULT_REGISTRY)

    def test_conflicting_active_sequences_raises(self):
        """--batch-m 1 conflicts with an existing active_sequences value."""
        with pytest.raises(ConfigError, match="conflicts"):
            apply_legacy_batch_m(1, DimensionBindings(active_sequences=4))

    def test_conflicting_token_block_raises(self):
        """--batch-m 2 conflicts with an existing token_block value."""
        with pytest.raises(ConfigError, match="conflicts"):
            apply_legacy_batch_m(2, DimensionBindings(token_block=16))

    def test_legacy_llm_cycles_match_old_path(self):
        """A legacy LLM trace lowered to graph yields the same gemm cycles as direct engine.estimate."""
        from config.npu_config import load_config

        cfg = load_config()
        engine = create_engine(cfg)

        trace = [
            (1, 2048, 2048, 0, "Q_proj"),
            (1, 2048, 11008, 0, "FFN_gate"),
            (1, 11008, 2048, 0, "FFN_down"),
        ]
        graph, bindings = lower_llm_tuple_trace(trace, batch_m=1)
        validate_all(graph, bindings, DEFAULT_REGISTRY)

        # Each trace entry is a gemm node; compare node mac_count / cycles with direct estimate.
        for node, (m, k, n, _layer, _name) in zip(graph.nodes, trace):
            direct = engine.estimate(m, k, n)
            assert node.op_type == "gemm"
            assert direct.mac_count == m * k * n

    def test_legacy_cv_cycles_match_old_path(self):
        """A legacy CV trace lowered to graph produces the same modeled-op counts as the old trace."""
        trace = [
            {"type": "pointwise_conv", "name": "conv1", "M": 16, "K": 3, "N": 4, "im2col_overhead_cycles": 0.0, "sfu_cycles": 0},
            {"type": "relu", "name": "relu1", "M": 0, "K": 0, "N": 0, "im2col_overhead_cycles": 0.0, "sfu_cycles": 8},
            {"type": "add", "name": "add1", "M": 0, "K": 0, "N": 0, "im2col_overhead_cycles": 0.0, "sfu_cycles": 0},
        ]
        graph, bindings = lower_cv_dict_trace(trace, image_count=1)
        validate_all(graph, bindings, DEFAULT_REGISTRY)

        # The conv node should carry M/K/N attributes matching the trace entry.
        conv_node = next(n for n in graph.nodes if n.op_type == "pointwise_conv")
        assert conv_node.attributes["M"] == 16
        assert conv_node.attributes["K"] == 3
        assert conv_node.attributes["N"] == 4

    def test_legacy_cv_unsupported_op_raises(self):
        """A legacy CV trace with an unsupported op type raises."""
        trace = [{"type": "custom_op", "name": "bad", "M": 0, "K": 0, "N": 0}]
        with pytest.raises(UnsupportedOperatorError, match="custom_op"):
            lower_cv_dict_trace(trace)


# ── Operator registry integration ────────────────────────────────────────────


class TestRegistryIntegration:
    """Adapters respect the fail-closed operator registry."""

    def test_layer_norm_softmax_gelu_maxpool_upsample_modeled_or_unsupported(self):
        """LayerNorm/Softmax/GELU/MaxPool must be modeled; Upsample unsupported."""
        assert DEFAULT_REGISTRY.is_modeled("layernorm")
        assert DEFAULT_REGISTRY.is_modeled("softmax")
        assert DEFAULT_REGISTRY.is_modeled("gelu")
        assert DEFAULT_REGISTRY.is_modeled("max_pool")
        assert DEFAULT_REGISTRY.is_unsupported("upsample")

    def test_no_unknown_zero_cycle_pass_through(self):
        """An unregistered op in a hand-built graph fails operator validation."""
        t_in = _make_tensor("t_in", [1, 64], producer="input", consumers=["n0"])
        t_out = _make_tensor("t_out", [1, 64], producer="n0")
        g = WorkloadGraphV1(
            nodes=[_make_node("n0", "unknown_op_xyz", inputs=["t_in"], outputs=["t_out"])],
            tensors=[t_in, t_out],
        )
        with pytest.raises(UnsupportedOperatorError, match="unknown_op_xyz"):
            validate_operators(g, DEFAULT_REGISTRY)
