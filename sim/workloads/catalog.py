"""Executable workload fixture catalog.

Discovers YAML fixtures under ``sim/config/workloads/``, loads each into a
validated ``WorkloadGraphV1`` + ``DimensionBindings``, and produces a coverage
manifest.  Fixtures may declare a full inline graph or reference a trace builder
function (``module.path:function_name``) that returns a ``WorkloadGraphV1``.

Every fixture must separate ``source_facts`` from ``engineering_assumptions``
and carry an external ``reference_uri`` for source facts.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from contracts.errors import ConfigError, UnsupportedOperatorError
from workloads.dimensions import (
    ACTION_HORIZON_EDGES,
    AXIS_ACTION_HORIZON,
    AXIS_BATCH,
    AXIS_FLOW_STEPS,
    AXIS_IMAGE_COUNT,
    AXIS_INFLIGHT_JOBS,
    AXIS_RESIDENT_MODELS,
    AXIS_SEQUENCES,
    AXIS_TOKEN_BLOCK,
    FLOW_STEPS_EDGES,
    IMAGE_COUNT_EDGES,
    INFLIGHT_JOBS_EDGES,
    RESIDENT_MODELS_EDGES,
    STANDARD_BATCH_EDGES,
    STRESS_BATCH_EDGES,
    TOKEN_BLOCK_EDGES,
    TOKEN_BLOCK_VLM_VLA_EXT,
    DimensionBindings,
)
from workloads.json_adapter import graph_digest
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
from workloads.validate import validate_all

__all__ = [
    "WorkloadFixture",
    "discover_fixtures",
    "load_fixture",
    "load_all_fixtures",
    "build_coverage_manifest",
]

# ── Constants ────────────────────────────────────────────────────────────────

CATALOG_DIR = Path(__file__).resolve().parent.parent / "config" / "workloads"
"""Default directory containing workload fixture YAML files."""

_CANONICAL_AXES = frozenset(
    {
        AXIS_BATCH,
        AXIS_SEQUENCES,
        AXIS_TOKEN_BLOCK,
        AXIS_IMAGE_COUNT,
        AXIS_ACTION_HORIZON,
        AXIS_FLOW_STEPS,
        AXIS_RESIDENT_MODELS,
        AXIS_INFLIGHT_JOBS,
    }
)

_EDGE_VALUES: dict[str, frozenset[int]] = {
    AXIS_BATCH: frozenset(STANDARD_BATCH_EDGES[AXIS_BATCH]) | frozenset(STRESS_BATCH_EDGES[AXIS_BATCH]),
    AXIS_SEQUENCES: frozenset(STANDARD_BATCH_EDGES[AXIS_SEQUENCES]) | frozenset(STRESS_BATCH_EDGES[AXIS_SEQUENCES]),
    AXIS_TOKEN_BLOCK: frozenset(TOKEN_BLOCK_EDGES) | frozenset(TOKEN_BLOCK_VLM_VLA_EXT),
    AXIS_IMAGE_COUNT: frozenset(IMAGE_COUNT_EDGES),
    AXIS_ACTION_HORIZON: frozenset(ACTION_HORIZON_EDGES),
    AXIS_FLOW_STEPS: frozenset(FLOW_STEPS_EDGES),
    AXIS_RESIDENT_MODELS: frozenset(RESIDENT_MODELS_EDGES),
    AXIS_INFLIGHT_JOBS: frozenset(INFLIGHT_JOBS_EDGES),
}


# ── Workload fixture ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WorkloadFixture:
    """A single loaded and validated workload fixture."""

    name: str
    version: str
    provenance: WorkloadProvenance
    graph: WorkloadGraphV1
    bindings: DimensionBindings
    scenario: dict[str, Any]
    source_facts: tuple[dict[str, Any], ...]
    engineering_assumptions: tuple[dict[str, Any], ...]
    footprint_digest: str

    def provenance_summary(self) -> dict[str, Any]:
        """Return a deterministic summary of provenance for golden comparison."""
        return {
            "source": self.provenance.source,
            "reference_uri": self.provenance.reference_uri,
            "source_fact_count": len(self.source_facts),
            "engineering_assumption_count": len(self.engineering_assumptions),
            "source_fact_refs": sorted(
                {str(f.get("reference_uri", "")) for f in self.source_facts if f.get("reference_uri")}
            ),
            "engineering_refs": sorted(
                {str(a.get("reference_uri", "")) for a in self.engineering_assumptions if a.get("reference_uri")}
            ),
        }

    def node_tensor_counts(self) -> dict[str, int]:
        """Return deterministic node and tensor counts."""
        return {
            "node_count": len(self.graph.nodes),
            "tensor_count": len(self.graph.tensors),
            "symbol_count": len(self.graph.symbols),
        }


# ── YAML parsing helpers ─────────────────────────────────────────────────────


def _require_str(obj: Any, path: str) -> str:
    if not isinstance(obj, str) or not obj.strip():
        raise ConfigError(f"{path} must be a non-empty string", field_path=path)
    return obj


def _require_mapping(obj: Any, path: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ConfigError(f"{path} must be a mapping", field_path=path)
    return obj


def _parse_provenance(data: dict[str, Any], path: str) -> WorkloadProvenance:
    """Parse a provenance block; trust_level defaults to T0 if absent."""
    source = _require_str(data.get("source"), f"{path}.source")
    reference_uri = data.get("reference_uri")
    if reference_uri is not None and not isinstance(reference_uri, str):
        raise ConfigError(f"{path}.reference_uri must be a string or null", field_path=f"{path}.reference_uri")
    return WorkloadProvenance(source=source, reference_uri=reference_uri)


def _parse_dimensions(data: dict[str, Any], path: str) -> DimensionBindings:
    """Parse the dimensions mapping into a ``DimensionBindings``."""
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a mapping", field_path=path)
    canonical_map = {
        "request_batch": "request_batch",
        "active_sequences": "active_sequences",
        "token_block": "token_block",
        "image_count": "image_count",
        "action_horizon": "action_horizon",
        "flow_steps": "flow_steps",
        "resident_models": "resident_models",
        "inflight_jobs": "inflight_jobs",
    }
    kwargs: dict[str, Any] = {}
    extra: dict[str, int] = {}
    for key, value in data.items():
        if key == "extra":
            nested = _require_mapping(value, f"{path}.extra")
            for extra_key, extra_value in nested.items():
                if not isinstance(extra_value, int) or isinstance(extra_value, bool):
                    raise ConfigError(
                        f"{path}.extra.{extra_key} must be an integer",
                        field_path=f"{path}.extra.{extra_key}",
                    )
                extra[extra_key] = extra_value
            continue
        if key in canonical_map:
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise ConfigError(
                    f"{path}.{key} must be a positive integer or null",
                    field_path=f"{path}.{key}",
                )
            kwargs[canonical_map[key]] = value
        else:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ConfigError(
                    f"{path}.{key} extra dimension must be an integer",
                    field_path=f"{path}.{key}",
                )
            extra[key] = value
    kwargs["extra"] = extra
    return DimensionBindings(**kwargs)


def _parse_facts(data: Any, path: str) -> tuple[dict[str, Any], ...]:
    """Parse source_facts / engineering_assumptions lists."""
    if not isinstance(data, list):
        raise ConfigError(f"{path} must be a list", field_path=path)
    result: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ConfigError(f"{path}[{idx}] must be a mapping", field_path=f"{path}[{idx}]")
        result.append(dict(item))
    return tuple(result)


# ── Graph builders ───────────────────────────────────────────────────────────


def _make_tensor(
    tensor_id: str,
    shape: list[int | str],
    *,
    producer: str = "",
    consumers: list[str] | None = None,
    precision: Precision = Precision.FP16,
    layout: Layout = Layout.ROW_MAJOR,
    provenance: WorkloadProvenance | None = None,
) -> TensorSpec:
    return TensorSpec(
        tensor_id=tensor_id,
        shape=shape,
        precision=precision,
        layout=layout,
        producer_node=producer,
        consumed_by=consumers or [],
        bytes=0,
        provenance=provenance,
    )


def _make_node(
    node_id: str,
    op_type: str,
    *,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    dependencies: list[str] | None = None,
    op_label: str = "",
    attributes: dict[str, Any] | None = None,
    provenance: WorkloadProvenance | None = None,
) -> NodeSpec:
    return NodeSpec(
        node_id=node_id,
        op_type=op_type,
        op_label=op_label or node_id,
        inputs=inputs or [],
        outputs=outputs or [],
        dependencies=dependencies or [],
        attributes=dict(attributes or {}),
        provenance=provenance,
    )


def _linear_chain_deps(nodes: list[NodeSpec]) -> list[NodeSpec]:
    """Add linear control dependencies between consecutive nodes (pure helper)."""
    updated: list[NodeSpec] = []
    for idx, node in enumerate(nodes):
        deps = list(node.dependencies)
        if idx > 0 and nodes[idx - 1].node_id not in deps:
            deps.append(nodes[idx - 1].node_id)
        updated.append(
            _make_node(
                node_id=node.node_id,
                op_type=node.op_type,
                inputs=list(node.inputs),
                outputs=list(node.outputs),
                dependencies=deps,
                op_label=node.op_label,
                attributes=dict(node.attributes),
                provenance=node.provenance,
            )
        )
    return updated


def _wire_consumers(nodes: list[NodeSpec], tensors: list[TensorSpec]) -> None:
    """Recompute ``consumed_by`` for every tensor from node inputs.

    This avoids manual consumer bookkeeping in graph builders.  It mutates
    ``tensors`` in place but leaves node objects unchanged.
    """
    tensor_map = {t.tensor_id: t for t in tensors}
    for tensor in tensors:
        tensor.consumed_by = []
    for node in nodes:
        for inp_id in node.inputs:
            tensor = tensor_map.get(inp_id)
            if tensor is not None and node.node_id not in tensor.consumed_by:
                tensor.consumed_by.append(node.node_id)


def build_qwen25_3b_graph(
    *,
    num_layers: int = 36,
    hidden: int = 2048,
    intermediate: int = 11008,
    num_heads: int = 16,
    head_dim: int = 128,
    vocab_size: int = 151936,
) -> WorkloadGraphV1:
    """Build a representative Qwen2.5-3B decode graph with symbolic batch/seq."""
    provenance = WorkloadProvenance(
        source="model_spec:qwen2.5-3b",
        reference_uri="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct",
    )

    tensors: list[TensorSpec] = []
    nodes: list[NodeSpec] = []

    # Embedding: [batch, seq_len] -> [batch, seq_len, hidden]
    t_in = _make_tensor("input_ids", ["batch", "seq_len"], producer="input", provenance=provenance)
    t_emb_w = _make_tensor("embed_weight", [vocab_size, hidden], producer="input", provenance=provenance)
    t_emb_out = _make_tensor("embed_out", ["batch", "seq_len", hidden], producer="embed", provenance=provenance)
    tensors.extend([t_in, t_emb_w, t_emb_out])
    nodes.append(
        _make_node("embed", "gemm", inputs=["input_ids", "embed_weight"], outputs=["embed_out"], provenance=provenance)
    )

    prev_out = "embed_out"
    for layer in range(num_layers):
        prefix = f"layer{layer}"
        t_ln1 = _make_tensor(
            f"{prefix}_ln1_out", ["batch", "seq_len", hidden], producer=f"{prefix}_ln1", provenance=provenance
        )
        tensors.append(t_ln1)
        nodes.append(
            _make_node(
                f"{prefix}_ln1", "layernorm", inputs=[prev_out], outputs=[f"{prefix}_ln1_out"], provenance=provenance
            )
        )

        for proj in ("q", "k", "v"):
            t_w = _make_tensor(f"{prefix}_{proj}_weight", [hidden, hidden], producer="input", provenance=provenance)
            t_out = _make_tensor(
                f"{prefix}_{proj}_out",
                ["batch", "seq_len", hidden],
                producer=f"{prefix}_{proj}_proj",
                provenance=provenance,
            )
            tensors.extend([t_w, t_out])
            nodes.append(
                _make_node(
                    f"{prefix}_{proj}_proj",
                    "gemm",
                    inputs=[f"{prefix}_ln1_out", f"{prefix}_{proj}_weight"],
                    outputs=[f"{prefix}_{proj}_out"],
                    provenance=provenance,
                )
            )

        t_attn = _make_tensor(
            f"{prefix}_attn_out", ["batch", "seq_len", hidden], producer=f"{prefix}_attn_softmax", provenance=provenance
        )
        tensors.append(t_attn)
        nodes.append(
            _make_node(
                f"{prefix}_attn_softmax",
                "softmax",
                inputs=[f"{prefix}_q_out", f"{prefix}_k_out", f"{prefix}_v_out"],
                outputs=[f"{prefix}_attn_out"],
                provenance=provenance,
            )
        )

        t_o_w = _make_tensor(f"{prefix}_o_weight", [hidden, hidden], producer="input", provenance=provenance)
        t_o_out = _make_tensor(
            f"{prefix}_o_out", ["batch", "seq_len", hidden], producer=f"{prefix}_o_proj", provenance=provenance
        )
        tensors.extend([t_o_w, t_o_out])
        nodes.append(
            _make_node(
                f"{prefix}_o_proj",
                "gemm",
                inputs=[f"{prefix}_attn_out", f"{prefix}_o_weight"],
                outputs=[f"{prefix}_o_out"],
                provenance=provenance,
            )
        )

        t_add1 = _make_tensor(
            f"{prefix}_add1_out", ["batch", "seq_len", hidden], producer=f"{prefix}_add1", provenance=provenance
        )
        tensors.append(t_add1)
        nodes.append(
            _make_node(
                f"{prefix}_add1",
                "add",
                inputs=[prev_out, f"{prefix}_o_out"],
                outputs=[f"{prefix}_add1_out"],
                provenance=provenance,
            )
        )

        # LayerNorm 2
        t_ln2 = _make_tensor(
            f"{prefix}_ln2_out", ["batch", "seq_len", hidden], producer=f"{prefix}_ln2", provenance=provenance
        )
        tensors.append(t_ln2)
        nodes.append(
            _make_node(
                f"{prefix}_ln2",
                "layernorm",
                inputs=[f"{prefix}_add1_out"],
                outputs=[f"{prefix}_ln2_out"],
                provenance=provenance,
            )
        )

        # FFN expand + gate (SwiGLU style)
        for gate in ("expand", "gate"):
            t_w = _make_tensor(
                f"{prefix}_ffn_{gate}_weight", [hidden, intermediate], producer="input", provenance=provenance
            )
            t_out = _make_tensor(
                f"{prefix}_ffn_{gate}_out",
                ["batch", "seq_len", intermediate],
                producer=f"{prefix}_ffn_{gate}",
                provenance=provenance,
            )
            tensors.extend([t_w, t_out])
            nodes.append(
                _make_node(
                    f"{prefix}_ffn_{gate}",
                    "gemm",
                    inputs=[f"{prefix}_ln2_out", f"{prefix}_ffn_{gate}_weight"],
                    outputs=[f"{prefix}_ffn_{gate}_out"],
                    provenance=provenance,
                )
            )

        # SwiGLU mul
        t_swiglu = _make_tensor(
            f"{prefix}_swiglu_out",
            ["batch", "seq_len", intermediate],
            producer=f"{prefix}_swiglu",
            provenance=provenance,
        )
        tensors.append(t_swiglu)
        nodes.append(
            _make_node(
                f"{prefix}_swiglu",
                "mul",
                inputs=[f"{prefix}_ffn_expand_out", f"{prefix}_ffn_gate_out"],
                outputs=[f"{prefix}_swiglu_out"],
                provenance=provenance,
            )
        )

        # FFN project down
        t_proj_w = _make_tensor(
            f"{prefix}_ffn_proj_weight", [intermediate, hidden], producer="input", provenance=provenance
        )
        t_proj_out = _make_tensor(
            f"{prefix}_ffn_proj_out", ["batch", "seq_len", hidden], producer=f"{prefix}_ffn_proj", provenance=provenance
        )
        tensors.extend([t_proj_w, t_proj_out])
        nodes.append(
            _make_node(
                f"{prefix}_ffn_proj",
                "gemm",
                inputs=[f"{prefix}_swiglu_out", f"{prefix}_ffn_proj_weight"],
                outputs=[f"{prefix}_ffn_proj_out"],
                provenance=provenance,
            )
        )

        # Residual add
        t_add2 = _make_tensor(
            f"{prefix}_add2_out", ["batch", "seq_len", hidden], producer=f"{prefix}_add2", provenance=provenance
        )
        tensors.append(t_add2)
        nodes.append(
            _make_node(
                f"{prefix}_add2",
                "add",
                inputs=[f"{prefix}_add1_out", f"{prefix}_ffn_proj_out"],
                outputs=[f"{prefix}_add2_out"],
                provenance=provenance,
            )
        )

        prev_out = f"{prefix}_add2_out"

    # Final layer norm + classifier
    t_final_ln = _make_tensor("final_ln_out", ["batch", "seq_len", hidden], producer="final_ln", provenance=provenance)
    tensors.append(t_final_ln)
    nodes.append(
        _make_node("final_ln", "layernorm", inputs=[prev_out], outputs=["final_ln_out"], provenance=provenance)
    )

    t_lm_w = _make_tensor("lm_head_weight", [hidden, vocab_size], producer="input", provenance=provenance)
    t_logits = _make_tensor("logits", ["batch", "seq_len", vocab_size], producer="classifier", provenance=provenance)
    tensors.extend([t_lm_w, t_logits])
    nodes.append(
        _make_node(
            "classifier", "gemm", inputs=["final_ln_out", "lm_head_weight"], outputs=["logits"], provenance=provenance
        )
    )

    nodes = _linear_chain_deps(nodes)
    _wire_consumers(nodes, tensors)
    symbols = [
        SymbolicDim(name="batch", description="Request batch / active sequences"),
        SymbolicDim(name="seq_len", description="Token sequence length"),
    ]
    return WorkloadGraphV1(
        version="1",
        graph_name="llm-qwen25-3b",
        nodes=nodes,
        tensors=tensors,
        symbols=symbols,
        provenance=provenance,
    )


def build_vit_b16_graph(
    *,
    image_size: int = 224,
    patch_size: int = 16,
    in_channels: int = 3,
    hidden: int = 768,
    mlp_hidden: int = 3072,
    num_heads: int = 12,
    num_layers: int = 12,
    num_classes: int = 1000,
) -> WorkloadGraphV1:
    """Build a ViT-B/16 classification graph."""
    provenance = WorkloadProvenance(
        source="model_spec:vit-b16",
        reference_uri="https://arxiv.org/abs/2010.11929",
    )

    num_patches = (image_size // patch_size) ** 2
    seq_len = num_patches + 1
    patch_embed_k = patch_size * patch_size * in_channels

    tensors: list[TensorSpec] = []
    nodes: list[NodeSpec] = []

    t_input = _make_tensor(
        "image", ["image_count", in_channels, image_size, image_size], producer="input", provenance=provenance
    )
    t_patch_w = _make_tensor("patch_embed_weight", [patch_embed_k, hidden], producer="input", provenance=provenance)
    t_patch_out = _make_tensor(
        "patch_embed_out", ["image_count", num_patches, hidden], producer="patch_embed", provenance=provenance
    )
    tensors.extend([t_input, t_patch_w, t_patch_out])
    nodes.append(
        _make_node(
            "patch_embed",
            "gemm",
            inputs=["image", "patch_embed_weight"],
            outputs=["patch_embed_out"],
            provenance=provenance,
        )
    )

    prev_out = "patch_embed_out"
    for layer in range(num_layers):
        prefix = f"block{layer}"
        t_ln1 = _make_tensor(
            f"{prefix}_ln1_out",
            ["image_count", seq_len, hidden],
            producer=f"{prefix}_ln1",
            consumers=[f"{prefix}_q_proj", f"{prefix}_k_proj", f"{prefix}_v_proj"],
            provenance=provenance,
        )
        tensors.append(t_ln1)
        prev_tensor = next(t for t in tensors if t.tensor_id == prev_out)
        prev_tensor.consumed_by.append(f"{prefix}_ln1")
        nodes.append(
            _make_node(
                f"{prefix}_ln1", "layernorm", inputs=[prev_out], outputs=[f"{prefix}_ln1_out"], provenance=provenance
            )
        )

        for proj in ("q", "k", "v"):
            t_w = _make_tensor(f"{prefix}_{proj}_weight", [hidden, hidden], producer="input", provenance=provenance)
            t_out = _make_tensor(
                f"{prefix}_{proj}_out",
                ["image_count", seq_len, hidden],
                producer=f"{prefix}_{proj}_proj",
                provenance=provenance,
            )
            tensors.extend([t_w, t_out])
            nodes.append(
                _make_node(
                    f"{prefix}_{proj}_proj",
                    "gemm",
                    inputs=[f"{prefix}_ln1_out", f"{prefix}_{proj}_weight"],
                    outputs=[f"{prefix}_{proj}_out"],
                    provenance=provenance,
                )
            )

        t_attn = _make_tensor(
            f"{prefix}_attn_out",
            ["image_count", num_heads, seq_len, seq_len],
            producer=f"{prefix}_attn_softmax",
            provenance=provenance,
        )
        tensors.append(t_attn)
        nodes.append(
            _make_node(
                f"{prefix}_attn_softmax",
                "softmax",
                inputs=[f"{prefix}_q_out", f"{prefix}_k_out"],
                outputs=[f"{prefix}_attn_out"],
                provenance=provenance,
            )
        )

        t_v_out = next(t for t in tensors if t.tensor_id == f"{prefix}_v_out")
        t_v_out.consumed_by.append(f"{prefix}_v_mul")
        t_v_mul = _make_tensor(
            f"{prefix}_v_mul_out", ["image_count", seq_len, hidden], producer=f"{prefix}_v_mul", provenance=provenance
        )
        tensors.append(t_v_mul)
        nodes.append(
            _make_node(
                f"{prefix}_v_mul",
                "mul",
                inputs=[f"{prefix}_attn_out", f"{prefix}_v_out"],
                outputs=[f"{prefix}_v_mul_out"],
                provenance=provenance,
            )
        )

        t_o_w = _make_tensor(f"{prefix}_o_weight", [hidden, hidden], producer="input", provenance=provenance)
        t_o_out = _make_tensor(
            f"{prefix}_o_out", ["image_count", seq_len, hidden], producer=f"{prefix}_o_proj", provenance=provenance
        )
        tensors.extend([t_o_w, t_o_out])
        nodes.append(
            _make_node(
                f"{prefix}_o_proj",
                "gemm",
                inputs=[f"{prefix}_v_mul_out", f"{prefix}_o_weight"],
                outputs=[f"{prefix}_o_out"],
                provenance=provenance,
            )
        )

        t_add1 = _make_tensor(
            f"{prefix}_add1_out",
            ["image_count", seq_len, hidden],
            producer=f"{prefix}_add1",
            consumers=[f"{prefix}_ln2"],
            provenance=provenance,
        )
        tensors.append(t_add1)
        prev_tensor = next(t for t in tensors if t.tensor_id == prev_out)
        prev_tensor.consumed_by.append(f"{prefix}_add1")
        t_o_out.consumed_by.append(f"{prefix}_add1")
        nodes.append(
            _make_node(
                f"{prefix}_add1",
                "add",
                inputs=[prev_out, f"{prefix}_o_out"],
                outputs=[f"{prefix}_add1_out"],
                provenance=provenance,
            )
        )

        t_ln2 = _make_tensor(
            f"{prefix}_ln2_out",
            ["image_count", seq_len, hidden],
            producer=f"{prefix}_ln2",
            consumers=[f"{prefix}_mlp_expand"],
            provenance=provenance,
        )
        tensors.append(t_ln2)
        nodes.append(
            _make_node(
                f"{prefix}_ln2",
                "layernorm",
                inputs=[f"{prefix}_add1_out"],
                outputs=[f"{prefix}_ln2_out"],
                provenance=provenance,
            )
        )

        t_expand_w = _make_tensor(
            f"{prefix}_mlp_expand_weight", [hidden, mlp_hidden], producer="input", provenance=provenance
        )
        t_expand_out = _make_tensor(
            f"{prefix}_mlp_expand_out",
            ["image_count", seq_len, mlp_hidden],
            producer=f"{prefix}_mlp_expand",
            provenance=provenance,
        )
        tensors.extend([t_expand_w, t_expand_out])
        nodes.append(
            _make_node(
                f"{prefix}_mlp_expand",
                "gemm",
                inputs=[f"{prefix}_ln2_out", f"{prefix}_mlp_expand_weight"],
                outputs=[f"{prefix}_mlp_expand_out"],
                provenance=provenance,
            )
        )

        t_gelu = _make_tensor(
            f"{prefix}_mlp_gelu_out",
            ["image_count", seq_len, mlp_hidden],
            producer=f"{prefix}_mlp_gelu",
            consumers=[f"{prefix}_mlp_proj"],
            provenance=provenance,
        )
        tensors.append(t_gelu)
        nodes.append(
            _make_node(
                f"{prefix}_mlp_gelu",
                "gelu",
                inputs=[f"{prefix}_mlp_expand_out"],
                outputs=[f"{prefix}_mlp_gelu_out"],
                provenance=provenance,
            )
        )

        t_proj_w = _make_tensor(
            f"{prefix}_mlp_proj_weight", [mlp_hidden, hidden], producer="input", provenance=provenance
        )
        t_proj_out = _make_tensor(
            f"{prefix}_mlp_proj_out",
            ["image_count", seq_len, hidden],
            producer=f"{prefix}_mlp_proj",
            provenance=provenance,
        )
        tensors.extend([t_proj_w, t_proj_out])
        nodes.append(
            _make_node(
                f"{prefix}_mlp_proj",
                "gemm",
                inputs=[f"{prefix}_mlp_gelu_out", f"{prefix}_mlp_proj_weight"],
                outputs=[f"{prefix}_mlp_proj_out"],
                provenance=provenance,
            )
        )

        t_add2 = _make_tensor(
            f"{prefix}_add2_out", ["image_count", seq_len, hidden], producer=f"{prefix}_add2", provenance=provenance
        )
        tensors.append(t_add2)
        nodes.append(
            _make_node(
                f"{prefix}_add2",
                "add",
                inputs=[f"{prefix}_add1_out", f"{prefix}_mlp_proj_out"],
                outputs=[f"{prefix}_add2_out"],
                provenance=provenance,
            )
        )

        prev_out = f"{prefix}_add2_out"

    t_final_ln = _make_tensor(
        "final_ln_out", ["image_count", seq_len, hidden], producer="final_ln", provenance=provenance
    )
    tensors.append(t_final_ln)
    nodes.append(
        _make_node("final_ln", "layernorm", inputs=[prev_out], outputs=["final_ln_out"], provenance=provenance)
    )

    t_cls_w = _make_tensor("classifier_weight", [hidden, num_classes], producer="input", provenance=provenance)
    t_logits = _make_tensor("logits", ["image_count", num_classes], producer="classifier", provenance=provenance)
    tensors.extend([t_cls_w, t_logits])
    nodes.append(
        _make_node(
            "classifier",
            "gemm",
            inputs=["final_ln_out", "classifier_weight"],
            outputs=["logits"],
            provenance=provenance,
        )
    )

    nodes = _linear_chain_deps(nodes)
    _wire_consumers(nodes, tensors)
    symbols = [SymbolicDim(name="image_count", description="Number of input images")]
    return WorkloadGraphV1(
        version="1",
        graph_name="cv-vit-b16",
        nodes=nodes,
        tensors=tensors,
        symbols=symbols,
        provenance=provenance,
    )


def build_yolov8n_graph(
    *,
    image_size: int = 640,
    num_classes: int = 80,
) -> WorkloadGraphV1:
    """Build a simplified YOLOv8n graph using only modeled/free-fused ops.

    Upsampling is modeled as a free reshape (engineering assumption) because
    nearest-neighbor upsampling is not yet in the modeled operator registry.
    """
    provenance = WorkloadProvenance(
        source="model_spec:yolov8n",
        reference_uri="https://github.com/ultralytics/ultralytics",
    )

    tensors: list[TensorSpec] = []
    nodes: list[NodeSpec] = []

    def add_conv(name: str, in_shape: list[int | str], weight_shape: list[int], out_shape: list[int | str]) -> str:
        out_id = f"{name}_out"
        t_in = next((t for t in tensors if t.tensor_id == in_shape[0]), None)
        if t_in is None:
            t_in = _make_tensor(
                str(in_shape[0]), in_shape, producer="input" if in_shape[0] == "image" else "", provenance=provenance
            )
            tensors.append(t_in)
        t_w = _make_tensor(f"{name}_weight", weight_shape, producer="input", provenance=provenance)
        t_out = _make_tensor(out_id, out_shape, producer=name, provenance=provenance)
        tensors.extend([t_w, t_out])
        nodes.append(
            _make_node(
                name, "conv", inputs=[str(in_shape[0]), f"{name}_weight"], outputs=[out_id], provenance=provenance
            )
        )
        return out_id

    def add_relu(name: str, in_id: str, shape: list[int | str]) -> str:
        out_id = f"{name}_out"
        t_out = _make_tensor(out_id, shape, producer=name, provenance=provenance)
        tensors.append(t_out)
        nodes.append(_make_node(name, "relu", inputs=[in_id], outputs=[out_id], provenance=provenance))
        return out_id

    def add_maxpool(name: str, in_id: str, shape: list[int | str]) -> str:
        out_id = f"{name}_out"
        t_out = _make_tensor(out_id, shape, producer=name, provenance=provenance)
        tensors.append(t_out)
        nodes.append(_make_node(name, "max_pool", inputs=[in_id], outputs=[out_id], provenance=provenance))
        return out_id

    def add_concat(name: str, in_ids: list[str], shape: list[int | str]) -> str:
        out_id = f"{name}_out"
        t_out = _make_tensor(out_id, shape, producer=name, provenance=provenance)
        tensors.append(t_out)
        nodes.append(_make_node(name, "concat", inputs=in_ids, outputs=[out_id], provenance=provenance))
        return out_id

    # Backbone: stem -> stage1 -> c2f -> stage3 -> c2f -> stage5 -> c2f -> sppf
    t_image = _make_tensor("image", ["image_count", 3, image_size, image_size], producer="input", provenance=provenance)
    tensors.append(t_image)

    s0 = add_conv("stem", ["image"], [16, 3, 3, 3], ["image_count", 16, 320, 320])
    s0 = add_relu("stem_act", s0, ["image_count", 16, 320, 320])
    s1 = add_conv("stage1", [s0], [32, 16, 3, 3], ["image_count", 32, 160, 160])
    s1 = add_relu("stage1_act", s1, ["image_count", 32, 160, 160])
    s2 = add_conv("stage2_cv1", [s1], [32, 32, 1, 1], ["image_count", 32, 160, 160])
    s2 = add_relu("stage2_cv1_act", s2, ["image_count", 32, 160, 160])
    s3 = add_conv("stage3", [s2], [64, 32, 3, 3], ["image_count", 64, 80, 80])
    s3 = add_relu("stage3_act", s3, ["image_count", 64, 80, 80])
    s4 = add_conv("stage4_cv1", [s3], [64, 64, 1, 1], ["image_count", 64, 80, 80])
    s4 = add_relu("stage4_cv1_act", s4, ["image_count", 64, 80, 80])
    s5 = add_conv("stage5", [s4], [128, 64, 3, 3], ["image_count", 128, 40, 40])
    s5 = add_relu("stage5_act", s5, ["image_count", 128, 40, 40])
    s6 = add_conv("stage6_cv1", [s5], [128, 128, 1, 1], ["image_count", 128, 40, 40])
    s6 = add_relu("stage6_cv1_act", s6, ["image_count", 128, 40, 40])
    s7 = add_conv("stage7", [s6], [256, 128, 3, 3], ["image_count", 256, 20, 20])
    s7 = add_relu("stage7_act", s7, ["image_count", 256, 20, 20])
    s8 = add_conv("stage8_cv1", [s7], [256, 256, 1, 1], ["image_count", 256, 20, 20])
    s8 = add_relu("stage8_cv1_act", s8, ["image_count", 256, 20, 20])

    # SPPF: maxpool x3 then concat
    m0 = add_maxpool("sppf_m0", s8, ["image_count", 256, 20, 20])
    m1 = add_maxpool("sppf_m1", m0, ["image_count", 256, 20, 20])
    m2 = add_maxpool("sppf_m2", m1, ["image_count", 256, 20, 20])
    sppf = add_concat("sppf_concat", [s8, m0, m1, m2], ["image_count", 1024, 20, 20])
    sppf = add_conv("sppf_cv2", [sppf], [256, 1024, 1, 1], ["image_count", 256, 20, 20])
    sppf = add_relu("sppf_cv2_act", sppf, ["image_count", 256, 20, 20])

    p5_up = _make_tensor("p5_upsample_out", ["image_count", 256, 40, 40], producer="p5_upsample", provenance=provenance)
    tensors.append(p5_up)
    nodes.append(
        _make_node("p5_upsample", "reshape", inputs=[sppf], outputs=["p5_upsample_out"], provenance=provenance)
    )

    p4_concat = add_concat("neck_p4_concat", [s6, "p5_upsample_out"], ["image_count", 384, 40, 40])
    p4 = add_conv("neck_p4_cv1", [p4_concat], [128, 384, 1, 1], ["image_count", 128, 40, 40])
    p4 = add_relu("neck_p4_cv1_act", p4, ["image_count", 128, 40, 40])

    p4_up = _make_tensor("p4_upsample_out", ["image_count", 128, 80, 80], producer="p4_upsample", provenance=provenance)
    tensors.append(p4_up)
    nodes.append(_make_node("p4_upsample", "reshape", inputs=[p4], outputs=["p4_upsample_out"], provenance=provenance))

    p3_concat = add_concat("neck_p3_concat", [s4, "p4_upsample_out"], ["image_count", 192, 80, 80])
    p3 = add_conv("neck_p3_cv1", [p3_concat], [64, 192, 1, 1], ["image_count", 64, 80, 80])
    p3 = add_relu("neck_p3_cv1_act", p3, ["image_count", 64, 80, 80])

    # Downsample P3 -> P4
    p3_d = add_conv("neck_p3_down", [p3], [64, 64, 3, 3], ["image_count", 64, 40, 40])
    p3_d = add_relu("neck_p3_down_act", p3_d, ["image_count", 64, 40, 40])
    p4_2_concat = add_concat("neck_p4_2_concat", [p4, p3_d], ["image_count", 192, 40, 40])
    p4_2 = add_conv("neck_p4_2_cv1", [p4_2_concat], [128, 192, 1, 1], ["image_count", 128, 40, 40])
    p4_2 = add_relu("neck_p4_2_cv1_act", p4_2, ["image_count", 128, 40, 40])

    # Downsample P4 -> P5
    p4_2_d = add_conv("neck_p4_2_down", [p4_2], [128, 128, 3, 3], ["image_count", 128, 20, 20])
    p4_2_d = add_relu("neck_p4_2_down_act", p4_2_d, ["image_count", 128, 20, 20])
    p5_2_concat = add_concat("neck_p5_2_concat", [sppf, p4_2_d], ["image_count", 384, 20, 20])
    p5_2 = add_conv("neck_p5_2_cv1", [p5_2_concat], [256, 384, 1, 1], ["image_count", 256, 20, 20])
    p5_2 = add_relu("neck_p5_2_cv1_act", p5_2, ["image_count", 256, 20, 20])

    # Head branches
    for scale, channels, h_size, feat in (("p3", 64, 80, p3), ("p4", 128, 40, p4_2), ("p5", 256, 20, p5_2)):
        b0 = add_conv(
            f"head_{scale}_box0", [feat], [channels, channels, 3, 3], ["image_count", channels, h_size, h_size]
        )
        b0 = add_relu(f"head_{scale}_box0_act", b0, ["image_count", channels, h_size, h_size])
        b1 = add_conv(f"head_{scale}_box1", [b0], [channels, channels, 3, 3], ["image_count", channels, h_size, h_size])
        b1 = add_relu(f"head_{scale}_box1_act", b1, ["image_count", channels, h_size, h_size])
        add_conv(f"head_{scale}_box2", [b1], [64, channels, 1, 1], ["image_count", 64, h_size, h_size])
        c0 = add_conv(f"head_{scale}_cls0", [feat], [80, channels, 3, 3], ["image_count", 80, h_size, h_size])
        c0 = add_relu(f"head_{scale}_cls0_act", c0, ["image_count", 80, h_size, h_size])
        c1 = add_conv(f"head_{scale}_cls1", [c0], [80, 80, 3, 3], ["image_count", 80, h_size, h_size])
        c1 = add_relu(f"head_{scale}_cls1_act", c1, ["image_count", 80, h_size, h_size])
        add_conv(f"head_{scale}_cls2", [c1], [num_classes, 80, 1, 1], ["image_count", num_classes, h_size, h_size])

    nodes = _linear_chain_deps(nodes)
    _wire_consumers(nodes, tensors)
    symbols = [SymbolicDim(name="image_count", description="Number of input images")]
    return WorkloadGraphV1(
        version="1",
        graph_name="cv-yolov8n",
        nodes=nodes,
        tensors=tensors,
        symbols=symbols,
        provenance=provenance,
    )


def build_smolvla_graph(
    *,
    vision_hidden: int = 384,
    vision_layers: int = 6,
    vl_hidden: int = 768,
    vl_layers: int = 12,
    action_dim: int = 32,
    action_horizon: int = 8,
    flow_steps: int = 10,
) -> WorkloadGraphV1:
    """Build a compact SmolVLA-class graph: tiny ViT + VLM backbone + flow expert."""
    provenance = WorkloadProvenance(
        source="model_spec:smolvla-class",
        reference_uri="https://arxiv.org/html/2506.01844v1",
    )

    tensors: list[TensorSpec] = []
    nodes: list[NodeSpec] = []

    t_image = _make_tensor("image", ["image_count", 3, 224, 224], producer="input", provenance=provenance)
    tensors.append(t_image)
    patch_w = _make_tensor("vit_patch_weight", [588, vision_hidden], producer="input", provenance=provenance)
    patch_out = _make_tensor(
        "vit_patch_out", ["image_count", 197, vision_hidden], producer="vit_patch_embed", provenance=provenance
    )
    tensors.extend([patch_w, patch_out])
    nodes.append(
        _make_node(
            "vit_patch_embed",
            "gemm",
            inputs=["image", "vit_patch_weight"],
            outputs=["vit_patch_out"],
            provenance=provenance,
        )
    )

    prev = "vit_patch_out"
    for layer in range(vision_layers):
        prefix = f"vit{layer}"
        ln = _make_tensor(
            f"{prefix}_ln_out", ["image_count", 197, vision_hidden], producer=f"{prefix}_ln", provenance=provenance
        )
        tensors.append(ln)
        nodes.append(
            _make_node(f"{prefix}_ln", "layernorm", inputs=[prev], outputs=[f"{prefix}_ln_out"], provenance=provenance)
        )
        q_w = _make_tensor(
            f"{prefix}_q_weight", [vision_hidden, vision_hidden], producer="input", provenance=provenance
        )
        q_out = _make_tensor(
            f"{prefix}_q_out", ["image_count", 197, vision_hidden], producer=f"{prefix}_q_proj", provenance=provenance
        )
        k_w = _make_tensor(
            f"{prefix}_k_weight", [vision_hidden, vision_hidden], producer="input", provenance=provenance
        )
        k_out = _make_tensor(
            f"{prefix}_k_out", ["image_count", 197, vision_hidden], producer=f"{prefix}_k_proj", provenance=provenance
        )
        v_w = _make_tensor(
            f"{prefix}_v_weight", [vision_hidden, vision_hidden], producer="input", provenance=provenance
        )
        v_out = _make_tensor(
            f"{prefix}_v_out", ["image_count", 197, vision_hidden], producer=f"{prefix}_v_proj", provenance=provenance
        )
        o_w = _make_tensor(
            f"{prefix}_o_weight", [vision_hidden, vision_hidden], producer="input", provenance=provenance
        )
        o_out = _make_tensor(
            f"{prefix}_o_out", ["image_count", 197, vision_hidden], producer=f"{prefix}_o_proj", provenance=provenance
        )
        tensors.extend([q_w, q_out, k_w, k_out, v_w, v_out, o_w, o_out])
        nodes.extend(
            [
                _make_node(
                    f"{prefix}_q_proj",
                    "gemm",
                    inputs=[f"{prefix}_ln_out", f"{prefix}_q_weight"],
                    outputs=[f"{prefix}_q_out"],
                    provenance=provenance,
                ),
                _make_node(
                    f"{prefix}_k_proj",
                    "gemm",
                    inputs=[f"{prefix}_ln_out", f"{prefix}_k_weight"],
                    outputs=[f"{prefix}_k_out"],
                    provenance=provenance,
                ),
                _make_node(
                    f"{prefix}_v_proj",
                    "gemm",
                    inputs=[f"{prefix}_ln_out", f"{prefix}_v_weight"],
                    outputs=[f"{prefix}_v_out"],
                    provenance=provenance,
                ),
                _make_node(
                    f"{prefix}_o_proj",
                    "gemm",
                    inputs=[f"{prefix}_v_out", f"{prefix}_o_weight"],
                    outputs=[f"{prefix}_o_out"],
                    provenance=provenance,
                ),
            ]
        )
        prev = f"{prefix}_o_out"

    # Project visual tokens to VLM hidden
    proj_w = _make_tensor("vision_proj_weight", [vision_hidden, vl_hidden], producer="input", provenance=provenance)
    proj_out = _make_tensor(
        "vision_proj_out", ["image_count", 197, vl_hidden], producer="vision_proj", provenance=provenance
    )
    tensors.extend([proj_w, proj_out])
    nodes.append(
        _make_node(
            "vision_proj",
            "gemm",
            inputs=[prev, "vision_proj_weight"],
            outputs=["vision_proj_out"],
            provenance=provenance,
        )
    )

    vl_ln = _make_tensor("vl_ln_out", ["image_count", 197, vl_hidden], producer="vl_ln", provenance=provenance)
    tensors.append(vl_ln)
    nodes.append(
        _make_node("vl_ln", "layernorm", inputs=["vision_proj_out"], outputs=["vl_ln_out"], provenance=provenance)
    )
    for vl_layer in range(vl_layers):
        prefix = f"vl{vl_layer}"
        ffn_w = _make_tensor(
            f"{prefix}_ffn_weight", [vl_hidden, vl_hidden * 4], producer="input", provenance=provenance
        )
        ffn_out = _make_tensor(
            f"{prefix}_ffn_out", ["image_count", 197, vl_hidden * 4], producer=f"{prefix}_ffn", provenance=provenance
        )
        gelu_out = _make_tensor(
            f"{prefix}_gelu_out", ["image_count", 197, vl_hidden * 4], producer=f"{prefix}_gelu", provenance=provenance
        )
        proj2_w = _make_tensor(
            f"{prefix}_proj_weight", [vl_hidden * 4, vl_hidden], producer="input", provenance=provenance
        )
        proj2_out = _make_tensor(
            f"{prefix}_proj_out", ["image_count", 197, vl_hidden], producer=f"{prefix}_proj", provenance=provenance
        )
        tensors.extend([ffn_w, ffn_out, gelu_out, proj2_w, proj2_out])
        nodes.extend(
            [
                _make_node(
                    f"{prefix}_ffn",
                    "gemm",
                    inputs=["vl_ln_out" if vl_layer == 0 else f"vl{vl_layer - 1}_proj_out", f"{prefix}_ffn_weight"],
                    outputs=[f"{prefix}_ffn_out"],
                    provenance=provenance,
                ),
                _make_node(
                    f"{prefix}_gelu",
                    "gelu",
                    inputs=[f"{prefix}_ffn_out"],
                    outputs=[f"{prefix}_gelu_out"],
                    provenance=provenance,
                ),
                _make_node(
                    f"{prefix}_proj",
                    "gemm",
                    inputs=[f"{prefix}_gelu_out", f"{prefix}_proj_weight"],
                    outputs=[f"{prefix}_proj_out"],
                    provenance=provenance,
                ),
            ]
        )
        prev = f"{prefix}_proj_out"

    t_action_in = _make_tensor("action_in", ["batch", action_dim], producer="input", provenance=provenance)
    tensors.append(t_action_in)
    for step in range(flow_steps):
        prefix = f"flow{step}"
        in_dim = action_dim if step == 0 else action_dim * action_horizon
        w1 = _make_tensor(f"{prefix}_w1", [in_dim, 256], producer="input", provenance=provenance)
        h = _make_tensor(f"{prefix}_hidden", ["batch", 256], producer=f"{prefix}_fc1", provenance=provenance)
        gelu_out = _make_tensor(f"{prefix}_gelu_out", ["batch", 256], producer=f"{prefix}_gelu", provenance=provenance)
        w2 = _make_tensor(f"{prefix}_w2", [256, action_dim * action_horizon], producer="input", provenance=provenance)
        out = _make_tensor(
            f"{prefix}_out", ["batch", action_dim * action_horizon], producer=f"{prefix}_fc2", provenance=provenance
        )
        tensors.extend([w1, h, gelu_out, w2, out])
        nodes.extend(
            [
                _make_node(
                    f"{prefix}_fc1",
                    "gemm",
                    inputs=["action_in" if step == 0 else f"flow{step - 1}_out", f"{prefix}_w1"],
                    outputs=[f"{prefix}_hidden"],
                    provenance=provenance,
                ),
                _make_node(
                    f"{prefix}_gelu",
                    "gelu",
                    inputs=[f"{prefix}_hidden"],
                    outputs=[f"{prefix}_gelu_out"],
                    provenance=provenance,
                ),
                _make_node(
                    f"{prefix}_fc2",
                    "gemm",
                    inputs=[f"{prefix}_gelu_out", f"{prefix}_w2"],
                    outputs=[f"{prefix}_out"],
                    provenance=provenance,
                ),
            ]
        )

    nodes = _linear_chain_deps(nodes)
    _wire_consumers(nodes, tensors)
    symbols = [
        SymbolicDim(name="image_count", description="Number of camera views"),
        SymbolicDim(name="batch", description="Inference batch"),
    ]
    return WorkloadGraphV1(
        version="1",
        graph_name="smolvla-class",
        nodes=nodes,
        tensors=tensors,
        symbols=symbols,
        provenance=provenance,
    )


def build_pi0_graph(
    *,
    image_count: int = 3,
    vision_hidden: int = 1152,
    vl_hidden: int = 2048,
    action_dim: int = 32,
    action_horizon: int = 50,
    flow_steps: int = 10,
) -> WorkloadGraphV1:
    """Build a pi0-class continuous VLA graph with multi-image VLM + flow expert."""
    provenance = WorkloadProvenance(
        source="model_spec:pi0-class",
        reference_uri="https://arxiv.org/html/2410.24164v1",
    )

    tensors: list[TensorSpec] = []
    nodes: list[NodeSpec] = []

    t_images = _make_tensor("images", ["batch", 3, 224, 224], producer="input", provenance=provenance)
    tensors.append(t_images)
    patch_w = _make_tensor("vit_patch_weight", [588, vision_hidden], producer="input", provenance=provenance)
    patch_out = _make_tensor(
        "vit_patch_out", ["batch", 197, vision_hidden], producer="vit_patch_embed", provenance=provenance
    )
    tensors.extend([patch_w, patch_out])
    nodes.append(
        _make_node(
            "vit_patch_embed",
            "gemm",
            inputs=["images", "vit_patch_weight"],
            outputs=["vit_patch_out"],
            provenance=provenance,
        )
    )

    ln_out = _make_tensor("vit_ln_out", ["batch", 197, vision_hidden], producer="vit_ln", provenance=provenance)
    tensors.append(ln_out)
    nodes.append(
        _make_node("vit_ln", "layernorm", inputs=["vit_patch_out"], outputs=["vit_ln_out"], provenance=provenance)
    )

    proj_w = _make_tensor("vision_proj_weight", [vision_hidden, vl_hidden], producer="input", provenance=provenance)
    proj_out = _make_tensor("vision_proj_out", ["batch", 197, vl_hidden], producer="vision_proj", provenance=provenance)
    tensors.extend([proj_w, proj_out])
    nodes.append(
        _make_node(
            "vision_proj",
            "gemm",
            inputs=["vit_ln_out", "vision_proj_weight"],
            outputs=["vision_proj_out"],
            provenance=provenance,
        )
    )

    vl_ln = _make_tensor("vl_ln_out", ["batch", 197, vl_hidden], producer="vl_ln", provenance=provenance)
    tensors.append(vl_ln)
    nodes.append(
        _make_node("vl_ln", "layernorm", inputs=["vision_proj_out"], outputs=["vl_ln_out"], provenance=provenance)
    )

    for vl_layer in range(4):
        prefix = f"vl{vl_layer}"
        ffn_w = _make_tensor(
            f"{prefix}_ffn_weight", [vl_hidden, vl_hidden * 4], producer="input", provenance=provenance
        )
        ffn_out = _make_tensor(
            f"{prefix}_ffn_out", ["batch", 197, vl_hidden * 4], producer=f"{prefix}_ffn", provenance=provenance
        )
        gelu_out = _make_tensor(
            f"{prefix}_gelu_out", ["batch", 197, vl_hidden * 4], producer=f"{prefix}_gelu", provenance=provenance
        )
        proj2_w = _make_tensor(
            f"{prefix}_proj_weight", [vl_hidden * 4, vl_hidden], producer="input", provenance=provenance
        )
        proj2_out = _make_tensor(
            f"{prefix}_proj_out", ["batch", 197, vl_hidden], producer=f"{prefix}_proj", provenance=provenance
        )
        tensors.extend([ffn_w, ffn_out, gelu_out, proj2_w, proj2_out])
        nodes.extend(
            [
                _make_node(
                    f"{prefix}_ffn",
                    "gemm",
                    inputs=["vl_ln_out" if vl_layer == 0 else f"vl{vl_layer - 1}_proj_out", f"{prefix}_ffn_weight"],
                    outputs=[f"{prefix}_ffn_out"],
                    provenance=provenance,
                ),
                _make_node(
                    f"{prefix}_gelu",
                    "gelu",
                    inputs=[f"{prefix}_ffn_out"],
                    outputs=[f"{prefix}_gelu_out"],
                    provenance=provenance,
                ),
                _make_node(
                    f"{prefix}_proj",
                    "gemm",
                    inputs=[f"{prefix}_gelu_out", f"{prefix}_proj_weight"],
                    outputs=[f"{prefix}_proj_out"],
                    provenance=provenance,
                ),
            ]
        )

    pool_out = _make_tensor("visual_pool_out", ["batch", vl_hidden], producer="visual_pool", provenance=provenance)
    tensors.append(pool_out)
    nodes.append(
        _make_node(
            "visual_pool",
            "global_avg_pool",
            inputs=["vl3_proj_out"],
            outputs=["visual_pool_out"],
            provenance=provenance,
        )
    )

    t_proprio = _make_tensor("proprioception", ["batch", action_dim], producer="input", provenance=provenance)
    tensors.append(t_proprio)
    fuse_concat = _make_tensor(
        "fuse_concat_out", ["batch", vl_hidden + action_dim], producer="fuse_concat", provenance=provenance
    )
    tensors.append(fuse_concat)
    nodes.append(
        _make_node(
            "fuse_concat",
            "concat",
            inputs=["visual_pool_out", "proprioception"],
            outputs=["fuse_concat_out"],
            provenance=provenance,
        )
    )
    fusion_w = _make_tensor("fusion_weight", [vl_hidden + action_dim, 1024], producer="input", provenance=provenance)
    fusion_out = _make_tensor("fusion_out", ["batch", 1024], producer="fusion", provenance=provenance)
    tensors.extend([fusion_w, fusion_out])
    nodes.append(
        _make_node(
            "fusion", "gemm", inputs=["fuse_concat_out", "fusion_weight"], outputs=["fusion_out"], provenance=provenance
        )
    )

    for step in range(flow_steps):
        prefix = f"flow{step}"
        in_dim = 1024 if step == 0 else action_dim * action_horizon
        w1 = _make_tensor(f"{prefix}_w1", [in_dim, 1024], producer="input", provenance=provenance)
        h = _make_tensor(f"{prefix}_hidden", ["batch", 1024], producer=f"{prefix}_fc1", provenance=provenance)
        gelu_out = _make_tensor(f"{prefix}_gelu_out", ["batch", 1024], producer=f"{prefix}_gelu", provenance=provenance)
        w2 = _make_tensor(f"{prefix}_w2", [1024, action_dim * action_horizon], producer="input", provenance=provenance)
        out = _make_tensor(
            f"{prefix}_out", ["batch", action_dim * action_horizon], producer=f"{prefix}_fc2", provenance=provenance
        )
        tensors.extend([w1, h, gelu_out, w2, out])
        nodes.extend(
            [
                _make_node(
                    f"{prefix}_fc1",
                    "gemm",
                    inputs=["fusion_out" if step == 0 else f"flow{step - 1}_out", f"{prefix}_w1"],
                    outputs=[f"{prefix}_hidden"],
                    provenance=provenance,
                ),
                _make_node(
                    f"{prefix}_gelu",
                    "gelu",
                    inputs=[f"{prefix}_hidden"],
                    outputs=[f"{prefix}_gelu_out"],
                    provenance=provenance,
                ),
                _make_node(
                    f"{prefix}_fc2",
                    "gemm",
                    inputs=[f"{prefix}_gelu_out", f"{prefix}_w2"],
                    outputs=[f"{prefix}_out"],
                    provenance=provenance,
                ),
            ]
        )

    nodes = _linear_chain_deps(nodes)
    _wire_consumers(nodes, tensors)
    symbols = [
        SymbolicDim(name="image_count", description="Number of input images"),
        SymbolicDim(name="batch", description="Inference batch"),
    ]
    return WorkloadGraphV1(
        version="1",
        graph_name="pi0-class",
        nodes=nodes,
        tensors=tensors,
        symbols=symbols,
        provenance=provenance,
    )


def build_openvla_graph(
    *,
    variant: str,
    vision_hidden: int = 1024,
    vl_hidden: int = 4096,
    action_dim: int = 32,
    action_horizon: int = 25,
    flow_steps: int = 10,
) -> WorkloadGraphV1:
    """Build an OpenVLA-class graph with variant-specific action head."""
    uris = {
        "baseline": "https://arxiv.org/abs/2406.09246",
        "oft": "https://arxiv.org/html/2502.19645v1",
        "fast": "https://arxiv.org/abs/2404.00644",  # FAST token compression proxy
    }
    provenance = WorkloadProvenance(
        source=f"model_spec:openvla-{variant}",
        reference_uri=uris.get(variant),
    )

    tensors: list[TensorSpec] = []
    nodes: list[NodeSpec] = []

    t_image = _make_tensor("image", ["image_count", 3, 224, 224], producer="input", provenance=provenance)
    tensors.append(t_image)
    patch_w = _make_tensor("vit_patch_weight", [588, vision_hidden], producer="input", provenance=provenance)
    patch_out = _make_tensor(
        "vit_patch_out", ["image_count", 197, vision_hidden], producer="vit_patch_embed", provenance=provenance
    )
    tensors.extend([patch_w, patch_out])
    nodes.append(
        _make_node(
            "vit_patch_embed",
            "gemm",
            inputs=["image", "vit_patch_weight"],
            outputs=["vit_patch_out"],
            provenance=provenance,
        )
    )
    t_image.consumed_by.append("vit_patch_embed")

    proj_w = _make_tensor("vision_proj_weight", [vision_hidden, vl_hidden], producer="input", provenance=provenance)
    proj_out = _make_tensor(
        "vision_proj_out", ["image_count", 197, vl_hidden], producer="vision_proj", provenance=provenance
    )
    tensors.extend([proj_w, proj_out])
    nodes.append(
        _make_node(
            "vision_proj",
            "gemm",
            inputs=["vit_patch_out", "vision_proj_weight"],
            outputs=["vision_proj_out"],
            provenance=provenance,
        )
    )

    prev = "vision_proj_out"
    for vl_layer in range(2):
        prefix = f"vl{vl_layer}"
        ln = _make_tensor(
            f"{prefix}_ln_out", ["image_count", 197, vl_hidden], producer=f"{prefix}_ln", provenance=provenance
        )
        tensors.append(ln)
        nodes.append(
            _make_node(f"{prefix}_ln", "layernorm", inputs=[prev], outputs=[f"{prefix}_ln_out"], provenance=provenance)
        )

        ffn_w = _make_tensor(
            f"{prefix}_ffn_weight", [vl_hidden, vl_hidden * 4], producer="input", provenance=provenance
        )
        ffn_out = _make_tensor(
            f"{prefix}_ffn_out", ["image_count", 197, vl_hidden * 4], producer=f"{prefix}_ffn", provenance=provenance
        )
        gelu_out = _make_tensor(
            f"{prefix}_gelu_out", ["image_count", 197, vl_hidden * 4], producer=f"{prefix}_gelu", provenance=provenance
        )
        proj2_w = _make_tensor(
            f"{prefix}_proj_weight", [vl_hidden * 4, vl_hidden], producer="input", provenance=provenance
        )
        proj2_out = _make_tensor(
            f"{prefix}_proj_out", ["image_count", 197, vl_hidden], producer=f"{prefix}_proj", provenance=provenance
        )
        tensors.extend([ffn_w, ffn_out, gelu_out, proj2_w, proj2_out])
        nodes.extend(
            [
                _make_node(
                    f"{prefix}_ffn",
                    "gemm",
                    inputs=[f"{prefix}_ln_out", f"{prefix}_ffn_weight"],
                    outputs=[f"{prefix}_ffn_out"],
                    provenance=provenance,
                ),
                _make_node(
                    f"{prefix}_gelu",
                    "gelu",
                    inputs=[f"{prefix}_ffn_out"],
                    outputs=[f"{prefix}_gelu_out"],
                    provenance=provenance,
                ),
                _make_node(
                    f"{prefix}_proj",
                    "gemm",
                    inputs=[f"{prefix}_gelu_out", f"{prefix}_proj_weight"],
                    outputs=[f"{prefix}_proj_out"],
                    provenance=provenance,
                ),
            ]
        )
        prev = f"{prefix}_proj_out"

    pool_out = _make_tensor("visual_pool_out", ["batch", vl_hidden], producer="visual_pool", provenance=provenance)
    tensors.append(pool_out)
    nodes.append(
        _make_node("visual_pool", "global_avg_pool", inputs=[prev], outputs=["visual_pool_out"], provenance=provenance)
    )

    head_hidden = 512 if variant == "baseline" else 1024
    if variant == "fast":
        compress_w = _make_tensor("token_compress_weight", [vl_hidden, 256], producer="input", provenance=provenance)
        compress_out = _make_tensor(
            "token_compress_out", ["image_count", 197, 256], producer="token_compress", provenance=provenance
        )
        tensors.extend([compress_w, compress_out])
        nodes.append(
            _make_node(
                "token_compress",
                "gemm",
                inputs=[prev, "token_compress_weight"],
                outputs=["token_compress_out"],
                provenance=provenance,
            )
        )
        pool_out_fast = _make_tensor("fast_pool_out", ["batch", 256], producer="fast_pool", provenance=provenance)
        tensors.append(pool_out_fast)
        nodes.append(
            _make_node(
                "fast_pool",
                "global_avg_pool",
                inputs=["token_compress_out"],
                outputs=["fast_pool_out"],
                provenance=provenance,
            )
        )
        action_input = "fast_pool_out"
    else:
        action_input = "visual_pool_out"

    for step in range(flow_steps):
        prefix = f"flow{step}"
        in_dim = head_hidden if step == 0 else action_dim * action_horizon
        w1 = _make_tensor(f"{prefix}_w1", [in_dim, head_hidden], producer="input", provenance=provenance)
        h = _make_tensor(f"{prefix}_hidden", ["batch", head_hidden], producer=f"{prefix}_fc1", provenance=provenance)
        gelu_out = _make_tensor(
            f"{prefix}_gelu_out", ["batch", head_hidden], producer=f"{prefix}_gelu", provenance=provenance
        )
        w2 = _make_tensor(
            f"{prefix}_w2", [head_hidden, action_dim * action_horizon], producer="input", provenance=provenance
        )
        out = _make_tensor(
            f"{prefix}_out", ["batch", action_dim * action_horizon], producer=f"{prefix}_fc2", provenance=provenance
        )
        tensors.extend([w1, h, gelu_out, w2, out])
        nodes.extend(
            [
                _make_node(
                    f"{prefix}_fc1",
                    "gemm",
                    inputs=[action_input if step == 0 else f"flow{step - 1}_out", f"{prefix}_w1"],
                    outputs=[f"{prefix}_hidden"],
                    provenance=provenance,
                ),
                _make_node(
                    f"{prefix}_gelu",
                    "gelu",
                    inputs=[f"{prefix}_hidden"],
                    outputs=[f"{prefix}_gelu_out"],
                    provenance=provenance,
                ),
                _make_node(
                    f"{prefix}_fc2",
                    "gemm",
                    inputs=[f"{prefix}_gelu_out", f"{prefix}_w2"],
                    outputs=[f"{prefix}_out"],
                    provenance=provenance,
                ),
            ]
        )

    nodes = _linear_chain_deps(nodes)
    _wire_consumers(nodes, tensors)
    symbols = [
        SymbolicDim(name="image_count", description="Number of input images"),
        SymbolicDim(name="batch", description="Inference batch"),
    ]
    return WorkloadGraphV1(
        version="1",
        graph_name=f"openvla-{variant}",
        nodes=nodes,
        tensors=tensors,
        symbols=symbols,
        provenance=provenance,
    )


def build_helix_graph(
    *,
    s2_hidden: int = 3584,
    s1_hidden: int = 768,
    s0_hidden: int = 256,
) -> WorkloadGraphV1:
    """Build a Helix-class multi-rate graph with S2 semantic + S1 reactive + optional S0."""
    provenance = WorkloadProvenance(
        source="model_spec:helix-multirate",
        reference_uri="https://www.figure.ai/news/helix",
    )

    tensors: list[TensorSpec] = []
    nodes: list[NodeSpec] = []

    # S2 semantic model: infrequent, large
    t_image = _make_tensor("image", ["image_count", 3, 224, 224], producer="input", provenance=provenance)
    tensors.append(t_image)
    s2_patch_w = _make_tensor("s2_patch_weight", [588, s2_hidden], producer="input", provenance=provenance)
    s2_patch_out = _make_tensor(
        "s2_patch_out", ["image_count", 197, s2_hidden], producer="s2_patch_embed", provenance=provenance
    )
    tensors.extend([s2_patch_w, s2_patch_out])
    nodes.append(
        _make_node(
            "s2_patch_embed",
            "gemm",
            inputs=["image", "s2_patch_weight"],
            outputs=["s2_patch_out"],
            provenance=provenance,
        )
    )
    t_image.consumed_by.append("s2_patch_embed")

    s2_ln = _make_tensor("s2_ln_out", ["image_count", 197, s2_hidden], producer="s2_ln", provenance=provenance)
    tensors.append(s2_ln)
    s2_patch_t = next(t for t in tensors if t.tensor_id == "s2_patch_out")
    s2_patch_t.consumed_by.append("s2_ln")
    nodes.append(
        _make_node("s2_ln", "layernorm", inputs=["s2_patch_out"], outputs=["s2_ln_out"], provenance=provenance)
    )

    s2_ffn_w = _make_tensor("s2_ffn_weight", [s2_hidden, s2_hidden * 4], producer="input", provenance=provenance)
    s2_ffn_out = _make_tensor(
        "s2_ffn_out", ["image_count", 197, s2_hidden * 4], producer="s2_ffn", provenance=provenance
    )
    s2_gelu_out = _make_tensor(
        "s2_gelu_out", ["image_count", 197, s2_hidden * 4], producer="s2_gelu", provenance=provenance
    )
    s2_proj_w = _make_tensor("s2_proj_weight", [s2_hidden * 4, s2_hidden], producer="input", provenance=provenance)
    s2_proj_out = _make_tensor(
        "s2_proj_out", ["image_count", 197, s2_hidden], producer="s2_proj", provenance=provenance
    )
    tensors.extend([s2_ffn_w, s2_ffn_out, s2_gelu_out, s2_proj_w, s2_proj_out])
    nodes.extend(
        [
            _make_node(
                "s2_ffn", "gemm", inputs=["s2_ln_out", "s2_ffn_weight"], outputs=["s2_ffn_out"], provenance=provenance
            ),
            _make_node("s2_gelu", "gelu", inputs=["s2_ffn_out"], outputs=["s2_gelu_out"], provenance=provenance),
            _make_node(
                "s2_proj",
                "gemm",
                inputs=["s2_gelu_out", "s2_proj_weight"],
                outputs=["s2_proj_out"],
                provenance=provenance,
            ),
        ]
    )

    s2_pool = _make_tensor("s2_latent", ["batch", s2_hidden], producer="s2_pool", provenance=provenance)
    tensors.append(s2_pool)
    nodes.append(
        _make_node("s2_pool", "global_avg_pool", inputs=["s2_proj_out"], outputs=["s2_latent"], provenance=provenance)
    )

    s1_input = _make_tensor("s1_input", ["batch", s1_hidden], producer="input", provenance=provenance)
    tensors.append(s1_input)
    s1_fused = _make_tensor("s1_fused", ["batch", s1_hidden], producer="s1_fuse", provenance=provenance)
    s1_w1 = _make_tensor("s1_w1", [s1_hidden, s1_hidden * 2], producer="input", provenance=provenance)
    s1_h = _make_tensor("s1_hidden_out", ["batch", s1_hidden * 2], producer="s1_fc1", provenance=provenance)
    s1_gelu_out = _make_tensor("s1_gelu_out", ["batch", s1_hidden * 2], producer="s1_gelu", provenance=provenance)
    s1_w2 = _make_tensor("s1_w2", [s1_hidden * 2, s1_hidden], producer="input", provenance=provenance)
    s1_out = _make_tensor("s1_out", ["batch", s1_hidden], producer="s1_fc2", provenance=provenance)
    tensors.extend([s1_fused, s1_w1, s1_h, s1_gelu_out, s1_w2, s1_out])
    nodes.extend(
        [
            _make_node("s1_fuse", "add", inputs=["s1_input", "s2_latent"], outputs=["s1_fused"], provenance=provenance),
            _make_node(
                "s1_fc1", "gemm", inputs=["s1_fused", "s1_w1"], outputs=["s1_hidden_out"], provenance=provenance
            ),
            _make_node("s1_gelu", "gelu", inputs=["s1_hidden_out"], outputs=["s1_gelu_out"], provenance=provenance),
            _make_node("s1_fc2", "gemm", inputs=["s1_gelu_out", "s1_w2"], outputs=["s1_out"], provenance=provenance),
        ]
    )

    s0_input = _make_tensor("s0_input", ["batch", s0_hidden], producer="input", provenance=provenance)
    tensors.append(s0_input)
    s0_w = _make_tensor("s0_w", [s0_hidden, s0_hidden], producer="input", provenance=provenance)
    s0_out = _make_tensor("s0_out", ["batch", s0_hidden], producer="s0_fc", provenance=provenance)
    tensors.extend([s0_w, s0_out])
    nodes.append(_make_node("s0_fc", "gemm", inputs=["s0_input", "s0_w"], outputs=["s0_out"], provenance=provenance))

    action_concat = _make_tensor(
        "action_concat_out", ["batch", s1_hidden + s0_hidden], producer="action_concat", provenance=provenance
    )
    tensors.append(action_concat)
    nodes.append(
        _make_node(
            "action_concat", "concat", inputs=["s1_out", "s0_out"], outputs=["action_concat_out"], provenance=provenance
        )
    )
    action_w = _make_tensor("action_w", [s1_hidden + s0_hidden, 64], producer="input", provenance=provenance)
    action_out = _make_tensor("action_out", ["batch", 64], producer="action_head", provenance=provenance)
    tensors.extend([action_w, action_out])
    nodes.append(
        _make_node(
            "action_head",
            "gemm",
            inputs=["action_concat_out", "action_w"],
            outputs=["action_out"],
            provenance=provenance,
        )
    )

    nodes = _linear_chain_deps(nodes)
    _wire_consumers(nodes, tensors)
    symbols = [
        SymbolicDim(name="image_count", description="Number of input images"),
        SymbolicDim(name="batch", description="Inference batch"),
    ]
    return WorkloadGraphV1(
        version="1",
        graph_name="helix-multirate",
        nodes=nodes,
        tensors=tensors,
        symbols=symbols,
        provenance=provenance,
    )


def build_physical_ai_multijob_graph(
    *,
    resident_models: int = 4,
) -> WorkloadGraphV1:
    """Build a Physical-AI multi-job DAG combining resident CV models."""
    provenance = WorkloadProvenance(
        source="scenario:physical-ai-multijob",
        reference_uri="https://www.ti.com/lit/wp/spradb4/spradb4.pdf",
    )

    tensors: list[TensorSpec] = []
    nodes: list[NodeSpec] = []

    # Three concurrent job classes share a resident model pool
    jobs = [
        ("critical_perception", "image_count", 3, 224, 224, 16),
        ("localization_fusion", "image_count", 2, 224, 224, 8),
        ("inspection_auxiliary", "image_count", 1, 224, 224, 4),
    ]

    for job_name, batch_axis, channels, h, w, out_channels in jobs:
        t_in = _make_tensor(f"{job_name}_input", [batch_axis, channels, h, w], producer="input", provenance=provenance)
        t_w = _make_tensor(
            f"{job_name}_weight", [out_channels, channels, 3, 3], producer="input", provenance=provenance
        )
        t_out = _make_tensor(
            f"{job_name}_feature", [batch_axis, out_channels, h, w], producer=f"{job_name}_conv", provenance=provenance
        )
        tensors.extend([t_in, t_w, t_out])
        nodes.append(
            _make_node(
                f"{job_name}_conv",
                "conv",
                inputs=[f"{job_name}_input", f"{job_name}_weight"],
                outputs=[f"{job_name}_feature"],
                provenance=provenance,
            )
        )

        t_pool = _make_tensor(
            f"{job_name}_pool", [batch_axis, out_channels], producer=f"{job_name}_pool", provenance=provenance
        )
        tensors.append(t_pool)
        nodes.append(
            _make_node(
                f"{job_name}_pool",
                "global_avg_pool",
                inputs=[f"{job_name}_feature"],
                outputs=[f"{job_name}_pool"],
                provenance=provenance,
            )
        )

    fusion_in = [f"{job_name}_pool" for job_name, *_ in jobs]
    t_fusion = _make_tensor("fusion_out", ["inflight_jobs", 28], producer="fusion", provenance=provenance)
    tensors.append(t_fusion)
    nodes.append(_make_node("fusion", "concat", inputs=fusion_in, outputs=["fusion_out"], provenance=provenance))

    nodes = _linear_chain_deps(nodes)
    _wire_consumers(nodes, tensors)
    symbols = [
        SymbolicDim(name="image_count", description="Per-job camera frames"),
        SymbolicDim(name="inflight_jobs", description="Concurrent job instances"),
    ]
    return WorkloadGraphV1(
        version="1",
        graph_name="physical-ai-multijob",
        nodes=nodes,
        tensors=tensors,
        symbols=symbols,
        provenance=provenance,
    )


# ── Trace builder registry ───────────────────────────────────────────────────

_TRACE_BUILDERS: dict[str, Any] = {
    "llm-qwen25-3b": build_qwen25_3b_graph,
    "cv-yolov8n": build_yolov8n_graph,
    "cv-vit-b16": build_vit_b16_graph,
    "smolvla-class": build_smolvla_graph,
    "pi0-class": build_pi0_graph,
    "openvla-baseline": lambda **kwargs: build_openvla_graph(variant="baseline", **kwargs),
    "openvla-oft": lambda **kwargs: build_openvla_graph(variant="oft", **kwargs),
    "openvla-fast": lambda **kwargs: build_openvla_graph(variant="fast", **kwargs),
    "helix-multirate": build_helix_graph,
    "physical-ai-multijob": build_physical_ai_multijob_graph,
}


def _resolve_trace_builder(ref: str) -> Any:
    """Resolve a trace builder reference of the form ``module.path:function_name``."""
    if ":" not in ref:
        raise ConfigError(
            f"trace_builder must be 'module.path:function_name', got {ref!r}",
            field_path="trace_builder",
        )
    module_path, func_name = ref.split(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ConfigError(
            f"cannot import trace builder module {module_path!r}: {exc}",
            field_path="trace_builder",
        ) from exc
    func = getattr(module, func_name, None)
    if func is None:
        raise ConfigError(
            f"module {module_path!r} has no function {func_name!r}",
            field_path="trace_builder",
        )
    if not callable(func):
        raise ConfigError(
            f"trace builder {ref!r} is not callable",
            field_path="trace_builder",
        )
    return func


def _load_inline_graph(data: dict[str, Any], path: str) -> WorkloadGraphV1:
    """Load an inline graph declaration from YAML."""
    try:
        return WorkloadGraphV1.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(
            f"invalid inline graph in {path}: {exc}",
            field_path=f"{path}.graph",
        ) from exc


# ── Public API ───────────────────────────────────────────────────────────────


def load_fixture(yaml_path: str | Path) -> WorkloadFixture:
    """Load a single workload fixture from YAML and validate it."""
    yaml_path = Path(yaml_path)
    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError(f"fixture {yaml_path.name} must be a mapping", field_path="")

    name = _require_str(raw.get("name"), "name")
    version = _require_str(raw.get("version"), "version")
    if version != "1":
        raise ConfigError(
            f"fixture {name} has unsupported version {version!r}",
            field_path="version",
        )

    provenance_data = _require_mapping(raw.get("provenance", {}), "provenance")
    provenance = _parse_provenance(provenance_data, "provenance")

    dimensions_data = _require_mapping(raw.get("dimensions", {}), "dimensions")
    bindings = _parse_dimensions(dimensions_data, "dimensions")

    scenario = _require_mapping(raw.get("scenario", {}), "scenario")
    source_facts = _parse_facts(raw.get("source_facts", []), "source_facts")
    engineering_assumptions = _parse_facts(raw.get("engineering_assumptions", []), "engineering_assumptions")

    # Graph loading
    graph_data = raw.get("graph")
    trace_builder_ref = raw.get("trace_builder")
    if graph_data is not None and trace_builder_ref is not None:
        raise ConfigError(
            f"fixture {name} cannot declare both 'graph' and 'trace_builder'",
            field_path="graph",
        )
    if graph_data is None and trace_builder_ref is None:
        raise ConfigError(
            f"fixture {name} must declare either 'graph' or 'trace_builder'",
            field_path="graph",
        )

    if graph_data is not None:
        graph = _load_inline_graph(graph_data, yaml_path.name)
    else:
        builder = _resolve_trace_builder(trace_builder_ref)
        builder_kwargs = raw.get("trace_builder_kwargs", {}) or {}
        graph = builder(**builder_kwargs)
        if not isinstance(graph, WorkloadGraphV1):
            raise ConfigError(
                f"trace builder {trace_builder_ref!r} did not return a WorkloadGraphV1",
                field_path="trace_builder",
            )

    # Validate no source-less market facts
    for fact in source_facts:
        if fact.get("category") == "market_source" and not fact.get("reference_uri"):
            raise ConfigError(
                f"fixture {name} has source-less market_source fact: {fact.get('param')}",
                field_path="source_facts",
            )

    # Run full validation
    try:
        validate_all(graph, bindings, DEFAULT_REGISTRY)
    except (ConfigError, UnsupportedOperatorError) as exc:
        raise ConfigError(
            f"fixture {name} failed validation: {exc}",
            field_path="graph",
        ) from exc

    footprint_digest = graph_digest(graph)

    return WorkloadFixture(
        name=name,
        version=version,
        provenance=provenance,
        graph=graph,
        bindings=bindings,
        scenario=scenario,
        source_facts=source_facts,
        engineering_assumptions=engineering_assumptions,
        footprint_digest=footprint_digest,
    )


def discover_fixtures(config_dir: str | Path | None = None) -> list[Path]:
    """Return all YAML fixture paths in the catalog directory."""
    config_dir = Path(config_dir) if config_dir is not None else CATALOG_DIR
    if not config_dir.exists():
        return []
    return sorted(config_dir.glob("*.yaml"))


def load_all_fixtures(config_dir: str | Path | None = None) -> dict[str, WorkloadFixture]:
    """Load every discovered fixture into a validated mapping (name -> fixture)."""
    fixtures: dict[str, WorkloadFixture] = {}
    for path in discover_fixtures(config_dir):
        fixture = load_fixture(path)
        if fixture.name in fixtures:
            raise ConfigError(
                f"duplicate workload name {fixture.name!r} in {path.name}",
                field_path="name",
            )
        fixtures[fixture.name] = fixture
    return fixtures


def build_coverage_manifest(fixtures: dict[str, WorkloadFixture]) -> dict[str, Any]:
    """Build a deterministic coverage manifest across all fixtures."""
    axis_values: dict[str, set[int]] = {axis: set() for axis in _CANONICAL_AXES}
    fixture_axes: dict[str, list[str]] = {}
    provenance_refs: set[str] = set()

    for name, fixture in sorted(fixtures.items()):
        bound = fixture.bindings.to_dict()
        used_axes: list[str] = []
        for axis in _CANONICAL_AXES:
            if axis in bound:
                axis_values[axis].add(bound[axis])
                used_axes.append(axis)
        fixture_axes[name] = sorted(used_axes)
        if fixture.provenance.reference_uri:
            provenance_refs.add(fixture.provenance.reference_uri)
        for fact in fixture.source_facts:
            ref = fact.get("reference_uri")
            if ref:
                provenance_refs.add(str(ref))

    coverage: dict[str, Any] = {
        "fixture_count": len(fixtures),
        "fixture_names": sorted(fixtures.keys()),
        "axis_coverage": {
            axis: {
                "active_values": sorted(axis_values[axis]),
                "edge_values": sorted(_EDGE_VALUES[axis]),
                "covered_all_edges": axis_values[axis] >= _EDGE_VALUES[axis],
            }
            for axis in sorted(_CANONICAL_AXES)
        },
        "fixture_axes": fixture_axes,
        "provenance_uris": sorted(provenance_refs),
    }

    # Required edge coverage from plan acceptance criteria
    required_axes = {
        AXIS_BATCH,
        AXIS_SEQUENCES,
        AXIS_TOKEN_BLOCK,
        AXIS_IMAGE_COUNT,
        AXIS_ACTION_HORIZON,
        AXIS_FLOW_STEPS,
        AXIS_RESIDENT_MODELS,
        AXIS_INFLIGHT_JOBS,
    }
    missing_edges: dict[str, list[int]] = {}
    for axis in required_axes:
        missing = sorted(_EDGE_VALUES[axis] - axis_values[axis])
        if missing:
            missing_edges[axis] = missing
    coverage["missing_required_edges"] = missing_edges
    coverage["complete"] = not missing_edges

    return coverage
