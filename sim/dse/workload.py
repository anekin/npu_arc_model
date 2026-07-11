"""Canonical decode/prefill workload construction for Arc Model."""

from typing import List, Tuple

from dse.types import WorkloadSpec
from model_specs import get_spec

GEMM = Tuple[int, int, int, str]


def load_workload(model_name: str, seq_len: int) -> WorkloadSpec:
    spec = get_spec(model_name)
    if spec.model_type != "llm":
        raise ValueError(f"{model_name} is not an LLM workload")
    return WorkloadSpec(
        model_name=model_name,
        hidden=spec.hidden,
        intermediate=spec.intermediate,
        layers=spec.layers,
        num_heads=spec.num_heads,
        kv_heads=spec.kv_heads,
        head_dim=spec.head_dim,
        seq_len=int(seq_len),
    )


def projection_trace(workload: WorkloadSpec, token_rows: int) -> List[GEMM]:
    """Seven weight-bearing GEMMs in one Transformer layer."""
    h = workload.hidden
    i = workload.intermediate
    q = workload.num_heads * workload.head_dim
    kv = workload.kv_heads * workload.head_dim
    m = int(token_rows)
    return [
        (m, h, q, "Q_proj"),
        (m, h, kv, "K_proj"),
        (m, h, kv, "V_proj"),
        (m, q, h, "O_proj"),
        (m, h, i, "FFN_gate"),
        (m, h, i, "FFN_up"),
        (m, i, h, "FFN_down"),
    ]
