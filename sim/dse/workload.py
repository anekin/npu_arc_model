"""Canonical decode/prefill workload construction for Arc Model."""

from typing import Any, Dict, List, Tuple

from dse.types import WorkloadSpec
from model_specs import get_spec

GEMM = Tuple[int, int, int, str]


def load_workload(
    model_name: str,
    seq_len: int,
    workload_config: Dict[str, Any] | None = None,
    weight_bits: int = 4,
) -> WorkloadSpec:
    spec = get_spec(model_name)
    if spec.model_type != "llm":
        raise ValueError(f"{model_name} is not an LLM workload")

    cfg = workload_config or {}
    prompt_tokens = int(cfg.get("prompt_tokens", cfg.get("seq_len", seq_len)))
    output_tokens = int(cfg.get("output_tokens", 128))
    concurrency = int(cfg.get("concurrent_requests", 1))
    batching = cfg.get("batching", {}) or {}
    requested_batch = int(cfg.get(
        "decode_batch_size", batching.get("max_batch_size", 1),
    ))
    decode_batch = min(max(1, requested_batch), max(1, concurrency))

    for name, value in (
        ("prompt_tokens", prompt_tokens),
        ("output_tokens", output_tokens),
        ("concurrent_requests", concurrency),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    return WorkloadSpec(
        model_name=model_name,
        hidden=spec.hidden,
        intermediate=spec.intermediate,
        layers=spec.layers,
        num_heads=spec.num_heads,
        kv_heads=spec.kv_heads,
        head_dim=spec.head_dim,
        seq_len=prompt_tokens,
        output_tokens=output_tokens,
        concurrent_requests=concurrency,
        decode_batch_size=decode_batch,
        weight_bits=int(cfg.get("weight_bits", weight_bits)),
        activation_bits=int(cfg.get("activation_bits", 8)),
        kv_bits=int(cfg.get("kv_bits", 16)),
        runtime_reserve_mb=float(cfg.get("runtime_reserve_mb", 256.0)),
        parameters_b=float(spec.parameters_b),
        context_window_tokens=int(cfg.get("max_context_tokens", 0)),
        cached_prefix_tokens=int(cfg.get("cached_prefix_tokens", 0)),
        attention_bits=int(cfg.get("attention_bits", 16)),
        causal_attention=bool(cfg.get("causal_attention", True)),
        vocab_size=int(spec.vocab_size),
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
