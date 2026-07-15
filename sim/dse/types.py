"""Structured data exchanged by the Arc Model DSE pipeline."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class WorkloadSpec:
    model_name: str
    hidden: int
    intermediate: int
    layers: int
    num_heads: int
    kv_heads: int
    head_dim: int
    seq_len: int
    output_tokens: int = 128
    concurrent_requests: int = 1
    decode_batch_size: int = 1
    weight_bits: int = 4
    activation_bits: int = 8
    kv_bits: int = 16
    runtime_reserve_mb: float = 256.0
    parameters_b: float = 0.0
    context_window_tokens: int = 0
    cached_prefix_tokens: int = 0
    attention_bits: int = 16
    causal_attention: bool = True
    vocab_size: int = 0

    @property
    def prompt_tokens(self) -> int:
        """Preferred name for the legacy ``seq_len`` field."""
        return self.seq_len

    @property
    def prefill_context_tokens(self) -> int:
        """K/V length seen by the incremental prefill query rows."""
        return self.cached_prefix_tokens + self.prompt_tokens

    @property
    def decode_context_tokens(self) -> int:
        """Initial K/V length seen by the first generated token."""
        return self.prefill_context_tokens

    @property
    def max_context_tokens(self) -> int:
        active = self.prefill_context_tokens + self.output_tokens
        return max(active, self.context_window_tokens)

    @property
    def causal_prefill_compute_context_tokens(self) -> int:
        """Equivalent visible K/V length for causal append compute."""
        if not self.causal_attention:
            return self.prefill_context_tokens
        return self.cached_prefix_tokens + (self.prompt_tokens + 2) // 2


@dataclass
class MemoryFootprint:
    weights_gb: float = 0.0
    kv_cache_gb: float = 0.0
    activations_gb: float = 0.0
    runtime_reserve_gb: float = 0.0
    required_gb: float = 0.0
    installed_gb: float = 0.0
    usable_gb: float = 0.0
    margin_gb: float = 0.0
    fits: bool = True
    capacity_specified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayerEstimate:
    total_cycles: int = 0
    compute_cycles: int = 0
    memory_cycles: int = 0
    attention_cycles: int = 0
    sfu_cycles: int = 0
    kv_cycles: int = 0
    transferred_bytes: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class ConstraintResult:
    passed: bool = True
    failed_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DSEPoint:
    """One auditable architecture candidate and its evaluated metrics."""

    # ``tok_s`` remains the compatibility name for per-request decode TPS.
    tok_s: float
    area_mm2: float
    power_w: float
    decode_tps: float = 0.0
    aggregate_tps: float = 0.0
    prefill_tps: float = 0.0
    itl_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    efficiency_tok_per_watt: float = 0.0
    efficiency_tok_per_mm2: float = 0.0
    config_label: str = ""
    sram_spill_mb: float = 0.0
    depthwise_util_pct: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    ttft_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    tops_int8: float = 0.0
    bandwidth_gbps: float = 0.0
    bandwidth_util_pct: float = 0.0
    memory_required_gb: float = 0.0
    memory_available_gb: float = 0.0
    memory_margin_gb: float = 0.0
    memory_fits: bool = True
    constraints_passed: bool = True
    maturity: str = "M1"
    raw_exploration_eligible: bool = True
    comparison_eligible: bool = False
    product_eligible: bool = False
    # Backward-compatible alias for comparison-ready architecture ranking.
    recommendation_eligible: bool = False
    failed_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    breakdown: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.decode_tps <= 0:
            self.decode_tps = self.tok_s
        if self.tok_s <= 0:
            self.tok_s = self.decode_tps
        if self.aggregate_tps <= 0:
            self.aggregate_tps = self.decode_tps
        self.efficiency_tok_per_watt = self.decode_tps / max(self.power_w, 0.1)
        self.efficiency_tok_per_mm2 = self.decode_tps / max(self.area_mm2, 0.1)

    def __repr__(self):
        status = "PASS" if self.constraints_passed else "FAIL"
        return (f"DSEPoint({status}, decode={self.decode_tps:.1f} tok/s, "
                f"aggregate={self.aggregate_tps:.1f} tok/s, "
                f"ttft={self.ttft_ms:.1f}ms, {self.area_mm2:.1f}mm2)")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
