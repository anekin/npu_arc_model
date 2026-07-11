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


@dataclass
class LayerEstimate:
    total_cycles: int = 0
    compute_cycles: int = 0
    memory_cycles: int = 0
    attention_cycles: int = 0
    sfu_cycles: int = 0
    kv_cycles: int = 0
    transferred_bytes: int = 0

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

    tok_s: float
    area_mm2: float
    power_w: float
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
    constraints_passed: bool = True
    failed_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    breakdown: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.efficiency_tok_per_watt = self.tok_s / max(self.power_w, 0.1)
        self.efficiency_tok_per_mm2 = self.tok_s / max(self.area_mm2, 0.1)

    def __repr__(self):
        status = "PASS" if self.constraints_passed else "FAIL"
        return (f"DSEPoint({status}, tok={self.tok_s:.1f}/s, "
                f"ttft={self.ttft_ms:.1f}ms, {self.area_mm2:.1f}mm2, "
                f"{self.power_w:.1f}W)")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
