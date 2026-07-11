#!/usr/bin/env python3
"""FSA Python Golden Reference — inline FlashAttention on systolic array.

Models the FSA（Fusing FlashAttention within a Single Systolic Array）approach:
  - Softmax operations (rowmax, exp, rowsum) executed INLINE within the
    systolic array dataflow — no external SFU or Vector unit needed.
  - CMP (comparator) columns for online max reduction.
  - PE Split units: reuse MAC hardware for exp piecewise linear interpolation.
  - Upward data path for reduce operations.

This serves as an architecture reference for comparison with CaduceusCore
(MXU + SFU + Vector approach) in the Arc Model.

Source: https://github.com/VCA-EPFL/FSA
Paper: http://arxiv.org/abs/2507.11331
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FSAConfig:
    """FSA architecture configuration."""
    sa_rows: int = 64           # systolic array rows
    sa_cols: int = 64           # systolic array columns
    frequency_mhz: int = 1000   # clock frequency
    weight_precision_bits: int = 4    # INT4
    activation_precision_bits: int = 8  # INT8
    accumulate_precision_bits: int = 32  # INT32
    dataflow: str = "weight_stationary"
    ops_per_mac: int = 2

    # FSA-specific
    cmp_overhead_pct: float = 2.0    # CMP column area overhead
    split_overhead_pct: float = 4.0  # PE Split unit area overhead
    upward_path_overhead_pct: float = 3.0  # upward data path overhead
    fsa_area_overhead_pct: float = 12.0  # total FSA-specific area overhead
    exp_pwl_segments: int = 8        # piecewise linear segments for exp

    def total_mac_per_cycle(self) -> int:
        return self.sa_rows * self.sa_cols * self.ops_per_mac

    def mac_per_cycle_effective(self, util_pct: float = 0.85) -> float:
        """Effective MACs/cycle accounting for inline softmax overhead."""
        return self.total_mac_per_cycle() * util_pct


# ═══════════════════════════════════════════════════════════════════
# FSA Hardware Model
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FSAHardwareModel:
    """Models FSA array behaviour: inline softmax via CMP + Split units."""

    config: FSAConfig = field(default_factory=FSAConfig)

    # Pre-computed piecewise linear intercepts for exp2 approximation
    # (FSA uses exp2(x) ≈ intercept_k + slope_k * x for k-th segment)
    _exp2_intercepts: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        n_seg = self.config.exp_pwl_segments
        x_range = np.linspace(-4, 4, n_seg + 1)
        self._exp2_intercepts = np.exp2(x_range[:-1])

    def estimate_area_mm2(self, technology_nm: int = 7) -> float:
        """Estimate die area for FSA array.

        Baseline: Caduceus MXU 64x64 area ≈ 0.85 mm² @ 7nm (empirical).
        FSA adds ~12% for CMP + Split + upward paths.
        """
        baseline_area = 0.85  # mm² for 64x64 MAC array @ 7nm
        fsa_area = baseline_area * (1 + self.config.fsa_area_overhead_pct / 100)
        return fsa_area

    def estimate_attention_latency(
        self, seq_q: int, seq_k: int, head_dim: int,
    ) -> int:
        """Estimate FlashAttention latency in cycles (FSA inline model).

        FSA executes attention in a single systolic array pass:
          S = QK^T (with inline rowmax/exp/rowsum), then P = softmax(S), O = PV.

        Key insight: because rowmax/exp/rowsum are overlapped with MAC ops,
        the latency overhead is minimal — about 3-5 extra cycles per row.

        Returns: total cycles
        """
        sa_r, sa_c = self.config.sa_rows, self.config.sa_cols

        # Phase 1: QK^T + online softmax (SA rows = head_dim, SA cols = head_dim)
        #   Each MAC cycle produces one partial S element
        #   rowmax/exp/rowsum overlapped via CMP + Split
        #   Tile iterations: ceil(seq_q / sa_r) × ceil(seq_k / sa_c)
        tiles_q = (seq_q + sa_r - 1) // sa_r
        tiles_k = (seq_k + sa_c - 1) // sa_c

        # Per-tile cost: sa_r MAC cycles + ~5 cycles CMP/Split overhead
        tile_cycles = sa_r + 5  # MAC fill + inline softmax overhead

        # Phase 1 total
        phase1_cycles = tiles_q * tiles_k * tile_cycles

        # Phase 2: O = PV (SA rows = head_dim, SA cols = seq_q)
        #   Each MAC cycle produces one O element
        #   Tile iterations: ceil(head_dim / sa_r) × ceil(seq_q / sa_c)
        tiles_hd = (head_dim + sa_r - 1) // sa_r
        tiles_oq = (seq_q + sa_c - 1) // sa_c
        phase2_cycles = tiles_hd * tiles_oq * (sa_r + 3)

        return phase1_cycles + phase2_cycles

    def estimate_mac_utilization(
        self, seq_q: int, seq_k: int, head_dim: int,
    ) -> float:
        """Estimate MAC utilization accounting for inline softmax overhead.

        During inline softmax, some MAC units are repurposed for exp PWL
        computation, reducing peak matmul throughput.
        """
        total_cycles = self.estimate_attention_latency(seq_q, seq_k, head_dim)
        mac_ops = 2 * seq_q * seq_k * head_dim + 2 * seq_q * head_dim * seq_k
        # FSA MACs work on matmul ~88% of the time, ~12% on exp approximation
        effective_mac_ops_per_cycle = self.config.total_mac_per_cycle() * 0.88
        peak_cycles = mac_ops / effective_mac_ops_per_cycle
        return min(peak_cycles / total_cycles, 1.0) if total_cycles > 0 else 0

    def compute_attention_ops(self, seq_q: int, seq_k: int, head_dim: int) -> int:
        """Total MAC operations for one attention head."""
        # S = QK^T: (seq_q × head_dim) × (head_dim × seq_k)
        # P = softmax(S): no MACs (just exp + normalize)
        # O = PV: (seq_q × seq_k) × (seq_k × head_dim)
        return (2 * seq_q * seq_k * head_dim) * 2  # two matmuls × MAC=2 ops


# ═══════════════════════════════════════════════════════════════════
# Caduceus Hardware Model (for comparison)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CaduceusHardwareModel:
    """Models CaduceusCore: MXU + SFU + Vector pipeline."""

    mxu_rows: int = 64
    mxu_cols: int = 64
    frequency_mhz: int = 1000
    sfu_width: int = 128
    vector_width: int = 128

    # Area estimates (mm² @ 7nm)
    mxu_area_mm2: float = 0.85
    sfu_area_mm2: float = 0.15   # softmax/layernorm/gelu/silu/rmsnorm/rope
    vector_area_mm2: float = 0.10  # SIMD ALU + reduce + convert + resid_add
    sram_area_mm2: float = 0.30    # shared SRAM

    def total_area_mm2(self) -> float:
        return (self.mxu_area_mm2 + self.sfu_area_mm2 +
                self.vector_area_mm2 + self.sram_area_mm2)

    def estimate_attention_latency(
        self, seq_q: int, seq_k: int, head_dim: int,
    ) -> dict:
        """Estimate CaduceusCore attention latency broken down by unit.

        Pipeline: MXU (QK^T) → SFU (softmax) → MXU (PV) → Vector (resid)
        """
        sa_r, sa_c = self.mxu_rows, self.mxu_cols
        sfu_w = self.sfu_width

        # Phase 1: QK^T on MXU
        tiles_q1 = (seq_q + sa_r - 1) // sa_r
        tiles_k1 = (seq_k + sa_c - 1) // sa_c
        mxu1_cycles = tiles_q1 * tiles_k1 * sa_r

        # Phase 2: softmax on SFU (row by row)
        sfu_elements_per_cycle = sfu_w
        softmax_rows = seq_q * seq_k  # each row of S
        sfu_cycles = (softmax_rows + sfu_elements_per_cycle - 1) // sfu_elements_per_cycle * 8

        # Phase 3: PV on MXU
        tiles_q2 = (seq_q + sa_r - 1) // sa_r
        tiles_hd2 = (head_dim + sa_c - 1) // sa_c
        mxu2_cycles = tiles_q2 * tiles_hd2 * sa_r

        total = mxu1_cycles + sfu_cycles + mxu2_cycles
        return {
            "mxu1": mxu1_cycles,
            "sfu": sfu_cycles,
            "mxu2": mxu2_cycles,
            "total": total,
        }


# ═══════════════════════════════════════════════════════════════════
# Architecture Comparison
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ArchComparisonReport:
    """Head-to-head architecture comparison."""

    model_name: str = ""
    seq_q: int = 1          # decode mode: 1 token
    seq_kv: int = 1024      # typical KV cache length
    head_dim: int = 128
    num_heads: int = 32
    num_kv_heads: int = 8
    num_layers: int = 28

    # FSA metrics
    fsa_area_mm2: float = 0.0
    fsa_attn_latency_cycles: int = 0
    fsa_attn_us: float = 0.0
    fsa_mac_util_pct: float = 0.0
    fsa_total_cycles_per_layer: int = 0

    # Caduceus metrics
    cad_area_mm2: float = 0.0
    cad_attn_latency_cycles: int = 0
    cad_attn_us: float = 0.0
    cad_mxu_util_pct: float = 0.0
    cad_total_cycles_per_layer: int = 0

    # Derived comparisons
    fsa_area_saving_pct: float = 0.0      # positive = FSA smaller
    fsa_latency_ratio: float = 0.0        # FSA / Caduceus (< 1 = FSA faster)
    fsa_general_score: float = 0.0        # 0-100 flexibility score

    def compute_derived(self, freq_mhz: int = 1000):
        """Compute derived comparison metrics."""
        self.fsa_attn_us = self.fsa_attn_latency_cycles / freq_mhz
        self.cad_attn_us = self.cad_attn_latency_cycles / freq_mhz
        self.fsa_area_saving_pct = (
            (self.cad_area_mm2 - self.fsa_area_mm2) / self.cad_area_mm2 * 100
        )
        self.fsa_latency_ratio = (
            self.fsa_attn_latency_cycles / self.cad_attn_latency_cycles
            if self.cad_attn_latency_cycles > 0 else 0
        )
        # FSA is attention-specialized; Caduceus has general-purpose SFU/Vector
        self.fsa_general_score = 45.0  # FSA: attention-only
        # Caduceus score reflects SFU handling ALL activation functions
        cad_general_score = 90.0  # MXU+SFU+Vector: general-purpose

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "config": f"{self.seq_q}q × {self.seq_kv}kv, hd={self.head_dim}",
            "fsa": {
                "area_mm2": round(self.fsa_area_mm2, 3),
                "attn_latency_cycles": self.fsa_attn_latency_cycles,
                "attn_us": round(self.fsa_attn_us, 2),
                "mac_util_pct": round(self.fsa_mac_util_pct, 1),
            },
            "caduceus": {
                "area_mm2": round(self.cad_area_mm2, 3),
                "attn_latency_cycles": self.cad_attn_latency_cycles,
                "attn_us": round(self.cad_attn_us, 2),
            },
            "comparison": {
                "fsa_area_saving_pct": round(self.fsa_area_saving_pct, 1),
                "fsa_latency_ratio": round(self.fsa_latency_ratio, 3),
                "fsa_wins_area": self.fsa_area_saving_pct > 0,
                "fsa_wins_latency": self.fsa_latency_ratio < 1.0,
            },
        }


def compare_architectures(
    model_name: str = "Qwen2.5-3B",
    seq_q: int = 1,
    seq_kv: int = 1024,
    head_dim: int = 128,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    num_layers: int = 28,
) -> ArchComparisonReport:
    """Run head-to-head architecture comparison.

    Compares FSA (inline softmax on systolic array) vs CaduceusCore
    (MXU + SFU + Vector pipeline) for FlashAttention latency and area.
    """
    report = ArchComparisonReport(
        model_name=model_name,
        seq_q=seq_q,
        seq_kv=seq_kv,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
    )

    # ── FSA model ────────────────────────────────────────────────
    fsa_cfg = FSAConfig()
    fsa_hw = FSAHardwareModel(config=fsa_cfg)
    report.fsa_area_mm2 = fsa_hw.estimate_area_mm2()

    # FSA latency per attention head
    fsa_per_head = fsa_hw.estimate_attention_latency(seq_q, seq_kv, head_dim)
    # All heads × all layers
    report.fsa_attn_latency_cycles = (
        fsa_per_head * num_kv_heads * num_layers
    )
    report.fsa_mac_util_pct = (
        fsa_hw.estimate_mac_utilization(seq_q, seq_kv, head_dim) * 100
    )

    # ── CaduceusCore model ───────────────────────────────────────
    cad_hw = CaduceusHardwareModel()
    report.cad_area_mm2 = cad_hw.total_area_mm2()
    cad_per_head = cad_hw.estimate_attention_latency(seq_q, seq_kv, head_dim)
    report.cad_attn_latency_cycles = (
        cad_per_head["total"] * num_kv_heads * num_layers
    )

    report.compute_derived(freq_mhz=fsa_cfg.frequency_mhz)
    return report


def print_comparison(report: ArchComparisonReport):
    """Print human-readable architecture comparison table."""
    d = report.to_dict()
    print(f"\n{'='*70}")
    print(f"  Architecture Comparison: {d['model']}")
    print(f"  Config: {d['config']}")
    print(f"{'='*70}")
    print(f"  {'':<20} {'FSA (inline)':>18} {'CaduceusCore':>18}")
    print(f"  {'─'*20} {'─'*18} {'─'*18}")
    print(f"  {'Die Area (mm²)':<20} {d['fsa']['area_mm2']:>18.3f} {d['caduceus']['area_mm2']:>18.3f}")
    print(f"  {'Attn Latency (μs)':<20} {d['fsa']['attn_us']:>18.1f} {d['caduceus']['attn_us']:>18.1f}")
    print(f"  {'Attn Latency (cycles)':<20} {d['fsa']['attn_latency_cycles']:>18,} {d['caduceus']['attn_latency_cycles']:>18,}")
    print(f"  {'MAC Utilization (%)':<20} {d['fsa']['mac_util_pct']:>18.1f} {'—':>18}")
    print(f"  {'─'*20} {'─'*18} {'─'*18}")
    print(f"  {'FSA vs Caduceus':<20}")
    print(f"    Area:   FSA is {d['comparison']['fsa_area_saving_pct']:+.1f}% {'smaller' if d['comparison']['fsa_wins_area'] else 'larger'}")
    print(f"    Latency: FSA is {d['comparison']['fsa_latency_ratio']:.2f}× Caduceus {'(faster)' if d['comparison']['fsa_wins_latency'] else '(slower)'}")
    print(f"{'='*70}")

    # Trade-off analysis
    print(f"\n  📐 Architecture Trade-off Analysis:")
    print(f"  {'─'*60}")
    print(f"  {'FSA wins:':<12} inline softmax eliminates MXU→SFU→Vector")
    print(f"  {'':<12} data movement, yielding {'better' if d['comparison']['fsa_wins_latency'] else 'comparable'}")
    print(f"  {'':<12} attention latency at {'lower' if d['comparison']['fsa_wins_area'] else 'higher'} die area.")
    print(f"  {'':<12}")
    print(f"  {'Caduceus wins:':<12} general-purpose SFU handles layernorm,")
    print(f"  {'':<12} rmsnorm, gelu, silu, rope — not just attention.")
    print(f"  {'':<12} Vector unit provides element-wise SIMD for ALL models.")
    print(f"  {'':<12}")
    print(f"  {'Verdict:':<12} FSA optimal for attention-dominated workloads;")
    print(f"  {'':<12} CaduceusCore optimal for general model serving.")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Default: Qwen2.5-3B decode (1 token query, 1024 KV cache)
    report = compare_architectures()
    print_comparison(report)

    # Extended: prefill scenario
    print(f"\n\n{'─'*70}")
    print("  Extended Scenario: Prefill (128 token query, 1024 KV cache)")
    print(f"{'─'*70}")
    prefill = compare_architectures(
        model_name="Qwen2.5-3B (prefill)",
        seq_q=128,
        seq_kv=1024,
    )
    print_comparison(prefill)
