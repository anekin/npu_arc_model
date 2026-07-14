"""FSA engine model with explicit upstream calibration boundaries.

FSA (Fusing FlashAttention within a Single Systolic Array) fuses QK,
online softmax, and PV in one augmented systolic array. The attention
schedule below follows the public RTL execution plans. Projection GEMMs
remain an architecture-stage proxy: the public implementation is FP16/FP32,
not the INT4/INT8 mixed-precision datapath assumed by scenario A.
"""

import math
from typing import Any, Dict

from engine.mac_engine import EngineResult, MACEngine

FSA_UPSTREAM_URL = "https://github.com/VCA-EPFL/FSA"
FSA_PAPER_URL = "https://arxiv.org/abs/2507.11331"


class FSAEngine(MACEngine):
    """Augmented systolic array with an inline FlashAttention schedule."""

    def _parse_config(self, config: Dict[str, Any]):
        super()._parse_config(config)
        mac = config.get("mac_engine", config.get("mxu", {}))
        self.exp2_pwl_pieces = int(mac.get("fsa_exp2_pwl_pieces", 8))
        self.reciprocal_latency = int(mac.get("fsa_reciprocal_latency", 12))
        self._area_overhead = float(mac.get("fsa_area_overhead_pct", 12.0))

    @property
    def engine_type(self) -> str:
        return "fsa"

    def estimate(self, M: int, K: int, N: int,
                 weight_preloaded: bool = False) -> EngineResult:
        """Architecture-stage projection GEMM proxy, not upstream INT4 RTL."""
        H, W = self.H, self.W
        macs = M * K * N * self.ops_per_mac
        tiles_k = max(1, math.ceil(K / H))
        tiles_m = max(1, math.ceil(M / H))
        tiles_n = max(1, math.ceil(N / W))
        total_tiles = tiles_k * tiles_m * tiles_n
        pipe_depth = H + M + W
        compute_cycles = total_tiles * pipe_depth
        weight_bytes = K * N * self.w_bits // 8
        act_bytes = M * K * self.a_bits // 8
        effective_weight_bytes = 0 if weight_preloaded else weight_bytes
        dma_total_bytes = effective_weight_bytes + act_bytes
        dma_cycles = max(
            dma_total_bytes / max(self.eff_bw, 1e-9),
            compute_cycles * 0.05,
        )
        total_cycles = max(compute_cycles, dma_cycles)
        utilization = macs / max(self.peak_macs_per_cycle * total_cycles, 1)
        return EngineResult(
            compute_cycles=math.ceil(compute_cycles),
            dma_cycles=math.ceil(dma_cycles),
            total_cycles=math.ceil(total_cycles),
            utilization=min(utilization, 1.0),
            ops=macs,
            num_tiles=total_tiles,
            weight_bytes=int(effective_weight_bytes),
            bottleneck="compute" if compute_cycles >= dma_cycles else "dma",
            details={
                "tiles_k": tiles_k,
                "tiles_m": tiles_m,
                "tiles_n": tiles_n,
                "pipe_depth": pipe_depth,
                "engine": "fsa",
                "projection_model": "v3_proxy_not_upstream_calibrated",
                "dram_eff": round(self.dram_efficiency, 3),
            },
        )

    def estimate_weight_cache_pair(self, M: int, K: int, N: int) -> EngineResult:
        """Gate and Up are distinct GEMMs with distinct weight tensors."""
        single = self.estimate(M, K, N)
        return EngineResult(
            compute_cycles=2 * single.compute_cycles,
            dma_cycles=2 * single.dma_cycles,
            total_cycles=2 * single.total_cycles,
            utilization=single.utilization,
            ops=2 * single.ops,
            num_tiles=2 * single.num_tiles,
            weight_bytes=2 * single.weight_bytes,
            bottleneck=single.bottleneck,
            details={
                "scheduler_fallback": "two_independent_gemms",
                "reason": "gate_and_up_weights_are_distinct",
            },
        )

    def estimate_attention(
        self,
        seq_q: int,
        seq_kv: int,
        head_dim: int,
        num_heads: int = 1,
        num_kv_heads: int = 1,
        kv_batch_size: int = 1,
        attention_bits: int = 16,
        causal: bool = False,
        cached_prefix_tokens: int = 0,
    ) -> EngineResult:
        """Estimate FSA from the public RTL instruction issue schedule.

        Upstream maps head_dim to array rows, Q blocks to array columns,
        and K/V blocks to array rows. Incremental cached-prefix attention
        lacks a query-position offset in the public causal kernel, so it is
        modeled as a full rectangle and explicitly marked uncalibrated.
        """
        if min(seq_q, seq_kv, head_dim, num_heads, num_kv_heads) <= 0:
            raise ValueError("FSA attention dimensions must be positive")

        q_tiles = math.ceil(seq_q / self.W)
        kv_tiles = math.ceil(seq_kv / self.H)
        mapping_compatible = self.H == head_dim
        native_causal = bool(
            causal
            and cached_prefix_tokens == 0
            and self.H == self.W
            and seq_q == seq_kv
        )
        if native_causal:
            inner_tiles = sum(min(kv_tiles, i + 1) for i in range(q_tiles))
        else:
            inner_tiles = q_tiles * kv_tiles

        # ExecutionPlan.scala issue points: score is conflict-free at
        # 2*rows + exp2_pieces + 3, then value takes 2*rows - 2 cycles.
        score_value_cycles = 4 * self.H + self.exp2_pwl_pieces + 1
        finalize_cycles = 2 + self.reciprocal_latency + self.H
        per_head_cycles = (
            inner_tiles * score_value_cycles
            + q_tiles * (self.W + finalize_cycles)
        )
        compute_cycles = num_heads * per_head_cycles

        elem_bytes = max(1, math.ceil(attention_bits / 8))
        attention_bytes = (
            2 * num_heads * seq_q * head_dim * elem_bytes
            + 2 * num_kv_heads * seq_kv * head_dim * elem_bytes
            * max(1, kv_batch_size)
        )
        dma_cycles = math.ceil(attention_bytes / max(self.eff_bw, 1e-9))
        total_cycles = max(compute_cycles, dma_cycles)

        if causal:
            visible_pairs = (
                cached_prefix_tokens * seq_q
                + seq_q * (seq_q + 1) // 2
            )
            visible_pairs = min(visible_pairs, seq_q * seq_kv)
        else:
            visible_pairs = seq_q * seq_kv
        logical_macs = 2 * num_heads * visible_pairs * head_dim

        return EngineResult(
            compute_cycles=math.ceil(compute_cycles),
            dma_cycles=dma_cycles,
            total_cycles=math.ceil(total_cycles),
            utilization=min(
                1.0,
                logical_macs / max(self.H * self.W * total_cycles, 1),
            ),
            ops=logical_macs * self.ops_per_mac,
            num_tiles=num_heads * inner_tiles,
            weight_bytes=0,
            bottleneck="compute" if compute_cycles >= dma_cycles else "dma",
            details={
                "engine": "fsa",
                "inline_softmax": True,
                "schedule_source": "upstream_execution_plan",
                "calibration_status": "paper_extrapolation",
                "source_repo": FSA_UPSTREAM_URL,
                "source_paper": FSA_PAPER_URL,
                "seq_q": seq_q,
                "seq_kv": seq_kv,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "kv_batch_size": kv_batch_size,
                "attention_bits": attention_bits,
                "causal_requested": causal,
                "native_causal_schedule": native_causal,
                "cached_prefix_tokens": cached_prefix_tokens,
                "incremental_causal_supported": not (
                    causal and cached_prefix_tokens > 0
                ),
                "mapping_compatible": mapping_compatible,
                "q_tiles": q_tiles,
                "kv_tiles": kv_tiles,
                "inner_tiles": inner_tiles,
                "score_value_cycles_per_tile": score_value_cycles,
                "finalize_cycles_per_q_tile": finalize_cycles,
                "attention_bytes": attention_bytes,
            },
        )
