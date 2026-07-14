"""FSA-inspired attention fusion layered onto existing NPU engines.

These engines are native Arc Model architecture candidates, not replicas of
the paper FSA. Projection and FFN operations are delegated unchanged to the
selected baseline engine. Only QK/online-softmax/PV use the fused path.
"""

import math
from typing import Any, Dict, Type

from engine.block_engine import BlockEngine
from engine.mac_engine import EngineResult, MACEngine
from engine.os_systolic_engine import OutputStationaryEngine


class _FusedAttentionEngine(MACEngine):
    """Common analytical model for an in-array fused-attention capability."""

    projection_engine_class: Type[MACEngine]

    def _parse_config(self, config: Dict[str, Any]):
        super()._parse_config(config)
        mac = config.get("mac_engine", config.get("mxu", {}))
        self.fusion_overlap_factor = float(
            mac.get("fused_attention_overlap_factor", 0.90)
        )
        self.control_cycles_per_tile = int(
            mac.get("fused_attention_control_cycles_per_tile", 2)
        )
        if not 0 < self.fusion_overlap_factor <= 1:
            raise ValueError("fused_attention_overlap_factor must be in (0, 1]")

    def _projection_engine(self) -> MACEngine:
        return self.projection_engine_class(self.config)

    def estimate(self, M: int, K: int, N: int,
                 weight_preloaded: bool = False) -> EngineResult:
        """Projection/FFN path is identical to the corresponding baseline."""
        result = self._projection_engine().estimate(
            M, K, N, weight_preloaded=weight_preloaded,
        )
        result.details.update({
            "composite_engine": self.engine_type,
            "projection_engine": self.projection_engine_class(self.config).engine_type,
            "fused_attention_active": False,
        })
        return result

    def estimate_weight_cache_pair(self, M: int, K: int, N: int) -> EngineResult:
        result = self._projection_engine().estimate_weight_cache_pair(M, K, N)
        result.details.update({
            "composite_engine": self.engine_type,
            "projection_engine": self.projection_engine_class(self.config).engine_type,
            "fused_attention_active": False,
        })
        return result

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
        """QK/online-softmax/PV using the baseline array plus fusion logic.

        The base matrix cycles come from Block or OS unchanged. Fusion removes
        the external softmax pass and exposes a configurable overlap factor for
        QK/online-softmax/PV. Cached-prefix causal attention uses the exact
        visible-pair equivalent context at architecture-model fidelity.
        """
        if min(seq_q, seq_kv, head_dim, num_heads, num_kv_heads) <= 0:
            raise ValueError("fused attention dimensions must be positive")

        if causal:
            visible_pairs = (
                cached_prefix_tokens * seq_q
                + seq_q * (seq_q + 1) // 2
            )
            visible_pairs = min(visible_pairs, seq_q * seq_kv)
            compute_context = max(1, math.ceil(visible_pairs / seq_q))
        else:
            visible_pairs = seq_q * seq_kv
            compute_context = seq_kv

        base = self._projection_engine()
        qk = base.estimate(seq_q, head_dim, compute_context)
        pv = base.estimate(seq_q, compute_context, head_dim)
        baseline_matrix_cycles = num_heads * (
            qk.compute_cycles + pv.compute_cycles
        )

        q_tiles = math.ceil(seq_q / self.W)
        kv_tiles = math.ceil(compute_context / self.H)
        inline_control_cycles = (
            q_tiles * kv_tiles * num_heads * self.control_cycles_per_tile
        )
        compute_cycles = (
            math.ceil(baseline_matrix_cycles * self.fusion_overlap_factor)
            + inline_control_cycles
        )

        elem_bytes = max(1, math.ceil(attention_bits / 8))
        attention_bytes = (
            2 * num_heads * seq_q * head_dim * elem_bytes
            + 2 * num_kv_heads * seq_kv * head_dim * elem_bytes
            * max(1, kv_batch_size)
        )
        dma_cycles = math.ceil(attention_bytes / max(self.eff_bw, 1e-9))
        total_cycles = max(compute_cycles, dma_cycles)
        logical_macs = 2 * num_heads * visible_pairs * head_dim

        return EngineResult(
            compute_cycles=compute_cycles,
            dma_cycles=dma_cycles,
            total_cycles=math.ceil(total_cycles),
            utilization=min(
                1.0,
                logical_macs / max(self.H * self.W * total_cycles, 1),
            ),
            ops=logical_macs * self.ops_per_mac,
            num_tiles=num_heads * (qk.num_tiles + pv.num_tiles),
            weight_bytes=0,
            bottleneck="compute" if compute_cycles >= dma_cycles else "dma",
            details={
                "engine": self.engine_type,
                "projection_engine": base.engine_type,
                "attention_architecture": "fsa_inspired_native_fusion",
                "calibration_status": "analytical_extrapolation",
                "inline_softmax": True,
                "external_softmax_cycles": 0,
                "fusion_overlap_factor": self.fusion_overlap_factor,
                "baseline_matrix_cycles": baseline_matrix_cycles,
                "inline_control_cycles": inline_control_cycles,
                "seq_q": seq_q,
                "seq_kv": seq_kv,
                "compute_context_tokens": compute_context,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "kv_batch_size": kv_batch_size,
                "attention_bits": attention_bits,
                "causal_requested": causal,
                "cached_prefix_tokens": cached_prefix_tokens,
                "visible_pairs": visible_pairs,
                "query_position_offset_required": cached_prefix_tokens > 0,
                "mapping_compatible": True,
                "attention_bytes": attention_bytes,
            },
        )


class BlockFusedAttentionEngine(_FusedAttentionEngine):
    projection_engine_class = BlockEngine

    @property
    def engine_type(self) -> str:
        return "block_fused_attention"


class OutputStationaryFusedAttentionEngine(_FusedAttentionEngine):
    projection_engine_class = OutputStationaryEngine

    @property
    def engine_type(self) -> str:
        return "os_systolic_fused_attention"