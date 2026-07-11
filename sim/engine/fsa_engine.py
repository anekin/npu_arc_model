"""FSA Engine — inline FlashAttention on systolic array.

FSA (Fusing FlashAttention within a Single Systolic Array) executes
complete FlashAttention — including softmax — within a single systolic
array pass. No external SFU or Vector unit needed.

This engine models:
  - Matmul throughput: same as a weight-stationary systolic array
  - Attention: inline softmax via CMP columns + PE Split units
  - Area overhead: +12% vs baseline systolic array
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from engine.mac_engine import EngineResult, MACEngine


class FSAEngine(MACEngine):
    """FSA: systolic array with inline softmax for FlashAttention.

    Configuration keys (under mac_engine):
      array_height, array_width : systolic array dimensions
      frequency_mhz             : clock frequency
      weight_precision_bits     : INT4
      activation_precision_bits : INT8
      fsa_softmax_overhead      : extra cycles per row for inline softmax (default 5)
      fsa_area_overhead_pct     : area overhead vs baseline SA (default 12%)
    """

    def _parse_config(self, config: Dict[str, Any]):
        super()._parse_config(config)
        mac = config.get("mac_engine", config.get("mxu", {}))
        self.softmax_overhead = int(mac.get("fsa_softmax_overhead", 5))
        self._area_overhead = float(mac.get("fsa_area_overhead_pct", 12.0))

    @property
    def engine_type(self) -> str:
        return "fsa"

    def estimate(self, M: int, K: int, N: int,
                 weight_preloaded: bool = False) -> EngineResult:
        """Estimate matmul with K-dimension tiling (weight-stationary systolic).

        Tiling:
          - K-tiles: ceil(K/H) — reduction dimension split across array rows
          - M-tiles: ceil(M/H) — batch dimension (usually 1 for decode)
          - N-tiles: ceil(N/W) — output dimension

        Pipeline per (K,N)-tile: H (weight load) + M (act stream) + W (drain)
        Activations broadcast across N-tiles, read once per K-tile.
        """
        H, W = self.H, self.W
        macs = M * K * N * self.ops_per_mac

        # K-dimension tiling (critical for large models)
        tiles_k = max(1, (K + H - 1) // H)
        tiles_m = max(1, (M + H - 1) // H)
        tiles_n = max(1, (N + W - 1) // W)
        total_tiles = tiles_k * tiles_m * tiles_n

        # Pipeline depth per K-tile: weight load + act stream + propagate
        pipe_depth = H + M + W

        # Compute: tiles_k × tiles_m × tiles_n × pipe_depth
        # N-tiles and M-tiles are sequential within each K-tile
        compute_cycles = total_tiles * pipe_depth

        # DMA: weights (K×N, loaded once per K-tile → K×N total, same as no tiling)
        weight_bytes = K * N * self.w_bits // 8
        # Activations: M×K total, broadcast across N-tiles (read once)
        act_bytes = M * K * self.a_bits // 8

        # SRAM-aware DRAM efficiency
        dram_eff = self._dram_eff_for_bytes(weight_bytes)
        if dram_eff <= 0:
            effective_weight_bytes = 0
        else:
            effective_weight_bytes = weight_bytes

        dma_total_bytes = effective_weight_bytes + act_bytes
        dma_cycles = max(
            dma_total_bytes / (self.eff_bw * max(dram_eff, 0.01)),
            compute_cycles * 0.05
        )

        total_cycles = max(compute_cycles, dma_cycles)
        peak = self.peak_macs_per_cycle
        utilization = macs / (peak * total_cycles) if total_cycles > 0 else 0

        return EngineResult(
            compute_cycles=int(compute_cycles),
            dma_cycles=int(dma_cycles),
            total_cycles=int(total_cycles),
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
                "inline_softmax": True,
                "dram_eff": round(dram_eff, 3),
            },
        )

    def estimate_weight_cache_pair(self, M: int, K: int, N: int) -> EngineResult:
        """FFN gate+up weight-cache: load gate weights, reuse for up."""
        # Same as single estimate but with shared weight load
        r = self.estimate(M, K, N)
        r.details["weight_cache"] = True
        return r

    def estimate_attention(
        self, seq_q: int, seq_kv: int, head_dim: int,
        num_heads: int = 1, num_kv_heads: int = 1,
    ) -> EngineResult:
        """Estimate full FlashAttention including inline softmax.

        FSA's key advantage: softmax is done INLINE in the systolic array
        via CMP columns and PE Split units. Only ~5 extra cycles per row.

        Returns:
            EngineResult with attention latency and utilization.
        """
        H, W = self.H, self.W
        overhead = self.softmax_overhead

        # Phase 1: S = QK^T + inline softmax (rowmax/exp/rowsum)
        # Tiling: ceil(seq_q / H) × ceil(seq_kv / W) tiles
        tiles_q = (seq_q + H - 1) // H
        tiles_kv = (seq_kv + W - 1) // W
        tiles_p1 = tiles_q * tiles_kv

        # Per-tile: H MAC cycles + overhead for CMP/Split inline ops
        phase1_cycles = tiles_p1 * (H + overhead) if tiles_p1 > 0 else 0

        # Phase 2: O = PV  (P: seq_q×seq_kv, V: seq_kv×head_dim)
        # Tiling: ceil(seq_q / H) × ceil(head_dim / W)
        tiles_oq = (seq_q + H - 1) // H
        tiles_hd = (head_dim + W - 1) // W
        tiles_p2 = tiles_oq * tiles_hd
        phase2_cycles = tiles_p2 * (H + 3) if tiles_p2 > 0 else 0  # slightly less overhead

        total_compute = (phase1_cycles + phase2_cycles) * num_kv_heads

        # Total MAC ops (both matmuls)
        mac_ops = (
            2 * num_kv_heads * seq_q * seq_kv * head_dim +
            2 * num_kv_heads * seq_q * head_dim * seq_kv
        )

        # DMA for K, V, Q loading
        elem_bytes = self.a_bits // 8
        dma_bytes = (
            num_kv_heads * seq_kv * head_dim * elem_bytes * 2 +  # K + V
            num_heads * seq_q * head_dim * elem_bytes             # Q
        )
        dma_cycles = dma_bytes / self.eff_bw if self.eff_bw > 0 else 0

        total_cycles = max(total_compute, int(dma_cycles))
        peak = self.peak_macs_per_cycle
        utilization = mac_ops / (peak * max(total_cycles, 1)) if total_cycles > 0 else 0

        return EngineResult(
            compute_cycles=int(total_compute),
            dma_cycles=int(dma_cycles),
            total_cycles=int(total_cycles),
            utilization=min(utilization, 1.0),
            ops=mac_ops,
            num_tiles=tiles_p1 + tiles_p2,
            weight_bytes=0,  # no weight — attention is activation-driven
            bottleneck="compute" if total_compute >= dma_cycles else "dma",
            details={
                "tiles_phase1": tiles_p1,
                "tiles_phase2": tiles_p2,
                "engine": "fsa",
                "inline_softmax": True,
                "softmax_overhead": overhead,
                "seq_q": seq_q,
                "seq_kv": seq_kv,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
            },
        )
