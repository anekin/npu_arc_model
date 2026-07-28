"""Output-Stationary Engine — Gemmini 风格

参考: Gemmini (UC Berkeley), "Architectural Insights: Comparing WS and OS" (IEEE 2024)

Output-stationary: 每个 PE 持有一个输出元素，权重和激活流动后累加。
与 WS-Systolic 不同，OS 没有 diagonal pipeline fill/drain；但每个 tile 仍需
广播同步（fan-out + PE latch enable）和累加/归约周期，不是理想化的零周期。
面积代价：每个 PE 带完整 accumulator + 双缓冲，约 4× 同规模 systolic。
"""

import math
from typing import Any, Dict

from engine.mac_engine import MACEngine, EngineResult
from engine.block_engine import BROADCAST_SYNC_CYCLES, _accumulate_cycles


class OutputStationaryEngine(MACEngine):
    """Output-stationary systolic array — Gemmini 风格.

    每个 PE 持有一个 output element (M,N)。
    权重和激活流入后，在 PE 内累加为部分和。

    对 M=1 decode:
      - 无 WS 的 fill/drain 开销
      - 但每 tile 仍需 broadcast-sync + accumulate 周期
      - DMA 模型与 BlockEngine 相同
    """

    @property
    def engine_type(self) -> str:
        return "os_systolic"

    def estimate(self, M: int, K: int, N: int,
                 weight_preloaded: bool = False) -> EngineResult:
        """OS GEMM estimate.

        Tiling 与 BlockEngine 对齐：
          - K 维度切成 K_tiles = ceil(K / H)
          - N 维度切成 N_tiles = ceil(N / W)
          - 每 tile 加载 H×W 权重 + M×H 激活

        Compute 模型：
          - 无 diagonal pipeline fill/drain
          - 但每 tile 支付 K-reduction depth (H) + broadcast_sync + accumulate/reduction 周期
          - 使用 BlockEngine 等效的外部 DRAM 聚合逻辑
        """
        K_tiles = math.ceil(K / self.H)
        N_tiles = math.ceil(N / self.W)
        total_tiles = K_tiles * N_tiles

        # ── Aggregated external-DRAM accounting (matching BlockEngine:99-136) ──
        total_weight_bytes = K * N * self.w_bits // 8
        weight_dram_eff = self._dram_eff_for_bytes(total_weight_bytes)
        if weight_dram_eff <= 0:
            weight_dma_cycles = 0
        else:
            weight_dma_cycles = total_weight_bytes / (self.eff_bw * weight_dram_eff)

        act_bytes = M * K * self.a_bits // 8
        act_dma_cycles = act_bytes / self.eff_bw
        total_dma_cycles = weight_dma_cycles + act_dma_cycles
        raw_dma_cycles = int(total_dma_cycles)

        # Per-tile compute with K-reduction depth (self.H)
        per_tile_compute = self.H + BROADCAST_SYNC_CYCLES + \
            _accumulate_cycles(self.w_bits, self.a_bits)
        total_compute_cycles = per_tile_compute * total_tiles
        k_reduction_cycles = self.H * total_tiles

        # Timing: max-model (no double-buffering approximation, matches BlockEngine)
        total_cycles = max(int(total_compute_cycles), raw_dma_cycles)
        total_macs = M * K * N

        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)
        util = ideal / total_cycles if total_cycles > 0 else 0.0

        if total_dma_cycles > total_compute_cycles:
            bottleneck = "dma"
            bottleneck_reason = (
                f"DMA ({raw_dma_cycles} cycles) dominates "
                f"compute ({total_compute_cycles} cycles)"
            )
        else:
            bottleneck = "compute"
            bottleneck_reason = (
                f"Compute ({total_compute_cycles} cycles) dominates "
                f"DMA ({raw_dma_cycles} cycles)"
            )

        return EngineResult(
            compute_cycles=int(total_compute_cycles),
            dma_cycles=raw_dma_cycles,
            total_cycles=total_cycles,
            utilization=util,
            ops=total_macs,
            num_tiles=total_tiles,
            weight_bytes=int(total_weight_bytes),
            bottleneck=bottleneck,
            details={
                "K_tiles": K_tiles, "N_tiles": N_tiles,
                "per_tile_compute": per_tile_compute,
                "broadcast_sync": BROADCAST_SYNC_CYCLES,
                "k_reduction_cycles": k_reduction_cycles,
                "raw_dma_cycles": raw_dma_cycles,
                "total_compute_cycles": int(total_compute_cycles),
                "bottleneck_reason": bottleneck_reason,
                "dataflow": "output_stationary",
            },
        )

    def estimate_weight_cache_pair(self, M: int, K: int, N: int) -> EngineResult:
        """Gate+Up pair with OS weight-cache behavior.

        OS keeps output partial sums stationary, so activations can remain in the
        PE array while gate and up weight tiles are streamed through. The pair
        loads both gate and up weights but only one activation per tile, then
        performs two accumulations back-to-back.

        This is *not* the same as WS weight-cache savings (PE dual weight
        registers); it is an activation-reuse benefit.
        """
        K_tiles = math.ceil(K / self.H)
        N_tiles = math.ceil(N / self.W)
        total_tiles = K_tiles * N_tiles

        # Aggregated external-DRAM accounting (matching BlockEngine external DRAM)
        total_weight_bytes = 2 * K * N * self.w_bits // 8  # gate+up
        weight_dram_eff = self._dram_eff_for_bytes(total_weight_bytes)
        if weight_dram_eff <= 0:
            weight_dma_cycles = 0
        else:
            weight_dma_cycles = total_weight_bytes / (self.eff_bw * weight_dram_eff)

        act_bytes = M * K * self.a_bits // 8
        act_dma_cycles = act_bytes / self.eff_bw
        total_dma_cycles = weight_dma_cycles + act_dma_cycles
        raw_dma_cycles = int(total_dma_cycles)

        # Two accumulations per tile (gate + up), each with K-reduction depth
        per_tile_compute = 2 * (self.H + BROADCAST_SYNC_CYCLES +
                                _accumulate_cycles(self.w_bits, self.a_bits))
        total_compute_cycles = per_tile_compute * total_tiles
        k_reduction_cycles = 2 * self.H * total_tiles

        # Timing: max-model (matches BlockEngine external DRAM)
        total_cycles = max(int(total_compute_cycles), raw_dma_cycles)
        total_macs = M * K * N * 2

        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)
        util = ideal / total_cycles if total_cycles > 0 else 0.0

        # Activation savings: one activation load shared between gate+up per tile
        activation_savings = M * K * self.a_bits // 8 / self.eff_bw

        if total_dma_cycles > total_compute_cycles:
            bottleneck = "dma"
            bottleneck_reason = (
                f"DMA ({raw_dma_cycles} cycles) dominates "
                f"compute ({total_compute_cycles} cycles)"
            )
        else:
            bottleneck = "compute"
            bottleneck_reason = (
                f"Compute ({total_compute_cycles} cycles) dominates "
                f"DMA ({raw_dma_cycles} cycles)"
            )

        return EngineResult(
            compute_cycles=int(total_compute_cycles),
            dma_cycles=raw_dma_cycles,
            total_cycles=total_cycles,
            utilization=util,
            ops=total_macs,
            num_tiles=total_tiles,
            weight_bytes=int(total_weight_bytes),
            bottleneck=bottleneck,
            details={
                "K_tiles": K_tiles, "N_tiles": N_tiles,
                "per_tile_compute": per_tile_compute,
                "broadcast_sync": BROADCAST_SYNC_CYCLES,
                "k_reduction_cycles": k_reduction_cycles,
                "raw_dma_cycles": raw_dma_cycles,
                "total_compute_cycles": int(total_compute_cycles),
                "bottleneck_reason": bottleneck_reason,
                "activation_reuse_savings": int(activation_savings),
                "dataflow": "output_stationary",
            },
        )
