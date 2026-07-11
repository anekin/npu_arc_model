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
          - 但每 tile 支付 broadcast_sync + accumulate/reduction 周期
        """
        K_tiles = math.ceil(K / self.H)
        N_tiles = math.ceil(N / self.W)
        total_tiles = K_tiles * N_tiles

        # Per-tile data movement (identical DMA model to BlockEngine)
        tile_weight_bytes = math.ceil(self.H * self.W * self.w_bits / 8)
        tile_act_bytes = math.ceil(M * self.H * self.a_bits / 8)
        per_tile_dma = (tile_weight_bytes + tile_act_bytes) / self.eff_bw

        # Per-tile compute: realistic broadcast-sync + accumulate latency
        per_tile_compute = BROADCAST_SYNC_CYCLES + \
            _accumulate_cycles(self.w_bits, self.a_bits)

        # Double-buffering between DMA and compute
        bottleneck = max(per_tile_compute, per_tile_dma)
        first_cold = per_tile_dma + per_tile_compute

        if total_tiles > 1:
            total = int(first_cold + (total_tiles - 1) * bottleneck)
        else:
            total = int(first_cold)

        total_macs = M * K * N
        total_weight_bytes = total_tiles * (tile_weight_bytes + tile_act_bytes)
        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)
        util = ideal / total if total > 0 else 0.0

        return EngineResult(
            compute_cycles=int(per_tile_compute * total_tiles),
            dma_cycles=int(total - per_tile_compute * total_tiles),
            total_cycles=total,
            utilization=util,
            ops=total_macs,
            num_tiles=total_tiles,
            weight_bytes=total_weight_bytes,
            bottleneck="dma" if per_tile_dma > per_tile_compute else "compute",
            details={
                "K_tiles": K_tiles, "N_tiles": N_tiles,
                "per_tile_dma": round(per_tile_dma, 1),
                "per_tile_compute": per_tile_compute,
                "broadcast_sync": BROADCAST_SYNC_CYCLES,
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

        tile_weight_bytes = math.ceil(self.H * self.W * self.w_bits / 8)
        tile_act_bytes = math.ceil(M * self.H * self.a_bits / 8)

        # Both gate and up weights, but one activation tile.
        dual_weight_bytes = 2 * tile_weight_bytes
        per_tile_dma = (dual_weight_bytes + tile_act_bytes) / self.eff_bw

        # Two accumulations per tile.
        per_tile_compute = 2 * (BROADCAST_SYNC_CYCLES +
                                _accumulate_cycles(self.w_bits, self.a_bits))

        bottleneck = max(per_tile_compute, per_tile_dma)
        first_cold = per_tile_dma + per_tile_compute

        if total_tiles > 1:
            total = int(first_cold + (total_tiles - 1) * bottleneck)
        else:
            total = int(first_cold)

        total_macs = M * K * N * 2
        total_weight_bytes = total_tiles * (dual_weight_bytes + tile_act_bytes)
        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)
        util = ideal / total if total > 0 else 0.0

        activation_savings = total_tiles * tile_act_bytes / self.eff_bw

        return EngineResult(
            compute_cycles=int(per_tile_compute * total_tiles),
            dma_cycles=int(total - per_tile_compute * total_tiles),
            total_cycles=total,
            utilization=util,
            ops=total_macs,
            num_tiles=total_tiles,
            weight_bytes=total_weight_bytes,
            bottleneck="dma" if per_tile_dma > per_tile_compute else "compute",
            details={
                "K_tiles": K_tiles, "N_tiles": N_tiles,
                "per_tile_dma": round(per_tile_dma, 1),
                "per_tile_compute": per_tile_compute,
                "activation_reuse_savings": int(activation_savings),
                "dataflow": "output_stationary",
            },
        )
