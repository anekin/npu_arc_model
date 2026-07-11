"""Block Engine — 全并行 MAC 阵列，纯空间映射

与 systolic 的本质区别:
  - 无 diagonal pipeline fill/drain: 数据广播到所有 MAC
  - 但仍有广播同步 + 累加/归约开销，不是理想化的 1 cycle/tile
  - 瓶颈: 通常是 DMA（加载权重+激活），但 compute 不再是零
  - 代价: 全互连 crossbar 广播总线，面积 3-5× 同规模 systolic

参考: NVIDIA Tensor Core, Google TPUv4 的 vector-matrix unit
"""

import math
from typing import Any, Dict

from engine.mac_engine import MACEngine, EngineResult


# Realistic broadcast-pipeline constants for BlockEngine.
# Block engine broadcasts weights + activations to all PEs simultaneously
# (no diagonal fill like systolic), but it still pays:
#   - broadcast synchronization overhead (fan-out + PE latch enable)
#   - accumulate/reduction latency (wider precision -> more cycles)
BROADCAST_SYNC_CYCLES = 2  # ~2 cycles for weight+activation fan-out


def _accumulate_cycles(w_bits: int, a_bits: int) -> int:
    """Accumulate/reduction cycles for block-engine MAC tile.

    Empirical mapping: INT4/INT8 mixed ~2 cycles, INT8/INT8 ~3 cycles,
    capped at 3 and floored at 1.  This replaces the old "1 cycle per tile"
    assumption that made BlockEngine ~8× faster than systolic reality.
    """
    return max(1, min(3, (w_bits + a_bits) // 8 + 1))


class BlockEngine(MACEngine):
    """Block MAC engine — all MACs fire in parallel per tile.

    Dataflow: broadcast weights + activations to all MAC units.
    Each tile processes H×K submatrix with broadcast-sync + accumulate
    latency; the old "1 cycle per tile" model was over-optimistic.

    Area model:
      - MAC array: same as systolic (H×W PEs)
      - Crossbar: ~2× MAC area for broadcast interconnect
      - Register file: ~1× MAC area for local weight storage
      - Total: ~4× systolic per MAC (conservative)
    """

    @property
    def engine_type(self) -> str:
        return "block"


    def estimate(self, M: int, K: int, N: int,
                 weight_preloaded: bool = False) -> EngineResult:
        """Block GEMM estimate. Supports on-chip 3D DRAM (weight resident) mode."""
        N_tiles = math.ceil(N / self.W)

        total_macs = M * K * N
        total_weight_bytes = K * N * self.w_bits // 8

        if self.weight_resident:
            # ── On-chip 3D DRAM: weights resident, no K-tiling ──
            # H handles batch (M) dimension. Each PE does K MACs for reduction.
            M_tiles = math.ceil(M / self.H)
            N_tiles = math.ceil(N / self.W)
            total_tiles = M_tiles * N_tiles

            # Per tile: K reduction cycles (one MAC/cycle/PE) + overhead
            per_tile_compute = K + BROADCAST_SYNC_CYCLES + \
                _accumulate_cycles(self.w_bits, self.a_bits)
            total_compute = per_tile_compute * total_tiles

            # Weight streaming from on-chip memory (once per token)
            act_bytes = M * K * self.a_bits // 8
            weight_stream_cycles = (total_weight_bytes + act_bytes) / self.on_chip_bw

            total_cycles = max(total_compute, weight_stream_cycles)

            return EngineResult(
                compute_cycles=int(total_compute),
                dma_cycles=int(weight_stream_cycles),
                total_cycles=int(total_cycles),
                utilization=total_macs / (self.peak_macs_per_cycle * total_cycles) if total_cycles > 0 else 0,
                ops=total_macs,
                num_tiles=total_tiles,
                weight_bytes=int(total_weight_bytes),
                bottleneck="compute" if total_compute >= weight_stream_cycles else "on_chip_bw",
                details={
                    "M_tiles": M_tiles,
                    "N_tiles": N_tiles,
                    "per_tile_compute": per_tile_compute,
                    "on_chip_mode": True,
                    "on_chip_bw": self.on_chip_bw,
                },
            )

        # ── External DRAM: time-multiplexed M ──
        # M tokens processed sequentially (M passes) sharing one weight load.
        M_tiles = math.ceil(M / self.H)   # M≤H → M_tiles=1 (one weight load pass)
        K_tiles = math.ceil(K / self.H)
        N_tiles = math.ceil(N / self.W)
        per_pass_tiles = K_tiles * N_tiles   # tiles per single-token pass

        # Per-tile weight (for reporting only)
        tile_weight_bytes = math.ceil(self.H * self.W * self.w_bits / 8)

        # Total weight for this matmul (loaded once per M-tile)
        total_weight_bytes = K * N * self.w_bits // 8

        # SRAM efficiency
        weight_dram_eff = self._dram_eff_for_bytes(total_weight_bytes)
        if weight_dram_eff <= 0:
            weight_dma_cycles = 0
        else:
            weight_dma_cycles = total_weight_bytes / (self.eff_bw * weight_dram_eff)

        # Activation DMA: per-token (one activation load per M pass)
        act_bytes_per_token = K * self.a_bits // 8
        act_dma_cycles = M * act_bytes_per_token / self.eff_bw

        total_dma_cycles = M_tiles * weight_dma_cycles + act_dma_cycles

        # Compute: M sequential passes, each processing one token
        per_tile_compute = self.H + BROADCAST_SYNC_CYCLES + \
            _accumulate_cycles(self.w_bits, self.a_bits)
        total_compute = M * per_tile_compute * per_pass_tiles

        total_cycles = max(total_compute, total_dma_cycles)
        total_macs = M * K * N
        total_tiles = M * per_pass_tiles   # total tiles across all passes

        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)
        util = ideal / total_cycles if total_cycles > 0 else 0.0

        return EngineResult(
            compute_cycles=int(total_compute),
            dma_cycles=int(total_dma_cycles),
            total_cycles=int(total_cycles),
            utilization=util,
            ops=total_macs,
            num_tiles=total_tiles,
            weight_bytes=int(total_weight_bytes),
            bottleneck="dma" if total_dma_cycles > total_compute else "compute",
            details={
                "M_tiles": M_tiles, "K_tiles": K_tiles, "N_tiles": N_tiles,
                "per_tile_compute": per_tile_compute,
                "weight_dram_eff": round(weight_dram_eff, 3),
                "token_multiplex": True,
            },
        )

    def estimate_weight_cache_pair(self, M: int, K: int, N: int) -> EngineResult:
        """Gate+Up with block-engine weight cache, time-multiplexed M.

        Time-multiplexed dataflow: weights (gate+up) loaded ONCE from DRAM,
        then M sequential compute passes through the array (one per token).
        Each pass processes both gate and up tiles with accumulator reset
        between tokens.  Activation is loaded per-token, not batched.
        """
        M_tiles = math.ceil(M / self.H)   # M≤H → one weight load
        K_tiles = math.ceil(K / self.H)
        N_tiles = math.ceil(N / self.W)
        per_pass_tiles = K_tiles * N_tiles
        total_tiles = M * per_pass_tiles  # for reporting

        tile_weight_bytes = math.ceil(self.H * self.W * self.w_bits / 8)
        dual_weight_bytes = 2 * tile_weight_bytes

        # Per-token activation per tile
        tile_act_per_token = math.ceil(self.H * self.a_bits / 8)

        # Weight cache: load gate+up weights once per M-tile
        total_weight_bytes = per_pass_tiles * dual_weight_bytes
        total_act_bytes = M * per_pass_tiles * tile_act_per_token

        # DMA: weight loaded once, activation loaded per token
        weight_dma_cycles = total_weight_bytes / self.eff_bw
        act_dma_cycles = total_act_bytes / self.eff_bw
        total_dma_cycles = M_tiles * weight_dma_cycles + act_dma_cycles

        # Compute: M sequential passes, each processing gate+up
        per_tile_compute = 2 * (BROADCAST_SYNC_CYCLES +
                                _accumulate_cycles(self.w_bits, self.a_bits))
        total_compute = M * per_tile_compute * per_pass_tiles

        total_cycles = max(total_compute, total_dma_cycles)
        total_macs = M * K * N * 2

        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)
        util = ideal / total_cycles if total_cycles > 0 else 0.0

        # Savings: one activation load saved per token (no separate gate+up loads).
        activation_savings = M * per_pass_tiles * tile_act_per_token / self.eff_bw

        return EngineResult(
            compute_cycles=int(total_compute),
            dma_cycles=int(total_dma_cycles),
            total_cycles=int(total_cycles),
            utilization=util,
            ops=total_macs,
            num_tiles=total_tiles,
            weight_bytes=int(total_weight_bytes),
            bottleneck="dma" if total_dma_cycles > total_compute else "compute",
            details={
                "M_tiles": M_tiles, "K_tiles": K_tiles, "N_tiles": N_tiles,
                "per_tile_compute": per_tile_compute,
                "weight_cache_savings": int(activation_savings),
                "token_multiplex": True,
            },
        )
