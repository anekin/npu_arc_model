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
        """Roofline estimate for a spatial HxW broadcast array.

        H MAC rows reduce one K tile in parallel and W columns produce W
        outputs.  This makes array height a real performance lever instead of
        charging K serial cycles regardless of H.
        """
        K_tiles = math.ceil(K / self.H)
        N_tiles = math.ceil(N / self.W)
        total_tiles = M * K_tiles * N_tiles
        per_tile_compute = BROADCAST_SYNC_CYCLES + _accumulate_cycles(
            self.w_bits, self.a_bits)
        total_compute = total_tiles * per_tile_compute

        weight_bytes = K * N * self.w_bits // 8
        activation_bytes = M * K * self.a_bits // 8
        if weight_preloaded:
            weight_cycles = 0.0
        elif self.weight_resident:
            weight_cycles = weight_bytes / max(self.eff_bw, 1e-9)
        else:
            # ``eff_bw`` already includes the scenario-level LPDDR protocol
            # efficiency. Do not apply a second engine-specific penalty.
            weight_cycles = weight_bytes / max(self.eff_bw, 1e-9)
        activation_cycles = activation_bytes / max(self.eff_bw, 1e-9)
        total_dma = weight_cycles + activation_cycles
        total_cycles = max(total_compute, total_dma)
        total_macs = M * K * N

        return EngineResult(
            compute_cycles=math.ceil(total_compute),
            dma_cycles=math.ceil(total_dma),
            total_cycles=math.ceil(total_cycles),
            utilization=min(1.0, total_macs / max(self.H * self.W * total_cycles, 1)),
            ops=total_macs * self.ops_per_mac,
            num_tiles=total_tiles,
            weight_bytes=weight_bytes,
            bottleneck="dma" if total_dma > total_compute else "compute",
            details={
                "M_rows": M, "K_tiles": K_tiles, "N_tiles": N_tiles,
                "per_tile_compute": per_tile_compute,
                "weight_resident": self.weight_resident,
                "roofline_overlap": True,
            },
        )

    def estimate_weight_cache_pair(self, M: int, K: int, N: int) -> EngineResult:
        """Evaluate Gate+Up with shared activation traffic and two weights."""
        K_tiles = math.ceil(K / self.H)
        N_tiles = math.ceil(N / self.W)
        total_tiles = M * K_tiles * N_tiles
        per_tile_compute = 2 * (BROADCAST_SYNC_CYCLES + _accumulate_cycles(
            self.w_bits, self.a_bits))
        total_compute = total_tiles * per_tile_compute
        weight_bytes = 2 * K * N * self.w_bits // 8
        activation_bytes = M * K * self.a_bits // 8
        if self.weight_resident:
            weight_cycles = weight_bytes / max(self.eff_bw, 1e-9)
        else:
            weight_cycles = weight_bytes / max(self.eff_bw, 1e-9)
        total_dma = weight_cycles + activation_bytes / max(self.eff_bw, 1e-9)
        total_cycles = max(total_compute, total_dma)
        total_macs = 2 * M * K * N

        # Weight-cache support is optional. The scheduler falls back to two
        # ordinary GEMMs when combining larger weight streams reduces DRAM
        # efficiency and would make the pair slower.
        single = self.estimate(M, K, N)
        if total_cycles >= 2 * single.total_cycles:
            return EngineResult(
                compute_cycles=2 * single.compute_cycles,
                dma_cycles=2 * single.dma_cycles,
                total_cycles=2 * single.total_cycles,
                utilization=single.utilization,
                ops=2 * single.ops,
                num_tiles=2 * single.num_tiles,
                weight_bytes=2 * single.weight_bytes,
                bottleneck=single.bottleneck,
                details={"scheduler_fallback": "two_independent_gemms"},
            )
        return EngineResult(
            compute_cycles=math.ceil(total_compute),
            dma_cycles=math.ceil(total_dma),
            total_cycles=math.ceil(total_cycles),
            utilization=min(1.0, total_macs / max(self.H * self.W * total_cycles, 1)),
            ops=total_macs * self.ops_per_mac,
            num_tiles=total_tiles,
            weight_bytes=weight_bytes,
            bottleneck="dma" if total_dma > total_compute else "compute",
            details={
                "K_tiles": K_tiles, "N_tiles": N_tiles,
                "per_tile_compute": per_tile_compute,
                "shared_activation": True,
                "weight_resident": self.weight_resident,
            },
        )
